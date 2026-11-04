"""Admission by expert-miss pressure (opt-in; `"admission": "pressure"` in the model's config).

With batch slots (`"parallel": N`) the server admits a request as soon as a slot is free.  On a card whose expert
cache holds a small share of the model, every admitted conversation routes to experts of its own, the CPU pool and the
link carry more misses per window, and every running request slows down.  This policy reads the engine's own account
of a window (the `BXP` telemetry: serve/server.py `expert_window_view`) and HOLDS a new admission while the running
windows are miss-bound.  It never rejects: a held request stays queued and goes in once the pressure drops or it has
waited `max_wait_s`.  A request that would run alone is never held (there is nothing to protect).

The signal is scale-free: `pressure` = the miss cost of a window (max(CPU pool ms, link estimate ms) + the host's wait
for the GPU) over the window's wall time, so a threshold means the same share of the inter-token time on a 12 GB and
on a 24 GB card.  It is the pressure NOW, not a prediction of the new request's own working set: that would need its
prompt read first, which is the cost an admission pays.

Everything here is pure (no server state) so it can be moved into the engine without a rewrite.
"""
from __future__ import annotations

DEFAULTS = {"threshold": 0.5, "max_wait_s": 30.0, "stale_s": 3.0, "min_windows": 4, "min_busy": 1}


class PressurePolicy:
    def __init__(self, threshold: float = 0.5, max_wait_s: float = 30.0, stale_s: float = 3.0, min_windows: int = 4,
                 min_busy: int = 1):
        self.threshold = float(threshold)     # hold while pressure is above this share of the window's wall time
        self.max_wait_s = float(max_wait_s)   # a held request is admitted after this many seconds whatever the pressure
        self.stale_s = float(stale_s)         # a sample older than this says nothing about now: no hold
        self.min_windows = int(min_windows)   # fewer windows in the recent samples: too little to judge, no hold
        self.min_busy = int(min_busy)         # hold only with at least this many slots busy (1: never hold a request that would run alone)

    @classmethod
    def from_config(cls, section) -> "PressurePolicy | None":
        """`"pressure"`, or `{"policy": "pressure", "threshold": 0.5, "max_wait_s": 30, "stale_s": 3, "min_windows": 4,
        "min_busy": 1}` (every key optional).  None for anything else, with the reason in `why` (a string) on the
        exception so the caller can print it."""
        if section is None or section is False:
            return None
        if isinstance(section, str):
            section = {"policy": section}
        if not isinstance(section, dict) or section.get("policy") != "pressure":
            raise ValueError(f'"admission" must be "pressure" or {{"policy": "pressure", ...}}, not {section!r}')
        kw = {}
        for k, default in DEFAULTS.items():
            v = section.get(k, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ValueError(f'"admission".{k} must be a number of 0 or more, not {v!r}')
            kw[k] = v
        if not 0.0 < kw["threshold"] <= 1.0:
            raise ValueError(f'"admission".threshold must be above 0 and at most 1 (a share of the window), not {kw["threshold"]!r}')
        return cls(**kw)

    def describe(self) -> dict:
        return {"policy": "pressure", "threshold": self.threshold, "max_wait_s": self.max_wait_s, "stale_s": self.stale_s,
                "min_windows": self.min_windows, "min_busy": self.min_busy}

    @staticmethod
    def pressure(view: dict | None) -> float | None:
        """The share of a window's wall time that is miss cost, from an `expert_window_view`; None without one."""
        if not view or not view.get("ms_per_window"):
            return None
        return float(view["miss_cost_ms_window"]) / float(view["ms_per_window"])

    def decide(self, view: dict | None, busy_slots: int, waited_s: float) -> tuple[bool, dict]:
        """(hold, why).  `view`: the server's expert_window_view (or None), `busy_slots`: slots decoding now,
        `waited_s`: how long this request has waited already."""
        p = self.pressure(view)
        info = {"pressure": round(p, 4) if p is not None else None, "threshold": self.threshold, "busy_slots": busy_slots,
                "waited_s": round(waited_s, 3)}
        if busy_slots < self.min_busy:
            return False, {**info, "why": "nothing to protect"}
        if view is None or p is None:
            return False, {**info, "why": "no telemetry"}
        if view.get("age_s", 0.0) > self.stale_s:
            return False, {**info, "why": "telemetry stale"}
        if view.get("windows", 0) < self.min_windows:
            return False, {**info, "why": "too few windows"}
        if p <= self.threshold:
            return False, {**info, "why": "below threshold"}
        if waited_s >= self.max_wait_s:
            return False, {**info, "why": "waited max_wait_s"}
        return True, {**info, "why": "pressure"}
