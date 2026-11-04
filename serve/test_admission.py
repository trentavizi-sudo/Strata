"""`"admission": "pressure"` (serve/admission.py): the policy's rules, its config forms, and the server holding a
new request while the fake engine's BXP samples say the windows are miss-bound - and admitting it after max_wait_s."""
import time
import unittest
import urllib.request
import json
import threading

from serve.admission import PressurePolicy
from serve.test_parallel import FakeEngineCase


def view(pressure_ms=4.5, window_ms=20.0, age_s=0.0, windows=10):
    return {"windows": windows, "ms_per_window": window_ms, "miss_cost_ms_window": pressure_ms, "age_s": age_s}


class Rules(unittest.TestCase):
    def test_pressure_is_the_miss_share_of_the_window(self):
        self.assertEqual(PressurePolicy.pressure(view(5.0, 20.0)), 0.25)
        self.assertIsNone(PressurePolicy.pressure(None))
        self.assertIsNone(PressurePolicy.pressure({"windows": 3, "ms_per_window": 0, "miss_cost_ms_window": 1}))

    def test_hold_only_when_every_condition_holds(self):
        p = PressurePolicy(threshold=0.2, max_wait_s=5.0, stale_s=3.0, min_windows=4)
        self.assertEqual(p.decide(view(5.0, 20.0), 1, 0.0), (True, {"pressure": 0.25, "threshold": 0.2, "busy_slots": 1,
                                                                   "waited_s": 0.0, "why": "pressure"}))
        self.assertEqual(p.decide(view(5.0, 20.0), 0, 0.0)[1]["why"], "nothing to protect")   # it would run alone
        self.assertEqual(p.decide(None, 2, 0.0)[1]["why"], "no telemetry")
        self.assertEqual(p.decide(view(5.0, 20.0, age_s=3.5), 2, 0.0)[1]["why"], "telemetry stale")
        self.assertEqual(p.decide(view(5.0, 20.0, windows=3), 2, 0.0)[1]["why"], "too few windows")
        self.assertEqual(p.decide(view(4.0, 20.0), 2, 0.0)[1]["why"], "below threshold")     # 0.2 is not above 0.2
        self.assertEqual(p.decide(view(5.0, 20.0), 2, 5.0)[1]["why"], "waited max_wait_s")
        self.assertFalse(p.decide(view(5.0, 20.0), 2, 5.0)[0])
        self.assertTrue(p.decide(view(5.0, 20.0), 2, 4.9)[0])
        self.assertFalse(PressurePolicy(min_busy=2).decide(view(19.0, 20.0), 1, 0.0)[0])

    def test_config_forms(self):
        self.assertIsNone(PressurePolicy.from_config(None))
        self.assertIsNone(PressurePolicy.from_config(False))
        p = PressurePolicy.from_config("pressure")
        self.assertEqual(p.describe(), {"policy": "pressure", "threshold": 0.5, "max_wait_s": 30.0, "stale_s": 3.0,
                                        "min_windows": 4, "min_busy": 1})
        p = PressurePolicy.from_config({"policy": "pressure", "threshold": 0.3, "max_wait_s": 10, "min_busy": 2})
        self.assertEqual((p.threshold, p.max_wait_s, p.min_busy, p.stale_s), (0.3, 10.0, 2, 3.0))
        for bad in ("other", {"policy": "x"}, {"policy": "pressure", "threshold": 0}, {"policy": "pressure", "threshold": 1.5},
                    {"policy": "pressure", "max_wait_s": -1}, {"policy": "pressure", "stale_s": True},
                    {"policy": "pressure", "min_windows": "4"}, 7):
            with self.assertRaises(ValueError, msg=repr(bad)):
                PressurePolicy.from_config(bad)


class Held(FakeEngineCase):
    """The fake engine prints a BXP sample (miss cost 4.5 ms of a 20 ms window: pressure 0.225) when its slots go idle."""

    def warm(self):
        """Two requests together, so the engine runs batch windows and leaves a sample behind."""
        replies = self.race("first LONGREPLY", "second LONGREPLY", delay=0.05)
        self.assertTrue(all("ok, " in r for r, _ in replies.values()))
        deadline = time.monotonic() + 2
        while not self.engine.expert_samples and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.engine.expert_samples, "no BXP sample arrived")

    def test_a_second_request_is_held_then_admitted(self):
        self.start(2, expert_sample=True)
        self.engine.admission = PressurePolicy(threshold=0.1, max_wait_s=0.6, stale_s=60.0, min_windows=1)
        self.warm()
        self.assertEqual(self.engine.admission_held, 0)         # no sample existed when the second one came
        res = self.race("third LONGREPLY", "fourth LONGREPLY", delay=0.05)
        self.assertTrue(all("ok, " in r for r, _ in res.values()))
        self.assertEqual(self.engine.admission_held, 1)         # the fourth waited while the third ran
        self.assertGreaterEqual(self.engine.admission_hold_s, 0.5)
        self.assertLess(self.engine.admission_hold_s, 3.0)
        self.assertEqual(self.engine.admission_last["threshold"], 0.1)
        m = self.get("/metrics")["live"]["admission"]
        self.assertEqual(m["held"], 1)
        self.assertEqual(m["policy"], "pressure")
        work = self.get("/v1/telemetry")["work"]
        self.assertEqual(work["admission"]["held"], 1)

    def test_below_the_threshold_nothing_is_held(self):
        self.start(2, expert_sample=True)
        self.engine.admission = PressurePolicy(threshold=0.9, max_wait_s=0.6, stale_s=60.0, min_windows=1)
        self.warm()
        self.race("third LONGREPLY", "fourth LONGREPLY", delay=0.05)
        self.assertEqual(self.engine.admission_held, 0)
        self.assertEqual(self.engine.admission_last["why"], "below threshold")

    def test_a_request_alone_is_never_held(self):
        self.start(2, expert_sample=True)
        self.engine.admission = PressurePolicy(threshold=0.1, max_wait_s=5.0, stale_s=60.0, min_windows=1)
        self.warm()
        t0 = time.monotonic()
        self.chat("fifth")
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(self.engine.admission_held, 0)
        self.assertEqual(self.engine.admission_last["why"], "nothing to protect")

    def test_a_held_request_keeps_its_queue_deadline(self):
        """A hold never outlives the server's queue timeout: the request still gets its 429 on time."""
        self.start(2, expert_sample=True)
        self.engine.admission = PressurePolicy(threshold=0.1, max_wait_s=30.0, stale_s=60.0, min_windows=1)
        self.svc.max_queue = 4
        self.svc.queue_timeout_s = 0.4
        self.warm()
        out = {}

        def go(k, text):
            try:
                out[k] = ("ok", self.chat(text, max_tokens=64))
            except urllib.error.HTTPError as e:
                out[k] = ("http", e.code, json.loads(e.read().decode()))
        a = threading.Thread(target=go, args=("a", "sixth LONGREPLY"))
        a.start()
        time.sleep(0.05)
        t0 = time.monotonic()
        go("b", "seventh")
        a.join(10)
        self.assertEqual(out["a"][0], "ok")
        self.assertEqual(out["b"][0], "http")
        self.assertEqual(out["b"][1], 429)
        self.assertLess(time.monotonic() - t0, 3.0)


if __name__ == "__main__":
    unittest.main()
