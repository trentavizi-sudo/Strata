#!/usr/bin/env python3
"""Aggregate throughput under concurrent requests, with the latency and expert-cache figures beside it.

    python tools/concurrency_bench.py --url http://127.0.0.1:8080 --clients 1 2 4 8 --out run.json
    python tools/concurrency_bench.py --config strata-<model>.json --parallel 4 --clients 1 2 4 8 --out run.json

For every concurrency level N the harness sends N requests at once, each with its own prompt of about
--prompt-tokens tokens carrying a hidden access code (tools/dgx_concurrent.py's prompts), streams the replies and
records, per level:

  aggregate_tok_s     completion tokens of all N requests over the wall time from the first send to the last reply
  ttft_s              time to the first content token, p50 / p95 / max over the N requests
  itl_ms              time between consecutive content tokens, p50 / p95 pooled over all tokens of all requests
  requests            the server's own /metrics history rows of these requests: hit_rate (VRAM share of the routed
                      experts), pcie_share, decode_tok_s, prompt_ms, decode_ms, first_engine_token_s
  expert_window       with STRATA_EXPERT_TELEMETRY=1 on the engine (config "expert_telemetry": true): the engine's
                      BXP totals over the level - PCIe bytes per generated token and per window, CPU pool ms and
                      GPU-reach wait ms per window, misses per window, the hit rate over routed entries
  own_code_named      how many replies name their own code and no other request's (a batch that mixed two requests'
                      state fails this); all_coherent: no reply is empty or one repeated word

The level N=1 is the single-request baseline: the acceptance rule for every concurrency change is an aggregate gain at
2/4/8 with TTFT and ITL at N=1 unchanged.  --rounds R repeats each level R times (the server stays up; the rows are
kept apart, with the median of the aggregate rates in the summary).  Nothing here measures the GPU: every number is the
server's or the client's clock.  --no-check-codes skips the code check (a test engine that cannot read the prompt).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dgx_concurrent import coherent, distinct_prompt, serve  # noqa: E402


def percentile(values, q: float):
    """The q-th percentile (0..100) by the nearest-rank rule; None without values."""
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1)))))
    return xs[k]


def _get(url: str, path: str, api_key: str | None = None, timeout: float = 30.0):
    req = urllib.request.Request(url.rstrip("/") + path, headers=_headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _headers(api_key: str | None, json_body: bool = False) -> dict:
    h = {"Content-Type": "application/json"} if json_body else {}
    if api_key:
        h["Authorization"] = "Bearer " + api_key
    return h


def stream_one(url: str, prompt: str, max_new: int, api_key: str | None = None, timeout: float = 3600.0) -> dict:
    """One streamed chat completion: the text, the time of every content token, the usage the server sent."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_new, "temperature": 0,
                       "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=_headers(api_key, True))
    t0 = time.perf_counter()
    stamps: list[float] = []
    text, finish, usage, timings = [], None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            c = json.loads(payload)
            if c.get("error"):
                raise RuntimeError(str(c["error"]))
            now = time.perf_counter()
            for ch in c.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    stamps.append(now)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            usage = c.get("usage") or usage
            timings = c.get("timings") or timings
    t1 = time.perf_counter()
    n = (usage or {}).get("completion_tokens")
    return {"text": "".join(text), "finish": finish, "latency_s": t1 - t0, "n": n if n is not None else len(stamps),
            "ttft_s": (stamps[0] - t0) if stamps else None,
            "itl_ms": [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])],
            "chunks": len(stamps), "usage": usage, "timings": timings, "error": None}


def _run_client(url, prompt, max_new, api_key, out, i):
    t0 = time.perf_counter()
    try:
        out[i] = stream_one(url, prompt, max_new, api_key)
    except Exception as e:                   # an error is a finding of the run, not a crash of the harness
        out[i] = {"error": f"{type(e).__name__}: {e}", "latency_s": time.perf_counter() - t0, "n": 0, "text": "",
                  "finish": None, "ttft_s": None, "itl_ms": [], "chunks": 0, "usage": None, "timings": None}


def _expert_totals(metrics: dict) -> dict | None:
    x = (metrics.get("live") or {}).get("expert_window")
    return (x or {}).get("total") if x else None


def _delta(a: dict | None, b: dict | None) -> dict | None:
    """b - a per key; an engine that had emitted no sample before the level counts from zero."""
    if not b:
        return None
    a = a or {}
    return {k: b.get(k, 0) - a.get(k, 0) for k in b}


def run_level(url: str, clients: int, prompt_tokens: int, max_new: int, api_key: str | None = None,
              check_codes: bool = True, label: str = "", round_no: int = 0) -> dict:
    """One concurrency level: N streamed requests at once, and what the server's /metrics says about them."""
    before = _get(url, "/metrics?requests=all", api_key)
    hist0 = {r["id"] for r in before.get("requests") or []}
    prompts = [distinct_prompt(prompt_tokens, i + 100 * round_no) for i in range(clients)]
    out: list = [None] * clients
    t0 = time.perf_counter()
    threads = [threading.Thread(target=_run_client, args=(url, prompts[i][0], max_new, api_key, out, i))
               for i in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    after = _get(url, "/metrics?requests=all", api_key)
    # the engine's last BXP sample of the level can land a moment after the last reply: read once more if it has not
    if _expert_totals(after) is not None and _expert_totals(after) == _expert_totals(before):
        time.sleep(0.3)
        after = _get(url, "/metrics?requests=all", api_key)
    ok = [r for r in out if not r["error"]]
    tokens = sum(r["n"] for r in ok)
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    itl = [x for r in ok for x in r["itl_ms"]]
    rows = [r for r in reversed(after.get("requests") or []) if r["id"] not in hist0]   # oldest first
    keep = ("id", "prompt_tokens", "output_tokens", "first_engine_token_s", "prompt_ms", "decode_ms", "decode_tok_s",
            "hit_rate", "pcie_share", "drafts_offered", "drafts_accepted", "finish")
    xt = _delta(_expert_totals(before), _expert_totals(after))
    level = {"label": label, "round": round_no, "clients": clients, "prompt_tokens": prompt_tokens, "max_new": max_new,
             "wall_s": round(wall, 3), "tokens": tokens,
             "aggregate_tok_s": round(tokens / wall, 2) if wall > 0 else None,
             "ttft_s": {"p50": percentile(ttft, 50), "p95": percentile(ttft, 95), "max": max(ttft) if ttft else None},
             "itl_ms": {"p50": percentile(itl, 50), "p95": percentile(itl, 95), "n": len(itl)},
             "latency_s": {"p50": percentile([r["latency_s"] for r in ok], 50),
                           "max": max((r["latency_s"] for r in ok), default=None)},
             "errors": [r["error"] for r in out if r["error"]],
             "finish": sorted({str(r["finish"]) for r in out}),
             "all_coherent": all(coherent(r["text"]) for r in ok) if ok else False,
             "requests": [{k: r.get(k) for k in keep} for r in rows],
             "server_hit_rate": None, "expert_window": None}
    rates = [r["hit_rate"] for r in rows if r.get("hit_rate") is not None]
    if rates:
        level["server_hit_rate"] = round(statistics.mean(rates), 4)
    if xt and xt.get("windows"):
        w = xt["windows"]
        ent = xt.get("entries") or 0
        level["expert_window"] = {
            "windows": w, "rows": xt.get("rows"),
            "pcie_bytes_per_token": round(xt["pcie_bytes"] / tokens, 1) if tokens else None,
            "pcie_bytes_per_window": round(xt["pcie_bytes"] / w, 1),
            "pool_ms_per_window": round(xt["pool_us"] / 1000.0 / w, 3),
            "wait_ms_per_window": round(xt["wait_us"] / 1000.0 / w, 3),
            "link_est_ms_per_window": round(xt["link_est_us"] / 1000.0 / w, 3),
            "misses_per_window": round(xt["misses"] / w, 3),
            "hit_rate": round(xt["hits"] / ent, 4) if ent else None,
            "offload_share": round(xt["offload"] / ent, 4) if ent else None,
            "ms_per_window": round(xt["elapsed_ms"] / w, 3)}
    if check_codes:
        right = [(prompts[i][1] in r["text"]) and not any(prompts[j][1] in r["text"] for j in range(clients) if j != i)
                 for i, r in enumerate(out) if not r["error"]]
        level["own_code_named"] = f"{sum(right)} of {clients}"
    level["answers"] = [r["text"][:80] for r in out]
    return level


def summary_table(levels: list[dict]) -> str:
    """One markdown table, one row per level (the median over rounds of the aggregate rate)."""
    by = {}
    for lv in levels:
        by.setdefault((lv["label"], lv["clients"]), []).append(lv)
    lines = ["| arm | clients | aggregate tok/s | TTFT p50 s | TTFT p95 s | ITL p50 ms | ITL p95 ms | hit rate | PCIe B/token | pool ms/win | wait ms/win | codes |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]

    def f(v, d=2):
        return "-" if v is None else f"{v:.{d}f}"
    for (label, n), rows in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        agg = statistics.median([r["aggregate_tok_s"] for r in rows if r["aggregate_tok_s"] is not None] or [0])
        last = rows[-1]
        x = last.get("expert_window") or {}
        hit = x.get("hit_rate") if x.get("hit_rate") is not None else last.get("server_hit_rate")
        lines.append(f"| {label or 'run'} | {n} | {agg:.2f} | {f(last['ttft_s']['p50'], 3)} | {f(last['ttft_s']['p95'], 3)} | "
                     f"{f(last['itl_ms']['p50'], 1)} | {f(last['itl_ms']['p95'], 1)} | {f(hit, 3)} | "
                     f"{f(x.get('pcie_bytes_per_token'), 0)} | {f(x.get('pool_ms_per_window'))} | "
                     f"{f(x.get('wait_ms_per_window'))} | {last.get('own_code_named', '-')} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="a running server (else --config starts one)")
    ap.add_argument("--config", help='a strata-<model>.json to start a server from (adds "parallel" and "expert_telemetry")')
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--parallel", type=int, default=None, help='with --config: "parallel": N batch slots')
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--no-check-codes", action="store_true")
    ap.add_argument("--out", required=True, help="the JSON file for every level's row")
    ap.add_argument("--md", help="also write the summary table here")
    a = ap.parse_args()
    if not a.url and not a.config:
        ap.error("--url or --config")
    proc = None
    url = a.url
    if not url:
        cfg = json.loads(Path(a.config).read_text())
        cfg["expert_telemetry"] = True
        tmp = Path(f"/tmp/concurrency-bench-{a.port}.json")
        tmp.write_text(json.dumps(cfg, indent=1))
        proc = serve(str(tmp), a.port, a.parallel)
        url = f"http://127.0.0.1:{a.port}"
    levels = []
    try:
        # one warm-up request, so the first level does not pay the engine's first-request costs
        stream_one(url, "Say hello in five words.", 16, a.api_key)
        for rnd in range(a.rounds):
            for n in a.clients:
                lv = run_level(url, n, a.prompt_tokens, a.max_new, a.api_key, not a.no_check_codes, a.label, rnd)
                levels.append(lv)
                print(json.dumps({k: v for k, v in lv.items() if k not in ("requests", "answers")}), flush=True)
                Path(a.out).write_text(json.dumps(levels, indent=1))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(180)
            except subprocess.TimeoutExpired:
                proc.kill()
    table = summary_table(levels)
    print(table)
    if a.md:
        Path(a.md).write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
