#!/usr/bin/env python3
"""A synthetic routing trace in `--dump-routing`'s format, for the offline cache-policy tools when no trace from a GPU
run is at hand.  It is NOT a model's routing: it is a Zipf draw with the two features the policies are about - a set of
hub experts every phase routes to, and phases whose other hot experts differ (a topic shift, or another conversation
taking over a batch slot).  Use it to compare policies against each other, never to predict a hit rate.

    python tools/synth_routing_trace.py OUT.bin [--windows 2000] [--rows 4] [--phases 4] [--hubs 48] [--zipf 1.1]
                                        [--k 10] [--layers 48] [--experts 512] [--seed 1]

Record = int32 layer, int32 k, k int32 ids, k float32 weights; one record per (window row, layer), the window's rows
one after another within a layer and the layers in order, so the replay tools see complete windows.
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np


def zipf_weights(n: int, s: float) -> np.ndarray:
    w = 1.0 / np.power(np.arange(1, n + 1, dtype=np.float64), s)
    return w / w.sum()


def generate(windows: int, rows: int, phases: int, hubs: int, zipf: float, k: int, layers: int, experts: int, seed: int):
    """Yields (layer, ids) per record.  Every layer has its own permutation of expert heat; `hubs` experts of each
    layer are the hubs (the hottest, shared by every phase); each phase re-draws the order of the rest."""
    rng = np.random.default_rng(seed)
    base = zipf_weights(experts, zipf)
    per_phase = max(1, windows // max(1, phases))
    hub_ids = [rng.permutation(experts)[:hubs] for _ in range(layers)]
    for ph in range(phases):
        # each phase: the hubs keep the top ranks, the other experts take the remaining ranks in a new order
        order = []
        for l in range(layers):
            rest = np.setdiff1d(np.arange(experts), hub_ids[l])
            rng.shuffle(rest)
            order.append(np.concatenate([hub_ids[l], rest]))
        n_w = per_phase if ph < phases - 1 else windows - per_phase * (phases - 1)
        for _ in range(n_w):
            for l in range(layers):
                for _r in range(rows):
                    ranks = rng.choice(experts, size=k, replace=False, p=base)
                    yield l, order[l][ranks]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=2000)
    ap.add_argument("--rows", type=int, default=4, help="rows per window (one token per batch slot)")
    ap.add_argument("--phases", type=int, default=4)
    ap.add_argument("--hubs", type=int, default=48)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    n = 0
    with open(a.out, "wb") as f:
        for layer, ids in generate(a.windows, a.rows, a.phases, a.hubs, a.zipf, a.k, a.layers, a.experts, a.seed):
            f.write(struct.pack("<ii", layer, a.k))
            f.write(struct.pack("<%di" % a.k, *[int(x) for x in ids]))
            f.write(struct.pack("<%df" % a.k, *([1.0 / a.k] * a.k)))
            n += 1
    print(f"{a.out}: {n} records, {a.windows} windows of {a.rows} rows, {a.phases} phases, {a.hubs} hubs per layer, zipf {a.zipf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
