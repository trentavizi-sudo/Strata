# Concurrency benchmark runbook

How to measure what the concurrency work ([CONCURRENCY_PLAN.md](CONCURRENCY_PLAN.md)) changes, on the boxes it is
for. Every number comes from these runs; nothing in this file is a measurement yet.

The metric that matters is **aggregate tokens per second** at 2, 4 and 8 concurrent requests. Beside it, every run
records what must not get worse: **TTFT and ITL at one client**. The acceptance rule for every arm is an aggregate
gain at 2/4/8 with the one-client TTFT and ITL unchanged within the run-to-run noise of the baseline.

## Boxes

| Box | GPUs | Role |
| --- | --- | --- |
| B | 1x RTX 3090 24 GB | primary baseline |
| A, one card | 1x RTX 3060 12 GB (`CUDA_VISIBLE_DEVICES=<the 3060>`) | floor: a small cache, most experts on the CPU |
| A, all cards | 2x RTX 3060 + 1x RTX 3090 | later, replica-level aggregate only |

## Step 0: the facts of the box (once per box)

Misses are computed on the CPU by default, so the CPU is a benchmark variable. Record these with the results:

```bash
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|Flags' | sed 's/Flags:.*\(avx512f\|avx2\)/& .../' ; lscpu | grep -o 'avx512[a-z_]*\|avx2\|amx[a-z_]*' | sort -u | tr '\n' ' '; echo
```

```bash
sudo dmidecode -t memory | grep -E 'Configured Memory Speed|Size:' | sort | uniq -c
```

```bash
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv
```

```bash
free -g && df -h . && nvcc --version | tail -1
```

Put the output in `bench/results/<date>-concurrency-<box>/hardware.txt`.

## Step 1: the model

The checkpoint is `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (the plan verified it exists). Primary quant Q2_0,
secondary IQ3_S; both are larger than 24 GB, so both stream experts on every card here. The download is 66 to 76 GB
for Q2_0 (shard 1 plus the 29 GB n-gram table), about 84 GB for IQ3_S, plus the 6 GB draft layer. **Check the
disk and ask before downloading** when the files are not on the box yet:

```bash
strata launch --plan --family qwen --model q2_0
```

The plan prints what would be downloaded and where. Nothing is changed. With the files in place:

```bash
strata launch --family qwen --model q2_0 --context 32k --yes --no-tui
```

Record the engine's `expert cache auto: ... -> N slots` line and `... MiB of VRAM free with everything loaded` from
the server log; they replace the plan's estimates. Stop the server (`strata launch --stop`) before the runs below:
the harness starts its own.

The CPU miss path these quants use is `src/kernels/cpu/q2_*.cpp` (Q2_0) and the i-quant kernels (IQ3_S); the
trellis kernels `tq_*.cpp` serve the Gyro packs only. If the trellis path is wanted, add an arm with
`--family gyro --model gyro-m` (TQ2_0); it is not in this runbook.

## Step 2: the config

The harness takes the server's `strata-<model>.json`, which `strata launch` writes into the Strata data folder
(`--data-dir`; `strata launch --save-preset conc` keeps the launch choices under a name). Make one copy per arm;
the harness adds `"parallel"` from `--parallel` and sets `"expert_telemetry": true` itself.

Floor card (3060): keep the context at 32K (`--context 32k`) or set `--kv-resident 32768`; four full slots at a
long context would take the cache (plan, "Batch slots cost the cache too").

## Step 3: the runs

Each level sends N streamed requests at once, each with its own prompt of about 2,000 tokens and a hidden code, and
checks every reply names its own code. Three rounds per arm, interleaved with the baseline (A B A B A B): the
server restarts between arms, never between levels.

Baseline (the engine as released; `parallel` 4 on the 3090, 2 on the 3060):

```bash
python tools/concurrency_bench.py --config strata-q2_0-base.json --parallel 4 --clients 1 2 4 8 --rounds 3 --label base --out bench/results/<dir>/base.json --md bench/results/<dir>/base.md
```

Arm 1, admission control (`"admission": "pressure"` in the config; `"threshold"` 0.5, then 0.3 and 0.7):

```bash
python tools/concurrency_bench.py --config strata-q2_0-admit.json --parallel 4 --clients 1 2 4 8 --rounds 3 --label admit-0.5 --out bench/results/<dir>/admit-0.5.json
```

Arm 2, the LFRU policy (`--adapt-policy lfru` in the config's `args`; with `--adapt-every 1 --adapt-swaps 24` as a
second variant, and `--adapt-predict-credit 1` as a third):

```bash
python tools/concurrency_bench.py --config strata-q2_0-lfru.json --parallel 4 --clients 1 2 4 8 --rounds 3 --label lfru --out bench/results/<dir>/lfru.json
```

Arm 3, the elastic K/V for slots (`--kv-grow-slots` in `args`; needs `--expert-profile`, one GPU, no K/V streaming,
every expert in RAM: the engine log says `--kv-grow-slots: the N batch slots' K/V is elastic too` or why not):

```bash
python tools/concurrency_bench.py --config strata-q2_0-kvslots.json --parallel 4 --clients 1 2 4 8 --rounds 3 --label kv-slots --out bench/results/<dir>/kv-slots.json
```

Record with each arm the engine's `expert cache auto` line: the elastic K/V arm should show more slots.

Item 4 of the plan (CPU/GPU overlap) is a measurement, not an arm: at 2/4/8 compare `wait_ms_per_window` and
`pool_ms_per_window` in each level's `expert_window`. A wait far above the pool time says the GPU waits on
something other than the CPU experts; a pool time near the window's `ms_per_window` says the CPU is the bound.

## Step 4: reading the results

`--md` writes one table per run; paste the arms together. The columns:

| column | what it is | from |
| --- | --- | --- |
| aggregate tok/s | completion tokens of all N requests over the level's wall time | the client |
| TTFT p50/p95 | time to the first content token | the client |
| ITL p50/p95 | time between consecutive content tokens, pooled over the requests | the client |
| hit rate | routed entries the VRAM cache served, over all routed entries, during the level | the engine's BXP totals |
| PCIe B/token | bytes the GPU read over the link for its PCIe share, per completion token | BXP |
| pool ms/win, wait ms/win | the host in the CPU pool, and waiting for the GPU, per window | BXP |
| codes | replies that named their own code and no other's | the client |

A level with an error in `errors`, a `codes` below N of N, or `all_coherent` false is a failed run, whatever its
rate. The JSON also carries the server's own per-request rows (`requests`: `hit_rate`, `pcie_share`,
`decode_tok_s`, `prompt_ms`, `decode_ms`).

Acceptance per arm: the median aggregate tok/s over the rounds at 2, 4 and 8 clients above the baseline's, and the
one-client TTFT p50 and ITL p50 within the baseline's round-to-round spread. Report the three rounds, not one.

## Step 5: the report

`bench/results/<date>-concurrency-<box>/README.md` with the hardware facts, the model and quant, the engine commit,
every config used, the table, and what was not run. Follow [COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)'s
rules: same-day baseline, the build under test named, nothing from another machine.

## Box A, all cards (later)

Replica routing (`serve/replicas.py`, `--replicas`) over the three cards, one engine per card, with the 3090's and
the 3060s' own `parallel`. The harness is the same against the front server's URL. No tensor split, no layer split:
the plan keeps multi-GPU out of the cache and admission design.
