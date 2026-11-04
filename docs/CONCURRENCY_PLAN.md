# Concurrency plan: aggregate throughput on 10 to 24 GB cards

An audit of how Strata schedules requests, caches and moves experts and manages the K/V cache today, what the
concurrency work changes, and how it is measured. Written from the code at 0.2.10 (engine 0.1.56); every claim
below names its file. Nothing in this plan is a measurement unless it says what it was measured on.

Scope: the main engine (`src/program/generate.cpp`, `src/core`) serving Qwen3.8-Flash-Next. The DeepSeek V4.1
engine (`src/dsv41`, its own `--parallel`) and the Qwen3.6 engine (`src/qwen35`) are separate paths and are not
covered. Multi-GPU is out of scope for the cache and admission design: a second card is server-side replica routing
(`serve/replicas.py`) when it comes, never a tensor split.

## The model (verified)

- Base model: Qwen3.8-Flash-Next (Alibaba, released August 2026). About 125B main-model parameters and 6B active
  per token, plus a 51B n-gram embedding table (176 to 180B on disk with the MTP head). 48 layers alternating Gated
  DeltaNet and QSA attention blocks, each with a MoE of **512 experts, 10 routed per token plus 1 shared**, expert
  width 640. These figures are from secondary coverage of the model card; the GPU box should confirm them against
  the card itself (runbook step 0).
- Quantized checkpoint used by Strata: [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  (exists; pinned by `setup.py` at revision `ed59f92`). Sizes: Q2_0, IQ2_XS, IQ3_XXS, IQ3_S. The cloud session
  could not reach huggingface.co directly (network policy); the repository was confirmed through a web search of
  the model pages, and `setup.py` pins the revision and SHA-256 of every file.
- Expert geometry from the pack indexes in `tests/data/native_experts/*.txt` (48 x 512 = 24,576 experts):

| Quant | Expert bytes (whole model) | Bytes per expert (average) | Experts per GiB of cache |
| --- | ---: | ---: | ---: |
| Q2_0 | 33,973,862,400 | 1,382,400 | 776 |
| IQ2_XS | 35,454,976,000 | 1,442,666 | 744 |
| IQ3_XXS | 42,912,972,800 | 1,746,133 | 614 |
| IQ3_S | 50,292,326,400 | 2,046,400 | 524 |

- **Streaming criterion.** Every size is larger than 24 GB, so every size forces expert streaming on the 3090 and
  the 3060. Q2_0 is the primary benchmark and IQ3_S the secondary, as agreed.
- **Which CPU kernels the misses use.** The trellis kernels `src/kernels/cpu/tq_*.cpp` serve agentionai's Gyro
  packs (TQ1_0 / TQ2_0, `setup.py` line ~356), not the ISTA-DASLab sizes. Q2_0 misses run on
  `src/kernels/cpu/q2_avx2.cpp` / `q2_avx512.cpp` (bit-plane kernels, parity test `q2_bitplane_parity`); the i-quants
  (IQ2_XS, IQ3_XXS, IQ3_S) run on `iq_avx2.cpp` / `iq_avx512.cpp` and ggml-cpu (`native_expert_parity`,
  `iq_nibble_parity`). AGENTS.md ranks Gyro as a slower memory-saving fallback, so the trellis path is not the
  primary benchmark. If you want the trellis miss path measured, add a Gyro-M (TQ2_0) arm; it is not in this plan.
- Parity fixtures exist for Q2_0 (`tests/data/native_experts/q2_0.txt`, `q2_bitplane_parity`,
  `tools/test_setup_choices.py`) and IQ3_S (`iq3_s.txt`, the native expert parity tests). No new parity test is
  needed before benchmarking as long as no kernel changes; the policy and allocator work here changes no arithmetic.

## Hardware matrix

| Box | GPUs | Role | Notes |
| --- | --- | --- | --- |
| B | 1x RTX 3090 24 GB | primary baseline | one card, no split |
| A (one card) | 1x RTX 3060 12 GB | floor | the stress case: a small cache, most experts on the CPU |
| A (all cards) | 2x RTX 3060 12 GB + 1x RTX 3090 24 GB, PCIe only | secondary, later | replica-level aggregate only; no tensor split |

CPU, memory speed and AVX-512 support of both boxes are unknown until `lscpu` and `dmidecode` run there (runbook
step 0). Misses are CPU compute by default, so the CPU is a benchmark variable: report it with every number.

## Corrected expert cache sizes

The brief assumed about 23,000 of 24,576 experts resident on a 32 GB card. Measured: an RTX 5090 32 GB with
IQ2_XS at 131K context (K/V streamed, 32K cells resident) started with 24.26 GiB free before the cache and filled
**17,463 slots (23.44 GiB)**; the prompt path borrows 3,421 of them while a prompt is read
(`bench/results/2026-09-30-community-rtx-5090/engine.log`). So about 7.7 GiB of a card is taken before the cache
(dense weights, the resident K/V window, the draft layer, the head, the CUDA context), and the reserve and prompt
buffers take 0.7 to 3 GiB more depending on the prompt chunk.

Estimates for this plan's cards, from that anchor and the table above (estimates, not measurements; the
runbook's first run replaces them with the engine's own `expert cache auto:` line):

| Card | Free before the cache | Q2_0 slots (share of 24,576) | IQ3_S slots (share) |
| --- | ---: | ---: | ---: |
| RTX 3090 24 GB, 131K context streamed | ~16 GiB | ~11,000 (45%) | ~7,500 (30%) |
| RTX 3060 12 GB, 131K context streamed | ~4 GiB | ~2,300 (9%) | ~1,400 (6%) |

Anchors for the small card: an RTX 2080 Ti 11 GB with IQ3_S at 262K context (32K cells resident) got 1,313
slots (`bench/results/2026-10-02-community-rtx2080ti-11gb`); a 2x RTX 3060 split with IQ3_S filled 3,579 slots
over both cards (`bench/results/2026-10-04-community-2x-rtx3060`).

**Batch slots cost the cache too.** Every `--batch` slot is a full session at `--max-context`: measured 0.52 GiB
each at 32K int8 K/V (`bench/results/2026-10-06-community-2x-rx-7900-gre`) and 2.88 GiB each at a long context
without K/V streaming (`bench/results/2026-10-06-community-rtx4090-opt-ins`). On the 3060, four slots at 32K take
2.1 GiB of a ~3 GiB cache, which is why `--kv-grow-slots` (item 3 below) matters most on the floor card, and why the
floor runs use `--max-context 32768` or `--kv-resident 32768`.

## How it works today (the audit)

### Request scheduling

- Server side (`serve/server.py`). Default: one request at a time. `"parallel": N` becomes the engine's `--batch N`
  (`parallel_args`, ~3581). `Engine.generate_batched` (~2050): a request alone runs the solo path with MTP drafts;
  when a second arrives the first is `STOP`ped and resumed in a batch slot (`BGEN`) from the prompt cache, and the
  new one is admitted beside it. Admission is a FIFO on the engine's control lines (`_take_control`, ~1900); opt-in
  `STRATA_ADMIT_SHORT_FIRST` reorders by prompt length (`_admit_next`, ~2014). `pick_slot` (~2423) always takes a
  free slot: longest held prefix, else empty, else least recently used. **No cache-pressure or PCIe signal exists in
  admission.** `BYIELD` preempts a long prompt read for a short one (#656); `STRATA_BATCH_DECODE_SHARE` lets the slots
  decode between a new prompt's chunks.
- Engine side (`src/program/generate.cpp`). Slot sessions are allocated whole (`session_bytes(g, o.max_context, ...)`,
  ~6606) before the cache is sized. A decode step is one verify window (`include/strata/core/verify.hpp`): every
  active slot's row through all 48 layers in one captured graph, at most 8 rows a window (`kVerifyMaxT`);
  `batch_step` (~13567 on) drives it. Timing: `STRATA_DECODE_TIMING` (solo line ~18689; the batch line ~14166),
  `STRATA_BATCH_TELEMETRY` (`BSTAT` lines, `include/strata/core/batch_telemetry.hpp`), `STRATA_PAR_COUNTERS`
  (`include/strata/program/par_counters.hpp`).

### Expert caching and movement

- `ExpertCache` (`include/strata/core/expert_cache.hpp`) is storage plus the residency table; `admit` never evicts
  (line ~157, by design). `--expert-cache auto` sizes it from free VRAM minus the reserve, the prompt buffers and
  the draft head (`generate.cpp` ~7454-7580, `auto_cache_slots`). The startup fill follows a frequency profile.
- Per window and layer, `ExpertDispatch` (`src/core/expert_source.cpp` ~4900-5320) decides where each routed
  expert runs. Entries are grouped by distinct expert (`job_of[e]` / `jobs_multi`, ~5180-5230; the PCIe groups
  `first_of` / `pcie_i0`, ~5005-5040): **an offloaded expert is read at most once per window across all slots**,
  as one CPU job of up to MAXT tokens or one PCIe staging copy. The once-per-step guarantee already holds.
- Misses run on the CPU pool by default (`include/strata/kernels/cpu/pool.hpp`), concurrently with the GPU's hits
  (`include/strata/core/hit_hook.hpp`: Launch, pool, H2D, Combine). PCIe carries only the `--pcie-frac` share
  (defaults in `generate.cpp` ~4589-4610); `PcieModel` (`expert_source.hpp` ~409) fits per-layer CPU and GPU cost
  models and picks the PCIe count that minimises max(GPU, CPU). The GPU waits at each layer's combine for the
  pool's parts (`verify.cpp` ~3500-3525, `ms_wait` / `ms_pool`); the residual chain makes that barrier inherent.
- The adaptive tier is the eviction policy that exists: the `adapt` lambda (`generate.cpp` ~12490-12560). Every
  `--adapt-every` windows, per layer, the most-routed non-resident experts (decayed usage >= 2) replace the
  least-routed residents where the gain clears `--adapt-min-gain`, at most `--adapt-swaps` (96) per round; usage
  decays by `--adapt-decay` (0.7) after each round. Counts live for the process, so hub experts survive across
  requests; `--expert-profile-save` persists the ranking. This is LFU with decay: no recency term, rounds every
  N windows, a cap per round. `--adapt-async 1` moves the round to a helper thread.
- Cross-layer prediction exists: `RouterLookahead` (`expert_source.hpp` ~313) applies layer l+1's router to layer
  l's input and feeds page warming of the file tier and, with `STRATA_FS_SLOTS`, the `ForesightSwap`
  (`include/strata/core/foresight_swap.hpp`): a few VRAM slots per layer outside the main cache, refilled with
  recently missed and predicted experts; a landed slot counts as a hit. Off by default.
- Offline policy harnesses (CPU only): `tools/bench_eviction.py` (fill-only, lru, lfu-decay, oracle on a
  `--dump-routing` trace) and `tools/replay_cache_policy.py` (simulates `adapt()`).

### K/V cache

- Every session, solo or slot, carves 12 QSA states and 36 GDN states from one contiguous device buffer at
  `--max-context` cells (`session_init`, `include/strata/core/session.hpp`; `qsa_state_bytes`, `layer.hpp`). The
  cost under concurrency is over-provisioning per slot, not fragmentation.
- Formats `--kv int8|q4_0|k8v4`; `--kv-streaming` / `--kv-resident N` keep a window in VRAM with the rest in pinned
  RAM (`kv_stream.hpp`).
- `--kv-grow` (`include/strata/core/vmm.hpp`, `src/core/layer.cpp` ~624-730, `generate.cpp` `kvg_*` ~9525-9700) is
  already a chunked allocator: each state's K/V pools live in a VMM address range whose 2 MiB chunks are mapped on
  demand, taken from the expert cache arena (the cache gives up its top slots, refills them when the K/V shrinks),
  addresses fixed so captured graphs stay valid. It is refused with `--batch` (`generate.cpp` ~6363: "no --batch,
  --vram-elastic or --peer-device"), with K/V streaming, a layer split, a helper GPU and on HIP.

## What the brief asked for and what the repo already has

| Brief item | Status in the repo | This plan |
| --- | --- | --- |
| 1a LFRU eviction | LFU-decay tier, cross-request; no recency term | `--adapt-policy lfru` (recency in the ranking, per-window rounds), default off |
| 1b cross-layer prediction / prefetch | `RouterLookahead` + `ForesightSwap` (side slots, opt-in) | opt-in: predicted experts may take the tier's victim slots in the main cache |
| 1c once per batch step | holds (`job_of`, PCIe groups) | nothing; the harness reports it |
| 2 admission control | none (free slot = admit) | server-side, telemetry-driven, hold never reject, default off |
| 3 K/V block allocator | `--kv-grow`, solo session only | `--kv-grow-slots`: the same chunks for slot sessions, one shared pool, default off |
| 4 CPU/GPU overlap | the design already (hit_hook.hpp) | measure `ms_wait` vs `ms_pool` at 2/4/8, then decide |
| 5 ternary | no ternary checkpoints anywhere | skipped |
| 5 fused q/k/v, gate/up | `wqkv` (`layer.hpp:10`), grouped expert kernels, fused GR/GDN | nothing |
| 5 recurrent-state prefix cache | turn checkpoints, parked conversations | nothing |
| 5 head on prefill tokens | the prompt path does not run the head per token (`docs/PREFILL_HEAD_ALIAS.md`) | nothing |
| 5 clustered output head | absent; changes numerics | last, opt-in with a parity test, only if still wanted |

## Admission signal

Window miss cost = max(pool ms, link ms) plus the GPU-reach wait, per window, from the engine's own counters
(`ver.ms_pool`, `ver.ms_wait`, the PCIe bytes and the fitted link time). The threshold is expressed as a share of
the current inter-token time (ms per window over the window's wall time), so the same setting means the same thing
on a 12 GB and a 24 GB card. The engine emits the fields (the `BXP` line, item 0); the Python policy consumes them,
so the policy can move into the engine later without a rewrite.

## Implementation order

0. Harness and instrumentation (`tools/concurrency_bench.py`, the `BXP` telemetry line, `docs/CONCURRENCY_BENCH.md`).
1. Admission control in `serve/server.py`, opt-in, unit-tested with the fake engine.
2. `--adapt-policy lfru` and the predicted-experts-into-main-cache switch, validated offline on routing traces.
3. `--kv-grow-slots`, host-side accounting tests, GPU validation pending.
4. Item 4 measurement; item 5 head last if wanted.

Acceptance: nothing is done until its GPU benchmark shows an aggregate gain at 2/4/8 with no single-request
regression (TTFT and ITL at 1 client unchanged within noise). Every new flag defaults to current behaviour.

## Cloud-session limits

No CUDA toolchain in the cloud session: the C++ changes are compile-checked only where a host-only target
exists, and are marked **pending GPU validation** in their commits. The Python side (server, harness, replay) is
unit-tested here. No benchmark number in this repository comes from the cloud session.
