# Obsolete evidence — retired from the reviewer evidence chain

Preserved provenance. **Not reviewer evidence.** Numerically unaltered; only the
classification and location changed. Nothing here is scheduled for deletion.

## `exp1_exp5/` — performance table and scaling fit

Retired by [SCIENTIFIC_DECISION_EXP1_EXP5.md](../../SCIENTIFIC_DECISION_EXP1_EXP5.md)
(2026-07-13). Classification: **D — obsolete after the current revision strategy.**

| File | Former role |
|---|---|
| `exp1_perf_table.m` | EXP1: timing / memory / iteration counts / `omega_1` across 3 methods x 4 meshes x 10 samples. Source of the withdrawn 8.6% gap, 7.1x speedup, and memory headline. |
| `exp5_scaling.m` | EXP5: log-log per-iteration scaling fit. Consumed EXP1 only. Source of the withdrawn `O(n_e^1.3)` claim. |

Their saved outputs live in:

- `../superseded_runs/pre_authoritative/` — legacy `exp1_perf_table_results.mat`,
  `exp5_scaling_results.mat`, `exp5_scaling_loglog.{fig,png}`
- `../superseded_runs/campaign_r1_full_20260701/exp1/` — the 2026-07-03 EXP1 stage output
  (15.5 h). It was marked "accepted" by the runner on **artifact presence only**; its
  Olhoff comparator terminated at the iteration cap and its timing decomposition is
  unsound.

### Why these are not merely "unfinished"

EXP1 is **construct-invalid**, not under-instrumented. The manuscript itself documents that
the local `OlhoffApproach` adds a Heaviside projection, a seven-level continuation
schedule, and a grayness penalty absent from the published method, and performs a **trial
eigensolve after every MMA update — roughly doubling the eigensolves per outer
iteration**. The local Yuksel-inspired implementation does not reproduce the published
iteration counts. A timing or scaling comparison against either measures implementation
choices, not methods, and licenses no inference about either published method **at any
level of instrumentation**.

Consequently the proposal to add convergence telemetry to those comparators was
**withdrawn**: it would have delivered a precise measurement of a meaningless quantity.

### These artifacts may NOT be used to

- claim or bound a speedup, runtime ratio, memory advantage, or scaling exponent;
- populate a manuscript table or figure;
- support any statement in the response letter other than the fact of withdrawal.

The frozen-eigenpair accuracy and cost question — the only scientifically live question
EXP1 ever proxied — is answered by **A4** (eigenpair-refresh sweep `N = {1,5,10,50,inf}`)
*within a single implementation*, with no unfaithful comparator.
