# DATA_INVENTORY — what evidence exists, and where it came from

Written so that no later brief can again assert the existence of data this
repository does not hold. Every row was verified on disk at task start.

## 1. Raw density trajectories available BEFORE this task

All under `analysis/OlhoffCurrent/diagnostics/`, verified by `whos -file`:

| mesh | file | `RHO` | move policy | stop | manifested? |
|---|---|---|---|---|---|
| 160×20 | `move_stop/runs/baseline_160x20.mat` | 3200 × 91 | ladder (production) | production | **yes** |
| 160×20 | `move_stop/runs/fixedmove_160x20.mat` | 3200 × 400 | **fixed 0.04** | production | **yes** |
| 160×20 | `move_transition/runs/armP_160x20.mat` | 3200 × 600 | ladder (production) | unstopped | no |
| 160×20 | `move_transition/runs/armU_160x20.mat` | 3200 × 600 | **fixed 0.04 throughout** (its rule never fired) | unstopped | no |
| 160×20 | `admission_rule/runs/unstopped_160x20.mat` | 3200 × 600 | ladder | unstopped | no |
| 320×40 | `move_stop/runs/baseline_320x40.mat` | 12800 × 131 | ladder (production) | production | **yes** |
| 320×40 | `move_stop/runs/fixedmove_320x40.mat` | 12800 × 216 | **fixed 0.04** | production | **yes** |
| 320×40 | `move_transition/runs/armP_320x40.mat` | 12800 × 600 | ladder (production) | unstopped | no |
| 320×40 | `move_transition/runs/armU_320x40.mat` | 12800 × 600 | fixed 0.04 **until 214**, then its rule descended | unstopped | no |
| 320×40 | `admission_rule/runs/unstopped_320x40.mat` | 12800 × 600 | ladder | unstopped | no |
| **400×50** | — | **NONE** | — | — | — |

**Established fact 6 confirmed** (160×20 and 320×40 raw histories are available
and hash-valid). **Established fact 7 confirmed** (no valid 400×50 raw
trajectory existed).

### The gap this task fills

* **400×50** had no production trajectory at all → **RUN A**.
* **320×40 fixed-move** evidence ran out at iteration 216 (`fixedmove`, stopped
  by the production rule) and 213 (`armU`, before its own rule descended) — in
  both cases *while still in coherent descent*, so the question "does 320×40
  eventually enter a period-2 regime?" was unanswerable → **RUN B**, cap 1200.

### Not a valid 400×50 source

`analysis/performance_campaign_forensic_audit/olhoff_histories/400x50.csv`
(1600 rows) is **not** usable as a production reference and is not used: it is
scalar-only (no densities, no `M_nd`) and was produced under a **constant
`move = 0.005`**, not the production ladder.

## 2. Raw trajectories produced BY this task

| run | file | `RHO` | policy | cap | manifested? |
|---|---|---|---|---|---|
| **A** | `runs/runA_400x50.mat` | 20000 × nOuter | production ladder + β-stall + production stop | 400 | **yes — `FINAL_SHA256.txt` + `DATA_MANIFEST.json`** |
| **B** | `runs/runB_320x40.mat` | 12800 × nOuter | **fixed 0.04**, unstopped | 1200 | **yes — same** |

Each `.mat` carries `out` (full per-iteration telemetry), `cfg` (the resolved
configuration) and `RHO` (the complete raw trajectory).

## 3. Retention

`runs/*.mat` matches a `.gitignore` rule, exactly as every previous study's raw
output did. The preceding study found that `move_transition`'s `armP/armU`
`.mat` files were gitignored **and** absent from that study's
`FINAL_SHA256.txt`, leaving the raw densities protected by nothing but their
presence on this disk. **That failure is not repeated here**: both new
trajectories are hashed into `FINAL_SHA256.txt` and listed in
`DATA_MANIFEST.json` with byte size and role, which is the durable mechanism
`move_stop` already uses for its own `.mat` files.

Prior-study `.mat` files were read only. Their manifest status is reported above
as found; correcting `move_transition`'s manifest is outside this task's scope
and is flagged in `REPORT.md` instead.

## 4. Read-only inputs used

* `move_transition/runs/armU_160x20.mat` — the 160×20 fixed-move reference.
* `move_transition/runs/armP_160x20.mat`, `armP_320x40.mat` — production references.
* `move_stop/runs/fixedmove_320x40.mat` — RUN B prefix reference (216 iterations).
* `move_transition/runs/armU_320x40.mat` — RUN B prefix reference (213 iterations).

All were verified hash-valid where a manifest covers them, and none was modified.
