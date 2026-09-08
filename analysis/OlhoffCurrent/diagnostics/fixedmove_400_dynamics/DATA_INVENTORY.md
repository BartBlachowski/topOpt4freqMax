# DATA_INVENTORY — evidence used and produced

## 1. The gap this task fills

Established fact 15: **there was no 400×50 fixed-move trajectory.** Verified at
task start — the only 400×50 raw trajectory in the repository was
`dynamical_regime/runs/runA_400x50.mat` (production, 139 outer, first descent
138, converged). Every other fixed-move arm was 160×20 or 320×40.

RUN C fills exactly that gap and nothing else.

## 2. Raw trajectories available BEFORE this task

| mesh | file | `RHO` | move policy | stop | manifested |
|---|---|---|---|---|---|
| 160×20 | `move_stop/runs/baseline_160x20.mat` | 3200 × 91 | ladder (production) | production | yes |
| 160×20 | `move_stop/runs/fixedmove_160x20.mat` | 3200 × 400 | **fixed 0.04** | production | yes |
| 160×20 | `move_transition/runs/armP_160x20.mat` | 3200 × 600 | ladder | unstopped | no¹ |
| 160×20 | `move_transition/runs/armU_160x20.mat` | 3200 × 600 | **fixed 0.04 throughout** | unstopped | no¹ |
| 160×20 | `admission_rule/runs/unstopped_160x20.mat` | 3200 × 600 | ladder | unstopped | no¹ |
| 320×40 | `move_stop/runs/baseline_320x40.mat` | 12800 × 131 | ladder (production) | production | yes |
| 320×40 | `move_stop/runs/fixedmove_320x40.mat` | 12800 × 216 | **fixed 0.04** | production (`NATIVE_CONVERGED`) | yes |
| 320×40 | `move_transition/runs/armP_320x40.mat` | 12800 × 600 | ladder | unstopped | no¹ |
| 320×40 | `move_transition/runs/armU_320x40.mat` | 12800 × 600 | fixed 0.04 to 213 | unstopped | no¹ |
| 320×40 | `admission_rule/runs/unstopped_320x40.mat` | 12800 × 600 | ladder | unstopped | no¹ |
| **320×40** | **`dynamical_regime/runs/runB_320x40.mat`** | **12800 × 1200** | **fixed 0.04, extended** | unstopped (`CAP_HIT`) | **yes** |
| **400×50** | **`dynamical_regime/runs/runA_400x50.mat`** | **20000 × 139** | ladder (production) | production (`CONVERGED`) | **yes** |
| **400×50 fixed move** | — | **NONE** | — | — | — |

¹ Pre-existing retention debt in the `move_transition` study: those `.mat` files
are gitignored **and** absent from that study's `FINAL_SHA256.txt`. Flagged again
here; repairing another study's manifest is outside this task's scope. They were
read only.

## 3. Produced BY this task

| run | file | `RHO` | policy | cap | manifested |
|---|---|---|---|---|---|
| **C** | `runs/runC_400x50.mat` | 20000 × nOuter | **fixed 0.04**, unstopped | 1200 | **yes — `FINAL_SHA256.txt` + `DATA_MANIFEST.json`** |

The `.mat` carries `out` (full per-iteration telemetry, including the inherited
native-stop predicate), `cfg` (the resolved configuration) and `RHO` (the
complete raw elementwise trajectory).

## 4. Retention

`runs/*.mat` matches a `.gitignore` rule, as every prior study's raw output does.
Durability is provided by hashing into `FINAL_SHA256.txt` and listing in
`DATA_MANIFEST.json` with byte size and role — the mechanism `move_stop` and
`dynamical_regime` already use. **No required raw trajectory exists only in an
ignored, unmanifested location**, and `FINAL_SHA256.txt` is re-verified after all
cleanup. No scratch artifact is cited as evidence.

## 5. Read-only inputs

* `dynamical_regime/runs/runA_400x50.mat` — the production counterfactual and
  common-prefix reference (SHA-256 `465bb7475342a2fe…`).
* `dynamical_regime/runs/runB_320x40.mat` — 320×40 extended fixed move
  (SHA-256 `196ce8c317648dd4…`).
* `dynamical_regime/evidence/dr_analysis.mat` — the preceding analysis object,
  reused so cross-mesh numbers are literally the ones already reported.
* `dynamical_regime/scripts/{dr_telemetry,dr_dyn,dr_classify,dr_spatial}.m` —
  the frozen dynamical definitions, **called by reference, never copied**.
* `move_transition/runs/armU_160x20.mat`, `armP_160x20.mat`, `armP_320x40.mat` —
  reached indirectly through `dr_analysis.mat`.

None was modified.
