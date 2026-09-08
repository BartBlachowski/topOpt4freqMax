# DATA_INVENTORY — evidence used and produced

## 1. The gap this task fills

Established fact 13: **240×30 has not been used to choose the two-branch
hypothesis.** Verified — no 240×30 fixed-move trajectory of any kind existed, and
no 240×30 numeric content was opened before the preregistration was frozen
(`PROVENANCE.md` §3).

This task adds exactly one arm: **240×30, fixed move 0.04**.

## 2. Fixed-move trajectories available BEFORE this task (the training set)

| mesh | file | `RHO` | policy | stop | manifested |
|---|---|---|---|---|---|
| 160×20 | `move_transition/runs/armU_160x20.mat` | 3200 × 600 | fixed 0.04 throughout | unstopped | no¹ |
| 160×20 | `move_stop/runs/fixedmove_160x20.mat` | 3200 × 400 | fixed 0.04 | production | yes |
| 320×40 | `dynamical_regime/runs/runB_320x40.mat` | 12800 × 1200 | fixed 0.04, extended | unstopped (`CAP_HIT`) | yes |
| 320×40 | `move_stop/runs/fixedmove_320x40.mat` | 12800 × 216 | fixed 0.04 | production (`NATIVE_CONVERGED`) | yes |
| 400×50 | `fixedmove_400_dynamics/runs/runC_400x50.mat` | 20000 × 1200 | fixed 0.04 | unstopped (`CAP_HIT`) | yes |
| **240×30** | — | **NONE** | — | — | — |

Production (ladder) references: `move_transition/runs/armP_160x20.mat`,
`armP_320x40.mat`, `dynamical_regime/runs/runA_400x50.mat`.

¹ Pre-existing retention debt in `move_transition`: those `.mat` files are
gitignored **and** absent from that study's `FINAL_SHA256.txt`. Flagged again;
repairing another study's manifest is outside this scope. Read only.

## 3. Produced BY this task

| run | file | `RHO` | policy | cap | manifested |
|---|---|---|---|---|---|
| **D** | `runs/runD_240x30.mat` | 7200 × nOuter | **fixed 0.04**, unstopped | 1200 | **yes — `FINAL_SHA256.txt` + `DATA_MANIFEST.json`** |

Carries `out` (full per-iteration telemetry including the inherited native-stop
predicate), `cfg` (resolved configuration) and `RHO` (complete raw trajectory).

## 4. Phase 8 — the 240×30 production reference

**No valid 240×30 production *trajectory* exists**, and per the strict one-run
limit **no second scientific run was authorized**. What exists is scalar or
non-canonical, inventoried here and used for reference only:

| source | content | verdict |
|---|---|---|
| `performance_campaign_forensic_audit/olhoff_histories/240x30.csv` (1600 rows) | scalar history, **constant `move = 0.005`**, `policy_stage = 1`; no densities, no `M_nd` | **NOT a production-ladder reference.** Different move policy entirely. |
| `iteration_count_audit/results/baseline_240x30.mat` (2026-07-30) | `xFinal` 7200×1 only, produced by `run_yuksel_audit.m` | **NOT an Olhoff production trajectory.** Final design only, Yuksel audit driver. |
| `Matlab/reproduction2007/baseline/FINAL_lp_240x30.mat` | independent MATLAB reproduction #3 | **Forbidden path** for this implementation; not used. |
| `examples/Performance/conference_benchmark/campaign_9mesh_r2/benchmark_results.json` and `examples/conference_benchmark_v1/campaign_9mesh/benchmark_results.json` | scalar record, "Du-Olhoff reconstruction (M4)": **104 outer, 2334 inner**; final `gray_fraction_01_09 = 0.1808`, volume 0.4999992, single connected component | **Scalar inventory only** — see caveats below |

The two campaign records **agree exactly** (104 / 2334), which is a useful
consistency check. But their provenance is *not* sufficient for causal use:

* they predate the establishment of canonical `OlhoffCurrent` (2026-09-07) and
  cannot be shown to have come from it;
* `campaign_9mesh_r2` is the subject of an unresolved forensic audit
  (`analysis/performance_campaign_forensic_audit/`);
* they carry no trajectory, no `M_nd`, no `ω₁`, and **no first-descent
  iteration**.

**Consequence, stated plainly:** this study cannot report a 240×30 production
first-descent iteration, nor a production/fixed-move common-prefix check, because
the required trajectory does not exist and the one-run limit forbids generating
it. That is an accepted, disclosed gap — not an omission. The scalar figure
(production converges at 104 outer) is reported as context only, never as a
causal reference.

## 5. Retention

`runs/*.mat` matches a `.gitignore` rule, as every prior study's raw output does.
Durability comes from hashing into `FINAL_SHA256.txt` and listing in
`DATA_MANIFEST.json` with byte size and role. **No required raw artifact exists
only in an ignored, unmanifested path**, and `FINAL_SHA256.txt` is re-verified
after cleanup. No scratch artifact is cited as evidence.

## 6. Read-only inputs

* `fixedmove_400_dynamics/evidence/fm_analysis.mat` — the preceding analysis
  object, reused so the 160/320/400 numbers are literally those already reported.
* `dynamical_regime/scripts/{dr_telemetry,dr_dyn,dr_classify}.m` — the frozen
  dynamical definitions, **called by reference, never copied**.
* `dynamical_regime/runs/runB_320x40.mat`, `fixedmove_400_dynamics/runs/runC_400x50.mat`,
  `move_transition/runs/armU_160x20.mat` — reached through `fm_analysis.mat`.

None was modified.
