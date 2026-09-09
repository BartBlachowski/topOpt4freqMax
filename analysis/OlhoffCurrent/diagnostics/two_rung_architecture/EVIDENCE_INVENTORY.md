# EVIDENCE INVENTORY — what this audit used, and what it could not

Zero scientific runs were executed. Every number comes from artifacts that
already existed on disk at HEAD `1438aa3f`. Digests are recorded in
`evidence/provenance_start.json` and re-verified in `DATA_MANIFEST.json`.

---

## 1. Raw causal-controller trajectories — present, hash-valid, sufficient

| artifact | bytes | role |
|---|---|---|
| `analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C160x20_trajectory.mat` | 8 858 000 | `RHO` (3200×219), `hist`, `cfg` — source of the 160×20 frozen replay, S1/S2/F densities |
| `.../C320x40_trajectory.mat` | 283 657 283 | `RHO` (12800×1600) — 320×40 |
| `.../C400x50_trajectory.mat` | 139 808 729 | `RHO` (20000×505) — 400×50 |
| `diagnostics/two_branch_controller_validation/runs/C160x20_iterations.csv` | — | 55-column per-iteration telemetry incl. the `ex*` controller trace |
| `.../runs/C320x40_iterations.csv` | — | 1600 rows |
| `.../runs/C400x50_iterations.csv` | — | 505 rows |
| `.../runs/C{160x20,320x40,400x50}_record.json` | — | run records: status, stage starts, descents, exhaustion struct |
| `.../evidence/baselines.json` | — | the three frozen production baselines |
| `diagnostics/move_ladder_necessity/METRICS.json` | — | prior S1 extraction and rung boundaries, used for cross-checking only |
| `analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat` | 36 576 273 | production 400×50 density field, the one surviving `P` topology |

Source of the frozen rule (read, not modified):
`+impl/architecture/+olh/+move/exhaustion.m`, `+olh/+move/limit.m`,
`architecture/olhoffSolve.m`.

**All 14 required artifacts present and hashed. `requiredEvidence` = PASS.**

## 2. Everything the audit needed, it had

| requirement | status |
|---|---|
| raw C160/C320/C400 trajectories | ✅ present |
| stage-resolved telemetry for the frozen replay | ✅ present (`RHO`, `hist.dxNorm2`) |
| the controller's own `ex*` trace, for element-wise comparison | ✅ present |
| production scalars for all three meshes | ✅ present |
| prior finalization gate on the source study | ✅ PASS |

No primary causal trajectory needed for the two-rung counterfactual is missing.
The Phase-0 gate returned **`TWO_RUNG_EVIDENCE_GATE_PASS`**.

## 3. Gaps — declared, not worked around

### 3.1 240×30 is unavailable for two-rung causal comparison

`analysis/OlhoffCurrent` contains **zero** files matching `*240x30*`. Three
independent facts, each sufficient on its own:

1. `diagnostics/two_branch_maturity_240/` has **no `runs/` directory**; its raw
   `.mat` artifacts are among the losses that motivated the finalization gate
   (that study's gate correctly reports **FAIL** today).
2. Even had it survived, that arm was a **fixed-move** arm — `move.policy = 'fixed'`,
   `move.initial = 0.04`, per its own preregistration §14 — so it never executed
   a `move = 0.02` stage and could not contain an S2.
3. No four-rung causal-controller run at 240×30 exists at all.

**Action taken: none.** No S2 is inferred for 240×30, no proxy is substituted,
and no run is made. Marked `UNAVAILABLE_FOR_TWO_RUNG_CAUSAL_COMPARISON` in
`METRICS.json`. The architecture decision rests on C160/C320/C400.

### 3.2 Production density fields for 160×20 and 320×40 are lost

`baselines.json` records `rho_available = false` and
`rho_sha256 = "UNAVAILABLE -- raw .mat lost (see EVIDENCE_POLICY.md)"` for both.
Consequence: the S2-versus-production **topology distance** is computable only at
400×50. Every other production comparison uses the recorded scalars, which are
complete. The gap is stated in `PHYSICS_SAFETY.md` §4 and is **not** imputed.

### 3.3 Wall-clock time is unreliable on all three runs

Seconds per inner MMA iteration drift 3.8×–5.2× within each run — a machine
property, not an algorithmic one. Reported, down-weighted, and never used to
decide anything. See `PERFORMANCE_READINESS.md` §1.

## 4. This study's own outputs

| artifact | kind |
|---|---|
| `PREREGISTRATION.md` + `evidence/PREREGISTRATION.{frozen,sha256}` | frozen `b50455fb…` |
| `evidence/provenance_start.json`, `evidence/provenance_final.json` | Phase-0 gate |
| `evidence/event_verification.json` | frozen-rule replay + §7 validity checks |
| `evidence/analysis.json` | full P/S1/S2/F extraction and decomposition |
| `evidence/figures.json` | figure digests |
| `METRICS.json` | frozen verdict mapping applied mechanically |
| `figures/F1…F11` | the eleven required figures |
| `scripts/tr_provenance.m`, `tr_frozen.py`, `tr_verify.py`, `tr_analyze.py`, `tr_figures.py`, `tr_metrics.py` | analysis code |
| `EVIDENCE.json`, `DATA_MANIFEST.json`, `FINAL_SHA256.txt` | fail-closed retention |

No `.mat` was produced. Nothing under `+impl/` was written.
