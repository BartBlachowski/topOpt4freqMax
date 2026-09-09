# EVIDENCE INVENTORY — what this audit stands on

Zero scientific runs were executed. Every number traces to an artifact that
already existed, referenced by immutable hash.

---

## 1. Primary evidence — present, hash-valid, sufficient

| artifact | bytes | role |
|---|---|---|
| `evidence/two_branch_controller_validation/C160x20_trajectory.mat` | 8 858 000 | 160×20 four-rung candidate: `RHO`, `DRHO`, `move`, full `hist` incl. controller trace, `cfg` |
| `evidence/two_branch_controller_validation/C320x40_trajectory.mat` | 283 657 283 | 320×40, same |
| `evidence/two_branch_controller_validation/C400x50_trajectory.mat` | 139 808 729 | 400×50, same |
| `evidence/move_activity_400/F400_400x50_trajectory.mat` | 102 057 786 | 400×50 fixed-move arm — the bitwise reference for the prefix argument |
| `evidence/move_activity_400/P400_400x50_trajectory.mat` | 36 576 273 | 400×50 production, with final density field |
| `two_branch_controller_validation/runs/C{160x20,320x40,400x50}_iterations.csv` | — | per-iteration telemetry (Phase-12 fields) |
| `two_branch_controller_validation/runs/C*_record.json` | — | scalar records, descents, terminal states |
| `two_branch_controller_validation/evidence/baselines.json` | — | frozen production baselines |
| `move_stop/runs/baseline_{160x20,320x40}_iterations.csv` | — | production per-iteration telemetry |
| `move_activity_400/runs/P400_400x50_iterations.csv` | — | production per-iteration telemetry |
| `two_branch_maturity_240/METRICS.json` | — | 240×30 fixed-move scalars (supporting mesh) |

All five `.mat` files verify against their declared SHA-256. The three candidate
trajectories pass `olhoffcurrent_evidence_gate` 3/3.

## 2. What each mesh can and cannot answer

| mesh | S vs F | rung decomposition | topology fields | note |
|---|---|---|---|---|
| 160×20 | **yes** | **yes** | S and F yes; **production ρ unavailable** | complete for this audit |
| 320×40 | **yes** | **yes** | S and F yes; **production ρ unavailable** | complete |
| 400×50 | **yes** | **yes** | S, F **and** production ρ all available | strongest case |
| 240×30 | **no** | **no** | none | scalars only — see §3 |

## 3. Gaps, declared not worked around

**240×30 is a supporting mesh only.** `runD_240x30.mat` is lost and
`two_branch_maturity_240` kept no per-iteration CSV, so all that survives is the
tracked `METRICS.json`. That is enough to report the fixed-move exhaustion event
(iteration 187, Branch B) and what continuing at `move = 0.04` bought afterwards,
and **not** enough for any S-vs-F ladder comparison. No four-rung 240×30 run has
ever existed, and none was made. The mesh is therefore excluded from the
materiality count by construction, exactly as the preregistration states.

**Production final density fields for 160×20 and 320×40 do not exist.** So
density-field distance and topology images comparing *production* against S or F
are available only at 400×50. S-vs-F topology comparison, which is what this
audit actually needs, is available at all three.

**Twelve hash-manifested artifacts are missing tree-wide** (`RETENTION_AUDIT.md`
§1). None of them is required by this audit: every primary quantity comes from
the five surviving trajectories and the tracked CSVs. Nothing was regenerated.

## 4. Chain of custody for the numbers that carry the verdict

The exhaustion events were **not** read back from the controller's own log. They
were recomputed from `RHO` and `hist.dxNorm2` by an independent implementation of
the frozen rule (`scripts/ml_frozen.py`), and only then compared:

```
                offline recomputation        controller's recorded trace
160x20          declare 102, branch A        declare 102     A/B identical elementwise
320x40          declare 274, branch A        declare 274     A/B identical elementwise
400x50          declare 388, branch B        declare 388     A/B identical elementwise
```

`A` and `B` agree **element-wise over every iteration of every `move = 0.04`
prefix**, and each solver-applied first descent equals declaration + 1
(103 / 275 / 389). Recorded in `evidence/event_verification.json`.
