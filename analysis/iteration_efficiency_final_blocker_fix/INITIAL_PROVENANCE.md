# Initial provenance — targeted blocker correction

## Starting state (recorded before any edit)

| item | value |
|---|---|
| branch | `benchmark-methodology-r2` |
| HEAD | `d4df2137e68851f91cff5e75de1ee4a99a6a7625` |
| final HEAD | `d4df2137e68851f91cff5e75de1ee4a99a6a7625` (no commits created) |
| tracked-modified | 22 |
| untracked | 26 |
| total status entries | 48 |
| MATLAB | R2025b |
| platform | darwin 25.5.0, arm64, threads = 1 |

The 48 entries equal the 47 recorded by the final pre-production audit plus the audit's own output directory `analysis/iteration_efficiency_final_audit/` (untracked), which the audit created. The 22 tracked-modified files are byte-identical to the audit's list. **No discrepancy.**

## Integrity verified before editing

- `analysis/iteration_efficiency_final/SHA256SUMS.txt` — **36/36 OK**
- `analysis/iteration_efficiency_final_audit/SHA256SUMS.txt` — **14/14 OK**
- All `PRODUCTION_MANIFEST.json` pinned hashes recomputed and matched (contract, evaluator, topology gate, four method sources, ten pinned components).

## Pre-existing unrelated work — preserved

`.gitignore`, `analysis/OlhoffReproduced2007/` (5 files), and `analysis/iteration_efficiency_phase2i_precision_qualification/` (11 modified + 24 untracked) are untouched by this task. `analysis/iteration_efficiency_final_audit/` was read only and never modified.

## Verification mesh floor

Per standing project rule, **no verification below 160×20**. Every live run in this task uses a production-floor mesh (160×20, 240×30, 320×40); cost is controlled by cutting the iteration cap, never the resolution. This also changed the harness smoke configuration from 16×2 to 160×20 (`+iefinal/config.m`), keeping `reference_horizon = 3`.

The one exception is *replay of frozen immutable evidence*: the `b_ref = 2100` / `B_meas = 3200` / `k_enter`–`k_cert` anchors live in the Phase-2I 96×12 H=3200 capture (`raw/reference_evaluation.mat`, `raw/capture_96x12_H3200.mat`). That evidence is historical and cannot be regenerated within this task; it is replayed, not newly generated. This is stated wherever those anchors are cited.
