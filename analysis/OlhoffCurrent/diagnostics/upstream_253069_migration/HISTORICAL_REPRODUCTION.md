# HISTORICAL_REPRODUCTION — Parts 10 and 20A

```
HISTORICAL_PRESET_REPRODUCTION_PASS
```

All runs are 160×20 on a single thread (MATLAB R2025b). Timing is excluded from every comparison.

**Standard.** Bitwise: numeric and logical arrays compared by class, size and raw bytes; structs and cells compared recursively.

**Excluded.** `tEig`, `tGrad`, `tInner`, `tOuter`, `wallclock`, `provenance.resolvedAt`.

Source of the tables: `evidence/comparisons.json` (`scripts/mig_compare.m`). Raw results are declared in `EVIDENCE.json`.

## A. `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` (historical production formulation)

| quantity | POST (migrated) | PRE (tree edbfe47) | frozen conference record | UP (253069) |
|---|---|---|---|---|
| status | CONVERGED | CONVERGED | NATIVE_CONVERGED | CONVERGED |
| outer / cumulative inner | 91 / 2241 | 91 / 2241 | 91 / 2241 | 91 / 2241 |
| ω₁ | 169.495227021538 | = (bitwise) | = (bitwise) | = |
| ω₂ / gap₁₂ | 171.959995918159 / 1.454 % | = | — | = |
| volume | 0.4999990088 | = | = (bitwise) | = |
| ρ SHA-256 (float64 bytes) | `22aa4447bc5f2855…` | = | ρ bitwise equal | = |
| M_nd / gray / mid | 13.4025 % / 0.149375 / 0.025 | = | — | = |
| ladder stage starts / levels | 1, 79, 90 / 0.04, 0.02, 0.01 | = | — | = |
| final ‖Δρ‖₂ / move / N | 2.3583e-02 / 0.01 / 2 | = | — | = |
| inner max / not converged | 51 / 0 | = | — | = |
| log | 4 lines, incl. "iter 90: … move limit just changed (0.02 -> 0.01); convergence NOT asserted" | identical | — | identical |

- **POST vs PRE**: no differing value anywhere in the result.
  - The PRE result additionally carries the 12 `hist.ex*` trace fields, all **empty**. The old target always created them; upstream creates them only when the controller is selected.
  - POST additionally carries `res.aux` (M_nd and mean box per iteration, reporting).
  - The configuration differs only in the six added schema rows.
  - These are exactly the preregistered allowances (§5/§6). Verdict: `passPrePost = true`.
- **POST vs frozen conference record** (`campaign_9mesh_r2/benchmark_records.mat`, SHA-256 `873125858df0…`, the record the 2026-09-11 campaign produced): ρ, ω₁ and volume bitwise; 91 / 2241; converged.
  - `tests/test_preset_equivalence` re-ran the same check independently: **PASS**, 6/6.
- **POST vs UP**: strict — identical field sets, all values bitwise, all 87 configuration rows equal.

## A′. `duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered` (historical diagnostic)

Diagnostics on and cap 1600, as in every historical run of this controller.

| quantity | POST | PRE (edbfe47) | upstream-audit target record `case_TARGET_EX3_160.mat` (edbfe47) | UP |
|---|---|---|---|---|
| status / outer / inner | CONVERGED / 180 / 4283 | = | = | = |
| ω₁ / ω₂ / gap | 169.975120289596 / 171.515809381821 / 0.906 % | = | = | = |
| volume / M_nd / gray / mid | 0.4999967704 / 12.7561 % / 0.145 / 0.026875 | = | = | = |
| ρ SHA-256 | `b779f036154576cd…` | = | = | = |
| stage starts | 1, 103, 142 | = | = | = |
| descents [applied, from, declIter, declBegin] | [103 1 102 83], [142 2 141 122] | = | = | = |
| event branches / terminal | A, A / declared at 180, window 161–180, branch B | = | = | = |
| every Δρ (180 vectors) | recorded | = | = | = |
| `res.exhaustion` | — | = | = | = |

- **POST vs PRE**: no differing value; POST adds only `res.aux`; configuration differs only in the added rows (`passPrePost = true`).
- **POST vs the upstream-audit target record**: strict pass on every field that record carries.
- **POST vs UP**: strict.
- `tests/test_named_preset_reproduction('stageExhaustion')`, run against the tracked fixture: **PASS**, 10/10.

## A″. Four-rung stage exhaustion vs the committed target record (additional)

This is the β-stall preset with `move.continuation.signal = stop.rule = stageExhaustion`, cap 1600, diagnostics on. It is a documented override, not a registered preset. It is the configuration of the committed two-branch validation record, the only 160×20 stage-exhaustion evidence tracked in this repository.

| check vs `two_branch_controller_validation` C160x20 (record JSON, iterations CSV, declared trajectory `.mat`) | result |
|---|---|
| ρ SHA-256 | `332c00a5181372bf…` = recorded `rho_sha256` |
| ρ = last column of the recorded RHO trajectory | bitwise |
| ω₁…ω₅ | exact (170.01131577599028, …) |
| outer / inner total / inner max | 219 / 5074 / 53 = record |
| volume | 0.49999913419341135 exact |
| all 36 non-timing `hist` fields vs the trajectory file's `hist` | 36/36 bitwise |
| every per-iteration Δρ (219) vs recorded DRHO | bitwise |
| CSV columns omega1, move, stage, beta, nInner, cumInner, multN, exA, exB, exE, exDecl, exAmp | exact |
| exhaustion record (stage starts 1/103/142/181; branches A, A, B; terminal 219, window 200–219, B) | `isequaln` |
| log (7 lines) vs record and vs trajectory file | identical |

## Comparison-script correction (recorded, criteria unchanged)

The first run of `mig_compare.m` (`evidence/comparisons.run1.json`) reported `beta_post_vs_pre = 0` and `ex3_post_vs_pre = 0`. Its `differing` lists were empty. The script had flagged the preregistered allowances — the six added schema rows and the dropped `hist.ex*` fields — as failures. It had also checked only the *names* of the dropped fields, not that they were empty.

The script was corrected to implement §5/§6 exactly, **including the emptiness check it had omitted**, and re-run. The data did not change. See PREREGISTRATION_AMENDMENT_1.md.
