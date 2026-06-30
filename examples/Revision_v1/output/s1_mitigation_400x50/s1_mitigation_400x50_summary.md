# S1 Mitigation 400x50 Summary

Scope: one mitigated Exp3 400x50 alpha=1.00 run only. No manuscript edits, no full Exp2/Exp3 rerun, and no CR2/A4/P1 run.

## Mitigation

Changed exactly one mitigation parameter: mass interpolation exponent `pmass` from baseline `1` to mitigated `6`.

Du-Olhoff `du2007_c1` is implemented in `analysis/OlhoffApproachExact/Matlab/mass_interp.m`, but not in `ourApproach`; this pilot uses the closest existing `ourApproach` mass-penalization option.

## Classification

**inconclusive.** Acceptance and S1 localization indicators are mixed.

## Acceptance Gates

| gate | pass | value |
|---|---|---:|
| not capped | pass | 1579/2000 |
| design_change <= tol | pass | 0.000515876883455 <= 0.001 |
| feasibility <= tol | pass | 0 <= 1e-08 |
| tracked MAC >= 0.8 | pass | 0.973514472548 |
| A5 lowest-mode check | pass | mode 1 |
| artifacts complete | pass | manifest written |

## Baseline vs Mitigated

| metric | baseline | mitigated | delta |
|---|---:|---:|---:|
| omega_1 rad/s | 64.3936516345 | 131.934195703 | 67.5405440684 |
| omega_2 rad/s | 88.1790950033 | 145.885500317 | 57.7064053138 |
| omega_3 rad/s | 101.804704818 | 147.107859545 | 45.3031547271 |
| tracked MAC | 0.786126687583 | 0.973514472548 | 0.187387784965 |
| grayness | 0.0607931823467 | 0.0537743707138 | -0.00701881163291 |
| iterations | 1750 | 1579 | -171 |
| final design change | 0.000903109639332 | 0.000515876883455 |  |
| topology MAD |  | 0.468193087895 |  |
| topology correlation |  | 0.0371852444955 |  |

## S1 Mode Diagnosis

Baseline S1: `likely localized/spurious low-density mode influence` with 8 localized low-density modes, 2 ambiguous modes.

Mitigated S1: `likely localized/spurious low-density mode influence` with 9 localized low-density modes, 0 ambiguous modes.

| mode | baseline class | mitigated class | baseline low-density S frac | mitigated low-density S frac | baseline eff-S frac | mitigated eff-S frac |
|---:|---|---|---:|---:|---:|---:|
| 1 | ambiguous | physical global mode | 0.00222282 | 0.0224214 | 0.013038 | 0.0763718 |
| 2 | localized low-density mode | localized low-density mode | 0.999899 | 0.998217 | 0.00201102 | 0.00188838 |
| 3 | localized low-density mode | localized low-density mode | 0.999586 | 0.998229 | 0.00184336 | 0.00189178 |
| 4 | localized low-density mode | localized low-density mode | 0.991826 | 0.547024 | 0.00126024 | 0.00901961 |
| 5 | ambiguous | localized low-density mode | 0.0549846 | 0.986475 | 0.0192101 | 0.00320796 |
| 6 | localized low-density mode | localized low-density mode | 0.963746 | 0.996084 | 0.00095744 | 0.00133241 |
| 7 | localized low-density mode | localized low-density mode | 0.998366 | 0.568959 | 0.00133842 | 0.0117621 |
| 8 | localized low-density mode | localized low-density mode | 0.999748 | 0.991289 | 0.00109254 | 0.00325009 |
| 9 | localized low-density mode | localized low-density mode | 0.999771 | 0.937072 | 0.00227594 | 0.00354821 |
| 10 | localized low-density mode | localized low-density mode | 0.999911 | 0.994246 | 0.0016978 | 0.000824848 |

## Artifacts

- result MAT: `examples/Revision_v1/output/s1_mitigation_400x50/s1_mitigation_400x50_result.mat`
- result JSON: `examples/Revision_v1/output/s1_mitigation_400x50/s1_mitigation_400x50_result.json`
- S1 mode diagnosis: `examples/Revision_v1/output/s1_mitigation_400x50/s1_postprocessing/s1_mitigation_400x50_mode_diagnosis.md`
- manifest: `examples/Revision_v1/output/s1_mitigation_400x50/s1_mitigation_400x50_manifest.json`
