# Eq. 4b S1 Summary

Scope: one controlled Exp3 400x50 alpha=1.00 benchmark; no parameter tuning and no manuscript edits.

## Acceptance Gates

| gate | pass | value |
|---|---|---:|
| not capped | fail | 2000/2000 |
| design_change <= tol | fail | 0.106568605444 <= 0.001 |
| feasibility <= tol | pass | 0 <= 1e-08 |
| tracked MAC >= 0.8 | pass | 0.924439952738 |
| A5 lowest-mode check | fail | mode 2 |
| artifacts complete | pass | manifest written |

## Three-Case Comparison

| metric | original 400x50 | pmass=6 | Eq. 4b |
|---|---:|---:|---:|
| omega_1 rad/s | 64.3936516345 | 131.934195703 | 77.0138805747 |
| tracked MAC | 0.786126687583 | 0.973514472548 | 0.924439952738 |
| grayness | 0.0607931823467 | 0.0537743707138 | 0.130512609407 |
| iterations | 1750 | 1579 | 2000 |
| localized low-density modes | 8 | 9 | 9 |
| physical global modes | 0 | 1 | 1 |

## Mode Classifications

- mode 1: localized low-density mode
- mode 2: physical global mode
- mode 3: localized low-density mode
- mode 4: localized low-density mode
- mode 5: localized low-density mode
- mode 6: localized low-density mode
- mode 7: localized low-density mode
- mode 8: localized low-density mode
- mode 9: localized low-density mode
- mode 10: localized low-density mode

## Final Scientific Conclusion

1. Does Eq. 4b improve the accepted tracked global response? **no**.
2. Does Eq. 4b reduce localized low-density modes? **no**.
3. Does Eq. 4b rescue Exp3 mesh validation? **no**.
4. Is Eq. 4b sufficient evidence for the missing mechanism? **no**. The single benchmark does not simultaneously improve the tracked response, reduce localized low-density modes, and establish mesh convergence.

S1 detailed diagnosis: `s1_postprocessing/eq4b_mode_diagnosis.md`.
