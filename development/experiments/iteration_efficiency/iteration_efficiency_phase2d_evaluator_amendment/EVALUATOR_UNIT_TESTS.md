# WP4 — Unit tests for the amended common evaluator
OFFLINE_AMENDMENT_VALIDATION — NO_NEW_OPTIMIZATION
Script: `scripts/validate.m`. Transcript: `scripts/validate.log`.

## Mass-law values at the required test points

| x | `g4(x)` = Eq. (4) | `g4a(x)` = Eq. (4a) | branch |
|---|---|---|---|
| 0 | 0.000000e+00 | 0.000000e+00 | low |
| 1e-3 (E3 density floor) | 1.000000e-18 | 1.000000e-13 | low |
| 0.05 | 1.562500e-08 | 1.562500e-03 | low |
| `nextbelow(0.1)` = 0.099999999999999992 | 1.000000e-06 | 1.000000e-01 | low |
| 0.1 | 1.000000e-06 | 1.000000e-01 | low (`<=` is closed) |
| `nextabove(0.1)` = 0.10000000000000002 | 1.000000e-01 | 1.000000e-01 | high |
| 0.2 | 2.000000e-01 | 2.000000e-01 | high |
| 0.5 | 5.000000e-01 | 5.000000e-01 | high |
| 1.0 | 1.000000e+00 | 1.000000e+00 | high |

## Assertions

| test | result |
|---|---|
| coefficient is exactly `1e5` | pass — literal `1e5` in both branches |
| `1e5*(0.1)^6 = 0.1` | `0.10000000000000003`, differs from `0.1` by **2.776e-17** = one double ULP |
| C0 continuity across the branch | `|g4a(0.1-) - g4a(0.1+)| =` **6.939e-17**, against **1.000e-01** for Eq. (4) |
| C1 continuity | **correctly absent**: low-side derivative `6*1e5*0.1^5 = 6.000000`, high-side `1.000000` |
| branch condition unchanged (`x <= 0.1` closed below) | pass — `0.1` itself still takes the low branch |
| E1 unchanged when low-density elements are present | pass — `|dE1| = 0.000e+00`, bit-identical |
| E2/E3 unchanged when no element is `<= 0.1` | pass — bit-identical (low branch never taken) |
| stiffness interpolation unchanged | pass — no stiffness line differs in the diff |
| exact-count binary projection unchanged | pass — `binary_solid_count` and `binary_volume` identical |
| only two functional lines differ from the frozen evaluator | pass — verified by `diff` |

## Note on the E3 floor

At `x = 1e-3` the amended value is `1e-13` rather than `1e-18`. This is the intended
consequence of the `1e5` coefficient and is not a separate change: `g4a` is uniformly `1e5`
times `g4` on the entire low branch. Void mass rises by five orders of magnitude in absolute
terms but remains eleven orders below solid, so the artificial-mode suppression the branch
exists to provide is preserved — which is precisely what Du & Olhoff intend by offering
Eq. (4a) as an equivalent alternative to Eq. (4).
