# Spatial localization against the oracle

| Iter | Class | N | Sign error (all signs) | Bound error | Share squared distance | Reduced-cost loss | Raw KKT RMS | Sensitivity RMS amplification | Amplification-distance Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | void | 10266 | 12.186% | 100.000% | 10.109% | 0.00066379 | 0.5776 | 30.6899 | -0.6008 |
| 19 | gray_shell | 4850 | 5.278% | 100.000% | 48.054% | 0.000323922 | 0.01617 | 0.8391 | -0.2102 |
| 19 | gray_core | 3422 | 4.939% | 100.000% | 34.093% | 0.000146777 | 0.01218 | 1.0440 | -0.2763 |
| 19 | solid | 10262 | 27.753% | 100.000% | 7.744% | 0.00064571 | 0.1712 | 0.9306 | -0.3191 |
| 500 | void | 10266 | 12.439% | 100.000% | 10.212% | 0.000399474 | 0.9196 | 30.6899 | -0.5551 |
| 500 | gray_shell | 4850 | 14.887% | 100.000% | 46.461% | 0.000269717 | 0.03872 | 0.8391 | -0.2274 |
| 500 | gray_core | 3422 | 22.122% | 100.000% | 36.213% | 0.000132153 | 0.03854 | 1.0440 | -0.2024 |
| 500 | solid | 10262 | 0.682% | 100.000% | 7.113% | 0.000278722 | 0.3398 | 0.9306 | -0.3150 |
| 5000 | void | 10266 | 9.868% | 100.000% | 11.145% | 0.000290829 | 0.9193 | 30.6899 | -0.5939 |
| 5000 | gray_shell | 4850 | 7.546% | 100.000% | 44.150% | 0.000210511 | 0.04728 | 0.8391 | -0.2571 |
| 5000 | gray_core | 3422 | 8.738% | 100.000% | 38.033% | 0.000109963 | 0.04689 | 1.0440 | -0.2362 |
| 5000 | solid | 10262 | 0.214% | 100.000% | 6.672% | 0.000210136 | 0.3898 | 0.9306 | -0.3171 |


At 5000, gray shell plus gray core contain
82.183%
of squared design error; void contributes only
11.145%. The disagreement
is broad, with a strong gray design-distance component, despite the earlier
void-dominated raw residual normalization. Void and solid regions still account
for substantial effective objective loss. All oracle-active bound agreement is
zero for B0 at the strict bound tolerance, across density classes.

Filtered first-mode sensitivity RMS amplification in void is 30.6899x, matching
the prior observation. The within-void Spearman correlation of amplification
ratio with absolute oracle disagreement is negative (-0.5939),
so these data do not support a simple positive amplification-causes-disagreement
story. Elementwise ratios are unstable near zero raw sensitivities and remain
associations. No filter intervention was made.

Reduced-cost class loss means sum(q_oracle,e*(drho_candidate,e-drho_oracle,e)),
a supporting-dual decomposition. It is not asserted to be an additive nonlinear
spectral objective decomposition; cone curvature and row slack are separate.
Raw exact-gradient KKT, complementarity RMS, large-increment sign errors,
class-normalized distances, and first-mode linear losses are in structure.json.
