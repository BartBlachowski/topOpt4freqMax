# MESH_TREND (Part 19): empirical mesh trend

```
PEDERSEN_ADAPTIVE_MESH_TREND_ACCEPTABLE
```

This is the rule outcome of PREREGISTRATION.md §10.5: none of flags S1–S7 fired (§3).

**Wording.** The document describes an **empirical mesh trend** over nine meshes that share one physical filter radius. It is not a proof of mesh convergence. The frequencies are native to the Pedersen/linear-mass model.

## 1. ω₁

| mesh | ω₁ [rad/s] | Δ to next mesh [rad/s] | Δ relative |
|---|---|---|---|
| 160x20 | 169.2106 | −1.868 | −1.104 % |
| 240x30 | 167.3424 | −1.486 | −0.888 % |
| 320x40 | 165.8568 | **+0.598** | **+0.361 %** |
| 400x50 | 166.4552 | −0.446 | −0.268 % |
| 480x60 | 166.0093 | −0.199 | −0.120 % |
| 560x70 | 165.8101 | −0.160 | −0.097 % |
| 640x80 | 165.6500 | −0.227 | −0.137 % |
| 720x90 | 165.4234 | +0.009 | +0.005 % |
| 800x100 | 165.4322 | — | — |

- **Coarse to fine (160x20 → 800x100):** −3.778 rad/s, **−2.23 %**.
- **Full range:** 3.787 rad/s, 2.28 % of the median.
- **Five finest meshes (480x60 … 800x100):** range 0.586 rad/s, **0.35 %**.
- **Largest adjacent change:** 1.10 % among the four coarsest pairs, and **0.14 %** among the four finest.
- **Shape.** ω₁ falls steeply from 160x20 to 320x40. 320x40 lies **below** the trend: it is the strongly left–right-asymmetric design (TOPOLOGY_AUDIT.md). From 400x50 to 720x90 ω₁ decreases monotonically in steps of ≤ 0.27 %, and 720x90 and 800x100 agree to 0.005 %.
- **Reading.** ω₁ is **empirically stable** at the ≈ 0.35 % level from 480x60 upward. The sequence still drifts slightly downward to 720x90 and is flat at the last step. Two meshes of flatness do not demonstrate a limit.

## 2. Other quantities

| mesh | ω₂ | Δω₂ rel | terminal gap | M_nd | gray fraction | \|volume error\| | outer | total wall [s] | wall / outer [s] |
|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 170.40 | — | 0.70 % | 0.1146 | 0.1419 | 1.0e-6 | 121 | 223.9 | 1.85 |
| 240x30 | 187.16 | +9.84 % | 11.84 % | 0.1227 | 0.1447 | 1.5e-6 | 111 | 354.9 | 3.20 |
| 320x40 | 195.14 | +4.26 % | 17.66 % | 0.1406 | 0.1656 | 1.7e-7 | 101 | 427.2 | 4.23 |
| 400x50 | 198.13 | +1.53 % | 19.03 % | 0.1216 | 0.1397 | 2.1e-7 | 93 | 570.1 | 6.13 |
| 480x60 | 203.44 | +2.68 % | 22.55 % | 0.1307 | 0.1509 | 2.3e-7 | 112 | 981.4 | 8.76 |
| 560x70 | 206.38 | +1.44 % | 24.47 % | 0.1329 | 0.1534 | 2.6e-7 | 130 | 1425.0 | 10.96 |
| 640x80 | 205.97 | −0.20 % | 24.34 % | 0.1330 | 0.1529 | 3.0e-7 | 156 | 2215.8 | 14.20 |
| 720x90 | 202.46 | −1.70 % | 22.39 % | 0.1617 | 0.1863 | 3.6e-7 | 204 | 3537.5 | 17.34 |
| 800x100 | 195.74 | −3.32 % | 18.32 % | 0.1645 | 0.1878 | 4.2e-7 | 246 | 4901.0 | 19.92 |

- **ω₂ is not mesh-stable.** It rises 21 % to a maximum at 560x70, then falls 5.2 % over the three finest meshes.
- **Terminal gap.** It follows ω₂ because ω₁ is nearly constant: 0.7 % → 24.5 % → 18.3 %.
- **M_nd and gray fraction.** Both sit on a plateau at 0.115–0.141 and 0.140–0.166 up to 640x80. They then **step up by about 22 %** at 720x90 and hold at 800x100. The designs were still getting less gray at the stop (TERMINATION_AUDIT.md).
- **Volume** is satisfied to ≤ 1.5e-6 at every mesh.
- **Outer iterations** are non-monotone: 121 → 93 (minimum at 400x50) → 246, rising steeply at the two finest meshes (156 → 204 → 246).
- **Total runtime and runtime per outer iteration:** see PERFORMANCE_AUDIT.md. Per-outer cost grows smoothly, p ≈ 0.75 against NE. Total wall time inherits the non-monotone outer count.

Figures: `figures/omega1_vs_mesh.png`, `omega2_vs_mesh.png`, `gap_vs_mesh.png`, `Mnd_vs_mesh.png`, `gray_fraction_vs_mesh.png`, `outer_iterations_vs_mesh.png`.

## 3. Preregistered flags

| flag | criterion | value | fired |
|---|---|---|---|
| S1 | any mesh not NATIVE_CONVERGED | 0 of 9 | no |
| S2 | any adjacent \|Δω₁\|/ω₁ > 2 % | max 1.104 % (160x20 → 240x30) | no |
| S3 | \|ω₁(800x100) − ω₁(160x20)\|/ω₁(160x20) > 5 % | 2.233 % | no |
| S4 | largest fine-4 adjacent change > 0.5 % and > largest coarse-4 | 0.137 % against 1.104 % | no |
| S5 | any \|volume error\| > 1e-3 | max 1.5e-6 | no |
| S6 | M_nd(800x100) > 2·M_nd(160x20) or gray(800x100) > 2·gray(160x20) | ratios 1.44 and 1.32 | no |
| S7 | spike event in the last 10 outer iterations, or evaluator status ≠ PASS | 0 spikes; all PASS | no |

## 4. Do 720x90 and 800x100 depart from the trend?

- **In ω₁: no.** They continue the small monotone decline and then flatten (165.42 / 165.43).
- **In the other quantities: yes, qualitatively.** Relative to the 400x50–640x80 plateau, the two finest meshes show:
  - **higher grayness:** M_nd +22 %, gray fraction +22 %;
  - **many more outer iterations:** 204 and 246, against 93–156;
  - **a longer near-multiplicity phase:** gap12 < 0.05 for 48 and 76 iterations, against 24–40;
  - **falling ω₂ and gap**;
  - **a different layout:** extra gray end-bay braces at 720x90, left–right asymmetry at 800x100. Neighbour IoU is 0.85 and 0.78, against 0.90–0.92 in the plateau.
- **Status.** None of this crosses a preregistered flag, and none of it is a failure. It is recorded as a **regime change at the finest meshes** whose cause is not established.
- **Candidates, not tested.** The filter radius reaching 5.4–6 elements; the unchanged per-element RMS stop tolerance; slower separation of the ω₁/ω₂ pair.
