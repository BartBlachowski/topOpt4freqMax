# Objective and scientific refinement

**OBJECTIVE_MESH_CONVERGENCE_INCONCLUSIVE.** There is no nine-point sequence for the intended three-rung method. Even the actual legacy sequence does not justify an asymptotic convergence claim.

The nine-mesh observations below are the **legacy beta/four-rung campaign**, not a three-rung E-controller campaign. This distinction applies to every numerical table and plot unless explicitly labelled historical E-controller evidence.

| Mesh | h/b | omega1 rad/s | omega2 rad/s | Relative gap | Mnd % | Volume error |
| --- | --- | --- | --- | --- | --- | --- |
| 160x20 | 0.05 | 169.495 | 171.96 | 0.0145418 | 13.4025 | -9.91222e-07 |
| 240x30 | 0.0333333 | 167.07 | 194.109 | 0.161837 | 15.6036 | -7.82598e-07 |
| 320x40 | 0.025 | 165.951 | 183.77 | 0.107375 | 23.3596 | -8.61382e-07 |
| 400x50 | 0.02 | 162.889 | 175.496 | 0.0773963 | 32.3283 | -8.50337e-07 |
| 480x60 | 0.0166667 | 161.906 | 173.183 | 0.0696574 | 34.6717 | -9.7885e-07 |
| 560x70 | 0.0142857 | 161.034 | 165.698 | 0.0289662 | 37.014 | -1.15965e-06 |
| 640x80 | 0.0125 | 159.725 | 161.161 | 0.0089898 | 39.7204 | -2.81187e-06 |
| 720x90 | 0.0111111 | 159.086 | 160.554 | 0.00922841 | 41.2418 | -7.72698e-06 |
| 800x100 | 0.01 | 153.302 | 154.223 | 0.00600634 | 50.6561 | -1.19385e-05 |


Omega1 decreases at every refinement: 169.495→167.070→165.951→162.889→161.906→161.034→159.725→159.086→153.302. The net loss is 9.55%; the last step alone loses 3.64%, larger than the preceding fine steps. This is not an observed plateau or a smooth asymptotic sequence. Monotone decrease is compatible with several possible limits; nine finite points cannot mathematically refute eventual convergence. The final discontinuity prevents a credible fitted limit or empirical order. **No y_inf+C h^p fit is forced.**

Mnd increases monotonically 13.40→50.66%; the intermediate-density fraction also rises strongly. The volume stays close to 0.5 but increasingly undershoots, with final error −1.194e−5 at 800. These are not feasibility failures on their own. They demonstrate that objective and topology quality cannot be reduced to accurate volume satisfaction.

Omega2 peaks at 240 then decreases. Gap12 is highly nonmonotonic across the full sequence, then settles below 1% at 640–800 while omega1 and topology worsen. Small spectral gap is therefore not evidence of successful overall refinement. Spectral and density metrics must be read jointly.

The five new fine legacy designs are not demonstrated scientific improvements over 160–400: their first frequencies are lower, grayness higher, and no mature-state or optimality comparison is available. FE discretization error and optimization endpoint error are confounded because each design changes with mesh. A fixed-design FE refinement study was not stored; none is run here.

[Figure F01](figures/F01_omega1_refinement.png) plots both NE and h/b=1/nely. [F02](figures/F02_spectrum.png) and [F07](figures/F07_grayness.png) show the other full-sequence measures.
