# Multiplicity and eigenvalue structure

**MULTIPLICITY_CROSS_MESH_MIXED** for the available evidence; intended fine three-rung behavior is unknown.

The code's `subspace` method fixes N=2 every iteration. The constant final N is therefore imposed, not an observed spectral multiplicity classification. Diagonal offsets retain separation and make treating two distinct modes within a subspace a documented reconstruction choice; gap>5% is not by itself a coding error in this mode. It is, however, incompatible with calling those frequencies a reproduced near-double eigenvalue.

| Mesh | omega1 | omega2 | Relative gap12 | Fixed N | Multiple-J warnings |
| --- | --- | --- | --- | --- | --- |
| 160x20 | 169.495 | 171.96 | 0.0145418 | 2 | 2 |
| 240x30 | 167.07 | 194.109 | 0.161837 | 2 | 3 |
| 320x40 | 165.951 | 183.77 | 0.107375 | 2 | 1 |
| 400x50 | 162.889 | 175.496 | 0.0773963 | 2 | 2 |
| 480x60 | 161.906 | 173.183 | 0.0696574 | 2 | 4 |
| 560x70 | 161.034 | 165.698 | 0.0289662 | 2 | 4 |
| 640x80 | 159.725 | 161.161 | 0.0089898 | 2 | 7 |
| 720x90 | 159.086 | 160.554 | 0.00922841 | 2 | 43 |
| 800x100 | 153.302 | 154.223 | 0.00600634 | 2 | 81 |


Four actual endpoints from 240 through 480 have substantial gaps (240=16.18%,320=10.74%,400=7.74%,480=6.97%; 560 drops to 2.90%). The three finest are 0.899%,0.923%,0.601%, suggesting local approach of the first two modes, not proof of exact coalescence or global stabilization. Historical S3 C320 is even more separated, at 22.3248%, despite a mature density field and credible terminal B. Neither A/B nor fixed N enforces bimodality.

A separate material caveat appears in **all nine** solver logs: the third mode J is sometimes itself near-multiple with the next mode under the 5% numerical warning criterion. Counts are 2,3,1,2,4,4,7,43,81. The solver expressly logs `(25b) undefined` and continues with the simple-J constraint; it does not repair the spectral cluster. At 800 this occurs on 81/170=47.65% of outer iterations. At 720 it is 43/223=19.28%. These warnings are not exceptions or NaNs and are not included in the reported inner failure count.

This is a documented domain-of-validity limitation of the next-mode constraint, increasingly exercised on fine meshes. The available logs prove occurrence, not its causal effect on objective/topology. No per-iteration eigenpairs, fourth-frequency history or native residual history is retained for the campaign, so cluster identity, eigenvector rotation and mode exchange cannot be independently checked. The final common evaluator's alternative interpolation results are not substituted for native mode histories.

The multiple-J caveat warrants disclosure in any benchmark, independently of controller failure. Changing mass, multiplicity handling or MMA is outside this audit, and no such repair was made.
