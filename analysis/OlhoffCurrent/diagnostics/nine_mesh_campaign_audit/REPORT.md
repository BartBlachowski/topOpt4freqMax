# BOTTOM LINE

**No, the struggle was not for nothing. But the supposed final test of its outcome did not happen.** The completed September 11 campaign ran the old beta/four-rung policy on all nine meshes. The three-rung E-controller was validated locally but never promoted into the production preset used by this campaign. Calling these nine results a failure—or a success—of that controller is scientifically wrong.

The owner's impression is **partly correct** about delivery: substantial development did not reach the final production experiment, and the actual fine-mesh outputs are increasingly gray and poorly supported as mature designs. It is **wrong** about scientific value: removing 0.005 demonstrably eliminated enormous wasted work at 240/320 with negligible endpoint change, and the earlier investigation exposed real deficiencies of beta stopping.

The one primary recommendation is **IMPLEMENTATION_OR_CAMPAIGN_INVALID** (option E: wrong campaign policy, not blanket numerical corruption). No new optimization, retuning, source repair, rerun or cap extension was performed. This audit does not automatically recommend another campaign.

## Evidence that decides the case

1. All nine MAT effective configs and manifest configs say `[0.04,0.02,0.01,0.005]`, `boundVariable`, `designChange`, cap 400. Independently recomputed config hashes match 9/9; source hashes match 21/21 and the 75-file implementation tree is exact.
2. Production config still delegates to the unpromoted old preset. The earlier promotion-closure report explicitly recorded promotion blocked. There is no hidden three-rung fallback explanation.
3. All nine legacy solves stop **one iteration after a move reduction**; seven stop at 0.02. Full Olhoff histories were discarded, so terminal E cannot be replayed for any campaign mesh.
4. Legacy omega1 decreases 169.495→153.302 while Mnd increases 13.40→50.66%. The final refinement loses 3.64% frequency and yields adjacent topology IoU 0.753 after the previous pair's 0.960.
5. Historical retained E trajectories independently reproduce every A/B decision on 160/240/320/400. Removing 0.005 saves 70034 inner steps at 320 and 38675 at 240 with negligible density/objective change. The separate validated C320 MAT is missing, but its recorded 352-row scientific CSV matches the retained oracle prefix.

## Final verdicts and their scope

| Category | Verdict |
| --- | --- |
| CAMPAIGN | NINE_MESH_CAMPAIGN_INTEGRITY_FAIL |
| CONTROLLER | THREE_RUNG_CONTROLLER_CROSS_MESH_INCONCLUSIVE |
| TERMINATION | TERMINATION_CROSS_MESH_NOT_CREDIBLE |
| MESH OBJECTIVE | OBJECTIVE_MESH_CONVERGENCE_INCONCLUSIVE |
| TOPOLOGY | TOPOLOGY_MESH_CONVERGENCE_INCONCLUSIVE |
| MULTIPLICITY | MULTIPLICITY_CROSS_MESH_MIXED |
| PERFORMANCE | PERFORMANCE_SCALING_COSTLY_BUT_INTERPRETABLE |
| VALUE OF THREE-RUNG WORK | THREE_RUNG_WORK_PARTIALLY_JUSTIFIED |
| PROGRAMME | IMPLEMENTATION_OR_CAMPAIGN_INVALID |


Campaign/controller/objective/topology conclusions address the requested intended method and the limits of the available evidence. Termination, multiplicity and performance describe the actual legacy records where measurements exist; they must not be misread as fine-mesh three-rung observations. The termination vocabulary lacks an inconclusive code, so NOT_CREDIBLE rejects the actual campaign's scientific-maturity claim while leaving the absent intended fine-mesh test unassessed.

## Observed endpoint summary

| Mesh | Outer | Inner MMA | omega1 | Mnd % | Final move | Wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 160x20 | 91 | 2241 | 169.495 | 13.4025 | 0.01 | 119.761 |
| 240x30 | 104 | 2334 | 167.07 | 15.6036 | 0.01 | 214.408 |
| 320x40 | 131 | 2614 | 165.951 | 23.3596 | 0.02 | 335.713 |
| 400x50 | 139 | 2918 | 162.889 | 32.3283 | 0.02 | 543.591 |
| 480x60 | 164 | 3463 | 161.906 | 34.6717 | 0.02 | 898.26 |
| 560x70 | 190 | 3922 | 161.034 | 37.014 | 0.02 | 1334.93 |
| 640x80 | 199 | 4324 | 159.725 | 39.7204 | 0.02 | 1773.79 |
| 720x90 | 223 | 4831 | 159.086 | 41.2418 | 0.02 | 2397.57 |
| 800x100 | 170 | 3713 | 153.302 | 50.6561 | 0.02 | 1987.06 |


The authoritative full table is [MASTER_TABLE.csv](MASTER_TABLE.csv), with blanks for unavailable data and definitions in the section reports. It never presents these as three-rung endpoints.

## Direct answers

**1. Are all nine runs trustworthy?** As legacy endpoint records, internally consistent with caveats: source/config hashes, raw/JSON scalars and final densities agree. As the requested three-rung campaign, no: all nine have the wrong policy and lack retained histories.

**2. Did all nine converge?** All nine report NATIVE_CONVERGED under legacy designChange. None demonstrated terminal persistent E at 0.01; seven stop at 0.02. A programmed stop is not proof of mature topology.

**3. Did the three-rung controller generalize?** Unknown. It was not executed in the nine-mesh campaign.

**4. First problematic mesh?** 160×20 for policy identity. 320×40 is the first legacy endpoint to stop at stage 2; 800×100 shows the strongest fine-mesh deterioration.

**5. Systematic refinement trend?** Omega1 falls and grayness rises throughout. Inner work grows through 720 then drops at 800; multiple-J warnings increase sharply at 720/800.

**6. Does omega1 converge?** No supported asymptotic conclusion. 169.495→153.302, with a 3.64% loss at the last refinement; no justified limit/order fit.

**7. Does topology converge?** A broad beam family persists, but density convergence is not demonstrated. Adjacent IoU improves to 0.960 at 640→720 then falls to 0.753 at 720→800.

**8. Do gap and multiplicity stabilize?** Only the last three legacy gaps are below 1%; full-sequence gaps are not stable. Fixed N=2 is imposed, and J=3 approximation warnings grow to 81/170 iterations.

**9. Is stopping physically credible?** Not as an across-mesh maturity claim for these records: every stop is one iteration after a move halving. The separate historical E endpoints have much better supporting trajectory/counterfactual evidence.

**10. How much cost is eigensolving?** Recorded assembly+eigensolve share rises from 2.15% to 9.53%; pure eigensolver time is not separable from FE assembly.

**11. How much is nested MMA?** 97.60% at 160 and 90.09% at 800, dominant on every mesh. Mean inner work per outer remains about 20–25.

**12. Fitted C and p?** Legacy total wall time in seconds with N=NE: all-nine C=0.03701857,p=0.9825066,R²log=0.98002; fine-five C=0.1217553,p=0.8765449,R²log=0.85324. Fine-four p=0.6503 is unstable. Full component fits, residuals and conditional intervals are provided.

**13. Does iteration count scale badly?** Not in the measured aggregate: all-nine outer p=0.2651; per-outer cost p=0.7174. The 800 count reduction is a problematic stopping-regime change, not efficiency validation.

**14. Are fine meshes scientifically better?** Not demonstrated: lower omega1, much higher grayness and no evidence of comparable terminal maturity. Smaller gap alone is insufficient.

**15. What did three-rung improve?** Against four-rung E, remove real wasted work with nearly unchanged useful designs. Against beta production, more mature density fields and a meaningful C400 frequency gain, at increased work.

**16. What did it fail to improve?** It did not establish exact paper reproduction, spectral degeneracy, optimality, continuum topology or fine-mesh practicality. Its production adoption also failed.

**17. Could something simpler do as well?** Retrospectively, keeping S1 E then fixed 39-update dwells at 0.02 and 0.01 reproduces all four known three-rung endpoints. Plain fixed 0.04 and earlier S2 termination are not universally supported; no fine-mesh result tests the simpler dwell alternative.

**18. Did 0.005 deserve removal?** Yes on all four known meshes: C320 saves 78.00% outer and 91.51% inner; C240 saves 79.09% outer and 87.54% inner, with negligible scientific differences.

**19. Was A/B investigation useful?** Yes. Independent replay confirms the different regimes and causal timing; objective stagnation and amplitude alone were inadequate universal maturity signals.

**20. Did this campaign show further controller tuning has low value?** It did not test controller tuning at all. Historical minimum-duration late stages suggest limited demonstrated value of late-stage adaptivity, but that is separate evidence.

**21. Good enough for a benchmark table?** Only as explicitly labelled legacy reconstruction endpoint/timing data with limitations. Not as a validated three-rung performance benchmark or equal-quality comparison.

**22. Good enough for a mesh-independent convergence claim?** No.

**23. Faithful reconstruction rather than new method?** Best called a documented reconstruction, with partial qualitative fidelity and explicit unsupported choices; “faithful” must not imply exact numerical or spectral reproduction.

**24. One high-value follow-up experiment?** No new controller experiment is justified now. The first need is correct campaign identity and retained evidence. The multiple-J warning concern is documented, without proposing a new experiment.

**25. Should the project stop here?** Stop the present audit without repair or rerun. Preserving the coarse-mesh result and ending the programme is defensible; condemning the three-rung controller on these nine legacy jobs is not. The primary decision is campaign invalidity, not a speculative tuning programme.

## What supports the recommendation

The audit distinguishes three comparisons. The actual nine-mesh legacy campaign has poor scientific-maturity support. Historical E-controller versus legacy beta shows that cleaner designs required additional work. Historical three-rung versus four-rung E shows that removing 0.005 retained those cleaner designs while eliminating largely useless continuation. Mixing those comparisons would make the three-rung method look either universally faster or scientifically ineffective, neither of which the evidence says.

The strongest unresolved fine-mesh numerical concern is the increasing incidence of a near-multiple third mode while the next-mode constraint assumes simplicity. It is logged on 81 of 170 outer iterations at 800. That is material disclosure, not proven causality and not a mandate to change multiplicity, mass, filter and MMA together.

No asymptotic objective limit was fitted. No missing terminal trajectory was synthesized. No historical wall-time ratio was passed off as a same-machine causal saving. Current hashes establish retained bytes, not an original missing campaign seal. Every plot was created from stored numerical evidence; the historical branch figures label the five fine meshes as absent.

## Artifact guide

Provenance/integrity: PROVENANCE.md, CAMPAIGN_INTEGRITY.md, verification.json, INTEGRITY_TABLE.csv, effective_configs.json.

Controller/science: CONTROLLER_GENERALIZATION.md, TERMINATION_QUALITY.md, MESH_REFINEMENT.md, TOPOLOGY_CONVERGENCE.md, FILTER_AUDIT.md, LITERATURE_FIDELITY.md, MULTIPLICITY_AUDIT.md, REGIME_CHANGE_AUDIT.md.

Value/cost/decision: THREE_RUNG_VALUE.md, PERFORMANCE_SCALING.md, INNER_MMA_SCALING.md, SCORECARD.md, RECOMMENDATION.md. Twelve figures are supplied in PNG and SVG, including the actual topology atlas and residual plots. Figures 8/9 necessarily use the four historical E meshes; the requested nine-mesh E data do not exist in the campaign.

Reproducibility: scripts/, INPUT_MANIFEST.json, DATA_MANIFEST.json, EVIDENCE.json, FINAL_SHA256.txt. The original paper and its erratum were inspected; no external optimization or evaluation was run.

# WHAT WE LEARNED

The local three-rung result has measurable value. Its adoption into production failed. The legacy nine-mesh campaign shows increasingly gray endpoints, schedule-linked stopping, dominant nested-MMA cost and a growing next-mode multiplicity warning regime. Source correctness, policy identity, stopping maturity and mesh convergence are separate claims.

# WHAT WE DID NOT LEARN

We did not learn whether the intended three-rung method generalizes to 480–800, whether its terminal E stays credible there, its fine-mesh computational savings, or a continuum objective/topology limit. The missing trajectories also prevent reconstructing exact late dynamics of the actual campaign. These are unavailable observations, not negative measurements.

# WHAT I WOULD DO NEXT

Preserve this audit and correct the description of the completed campaign. Keep the coarse-mesh three-rung finding as a useful, bounded result. Make no controller change or new scientific run on the strength of this audit. If the owner chooses to stop the programme, stop without pretending the legacy campaign refuted the untested controller. If the owner later chooses more work, campaign identity and durable evidence must be resolved first; this report does not authorize that work.
