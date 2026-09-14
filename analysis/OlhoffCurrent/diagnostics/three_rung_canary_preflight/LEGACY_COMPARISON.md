# LEGACY_COMPARISON — Part I

## Status

480×60 compared; 800×100 pending its run. The legacy records are **historical
comparators, not counterfactuals of equal status** — §2 says why.

## 0. Headline comparison at 480×60

| | legacy 480×60 | three-rung canary 480×60 | change |
|---|---|---|---|
| policy | beta / four-rung / designChange | stageExhaustion / three rungs | the factor under test |
| status | NATIVE_CONVERGED | CONVERGED | — |
| outer | 164 | 386 | +222 |
| inner MMA | 3 463 | 7 300 | +3 837 |
| final move / stage | 0.02 / 2 | **0.01 / 3** | reaches the last rung |
| final event | designChange_with_settledMove | terminal persistent E, branch B | — |
| ω₁ | 161.9055204201859 | **163.93225938567002** | **+2.0267 (+1.25 %)** |
| ω₂ | 173.18344069950192 | 185.21032464590309 | +12.03 |
| ω₃ | 404.0154434349261 | 401.39946949684344 | −2.62 |
| gap12 | 0.06965741656026896 | **0.12979791372346** | ×1.86 wider |
| gap23 | 1.3329 | 1.1673 | narrower |
| M_nd | 34.67169512013076 % | **26.34156302529312 %** | **−8.330 points** |
| grayness | 0.34672 | 0.28729 | −0.0594 |
| volume | 0.4999990211500618 | 0.4999984544949872 | both at target |
| multJ warnings | 4 of 164 (2.44 %) | 4 of 386 (**1.04 %**) | lower incidence |
| inner non-converged | 0 | 0 | — |
| wall | 898.259887875 s | 2799.278047375 s | ×3.12 — **not like-for-like**, see §2 |

The legacy run stopped at stage 2 one iteration after a move halving; the
canary reached the terminal rung and stopped on a 20-iteration persistent
exhaustion window. Direction of every scientific quantity matches the four
in-range meshes: more work, better ω₁, less gray, and here also a wider gap12.

**Do not read the ×3.12 wall ratio as a cost of the controller.** It mixes
three things: the recorder (`diagnostics` on vs off), the stage mix (the canary
spends 20 % of its iterations in stages that cost 2–2.3× a stage-1 iteration),
and a different session. `PERFORMANCE_DECOMPOSITION.md` §5 separates what can
be separated.

## 1. Legacy comparators, exact repository values

Source: `nine_mesh_campaign_audit/MASTER_TABLE.csv`, policy
`legacy_beta_four_rung` (ladder `[0.04 0.02 0.01 0.005]`, `boundVariable`
continuation, `designChange` stopping, cap 400, `runtime.diagnostics = false`).

| quantity | 480×60 | 800×100 |
|---|---|---|
| status | NATIVE_CONVERGED | NATIVE_CONVERGED |
| outer | 164 | 170 |
| inner MMA | 3 463 | 3 713 |
| mean inner / outer | 21.116 | 21.841 |
| ω₁ | 161.9055204201859 | 153.3020068531485 |
| ω₂ | 173.18344069950192 | 154.2227913711982 |
| ω₃ | 404.0154434349261 | 421.654662666513 |
| gap12 | 0.06965741656026896 | 0.0060063435368576425 |
| gap23 | 1.3329 | 1.7341 |
| multiplicity N | 2 | 2 |
| volume | 0.4999990211500618 | 0.4999880615005176 |
| M_nd [%] | 34.67169512013076 | 50.65606664059261 |
| grayness | 0.3467169512013076 | 0.5065606664059261 |
| final move | 0.02 | 0.02 |
| final stage | 2 | 2 |
| final event | designChange_with_settledMove | designChange_with_settledMove |
| eps (l2) | 0.15 | 0.25 |
| terminal ‖dρ‖₂ | 0.09681650704869162 | 0.06475930168641426 |
| amp / tol | 0.6454 | 0.2590 |
| last move change at | 163 | 169 |
| terminal stage observed duration | 2 | 2 |
| wall total [s] | 898.259887875 | 1987.063825 |
| wall / outer [s] | 5.477194438262195 | 11.688610735294118 |
| eigen(+assembly) / outer [s] | 0.270224725355691 | 1.1142149950980391 |
| gradient / outer [s] | 0.012431164380081303 | 0.03570811029411765 |
| inner MMA / outer [s] | 5.19189813262195 | 10.529919581862746 |
| s per MMA step | 0.24587678133121565 | 0.4821132046637939 |
| inner share [%] | 94.79 | 90.09 |
| eigen share [%] | 4.93 | 9.53 |
| multJ warnings | 4 | 81 |
| inner non-converged | 0 | 0 |
| effective config hash | `a49417d0571d3c24…` | `9321858983a3d7d3…` |

Both stopped **one iteration after a move halving**, at stage 2, with the
fourth rung never reached — the pattern the nine-mesh audit flagged across
seven of nine meshes.

## 2. Why these are not equal-status counterfactuals

The task permits calling them counterfactuals only if the configurations differ
**solely** in the controller fields relevant to the comparison. They do not:

| field | legacy | canary | relevant to the comparison? |
|---|---|---|---|
| `move.levels` | `[0.04 0.02 0.01 0.005]` | `[0.04 0.02 0.01]` | yes — under test |
| `move.continuation.signal` | `boundVariable` | `stageExhaustion` | yes — under test |
| `stop.rule` | `designChange` | `stageExhaustion` | yes — under test |
| `runtime.maxOuter` | 400 | 1600 | **no** — a budget, but it bounds the legacy result: a legacy run that wanted more than 400 could not have it |
| `runtime.diagnostics` | false | **true** | **no** — but it changes wall time materially, so every timing comparison is confounded |

The last two rows are why the legacy values are labelled **historical
comparators** throughout this study and never "the counterfactual". The
scientific fields (18 of 18 formulation settings) are identical — that part of
the comparison *is* clean.

## 3. The in-range three-rung results, which bound expectation

Validated three-rung endpoints, all `CONVERGED` on branch B at `move = 0.01`
(`nine_mesh_campaign_audit/LEGACY_VS_THREE_RUNG.csv`, sources in
`two_branch_controller_validation` and `three_rung_*`):

| mesh | outer (legacy → three-rung) | inner (legacy → three-rung) | ω₁ (legacy → three-rung) | M_nd % (legacy → three-rung) |
|---|---|---|---|---|
| 160×20 | 91 → 180 | 2 241 → 4 283 | 169.495 → 169.975 | 13.40 → 12.76 |
| 240×30 | 104 → 284 | 2 334 → 5 506 | 167.070 → 167.039 | 15.60 → 12.92 |
| 320×40 | 131 → 352 | 2 614 → 6 498 | 165.951 → 166.426 | 23.36 → 12.94 |
| 400×50 | 139 → 466 | 2 918 → 8 848 | 162.889 → 166.452 | 32.33 → 15.37 |

The pattern in range: **more work, markedly less gray, ω₁ equal or better**, and
the M_nd advantage widens with refinement (−0.6, −2.7, −10.4, −17.0 points).
Whether it continues at 480 and 800 is the question the canaries exist to
answer, and it is open.

**Do not infer improvement from more iterations.** The three-rung runs are
longer by construction — the terminal rung must satisfy a 20-iteration
persistence before it may stop. Length is a property of the rule, not evidence
of quality; the M_nd and ω₁ columns are the evidence, and only at meshes where
both were measured.

## 4. The level, not just the direction

Improvement over legacy is not the same as a good design. Across the five
meshes where three-rung has now been measured:

| mesh | three-rung M_nd % | three-rung ω₁ | legacy M_nd % | advantage |
|---|---|---|---|---|
| 160×20 | 12.756 | 169.975 | 13.402 | −0.65 pts |
| 240×30 | 12.917 | 167.039 | 15.604 | −2.69 pts |
| 320×40 | 12.940 | 166.426 | 23.360 | −10.42 pts |
| 400×50 | 15.373 | 166.452 | 32.328 | −16.96 pts |
| **480×60** | **26.342** | **163.932** | 34.672 | **−8.33 pts** |

Three-rung M_nd sat near 13 % through 320×40, rose to 15.4 % at 400×50, then
**jumps to 26.3 % at 480×60**, while the advantage over legacy *narrows* from
17.0 to 8.3 points and ω₁ falls from 166.45 to 163.93 after being flat across
240–400.

So the controller generalizes at 480×60 — every gate passes — but the design it
produces is markedly grayer than the same controller produced one mesh coarser.
That is the first quantitative evidence that fine-mesh degradation is not
purely a stopping artefact. One mesh is not a trend; 800×100 is the test.

## 5. 800×100

| | legacy 800×100 | three-rung canary 800×100 | change |
|---|---|---|---|
| status | NATIVE_CONVERGED | CONVERGED | — |
| outer | 170 | 468 | +298 |
| inner MMA | 3 713 | 10 404 | +6 691 |
| final move / stage | 0.02 / 2 | **0.01 / 3** | reaches the last rung |
| final event | designChange_with_settledMove | terminal persistent E, branch B | — |
| ω₁ | 153.3020068531485 | **161.94583808332942** | **+8.644 (+5.64 %)** |
| ω₂ | 154.2227913711982 | 161.95148607903909 | +7.73 |
| ω₃ | 421.654662666513 | 382.47940150449614 | −39.18 |
| gap12 | 0.0060063435368576425 | **3.4875831182270845e−05** | both near-degenerate |
| gap23 | 1.7341 | 1.3617 | — |
| M_nd | 50.65606664059261 % | **34.41232077776818 %** | **−16.244 points** |
| grayness | 0.50656 | 0.37763 | −0.1289 |
| volume | 0.4999880615005176 | 0.49999367812652545 | both at target |
| multJ warnings | 81 of 170 (47.65 %) | 81 of 468 (**17.31 %**) | same count, 2.8× the iterations |
| inner non-converged | 0 | 0 | — |
| wall | 1987.063825 s | 7554.096479416667 s | ×3.80 — **not like-for-like** |

Topology, measured directly from both density fields
(`evidence/topology_800_comparison.json`, `figures/FIG_12_*`):

| metric | value |
|---|---|
| IoU at threshold 0.5 | 0.7021 |
| threshold flip fraction | 0.15453 |
| density L1 distance | 0.09886 |

The two designs share a topology family but differ substantially in the details:
15.5 % of elements fall on opposite sides of the 0.5 threshold. The legacy field
is a gray blur; the canary field has resolved flanges, a defined central void
and distinct diagonal members.

**The wall ratio is not a controller cost.** The canary ran 2.75× more
iterations, spent 16.7 % of them in stages costing 1.5–2.4× a stage-1 iteration,
and retained a full trajectory (`diagnostics = true`) the legacy campaign did
not. Per *outer iteration* the ratio is 1.38, not 3.80.

## 6. The trend that matters

Improvement over legacy at every mesh — and a design that keeps getting worse:

| mesh | three-rung M_nd % | three-rung ω₁ | legacy M_nd % | legacy ω₁ | advantage |
|---|---|---|---|---|---|
| 160×20 | 12.756 | 169.975 | 13.402 | 169.495 | −0.65 pts |
| 240×30 | 12.917 | 167.039 | 15.604 | 167.070 | −2.69 pts |
| 320×40 | 12.940 | 166.426 | 23.360 | 165.951 | −10.42 pts |
| 400×50 | 15.373 | 166.452 | 32.328 | 162.889 | −16.96 pts |
| 480×60 | **26.342** | 163.932 | 34.672 | 161.906 | −8.33 pts |
| 800×100 | **34.412** | 161.946 | 50.656 | 153.302 | −16.24 pts |

Under the correct controller, held fixed: M_nd rises monotonically and
accelerates (12.8 → 34.4 %), ω₁ falls monotonically (169.98 → 161.95), and
neither shows an asymptote. The three-rung 800×100 design is **as gray as the
legacy 480×60 design** (34.41 % vs 34.67 %).

So the controller wins every head-to-head comparison and still does not deliver
a mesh-converged design. That is the central finding of this study, and it is
why the campaign decision turns on spectral/discretization questions rather than
on stopping. See `CAMPAIGN_DECISION.md`.

## 7. What was NOT inferred

No three-rung result at 560×70, 640×80 or 720×90 — those meshes were not run and
nothing is interpolated across them. No claim that legacy wall times and canary
wall times are comparable without the caveats in §2. No claim that the legacy
records are counterfactuals of equal status.
