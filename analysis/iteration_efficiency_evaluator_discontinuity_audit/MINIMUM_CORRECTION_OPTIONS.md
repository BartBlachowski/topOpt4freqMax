# WP16 — Minimum-correction options
READ_ONLY_AUDIT — NOT_NEW_OPTIMIZATION_EVIDENCE — NOTHING IMPLEMENTED

Enumerated only after the audit verdict. No option is implemented, and none is recommended
on grounds of convenience.

A decisive practical fact governs every option: **E1/E2/E3 are post-hoc functions of stored
density fields.** They are not part of any optimizer. Any evaluator change can therefore be
applied by re-evaluating existing trajectories offline, with no optimization rerun,
provided the stored densities suffice.

| # | Option | Scientific justification | Methodology change | Effect on frozen results | Offline re-evaluation possible? | Optimizer reruns? | Publication implication |
|---|---|---|---|---|---|---|---|
| A | Implementation correction on source-fidelity grounds | **Not available.** WP14 finds E2 and E3 already faithful to Yuksel Eq. (10) and Du & Olhoff Eq. (4). There is no defect to correct. | none | none | n/a | no | none |
| B | Adopt the continuous form the source itself specifies — Du & Olhoff Eq. (4a), c0 = 1e5 — **in the common evaluator only**, leaving every optimizer untouched | Strongest option. (4a) is in the same paper, same equation family, offered by the authors as an explicit improvement; the authors report "negligible differences in the final results" between (4), (4a), (4b). Already implemented in this repo as `massScale` case `'4a'` and `mass_interp` `du2007_c0`. Removes the 1e5 jump; residual sensitivity near 0.1 falls to ~1e-6 relative. | Evaluator definition in contract + IMPLEMENTATION_REQUIREMENTS; **quality/reference subsystem only** | E2/E3 values shift; b_ref, k_enter, k_cert must be recomputed | **Yes** | **No** | Must be stated: common evaluator uses the C0 variant, optimizers use their native variant |
| C | Retain E2/E3 as native diagnostics; remove them from the common robust acceptance | Defensible: it concedes that a method-specific artificial-mode suppression device is not a neutral scorer. But it silently makes the study single-evaluator. | Acceptance rule (robust min over 3 -> 1) | k_enter/k_cert recomputed from E1 alone | Yes | No | Weakens the co-primary neutrality claim that motivated three evaluators |
| D | E1 as common trajectory evaluator; E2/E3 retained for endpoint sensitivity reporting only | Same concession as C but preserves E2/E3 as reported evidence at endpoints, where the frozen prior absolute-quality context already lives | Acceptance rule + reporting | as C | Yes | No | Must justify why trajectory and endpoint use different evaluator sets |
| E | Construct a genuinely neutral continuous common evaluator | Cleanest in principle, but it is new methodology, not a correction, and would need its own derivation, validation and audit | Substantial | All quality results regenerated | Yes | No | Largest revision; hardest to defend as "the frozen study" |
| F | Retain E2/E3 and add a branch-ambiguity rule (e.g. censor states with elements within a tolerance of 0.1) | Avoids changing the evaluator, but WP7 shows 38.8%–78.3% of production states are exposed, so censoring discards most of the trajectory | Acceptance/censoring rule | Most states censored; estimands likely unobtainable | Yes | No | Hard to defend: the censoring rate is driven by an arithmetic artifact |
| G | Keep everything and document the instability as a stated uncertainty | Requires claiming the estimand uncertainty is acceptable. It is not: k_cert at q = 0.995 moves 85 iterations (13.6%) | none | none | n/a | no | Would require reporting iteration counts with ~14% representation uncertainty |

## Minimum scientifically defensible correction

**Option B.** It is the smallest change that removes the defect, and it is the only option
whose replacement function is specified by the same source that specifies the current one.
It requires no optimizer rerun, no change to any protected numerical source, no change to
the topology gate, iteration accounting, timing, scaling, persistence semantics, method
profiles or mesh sequence, and it leaves each method's native optimization untouched.

Its cost must be stated plainly: it breaks the literal identity "E2 ≡ Yuksel native
interpolation, E3 ≡ Olhoff native interpolation". After Option B the common evaluators
would mirror each method's *material model family* while using the source's continuous
variant of the shared low-density device. Whether that is acceptable is a methodology
decision, not an audit finding, and it is exactly the decision a delta audit should take.

## Cross-link to Phase 2B, recorded but not acted on

Under Option B the float32 sensitivity that produced the Phase-2B failure is largely
removed: with c0 = 1e5 the low branch has slope `6·c0·x^5 = 6` at x = 0.1, so a float32
perturbation of 2.9e-8 changes g by about 1.7e-7, i.e. ~1.7e-6 relative — against the
1e5-fold jump today. A corrected methodology would therefore very likely permit
single-precision Olhoff trajectory storage. That is a reason to sequence the evaluator
decision **before** any repeat precision qualification, not a reason to prejudge either.
