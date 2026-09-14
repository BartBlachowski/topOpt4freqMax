# Phase 1C scaling and figure specification

Status: empirical-analysis design for independent delta audit; no production output.

## 1. Layered story

Keep four quantities separate:

1. self-referenced maturation: `k_enter(Ne|q)`;
2. certification: `k_cert=k_enter+P-1`;
3. native algorithm structure and mean update cost;
4. reference-platform `T_enter/T_cert` and separately disclosed reference cost.

No scalar combines them. Counts must be adjacent to absolute achieved/reference quality.

## 2. Canonical observations and censoring

All points come from canonical endpoint/status/timing tables with method, mesh, `Ne`, q,
evaluator semantics, endpoint/stage, value, unit, status, fit eligibility/exclusion,
`k_native`, `k_gate`, source hashes, and replay ID.

Fit only positive finite certified R values. `PASS_WITH_LATER_SOLVER_TERMINATION` remains
eligible. Reference-not-established, solver termination, quality/topology failure,
persistent nonacceptance, cap, missing data, and fingerprint mismatch are visibly censored
at the observed boundary and excluded. Carry the exact backend subclass; known Olhoff LP
events say `GENERIC_LP_ITERATION_LIMIT_ONLY`, not formulation failure. Never impute, pool,
drop an eligible outlier, or choose a range after seeing slopes.

## 3. Primary quality-dependent count fits

The primary count family is robust-all-evaluator

\[
k_{enter}(N_e\mid q)=C(q)N_e^{p(q)},\quad q\in\{0.980,0.990,0.995\}.
\]

Report E1-, E2-, and E3-only fits as co-equal mandatory decompositions. Put the three q
values together; any material dependence of `p` on q or evaluator is a result, not a
supplementary nuisance. `k_cert` raw points remain a prominent certification companion,
but its power fit is secondary/descriptive because it fits a power law plus the fixed
constant `P-1`. If shown, its caption states this and the pipeline verifies that fitting
`k_cert-(P-1)` reproduces the `k_enter` fit exactly.

## 4. Fit discipline

For every eligible series, use unit-weight OLS on
`log(y)=log(C)+p log(Ne)`. Report:

- `C`, `p`, `R2_log`, `n_valid`;
- valid `Ne_min/Ne_max` and exact included meshes;
- leave-one-valid-mesh-out `p_LOO_min/p_LOO_max`;
- full-range or common-support label;
- q, evaluator semantics, endpoint/stage, and exclusions.

Require at least three distinct meshes. With fewer, show observations and status but no
fit. Fit lines span only the observed valid range. `C` is the formal intercept at `Ne=1`
and has no physical interpretation here.

Every per-method full-range fit has a **common-support companion fit** restricted to the
meshes valid for every compared method at the same q/evaluator/quantity. Cross-method
exponent comparison is prohibited outside this common support. If common support has
fewer than three meshes, state that no comparative exponent exists.

Label a fit `WEAKLY_IDENTIFIED` if any preregistered condition holds:

- `R2_log < 0.80`;
- the leave-one-out p range includes zero; or
- `(p_LOO_max-p_LOO_min) > abs(p)`.

Always show the underlying values, even if no label fires. These deterministic diagnostics
do not estimate sampling uncertainty. Nine meshes span only 1.398 decades, five lie in
the last half-decade, and there is one deterministic trajectory per cell; no confidence
interval or asymptotic-complexity claim is supported.

## 5. Mandatory figure set and compact placement

### F1 — quality versus native method-level updates (main)

For every method and representative mesh, show E1/E2/E3 raw quality, evaluator-specific
sustained floors, frozen references, q=98/99/99.5 lines, persistent enter/cert markers,
native stop, and method-gate satisfaction. Use small multiples rather than overlaying
incommensurate native counts. This is the primary scientific figure.

### F2 — `k_enter` versus `Ne` by q (main, adjacent to F3)

Three q panels or facets, robust-all raw points, fits, censored markers, common-support
fit distinctions, LOO ranges, and absolute reference-quality cue. E1/E2/E3 component
versions are mandatory companion panels. Put this in-axes note:

`Method-level design-update counts are not equal work; frozen 800x100 per-update cost differed by up to 3.8x. See adjacent F3.`

### F3 — mean native-update cost versus `Ne` (main, adjacent to F2)

Proposed OC, Yuksel Stage 1/Stage 2 plus labelled combined mean, and Olhoff outer updates.
State that Proposed has no per-loop eigensolve while the frozen Olhoff eigensolve was 75%
of outer cost at 800x100.

### F4 — absolute reference/endpoint quality versus `Ne` (main)

Show E1/E2/E3 `Q_ref`, accepted endpoint values, ratio to best-observed, and statuses.
The caption states the frozen evidence: Olhoff led Proposed by 6.2–8.5% and Yuksel by
5.9–7.7% in common raw-E1 `omega1` across the eight meshes with a complete method triple.
The Olhoff 800x100 endpoint is `RUN_ERROR`/E1 `N/A` and is explicitly excluded; no
nine-mesh value is inferred. This prevents “fewer updates means better method.”

### F5 — `T_enter` versus `Ne` by q (main)

Reference-platform medians/ranges with platform key. `T_cert` is a companion panel;
`T_reference` is a distinct calibration-cost panel and never added to endpoints.

### F6 — Yuksel decomposition (main)

Stage 1, Stage 2 to each q-level enter, chronological total, and certification companions.
Mark the three meshes whose frozen Stage 1 hit 1000 and state that the Phase-2 2000 cap
changes their handoff and breaks frozen-count comparability.

### F7 — Olhoff work decomposition (supplementary mandatory)

Outer updates and LP calls to each endpoint; they normally overlap. Solver-internal
iterations appear only when a diagnostic mirror genuinely records them, never from
`nInner=1`, and remain backend-specific supplementary diagnostics.

### F8 — topology grid (main)

Standardized raw/binary accepted or last-observed states with support markers, repaired
physical-area metrics, q/status labels, no visually substituted earlier design. Its caption
states the delta-audit observation for repaired-gate-passing Olhoff states at 640x80:
aggregate detached area ranged up to 674 elements (2.633% of solid volume), with median 65
and p95 148, while every individual detached component remained below `a_sig=64`.

### Certification count figure (main companion)

Show `k_cert` beside F2 as raw/descriptive panels, not as a co-equal exponent claim. State
the `+99` identity, proportional burden, unequal seconds, and P=50/200 rescan.

## 6. Old performance artifact mapping

| Existing artifact/concept | Decision | Phase-1C equivalent |
|---|---|---|
| `table1_complexity_fit` total wall time | REDEFINE | q-conditioned F5 endpoint time |
| fixed `p=1.5` plot | RETIRE | no neutral role |
| iteration count versus `Ne` | REDEFINE | robust `k_enter(Ne|q)` primary; `k_cert` descriptive |
| per-iteration loop time | RETAIN/REDEFINE | F3 with native units and stages |
| Yuksel stage plot | RETAIN/REDEFINE | F6 with corrected continuous handoff |
| Olhoff outer/inner plot | REDEFINE | F7 outer, LP calls, genuine solver iterations |
| native omega versus mesh | SUPPLEMENT ONLY | native-model context |
| common evaluator quality | PROMOTE/REDEFINE | F1/F4 E1/E2/E3 co-equal |
| grayness/binary plots | SUPPLEMENT ONLY | representation diagnostics |
| topology grids | REDEFINE | F8 repaired gate and endpoints |
| RAM summaries | RETIRE AS QUANTITATIVE EVIDENCE | platform RAM record only |
| failure neighborhoods | SUPPLEMENT ONLY | exact solver/status provenance |

## 7. Automatic checks and products

Generate observations/fits/source slices, vector PDF/SVG and >=300-dpi PNG, plus:

- table/plot endpoint identity;
- every q/evaluator/status point shown exactly once;
- censored points absent from fits;
- common-support mesh identity across methods;
- `k_cert-k_enter=P-1` and transformed-fit equality;
- LOO fits and weak-identification labels;
- no `nInner` relabelling;
- source/software hashes and output checksums.

Captions say “empirical scaling over the tested mesh range.” “Intrinsic complexity,”
“asymptotic complexity,” “order-optimal,” and unqualified “scales better” are prohibited.
