# Methodology freeze gate

## Decision

**DO NOT FREEZE — SPECIFIC BLOCKERS REMAIN.**

Two blockers, both with forced, parameter-free corrections. This is not a request for redesign:
all 24 original findings are closed, both CRITICAL repairs verify independently, and no
accepted repair is being reopened. The corrective pass should be short.

## Blockers

### N1 (MAJOR) — measurement horizon is not tied to the reference-freeze index

**Defect.** `B_ref = 3200` is uniform across methods, but measurement horizons are `B0 = 900`
(Proposed), `2000` (Yuksel Stage 2), `3200` (Olhoff), and nothing requires `B0` to cover
`b_ref`. The earliest possible `b_ref` is 600, and it may legitimately fall anywhere in
`[600, 3200]`. Compounding this, the single permitted extension requires (condition 3) that
the sustained floor **still exceed** the preceding block — the exact logical negation of the
stabilisation condition that defines `b_ref`. A cell that stabilises and then hits its
measurement cap is therefore denied its extension *because* it stabilised.

**Why blocking.** A validly frozen `Q_ref` can be unreachable within its own measurement
budget, producing `NOT_REACHED` for budget-bookkeeping reasons rather than maturation. The
exposure is method-correlated and unequal — Proposed largest, Olhoff none — and it did not
exist in Phase 1A, where the reference lived inside the measurement horizon. It is a defect
created by the C2 repair. Left unfixed, an implementer meets the case mid-production and must
choose between censoring and extending with method identity and consequences visible: a
results-affecting decision taken after seeing data.

**Minimum correction.** Set the measurement horizon per cell to
`min( max(B0, b_ref + P - 1), B_ref )`, and harmonise extension condition 3 so a
stabilised-but-uncertified cell is not denied its tranche. Both use only already-frozen
quantities; no new threshold is introduced. The reference trajectory has already demonstrated
the run to that length, so the cost is nil.

### N2 (MAJOR) — three live contradictions in the master protocol document

**Defect.** `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` narrative sections assert superseded
rules in the present tense: (a) "every evaluator, sensitivities ... remain supplementary"
(contradicts M4); (b) `k_enter` and `k_cert` as co-equal fitted layers (contradicts Mo1);
(c) "T1/T0 sensitivity" as a live control (contradicts C1 and Mo6).

**Why blocking.** A frozen normative set may not contradict itself on three points that were
themselves audit findings, and (a) governs the paper package — the exact surface M3 and M4
were raised to protect. Mitigation is real but partial: each section defers binding detail to a
correct spec, so the acceptance engine is not at risk.

**Minimum correction.** Rewrite the three passages to match the binding specs; annotate the
WP0 "E1 raw was preregistered as primary" sentence as historical. **Zero scientific content
changes.**

## Not blockers

`N3` (aggregate detached area absent from paper-facing tables) and `N4` (the 6.1–7.2%
disclosure is numerically wrong; actual 6.2–8.5% over Proposed across eight meshes, and the
error originates in my own original M3 wording) are mandatory reporting corrections that do
not gate implementation. Apply them in the same pass.

## Re-verification scope

The corrective pass requires confirmation of **N1, N2, N3, N4 only**. No re-audit of the 24
closed findings, no new design cycle, and no reopening of `A_sig`, the stabilisation
constants, the q-family, `P`, the evaluator rule, or the method-specific gates — all of which
this delta audit accepts.

---

# What the freeze will cover, once the blockers clear

Recorded now so the corrective pass is unambiguous.

## Normative documents constituting the freeze

1. `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` — master protocol (after N2)
2. `ACCEPTANCE_GATE_SPEC.md` — endpoint definition and status precedence
3. `REFERENCE_QUALITY_SPEC.md` — reference construction
4. `QUALITY_EFFORT_SPEC.md` — quality levels and estimand semantics
5. `TOPOLOGY_SANITY_SPEC.md` — topology gate
6. `ITERATION_ACCOUNTING_SPEC.md` — counting semantics
7. `TIMING_SPEC.md` — timing architecture
8. `SCALING_AND_FIGURE_SPEC.md` — fit discipline and figures
9. `PROPOSED_TABLE_LAYOUTS.md` — table layouts (after N3)
10. `IMPLEMENTATION_REQUIREMENTS.md` — implementation contract (after N1)
11. `FAIRNESS_RISK_REGISTER.md` — F01–F31
12. `EVIDENCE_AVAILABILITY_MATRIX.csv` — evidence classification

On freeze, hash all twelve into an immutable protocol manifest before the first production
identity is unblinded.

## Quantities that may NOT change after seeing Phase-2 results

Changing any of these post-hoc reopens methodology review in full:

**Acceptance and topology** — `A_sig = 0.01` and the `a_sig(j) = ceil(A_sig/A_e(j))` rule; the
absence of an aggregate-area veto; support-to-support connectivity and its even-`nely`
precondition; the exact-count projection and its tie-break; volume tolerance `1e-3` on the raw
field via `H.rV`.

**Reference** — `P = 100`, `L_ref = 500`, `epsilon_ref = 0.001`, `B_ref = 3200`; the causal
first-passage rule; **the absence of any cap fallback**; `REFERENCE_NOT_ESTABLISHED` and
`REFERENCE_SOLVER_TERMINATION` as valid published outcomes.

**Quality** — `q ∈ {0.980, 0.990, 0.995}` as co-primary; the acceptance rule
`min_e [Q_e/Q_ref_e] >= q`; E1/E2/E3 as co-equal decompositions; the mandatory best-observed
benchmark; `A_NOT_INSTANTIATED` unless an independently provenance-locked `Omega_req` predates
production.

**Counts and method gates** — `k_enter` as primary and `k_cert = k_enter + P - 1` as the paired
certification location; the per-method counting units; Yuksel Stage-2-only eligibility; Olhoff
`policyStage=2 AND N=2 AND gap12<=0.01`; Proposed's absence of an added gate; `k_gate` and
`k_native` reporting.

**Analysis** — `k_enter` scaling primary and `k_cert` fits descriptive; mandatory
common-support companion fits and the prohibition on cross-method exponent comparison outside
common support; LOO ranges; the `WEAKLY_IDENTIFIED` criteria (`R2_log<0.80`, LOO range spans
zero, LOO width `> |p|`); the three-valid-mesh minimum; censoring discipline and status
precedence; "empirical scaling over the tested mesh range" language.

**Measurement horizons** — after N1, the horizon rule itself becomes frozen and may not be
adjusted per cell once results are visible.

## What may change during Phase 2 without reopening review

Storage and caching layout; whether the LP diagnostic mirror is built (`NA` if not); whether
the measurement trajectory is re-run or proven bit-identical to the reference prefix;
scheduling and run order for timing replays within the frozen no-mixing rule; and any
engineering defect fix that passes an identity test and is applied uniformly without
inspecting comparative ranks.
