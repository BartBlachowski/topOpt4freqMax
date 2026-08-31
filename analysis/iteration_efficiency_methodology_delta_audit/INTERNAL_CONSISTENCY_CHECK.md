# Delta audit — internal consistency check (D12)

This check matters because four files in the package were still at Phase-1A state until late
in the repair. Every normative document was swept for the seven obsolete concepts, and each
hit was classified as **historical** (an explicitly marked delta trail, a superseded-quote, or
the "Phase 1A" column of a delta table) or **live normative** (asserted in the present tense
as a current rule).

Documents swept: `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md`, `ACCEPTANCE_GATE_SPEC.md`,
`ITERATION_ACCOUNTING_SPEC.md`, `TOPOLOGY_SANITY_SPEC.md`, `REFERENCE_QUALITY_SPEC.md`,
`QUALITY_EFFORT_SPEC.md`, `TIMING_SPEC.md`, `FAIRNESS_RISK_REGISTER.md`,
`PROPOSED_TABLE_LAYOUTS.md`, `IMPLEMENTATION_REQUIREMENTS.md`, `SCALING_AND_FIGURE_SPEC.md`,
`EVIDENCE_AVAILABILITY_MATRIX.csv`.

## Result by obsolete concept

| Obsolete concept | Live normative survivals | Verdict |
|---|---:|---|
| `a_res = 5` baseline | 0 | clean — all occurrences are marked superseded, in the Phase-1A column of the delta table, or in the audit-response narrative |
| `r_common` derived from Olhoff filter | 0 | clean — retired, and every mention states it was retired and why |
| aggregate detached-area **hard gate** | 0 | clean — every occurrence is either the superseded description or the "diagnostic only" mandate |
| E1 as unqualified universal evaluator | 0 normative; 1 borderline | see A below |
| horizon-relative `Q_ref` | 0 | clean — all occurrences describe the defect being repaired |
| Stage 1 not carried into Yuksel Stage 2 | 0 | clean — the false sentence is deleted, not softened |
| single δ_R presented as canonical | 0 | clean — 99% survives only as a permitted compact anchor that "may never stand alone" |

The seven concepts named in D12 are therefore **absent from the binding specifications**. The
delta trails are properly marked and are historical discussion, not live rules.

## But: three live contradictions in the master protocol document

`ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` was rewritten in its rule-bearing sections but three
narrative sections were not harmonised, and they assert superseded rules in the present tense.
Collected as **N2 (MAJOR)**.

**(a) "Paper evidence package" — evaluators demoted.** The section reads:

> "Olhoff outer/LP/solver work, full timing decomposition, **every evaluator, sensitivities**,
> and all 27 quality rows remain supplementary."

This directly contradicts the M4 closure. `QUALITY_EFFORT_SPEC.md` Sec 2 makes E1/E2/E3
co-equal primary decompositions and `PROPOSED_TABLE_LAYOUTS.md` states "No evaluator is called
a sensitivity." This is the single most consequential survival: a paper drafted from this
section would reintroduce precisely the presentation defect M4 was raised to remove. The
section also predates the q-family and the mandatory absolute-quality columns.

**(b) "Mandatory efficiency and scaling story" — `k_cert` restored to co-equal.** The section
lists `k_enter(Ne)` and `k_cert(Ne)` as layers 1 and 2 and states "Each eligible series uses
the descriptive free fit `y=C Ne^p`", treating both as co-equal fitted series. This contradicts
the Mo1 closure in `SCALING_AND_FIGURE_SPEC.md` Sec 3, which makes the `k_cert` power fit
secondary/descriptive with a mandatory caption caveat. The section also omits `q`-conditioning,
common-support companions (M5) and LOO ranges (Mo2).

**(c) "Hostile-review conclusions" — T1/T0 cited as live controls.** The bullet

> "exact binarization can penalize gray or fragmented designs; **T1/T0 sensitivity** and
> raw/binary evidence expose this"

asserts a control that no longer exists. T1 was deleted under C1; T0 was demoted from
"sensitivity" to a known strict diagnostic under Mo6, with the permissive-direction sensitivity
now being 1x1/3x3 FE patch scales. The stated safeguard is therefore not the one the frozen
methodology actually provides.

**(d) Borderline, not counted as a contradiction.** WP0 evidence item 3 reads "E1 raw was
preregistered as primary before this new study; no common model is truth." The past tense
explicitly scopes it to the prior study and the second clause is correct, so it is a historical
statement. It should nonetheless carry a Phase-1C annotation, since an implementer skimming
the evidence section could read "E1 primary" as current. Folded into N2 as a wording item.

## Severity assessment

Each of (a)-(c) sits in a section that defers binding detail to a named spec ("Binding details
... are in `SCALING_AND_FIGURE_SPEC.md`"; "table layouts are in `PROPOSED_TABLE_LAYOUTS.md`"),
and every one of those specs is correct. An implementer building the acceptance engine from
`ACCEPTANCE_GATE_SPEC.md`, `TOPOLOGY_SANITY_SPEC.md`, `REFERENCE_QUALITY_SPEC.md` and
`IMPLEMENTATION_REQUIREMENTS.md` would not be misled. That materially mitigates the risk.

It does not eliminate it. The reporting pipeline is auto-generated from the table/figure specs
(`IMPLEMENTATION_REQUIREMENTS.md` Sec 8), and the paper package is drawn from the very section
in (a). A frozen normative set must not contradict itself on three points that were themselves
audit findings. Rated **MAJOR** and **blocking for freeze**, with the qualification that the
correction is pure wording carrying **zero scientific content** — no threshold, rule, or
semantic changes.

## Cross-document consistency of the frozen constants

Independently confirmed that the frozen constants propagate consistently and without variant
values across the package: `A_sig`/`a_sig`, `P=100`, `L_ref=500`, `epsilon_ref=0.001`,
`B_ref=3200`, `q ∈ {0.980,0.990,0.995}`, `min_e` acceptance, `k_gate`, `k_native`,
`REFERENCE_NOT_ESTABLISHED`, `SOLVER_TERMINATION` + `GENERIC_LP_ITERATION_LIMIT_ONLY`. No
document carries a competing numeric value for any of them.

**One genuine numeric inconsistency exists** and it is not cosmetic: the measurement horizons
(`900 / 2000 / 3200`) versus the uniform reference horizon (`B_ref = 3200`), with no rule
tying the former to `b_ref`. That is **N1**, treated as a budget-harmonisation defect rather
than a consistency-of-wording issue.

## Verdict

- Obsolete Phase-1A concepts surviving normatively in the **binding specifications**: **none**.
- Live contradictions in the **master protocol document's narrative**: **three** (N2).
- Numeric/budget inconsistency across specs: **one** (N1).
