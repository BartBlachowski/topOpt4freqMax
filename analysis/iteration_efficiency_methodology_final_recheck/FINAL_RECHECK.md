# Phase 1F — restricted N1–N4 re-verification

**Reviewer:** the same independent methodologist who issued the original hostile audit and
the delta audit.

**Scope:** N1–N4 only. The 24 original findings remain closed and were not reopened; nothing
in the Phase-1E repair invalidates any of their closures (see R5 below).

**Constraints honoured:** read-only. No specification, audit directory, MATLAB source, or
frozen artifact was modified; no optimizer was run and no trajectory generated. Verified at
the end of this check: both audit directories retain their original timestamps, the frozen
`s1_800x100.mat` remains 0 bytes, and `git status` shows no tracked modification.

---

## Verdict

All four items **CLOSED**. No new blocker. Two immaterial precision notes carried forward as
non-blocking obligations.

| Item | Was | Verdict |
|---|---|---|
| N1 — measurement-budget contract | MAJOR, blocking | **CLOSED** |
| N2 — master-protocol synchronization | MAJOR, blocking | **CLOSED** |
| N3 — aggregate topology diagnostics | MODERATE | **CLOSED** |
| N4 — absolute-quality numbers | MODERATE | **CLOSED** |

---

## R1 — N1: measurement-budget contract

The binding rule now reads, in `IMPLEMENTATION_REQUIREMENTS.md` Sec 3.2 and identically in
`REFERENCE_QUALITY_SPEC.md` Sec 6:

```
B_meas = min( max(B0, b_ref + P - 1), B_ref )
```

### Mathematical verification

Checked exhaustively over every admissible `b_ref` (block endpoints 600…3200 in steps of
`P=100`; `b_ref >= L_ref + P = 600` because `F_e(b-L_ref)` must exist) against all three
frozen `B0` values (900 / 2000 / 3200):

| Required property | Result |
|---|---|
| 1. `B_meas >= B0` unless bounded by `B_ref` | **PASS** (and `B0 <= B_ref` holds for all three) |
| 2. persistence tail through `b_ref+P-1` whenever it lies within `B_ref` | **PASS** |
| 3. `B_meas <= B_ref` | **PASS** |
| 4. no method-identity-dependent extension | **PASS** — one equation, method-blind; only frozen inputs differ |
| 5. no trajectory-dependent "still improving" decision | **PASS** — zero live occurrences package-wide |
| 6. deterministic given `B0`, `b_ref`, `P`, `B_ref` | **PASS** — also monotone non-decreasing in `b_ref` |
| 7. `NOT_REACHED` remains possible | **PASS** — "reaching `B_meas` without a certified window produces the applicable final `NOT_REACHED` subclass" |
| 8. does not recreate C2 | **PASS** — see below |

**The original defect is resolved.** My N1 scenario was Proposed with `b_ref` beyond its
`B0 = 900`, censored for budget bookkeeping while Olhoff (`B0 = B_ref = 3200`) had no such
exposure. Under the new rule, `b_ref = 1500` yields `B_meas = 1599`; the tail binds for 24 of
27 admissible `b_ref` for Proposed and 13 of 27 for Yuksel Stage 2. The asymmetry I identified
is gone.

**The contradictory extension is gone.** The progress-triggered tranche and its condition that
the sustained floor "exceeds that of the preceding block" — the logical negation of the
stabilization condition defining `b_ref` — are deleted. A package-wide grep for
`extension tranche`, `0.5*B0`, `1.5*B0`, `INSUFFICIENT_INITIAL_BUDGET`, masked progress review
and "still improving" returns only negations ("no progress-triggered extension") and unrelated
M2 diagnostic text about whether a method plateaued. No discretionary review survives.

**C2 is not recreated.** The causal direction is verified to run one way only:
`REFERENCE_QUALITY_SPEC.md` Sec 6 states that the trajectory runner receives `b_ref` *solely
to calculate `B_meas`*, while the acceptance scan receives only the frozen triplet
`(Q_ref_E1, Q_ref_E2, Q_ref_E3)` and provenance, and "neither component can recompute
reference values from the measurement horizon." `Q_ref` is frozen before `B_meas` exists, so
no horizon can move a quality bar. A late-stabilizing method receives a longer horizon *and* a
higher bar; an early-stabilizing one receives its floor *and* a lower bar. These partially
offset and neither is assigned by method identity.

### Edge case `b_ref + P - 1 > B_ref`

Because `b_ref <= B_ref` and `b_ref` is a multiple of `P`, this arises at exactly one point:
`b_ref = 3200`. Then `B_meas = B_ref = 3200` and the persistence tail is truncated by exactly
**99 updates — identically for all three methods**, since `B_meas` collapses to `B_ref`
regardless of `B0`. The truncation is therefore method-blind.

**Status semantics:** determinate without a new decision. The cell runs to `B_meas = B_ref`;
if no certified window completes, the general clause applies and the cell takes the applicable
`NOT_REACHED` subclass under the `ACCEPTANCE_GATE_SPEC.md` Sec 8 precedence — normally
`QUALITY_NOT_REACHED` where base-valid states persist. This is the correct outcome: the
reference was established at the very last demonstrated block, there is no evidence beyond
`B_ref`, and extending would require optimization past the frozen common cap — precisely the
discretionary horizon extension C2 forbids. The case is covered by the general rule rather
than named separately, which is adequate.

### Is `B_ref` silently reinterpreted?

No. `IMPLEMENTATION_REQUIREMENTS.md` Sec 3.1 retains verbatim its Phase-1C semantics —
"a censoring/resource boundary only", "**There is no terminal-cap fallback**",
`REFERENCE_NOT_ESTABLISHED` / `REFERENCE_SOLVER_TERMINATION` on failure, and the engine "must
never substitute the best floor at the cap." Its new role as the measurement ceiling is
*additive* and consistent: `B_ref` is the largest demonstrated common horizon, so forbidding
measurement beyond it forbids extrapolation past the evidence. It still supplies no quality
value.

**N1: CLOSED.**

---

## R2 — N2: master-protocol synchronization

All four passages verified directly in `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md`:

- **WP0 evidence item 3** — now "E1 raw was preregistered as primary **in the prior campaign
  only**; Phase 1C privileges no evaluator, and no common model is truth."
- **Mandatory efficiency and scaling story** — `k_enter(Ne|q)` is the "primary empirical
  maturation count and primary count-fit family"; `k_cert(Ne|q)` is a "prominent conservative
  certification companion … its power fit is secondary/descriptive"; q-conditioning,
  common-support companions, `C,p,R2_log,n_valid`, fitted range and LOO `p` range all
  mandatory.
- **Paper evidence package** — item 3 is now "co-equal E1/E2/E3 endpoint decompositions plus
  robust all-evaluator acceptance"; item 5 adds mandatory absolute reference/endpoint quality
  and ratio-to-best; the closing states "no evaluator is demoted to a sensitivity or omitted
  from the primary decomposition." The offending sentence is gone.
- **Hostile-review conclusions** — the T1/T0 bullet now cites "the repaired physical-area
  gate, aggregate diagnostics, raw/binary evidence, known-strict T0 diagnostic, and
  method-neutral 1x1/3x3 FE-patch OAT sensitivity."

### Systematic sweep

Ten obsolete concepts swept across all twelve normative documents, with each hit classified
historical vs live:

| Obsolete concept | Live normative survivals |
|---|---:|
| `a_res = 5` baseline | 0 |
| filter-derived `r_common` | 0 |
| aggregate detached area as an acceptance condition | 0 |
| E1 as unqualified universal/neutral evaluator | 0 |
| E1/E2/E3 as merely supplementary | 0 |
| horizon-relative `Q_ref` | 0 |
| progress-triggered measurement extension | 0 |
| "still improving" extension condition | 0 |
| Stage 1 not carried into Yuksel Stage 2 | 0 |
| one canonical δ_R | 0 |

Three apparent hits were inspected individually and are regex false positives: a negation
("E1 is Proposed's own model, so calling it neutral is withdrawn"), the correct prohibition
itself ("No evaluator is called a sensitivity"), and the delta table's Phase-1A column
explicitly marked "(**false**)". Remaining historical descriptions are unmistakably labelled
superseded.

**Non-blocking wording nit:** the first hostile-review bullet still says E1 "may resemble
Proposed's mass model," where the audit verified E1 *is* Proposed's interpolation to 9–11
significant digits. It asserts no superseded rule and the binding specs
(`QUALITY_EFFORT_SPEC.md` Sec 2, `ACCEPTANCE_GATE_SPEC.md` Sec 3) state the verified fact
plainly, so this is a hedge, not a contradiction.

**N2: CLOSED.**

---

## R3 — N3: topology diagnostic reporting

**Added.** `PROPOSED_TABLE_LAYOUTS.md` Main Table 2 now carries `aggregate detached area` and
`n_islands_all` beside `largest detached physical area`. `SCALING_AND_FIGURE_SPEC.md` F8
requires the caption to state the observed 640x80 range.

**Separation intact — diagnostics did not become acceptance conditions.** Verified in both
normative locations:

- `TOPOLOGY_SANITY_SPEC.md` Sec 5: `H_T(k)=[C_required(k)=1] ∧ [max_c A_c^detached(k) < A_sig]`,
  followed by "There is **no aggregate detached-area veto**" and "Aggregate area, component
  count, and LCC remain mandatory diagnostics."
- `ACCEPTANCE_GATE_SPEC.md` normative pseudocode: `H_T(k) := support_component(xb(k)) exists
  AND max physical area of each detached component < A_sig=0.01  # no aggregate-area veto`.

Individual-component validity remains the sole normative topology acceptance rule; the
aggregate quantities are paper-facing descriptors only.

**Statistics checked against frozen evidence.** Recomputed with my independent union-find
implementation over the frozen 640x80 Olhoff trajectory:

| statistic | caption | full-trajectory (all 1014 passing states) |
|---|---:|---:|
| maximum aggregate detached area | 674 (2.633% solid) | **674 (2.633%) — exact** |
| median | 65 | 64 |
| p95 | 148 | 147 |

The maximum and its volume share are exact. Median and p95 differ by one element because the
caption cites my delta-audit values, which were computed on a stride-4 sample that I did not
label as sampled. A one-element difference in a descriptive diagnostic has no methodological
consequence; recorded as a non-blocking precision obligation.

**N3: CLOSED.**

---

## R4 — N4: absolute quality numbers

Recomputed independently from `examples/Performance/final_campaign/common_evaluators.csv`:

| check | result |
|---|---|
| normalization | `(Olhoff / other) − 1` — confirmed |
| complete method triples | **8** |
| Olhoff over Proposed | **6.2337% – 8.5048%** |
| Olhoff over Yuksel | **5.9356% – 7.7120%** |
| rows with E1 `N/A` | exactly one: `800x100 / Olhoff / RUN_ERROR` |

These reproduce the Phase-1E values to four decimals. All four stated one-decimal bounds are
correct roundings (6.2337→6.2, 8.5048→8.5, 5.9356→5.9, 7.7120→7.7).

**Consistency.** The corrected range appears identically in `QUALITY_EFFORT_SPEC.md`,
`SCALING_AND_FIGURE_SPEC.md` F4, `PROPOSED_TABLE_LAYOUTS.md` Main Table 2 caption,
`ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md`, and `PHASE1C_AUDIT_RESPONSE.md`. No surviving
"6.1–7.2%", "4.9–7.6%" or nine-mesh quality claim; the residual "nine meshes" occurrences all
refer to the mesh family for budgets, fits and storage estimates, not to the quality gap.

**No inferred 800x100 value.** `TOPOLOGY_SANITY_SPEC.md` Sec 6 now shows the row as
`N/A | 100 | N/A | UNVERIFIABLE_AT_PRESENT | N/A | N/A | N/A`, with the explicit statement that
"no state count, pass fraction, persistence run, or final topology measurement is inferred
from the missing case." The former `>= 27.38%` and `>= 298` bounds are gone package-wide. The
evidence matrix carries `ALREADY_AVAILABLE_EXCEPT_800x100` with the zero-byte and `RUN_ERROR`
facts.

The evaluator-agreement figure is unchanged and remains correct at **0.429%** maximum spread
with ordering preserved, now carrying the qualification that E2 and E3 share the same
piecewise `x^6` mass law and differ only in stiffness floor.

**N4: CLOSED.**

---

## R5 — collateral damage

Only two of the fourteen design files were left untouched by Phase 1E
(`ITERATION_ACCOUNTING_SPEC.md` and `TIMING_SPEC.md`, both still at their Phase-1C
timestamps), so iteration accounting and the timing architecture are unchanged by
construction. For the remainder I re-read the load-bearing definitions verbatim:

| Frozen semantics | Status |
|---|---|
| C1 topology repair — `A_sig = 4A_e0 = 0.01`, `a_sig = ceil(A_sig/A_e)`, `H_T` without aggregate term | unchanged |
| C2 reference — `b_ref = min{b : g_e(b) <= ε_ref ∀e}`, no cap fallback | unchanged |
| quality-effort axis — `q ∈ {0.980, 0.990, 0.995}` co-primary | unchanged |
| evaluator robustness — `r_all(k) = min_e r_e(k)`, `S_q = [r_all >= q]` | unchanged |
| persistence — `P = 100`, `P = 50/200` sensitivity at every q | unchanged |
| `k_enter(q) = min{a >= 1 : A_q(k)=1 ∀k∈[a,a+P−1]}` | unchanged |
| `k_cert(q) = k_enter(q) + P − 1` | unchanged |
| iteration accounting | file untouched |
| timing architecture | file untouched |
| scaling semantics — `k_enter` primary, `k_cert` descriptive, common support, LOO, weak-identification | unchanged (narrative now agrees) |

Integrity: `PHASE1C_FINDING_CLOSURE.csv` still parses with exactly 24 rows
(2 CRITICAL / 8 MAJOR / 9 MODERATE / 5 MINOR, all ACCEPT);
`EVIDENCE_AVAILABILITY_MATRIX.csv` parses with 33 rows × 7 columns and no ragged rows, after
the declared LF normalization.

**NO COLLATERAL METHODOLOGICAL CHANGE DETECTED.**

---

## New blockers

**None.** Phase 1E closes N1–N4, introduces no contradiction into the frozen methodology, and
leaves no scientific choice outstanding. `NEW_BLOCKER.md` is therefore deliberately not
produced.
