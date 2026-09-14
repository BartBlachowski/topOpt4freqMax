# Methodology freeze record

**Decision:** METHODOLOGY FROZEN WITH NON-BLOCKING QUALIFICATIONS — PHASE 2 IMPLEMENTATION
MAY BEGIN.

**Authority:** the independent methodologist who issued the original hostile audit
(`analysis/iteration_efficiency_methodology_audit/`, verdict *NOT READY*) and the delta audit
(`analysis/iteration_efficiency_methodology_delta_audit/`, verdict *DO NOT FREEZE — N1, N2*).

**Basis:** all 24 original findings CLOSED at delta audit; N1–N4 CLOSED at this restricted
re-verification; no collateral methodological change; no new blocker.

---

## 1. Normative documents constituting the frozen methodology

Twelve documents. Hash all of them into an immutable protocol manifest **before the first
production identity is unblinded**.

| # | Document | Governs |
|---|---|---|
| 1 | `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` | master protocol, claim boundaries, budgets |
| 2 | `ACCEPTANCE_GATE_SPEC.md` | endpoint definition, normative expression, status precedence |
| 3 | `REFERENCE_QUALITY_SPEC.md` | reference construction, `b_ref`, `B_meas`, measurement independence |
| 4 | `QUALITY_EFFORT_SPEC.md` | quality levels, evaluator-robust acceptance, estimand semantics |
| 5 | `TOPOLOGY_SANITY_SPEC.md` | topology acceptance rule and diagnostics |
| 6 | `ITERATION_ACCOUNTING_SPEC.md` | counting semantics per method |
| 7 | `TIMING_SPEC.md` | timing boundaries and replay policy |
| 8 | `SCALING_AND_FIGURE_SPEC.md` | fit discipline, censoring, mandatory figures |
| 9 | `PROPOSED_TABLE_LAYOUTS.md` | table layouts and pairing rules |
| 10 | `IMPLEMENTATION_REQUIREMENTS.md` | implementation contract, budgets, engine, sequence |
| 11 | `FAIRNESS_RISK_REGISTER.md` | F01–F31 controls and required wording |
| 12 | `EVIDENCE_AVAILABILITY_MATRIX.csv` | evidence classification |

Supporting, not normative: `PHASE1C_AUDIT_RESPONSE.md`, `PHASE1C_FINDING_CLOSURE.csv`,
`PHASE1E_DELTA_CLOSURE.md`, `verify_repaired_topology_gate.py`.

Immutable review evidence, never to be edited: both audit directories and this one.

## 2. Quantities and semantics that may NOT change after seeing Phase-2 results

Changing any of the following post-hoc reopens methodology review in full.

### Reference-quality construction
- Separation of reference generation from measurement, on distinct trajectories from an
  identical frozen initialization.
- `F_e(b)` as the cumulative best sustained floor over base-valid `P`-windows.
- `g_e(b) = [F_e(b) − F_e(b−L_ref)] / F_e(b)` evaluated at block endpoints `b = tP`.
- `b_ref = min{ b : g_e(b) <= ε_ref for all e ∈ {E1,E2,E3} }` — causal first passage; the scan
  may not inspect later quality to select a different `b_ref`.
- `Q_ref_e = F_e(b_ref)`, frozen with provenance hashes before measurement.
- **The absence of any cap-based fallback.** `REFERENCE_NOT_ESTABLISHED` and
  `REFERENCE_SOLVER_TERMINATION` are valid published outcomes.
- Constants: `P = 100`, `L_ref = 500`, `ε_ref = 0.001`.

### Budgets
- `B_ref = 3200` acceptance-eligible method-level updates, as a censoring/resource boundary
  that never supplies a quality value.
- `B0 = ceil_to_100(max(2·K_prior, K_prior + 5P))` and its frozen values
  900 / 2000 / 3200, plus Yuksel's separate `B0_stage1 = 2000` handoff budget.
- **`B_meas = min( max(B0, b_ref + P − 1), B_ref )`** — the complete measurement-budget
  contract. No discretionary, progress-triggered or result-contingent extension exists or may
  be reintroduced.

### Quality levels and evaluator-robust quality
- `q ∈ {0.980, 0.990, 0.995}`, co-primary. No level may be promoted or dropped after results.
- `r_e(k) = Q_e(k) / Q_ref_e`; `r_all(k) = min_e r_e(k)`; `S_q(k) = [r_all(k) >= q]`.
- E1/E2/E3 as co-equal primary decompositions; no evaluator may be demoted to a sensitivity.
- The mandatory best-observed benchmark `Q_BO_e,j = max_m Q_ref_e,mj`, never named A.
- `A_NOT_INSTANTIATED` unless an independently provenance-locked `Omega_req` predates
  production.

### Topology acceptance rule
- `H_T(k) = [C_required(k) = 1] ∧ [max_c A_c^detached(k) < A_sig]`.
- `A_sig = 4·A_e(160x20) = 0.01`; `a_sig(j) = ceil(A_sig / A_e(j))`.
- **No aggregate detached-area veto.** Aggregate area, `n_islands_all` and LCC are mandatory
  diagnostics and may never become acceptance conditions.
- Support-to-support connectivity via Q4 element footprints of both prescribed support nodes;
  even `nely` precondition.
- Exact-count volume-preserving binary projection with increasing-global-index tie-break.
- T0 as a known strict diagnostic; 1×1/3×3 FE-patch scales as the OAT sensitivity.

### Persistence and endpoints
- `P = 100` common to all methods; `P = 50/200` OAT sensitivity at every `q`. Method-specific
  `P` is prohibited.
- `k_enter(q) = min{ a >= 1 : A_q(k) = 1 ∀k ∈ [a, a+P−1] }` — primary.
- `k_cert(q) = k_enter(q) + P − 1` — paired certification location.
- Instantaneous crossings are diagnostics only and are never endpoints.

### Iteration accounting
- Proposed OC updates; Yuksel Stage 1 / Stage 2 / qualified chronological total; Olhoff outer
  updates and LP calls.
- `nInner = 1` means one `linprog` call and may never be presented as a simplex/HiGHS
  iteration; no `outer + inner` composite metric.
- Genuine solver-internal iterations only where actually instrumented via a mirror outside the
  frozen tree.
- `k_gate` and `k_native` reported beside every endpoint.

### Censoring and status
- The full precedence order in `ACCEPTANCE_GATE_SPEC.md` Sec 8, with mandatory backend
  subclasses; `GENERIC_LP_ITERATION_LIMIT_ONLY` wherever the known Olhoff LP event appears.
- Reaching `B_meas` uncertified yields the applicable `NOT_REACHED` subclass.
- Censored cells are shown at the observed boundary and excluded from fits; no imputation, no
  pooling, no post-hoc range selection.
- Olhoff 800x100 stays `RUN_ERROR` / E1 `N/A` / `UNVERIFIABLE_AT_PRESENT`; no value inferred.

### Timing
- The `T_init` / `T_loop_to_enter` / `T_loop_to_cert` / `T_native_finalize` decomposition and
  the paired `T_result_to_enter` / `T_result_to_cert`.
- `T_reference` separately reported; never folded into endpoint times.
- `T_gate_offline` and `T_observer_after_cert` disclosed but never charged.
- Fixed-horizon lightweight replays at determined endpoints; the no-mixing rule across
  methods, meshes, q levels and endpoints.

### Scaling-fit discipline
- `k_enter(Ne|q) = C(q)·Ne^{p(q)}` primary; `k_cert` power fit secondary/descriptive with the
  `+(P−1)` caveat and the identity check.
- Mandatory common-support companion fits; cross-method exponent comparison prohibited outside
  common support.
- Reported `C`, `p`, `R2_log`, `n_valid`, fitted range, LOO `p` range.
- `WEAKLY_IDENTIFIED` criteria: `R2_log < 0.80`, LOO range spans zero, or LOO width `> |p|`.
- Minimum three valid meshes; no extrapolation; "empirical scaling over the tested mesh range"
  language only.

## 3. Not frozen — no scientific effect

Storage and caching layout; whether the LP diagnostic mirror is built; whether the measurement
trajectory is re-run or proven bit-identical to the reference prefix; eigensolver bookkeeping
within fixed deterministic settings; replay scheduling and parallelism within the no-mixing
rule; and uniform engineering defect fixes that pass an identity test and are applied without
inspecting comparative ranks.

## 4. Conditions attached to the freeze

1. The four non-blocking reporting obligations in `PHASE2_FINAL_READINESS.md` are carried into
   implementation and the paper.
2. The twelve normative documents are hashed into the protocol manifest before unblinding.
3. Any proposal to alter a Section-2 quantity after results are visible requires formal
   reopening of methodology review — it is not an engineering correction.

## 5. Result firewall

Production results may not trigger a change to any acceptance constant, the reference or
budget rules, the evaluator set, profiles, fit ranges, or table/figure inclusion. Completed
campaign data remain read-only contextual evidence. Any necessary engineering correction is
documented, identity-tested, and rerun uniformly without inspecting comparative ranks.
