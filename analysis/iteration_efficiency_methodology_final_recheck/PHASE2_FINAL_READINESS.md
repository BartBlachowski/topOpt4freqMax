# Phase 2 final readiness (R6)

## The decisive question

> After the Phase-1E corrections, can Phase 2 be implemented without making any further
> **scientific** methodology choice after observing results?

**Yes.**

The single gap that prevented this at the delta audit was N1: when `b_ref` exceeded a
method's measurement budget, an implementer met the case mid-production and had to choose
between censoring and extending, with method identity and consequences visible. That choice
no longer exists. `B_meas = min(max(B0, b_ref+P-1), B_ref)` is evaluated automatically from
four already-frozen quantities before the measurement scan begins, the discretionary tranche
is deleted, and the outcome of reaching `B_meas` uncertified is fixed by the status
precedence.

Every remaining decision an implementer faces is engineering. The chain from raw state to
reported endpoint is closed:

- **State and projection** — return-equivalent field, exact-count binary projection with
  index tie-break, matching the repository evaluator's own convention.
- **Pointwise gate** — `H0 = H_health ∧ H_V ∧ H_T ∧ H_method`, each term fully specified,
  `H_V` naming `H.rV` explicitly to avoid the absolute-residual field.
- **Reference** — causal first-passage `b_ref`, frozen `(Q_ref_E1, Q_ref_E2, Q_ref_E3)`, no
  cap fallback, explicit failure classes.
- **Budget** — `B_meas`, deterministic.
- **Acceptance** — `r_all = min_e r_e`, `S_q`, `A_q`, at three preregistered `q`.
- **Endpoints** — `k_enter(q)`, `k_cert(q) = k_enter(q) + P − 1`.
- **Status** — total precedence order with mandatory backend subclasses.
- **Analysis** — fit discipline, censoring, common support, LOO, weak-identification labels.

## Classification of remaining items

### BLOCKING SCIENTIFIC DECISION

**None.**

### NON-BLOCKING REPORTING OBLIGATION

Must be carried into implementation and the paper; none prevents freeze.

1. **F8 caption precision (from N3).** The caption cites median 65 / p95 148 for the 640x80
   aggregate detached area; these are stride-4 sample values from the delta audit. The
   exhaustive full-trajectory figures over all 1014 passing states are **median 64, p95 147**;
   the maximum 674 elements / 2.633% of solid volume is exact. Cite the exhaustive values and
   state that they are exhaustive.
2. **E1 wording (from N2).** One hostile-review bullet says E1 "may resemble" Proposed's mass
   model. The verified fact is that E1 *is* Proposed's interpolation up to floor values, to
   9–11 significant digits. Align the hedge with the binding specs.
3. **Edge-case visibility (from N1).** `b_ref = 3200` truncates the persistence tail by 99
   updates for every method. The semantics are determinate, but any cell that lands there
   should be reported with its `b_ref`, `B_meas` and the truncation explicitly noted, so a
   reader can see that the censoring is horizon-bounded rather than maturation-based.
4. **Evaluator independence qualification (from M4).** State once, where the 0.429% agreement
   is quoted, that E2 and E3 share the same piecewise `x^6` mass law and differ only in
   stiffness floor, so the three-evaluator minimum is closer to two-way in evidential terms.
5. **All previously mandated disclosures**, unchanged: `k_gate` beside every endpoint;
   `k_native` beside `k_cert`; the Yuksel Stage-1 cap non-comparability at 640x80/720x90/800x100;
   `GENERIC_LP_ITERATION_LIMIT_ONLY` wherever an Olhoff solver termination appears; the `+99`
   convention caveat on `k_cert`; `WEAKLY_IDENTIFIED` labels; LOO ranges; the prohibition on
   cross-method exponent comparison outside common support; `T_reference` as separately
   reported calibration cost; and the absolute-quality pairing rule.
6. **Olhoff 800x100** remains `RUN_ERROR` / E1 `N/A` / `UNVERIFIABLE_AT_PRESENT`. No value may
   be inferred for it in any table, figure or fit.

### IMPLEMENTATION ENGINEERING DECISION

Resolvable during Phase 2 without changing scientific semantics.

- Storage layout, chunking, and the density-fingerprint evaluation cache.
- Whether the LP diagnostic mirror outside the frozen `reproduction2007` tree is built; if it
  is not, or its identity test fails, the quantity is `NA` and the solver-internal curve stays
  supplementary. The frozen tree and its hash are never edited either way.
- Whether the measurement trajectory is physically re-run or proven bit-identical to the
  reference prefix. The specification requires logical separation and fingerprint
  verification; proving identity and reusing the prefix is an optimisation that changes no
  semantics and would roughly halve campaign cost.
- Eigensolver start-vector and tolerance bookkeeping within the already-fixed deterministic
  settings.
- Scheduling, parallelism and run order for timing replays, subject to the frozen no-mixing
  rule across methods, meshes, q levels and endpoints.
- Any engineering defect fix that passes an identity test and is applied uniformly without
  inspecting comparative ranks.

### OPTIONAL FOLLOW-UP

- Regenerate the Olhoff 800x100 trajectory to complete the diagnostic grid. Not required for
  freeze; `a_sig(800x100) = 100` follows from `A_sig` by the same formula as every other mesh.
- Widen the mesh family or add seed replication to address the leverage and identifiability
  limits recorded under Mo2/Mo4. Out of scope for this study.
- Derive `P` from a preregistered false-certification-rate target in a successor study, as
  `QUALITY_EFFORT_SPEC.md` Sec 7 contemplates.

## Expected campaign shape

Two trajectories per cell (reference then measurement) across 27 cells, with the reference
phase separately costed. The Phase-A budget of roughly 75 h offline evaluation and 40 GB
storage stands; `B_meas <= B_ref` guarantees the measurement phase never exceeds the
reference horizon, so the estimate is an upper bound rather than an open-ended commitment.

Outcomes that must be accepted without renegotiation include `NOT_REACHED`,
`REFERENCE_NOT_ESTABLISHED`, `QUALITY_NOT_REACHED`, `MODEL_DEPENDENT`, rankings that change
with `q`, quality–effort curves that cross, and `WEAKLY_IDENTIFIED` scaling fits.
