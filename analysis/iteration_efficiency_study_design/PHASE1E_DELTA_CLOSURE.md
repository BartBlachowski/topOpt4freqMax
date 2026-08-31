# Phase 1E delta closure

Scope: N1–N4 only. No Phase-2 implementation or optimization was performed, and none of the
24 Phase-1C findings was reopened. Post-patch line numbers are used below.

## N1 — measurement-budget contract

For every cell with an established frozen reference, the measurement horizon is now

\[
B_{meas}=\min\{\max(B_0,b_{ref}+P-1),B_{ref}\}.
\]

The inner maximum preserves the already-frozen prior-evidence lower bound `B0` and otherwise
reserves the `P-1` persistence tail after `b_ref`. The outer minimum prevents an extrapolated
measurement horizon beyond the common demonstrated reference cap `B_ref`. Thus
`B_meas >= min(b_ref+P-1,B_ref)`, `B_meas >= B0` when `B0 <= B_ref`, and
`B_meas <= B_ref`. The transformation is identical for hidden method labels, introduces no
new parameter, and is fixed from `B0`, `b_ref`, `P`, and `B_ref` before the measurement scan.
A cell's `B0` is the frozen lower bound for its acceptance-eligible stream; Yuksel Stage 1
retains its separate frozen 2000-update handoff budget.
A cell without `b_ref` retains `REFERENCE_NOT_ESTABLISHED` and does not enter measurement.
The former progress-triggered tranche is deleted; no improvement test or discretionary
extension remains.

Changed locations:

- `IMPLEMENTATION_REQUIREMENTS.md` Sec 3.2, lines 53–89; Sec 9, lines 219–238; Sec 10
  Phase A/Phase E, lines 242–249 and 284–287; Sec 11, lines 294–303.
- `REFERENCE_QUALITY_SPEC.md` Sec 6, lines 102–119.
- `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` “Safety budgets and NOT_REACHED policy,” lines
  315–344; required answers 18–19, lines 535–539; auditor challenge 9, lines 606–608.
- `FAIRNESS_RISK_REGISTER.md` row F18, line 30.

## N2 — master-protocol narrative synchronization

- `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` WP0 evidence item 3, lines 96–98: prior-campaign
  E1 primacy is explicitly historical; Phase 1C privileges no evaluator.
- “Mandatory efficiency and scaling story,” lines 384–402: `k_enter(Ne|q)` fitting is
  primary; `k_cert` fitting is descriptive; q conditioning, common support, and LOO ranges
  are mandatory.
- “Paper evidence package,” lines 404–420: E1/E2/E3 are co-equal primary decompositions;
  the q family and absolute/ratio-to-best quality are main evidence.
- “Hostile-review conclusions,” lines 435–450: the live controls are the repaired physical
  FE gate, aggregate diagnostics, raw/binary evidence, known-strict T0 diagnostic, and
  1x1/3x3 OAT sensitivity.

No binding Phase-1C acceptance, evaluator, topology, or persistence rule was changed by N2.

## N3 — aggregate detached-area reporting

- `PROPOSED_TABLE_LAYOUTS.md` Main Table 2, lines 28–37: added aggregate detached area and
  `n_islands_all` beside largest detached area.
- `SCALING_AND_FIGURE_SPEC.md` F8, lines 123–129: caption must report the independently
  observed 640x80 Olhoff aggregate maximum 674 elements (2.633% solid volume), median 65,
  and p95 148, while each component remains below `a_sig=64`.
- `PHASE1C_AUDIT_RESPONSE.md` WP12, lines 542–556: historical table description synchronized.

## N4 — corrected eight-mesh evidence and unavailable 800x100 case

- Normative disclosure: `QUALITY_EFFORT_SPEC.md` Secs 2 and 5, lines 25–32 and 105–125;
  `SCALING_AND_FIGURE_SPEC.md` F4, lines 98–104; `PROPOSED_TABLE_LAYOUTS.md` Main Table 2,
  lines 28–37; `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` common evaluators/absolute A,
  lines 169–187 and 213–226.
- Evidence status: `TOPOLOGY_SANITY_SPEC.md` Secs 1 and 6, lines 3–14 and 99–121;
  `IMPLEMENTATION_REQUIREMENTS.md` Sec 4 historical evidence qualifier, lines 120–130;
  `EVIDENCE_AVAILABILITY_MATRIX.csv` rows 7–15, 18, 20, and 23; and
  `verify_repaired_topology_gate.py` comments, lines 84–86.
- Master-protocol evidence wording: `ITERATION_EFFICIENCY_PROTOCOL_DRAFT.md` topology and
  evidence-reuse passages, lines 248–278, 422–433, 547–556, and 598–599.
- Closure/history synchronization: `PHASE1C_AUDIT_RESPONSE.md` WP1, WP4, WP5, WP12, WP14,
  WP16, and final summary; `PHASE1C_FINDING_CLOSURE.csv` rows C1, M3, M4, Mo6, and Mo7;
  `ACCEPTANCE_GATE_SPEC.md` Sec 3, line 49; `FAIRNESS_RISK_REGISTER.md` rows F01, F07, and
  F29. The two edited CSVs were normalized to LF line endings so their targeted rows parse
  consistently.

Every live absolute-quality statement now says: Olhoff leads common raw-E1 `omega1` by
**6.2–8.5% over Proposed** and **5.9–7.7% over Yuksel** across eight complete method triples.
Olhoff 800x100 is explicitly `RUN_ERROR`, E1 `N/A`, topology
`UNVERIFIABLE_AT_PRESENT`; no ninth-mesh quality or topology value is inferred. The verified
maximum evaluator spread remains 0.429%, with the qualification that E2/E3 share the same
piecewise `x^6` mass law and differ only in stiffness floor.

## Static closure checks

- Source CSV recomputation: 8 complete triples; Proposed gap 6.2337–8.5048%; Yuksel gap
  5.9356–7.7120%; excluded row `800x100/RUN_ERROR/N/A`.
- Both edited CSVs parse with fixed column counts; the finding ledger still contains exactly
  24 rows.
- Markdown table column counts are consistent after ignoring literal pipes inside inline
  code.
- No superseded numeric range, inherited 800x100 bound, old extension status/tranche, or any
  of the three N2 narrative phrases remains in the design package.

**Closure verdict:** N1, N2, N3, and N4 are ready for the independent auditor's restricted
re-verification. This verdict authorizes neither Phase 2 nor optimization.
