# Delta audit — Phase 2 readiness (D13, D14, D16)

## 1. Evidence matrix and implementability (D13)

`EVIDENCE_AVAILABILITY_MATRIX.csv` (33 rows) uses the six-class taxonomy plus two honest
qualified variants (`ALREADY_AVAILABLE_EXCEPT_800x100`,
`DERIVABLE_OFFLINE_PROVISIONAL_ONLY`). Class counts across the three method columns:

| class | count |
|---|---:|
| ALREADY_AVAILABLE | 32 |
| ALREADY_AVAILABLE_EXCEPT_800x100 | 1 |
| DERIVABLE_OFFLINE | 7 |
| DERIVABLE_OFFLINE_PROVISIONAL_ONLY | 2 |
| NEEDS_OBSERVATIONAL_INSTRUMENTATION | 11 |
| NEEDS_REFERENCE_PHASE_RUN | 13 |
| NEEDS_TIMING_REPLAY | 5 |
| NEEDS_NEW_OPTIMIZATION_TRAJECTORY | 28 |

### Is anything classified too optimistically?

I checked the rows where over-optimism would matter most and found **none**. In several places
the matrix is *more* conservative than it needs to be, which is the right direction:

- **Olhoff per-state densities** are `ALREADY_AVAILABLE_EXCEPT_800x100`, not a blanket
  AVAILABLE. Correct — I confirmed `s1_800x100.mat` is 0 bytes.
- **Olhoff q-level endpoints** are `DERIVABLE_OFFLINE_PROVISIONAL_ONLY`, not
  `DERIVABLE_OFFLINE`. Correct and important: a production endpoint requires a paired declared
  reference run, so the existing snapshots can support engineering validation but not a
  publishable `k_enter`. An earlier draft classified this as simply derivable; the correction
  is sound.
- **LP solver-internal iterations** are `NEEDS_OBSERVATIONAL_INSTRUMENTATION` with the
  diagnostic-mirror constraint. I verified this is genuinely required: `innerLoopLP.m` line 65
  is `[x, ~, flag] = linprog(...)` — the fourth output is not captured, and the file sits
  inside the hash-guarded frozen tree.
- **Reference-phase rows (13)** are new and correctly classified. They are the real cost of
  the C2 repair and are not disguised as derivable.
- **Conditional A** is `NEEDS_NEW_OPTIMIZATION_TRAJECTORY` for all three methods with the note
  that no independent `Omega_req` exists. Correct, and A is not a blocker.

`IMPLEMENTATION_REQUIREMENTS.md` specifies the two-pass engine (causal reference freeze, then
measurement scan against the frozen triplet), the structural prohibition on recomputing a
reference from the measurement horizon, the repaired topology gate, fingerprint identity
tests, and the Phase A/B/C/D/E sequence with a pilot that exercises the full two-phase flow.

### Can Phase 2 be implemented without another scientific decision?

**Almost, but not quite — one gap (N1).** Everything else is determinate: the acceptance
expression in `ACCEPTANCE_GATE_SPEC.md` Sec 6 is mechanically implementable, every constant is
frozen, and the status precedence is total.

The exception is the interaction described in N1. When `b_ref` exceeds a method's measurement
horizon, the protocol as written yields `NOT_REACHED`, and the one permitted extension is
*denied* because condition 3 requires the floor to still be improving — which stabilisation
has just falsified. An implementer meeting that case in production would face a choice
(accept the censoring, or extend) whose consequences are known and method-correlated at the
moment of choosing. That is a results-affecting decision taken after seeing data, which is
exactly what the freeze exists to prevent. It must be settled in the specification first.

## 2. The zero-length 800x100 artifact (D14)

**Verified facts.** `examples/Performance/final_campaign/raw/olhoff/s1_800x100.mat` is
**0 bytes**. Separately, the frozen endpoint table records the Olhoff 800x100 row as status
`RUN_ERROR` with `omega1_common_raw_E1 = N/A`. So there is at present **no** verifiable Olhoff
800x100 evidence of either kind — neither trajectory nor endpoint.

**Classification: (B) and (C) — affects only completeness of frozen diagnostic verification,
and can be regenerated later under the already-frozen methodology. Not (A), not (D).**

Reasoning:

- **Does any freeze-critical claim depend on it? No.** C1's closure rests on the aggregate
  clause being the binding constraint, which is anchored at 640x80 — a mesh I fully reproduced
  independently (T1-with-aggregate 0.56% vs per-component-only 45.74%; `det_max=4`,
  `det_tot=20`). My original audit said the same: "The finding does not depend on 800x100."
  The repaired gate's satisfiability is demonstrated at four meshes.
- **Does a claimed closure depend on missing evidence? Partly, and it must be relabelled.**
  `TOPOLOGY_SANITY_SPEC.md` Sec 6 carries the 800x100 row as `>= 27.38%` pass and `>= 298`
  longest run, justified as a conservative bound from the original audit's per-component-five
  recomputation, and states that "the double-precision final state independently has largest
  detached area four." Neither can be checked today, and the original audit's own 800x100 row
  rested on a single implementation of a file that was already truncated. The *bound logic* is
  sound (`a_sig=100` is strictly more permissive than 5, so the pass fraction cannot be
  smaller), but its input is currently unverifiable.
- **Does it require a methodological decision? No.** `a_sig(800x100) = 100` follows from
  `A_sig` by the same formula as every other mesh; nothing about the rule depends on observing
  that trajectory.

**Obligation (non-blocking):** relabel the 800x100 row in `TOPOLOGY_SANITY_SPEC.md` Sec 6 as
`UNVERIFIABLE_AT_PRESENT — bound inherited from the original audit, input artifact currently
0 bytes; frozen-campaign endpoint is RUN_ERROR/N-A`, rather than carrying `>=` figures that
read as reproduced measurements. The author did not paper this over — the truncation is
disclosed in three places — so this is a labelling refinement, not a correction of intent.
Regeneration is not required for freeze.

## 3. Classification of remaining items (D16)

### BLOCKING SCIENTIFIC DECISION — must be resolved before methodology freeze

- **N1** — tie the measurement horizon to `b_ref` and harmonise the extension rule with the
  stabilisation rule. Parameter-free; uses only already-frozen quantities.
- **N2** — remove the three superseded assertions from the master protocol document's
  narrative sections. Zero scientific content; wording only. Blocking because a frozen
  normative set may not contradict itself on three prior audit findings.

### NON-BLOCKING REPORTING OBLIGATION — methodology may freeze; must be honoured

- **N3** — add aggregate detached area (and `n_islands_all`) to Main Table 2 and state the
  observed range in the F8 caption. The deleted CRITICAL clause's quantity must be visible for
  accepted endpoints; measured up to 2.633% of solid volume at 640x80.
- **N4** — replace "6.1–7.2% ... nine meshes" with the recomputed 6.2%–8.5% over Proposed /
  5.9%–7.7% over Yuksel across the eight meshes with a complete triple, noting the Olhoff
  800x100 `RUN_ERROR`. The evaluator-agreement figure (0.429%, ordering preserved) needs no
  change.
- **M4 qualification** — state once, where the 0.429% agreement is quoted, that E2 and E3
  share the same piecewise `x^6` mass law and differ only in stiffness floor, so the
  three-evaluator minimum is closer to two-way in evidential terms.
- **D14 relabel** — mark the 800x100 topology row `UNVERIFIABLE_AT_PRESENT` as above.
- Carry forward every disclosure already mandated: `k_gate` beside each endpoint, `k_native`
  beside `k_cert`, the Yuksel Stage-1 cap non-comparability at three meshes, the
  `GENERIC_LP_ITERATION_LIMIT_ONLY` subclass, the `+99` convention caveat, `WEAKLY_IDENTIFIED`
  labels, LOO ranges, common-support fit restrictions, and `T_reference` as separate
  calibration cost.

### IMPLEMENTATION ENGINEERING DECISION — resolvable during Phase 2

- Storage layout, chunking, and the density-fingerprint cache.
- Whether the LP diagnostic mirror is built at all; if its identity test fails, the quantity is
  `NA` and the curve stays supplementary. No scientific semantics change either way.
- Eigensolver start-vector and tolerance bookkeeping inside the already-fixed deterministic
  settings.
- Scheduling, parallelism, and run order for timing replays, subject to the frozen
  no-mixing rule.
- Whether the measurement trajectory is physically re-run or proven bit-identical to the
  reference prefix. The specification requires separation and fingerprint verification; if
  identity is proven, reusing the prefix is an engineering optimisation that changes no
  semantics and would roughly halve the campaign cost.

### OPTIONAL FOLLOW-UP — not required for freeze or implementation

- Regenerate the Olhoff 800x100 trajectory to complete the diagnostic grid.
- Widen the mesh family or add seed replication to address the leverage/identifiability limits
  recorded under Mo2/Mo4. Out of scope for this study.
- Derive `P` from a preregistered false-certification-rate target in a successor study, as
  `QUALITY_EFFORT_SPEC.md` Sec 7 already contemplates.
