# Phase 2 readiness assessment

Audit-only. This document does not authorize implementation or production.

## 1. Readiness verdict

**NOT READY — MAJOR METHODOLOGICAL REVISION REQUIRED.**

The revision is bounded and fully identified: **one clause of one gate**, **one quantifier
in one definition**, and a set of presentation rules that require no new science. Nothing in
the engineering design, the isolation strategy, the instrumentation plan, the timing
architecture or the governance needs rework — those are sound and in several places better
than standard practice.

## 2. Blocking items (must be resolved before Phase A)

| ref | item | why blocking | resolution requires |
|---|---|---|---|
| **C1** | T1's aggregate detached-area clause | Verified unsatisfiable for Du–Olhoff at 800x100 (0.00 % of 1601 states pass; longest run 0) and at three further meshes. The comparison would not exist at the top of the mesh range. | a new scientific choice: delete the clause, or re-derive an aggregate rule that is resolution- and size-consistent |
| **C2** | `Q_ref` horizon dependence with 3.5× asymmetric budgets | The primary estimand's reference is computed over different observation horizons per method, so `k_enter` is not cross-method comparable — which is the study's entire purpose. | a new scientific choice: common horizon, reference-stability requirement, or explicit horizon-sensitivity reporting |

Both are human decisions. An auditor cannot make them, and neither should be resolved by
picking whichever option produces a preferred ordering.

## 3. Non-blocking corrections (before the protocol hash is frozen)

Each requires no new scientific choice.

**Definition and disclosure**
- Correct the Yuksel Stage-1 handoff sentence in `ITERATION_ACCOUNTING_SPEC.md` (`x = xPhys`
  at line 237 carries the design field forward); restate the Stage-1 exclusion on
  objective-mismatch grounds, which are sound (M7).
- Publish the assembled acceptance expression verbatim in the protocol (Q14).
- Rename `r_common`; state `a_res` as "the strictest of the three frozen filter footprints
  (5 / 9 / 21 cells)" (Mo7).
- Disclose the Yuksel Stage-1 cap truncation at 640x80 / 720x90 / 800x100 and its effect on
  the Stage-2 starting design (M6).
- Record that a Proposed iteration contains no eigensolve, and that the Olhoff eigensolve is
  75 % of its outer-iteration cost at 800x100 (Mi5).
- State even `nely` as a precondition of the support-footprint definition (Mi1).
- Name `H.rV` (or the explicit relative formula) as the volume gate's source quantity rather
  than the evaluator's absolute `volume_residual` field (Mi2).

**Presentation**
- Move achieved E1 and E1-ratio-to-best-observed into Main Table 1; free the space by
  demoting the exactly-redundant `Olhoff LP calls` column to a footnote (M3).
- Promote the best-observed benchmark from optional to mandatory (M3).
- Report R under E1, E2 and E3 as three co-equal primary columns (M4).
- Report each method's gate-satisfaction iteration beside `k_enter` (M8).
- Report `delta_R` = 0.5 / 1 / 2 % as primary columns rather than a supplementary rescan (M1).
- Add the in-axes per-update-cost qualifier to F1 and require F2 adjacency (Q8).
- Carry the `GENERIC_LP_ITERATION_LIMIT_ONLY` classification wherever
  `GENUINE_SOLVER_FAILURE` appears (Mo5).
- Bind any single-number statement of iteration effort to name its endpoint and carry its
  quality context (Q6).

**Fitting**
- Require a common-mesh-support companion fit; forbid cross-method exponent comparison
  outside common support (M5).
- Subordinate `k_cert` scaling to `k_enter` scaling; add the `k_cert − (P−1)` identity check
  (Mo1).
- Report a leave-one-out range of `p` with every fit (Mo2).
- Add a preregistered "weakly identified" label for low-`R²_log` or wide-jackknife fits (Mo4).

**Governance**
- Publish a directional-bias register naming, for each frozen choice, the method it is
  expected to favour and whether the direction was checked (Q13).
- Add a mechanical provenance gate so no `Omega_req` lacking a pre-production hash can enter
  the acceptance engine (Q3).
- Either declare T0's known outcome up front or replace it with a sensitivity that can
  discriminate in the permissive direction (Mo6).

## 4. What is already sound and should not be touched

Recorded explicitly so that revision does not damage what works.

- **Isolation and immutability** (`IMPLEMENTATION_REQUIREMENTS.md` §1). New namespace,
  read-only treatment of the completed campaign and both audit directories, hash-bound
  profiles, refusal of output paths outside the new result root, plots and evaluators
  disabled inside optimization loops. Nothing to change.
- **Frozen profiles** (§2). Selected before this study by a preregistered Pareto rule with
  recorded holdout and forbidden mesh sets, and `retuning_after_this_point: PROHIBITED`.
- **The extension rule** (§3). One tranche, four simultaneous formulaic conditions, masked
  labels, published decisions, no second automatic tranche, clean-rerun requirement with
  full prefix match. Genuinely well built.
- **Observational invariance** (§5). Suppress only native termination, never Yuksel's
  Stage-1 handoff or Olhoff's S1 trigger; no common eigensolves inside optimization loops;
  observer-on/off prefix comparison at one coarse and one fine mesh per method. Already
  backed by passing bitwise evidence at 160x20 for all three methods.
- **The timing architecture** (`TIMING_SPEC.md` entire). See
  `TIMING_AND_SCALING_AUDIT.md` Part A — the strongest section of the package.
- **Censoring discipline** (`SCALING_AND_FIGURE_SPEC.md` §2). Status-aware markers, never
  connected, never regressed, no imputation, no pooling, no post-hoc subranges.
- **The prohibition on complexity language** (§6). Correct and should not be relaxed under
  any reviewer pressure.
- **`nInner` handling** throughout. Every claim verified; the treatment is more careful than
  the literature's.
- **Raw/binary role separation.** Raw for the spectral gate, binary for the hard topology
  gate. This is what stops the Yuksel 400x50-style binarization collapse (ω₁ 159.968 → 109.440)
  from contaminating `Q(k)`.
- **`Q_ref` as a window minimum.** The right construction; it is what makes R applicable to
  a method that never becomes stationary.
- **Refusing to instantiate A.** Correct, and it should survive review pressure.
- **The asymmetric-method-condition defence** (F11–F13). "False symmetry would be less
  fair" is right; keep the wording.

## 5. Implementability without contamination

The four-layer separation holds for everything except one item.

| layer | verdict |
|---|---|
| observational instrumentation | acceptable — the recorders exist and are scalar-only, so adding a density column is additive; `topopt_freq.m` already passes `xPhys` into `histRec` |
| evaluator logic | acceptable — entirely outside the optimizers; `evaluate_stabilized_e1.m` is a working precedent that already asserts agreement with `study_evaluate_design.m` to `1e-8` |
| experiment harness | acceptable |
| numerical algorithm changes | **one item** — see below |

**The one item.** Capturing `linprog`'s solver-internal iteration count requires taking the
fourth output of the call in `Matlab/reproduction2007/algo/innerLoopLP.m`. The call is
behaviourally identical, so this is observational instrumentation — but the file sits inside
the frozen Du–Olhoff reproduction tree guarded by `repro2007_tree_hash.m`, and editing it
breaks that hash. Resolve by instrumenting through a diagnostic mirror outside the frozen
tree, following the existing `innerLoopLPDiagnostic.m` and `olhoffOptStabilizedDiagnostic.m`
pattern, and say so explicitly (Mo8).

**Nothing else requires changing algorithm behaviour.**

## 6. Resource estimate (missing from the package)

The offline acceptance engine must evaluate `Q(k)` at **every** state — `Q_ref` is a max
over windows of a window-min, so no bisection is possible.

Estimated from frozen Olhoff `tEig` medians across the nine meshes, which fit
`t ≈ 1.316e-6 · N_e^1.214` s for a 5-mode generalized eigensolve. Scaling to 3 modes and
applying the full budgets (Proposed 900, Yuksel 4000, Olhoff 3200 states per mesh):

| item | estimate |
|---|---|
| E1-raw only, all 9 meshes, all 3 methods | **≈ 6.4 h** single-threaded |
| complete E1/E2/E3 × raw/binary set | **≈ 38 h** single-threaded |
| per-mesh E1-only, 800x100 | ≈ 1.86 h |
| trajectory storage, double precision, uncompressed | **≈ 20 GB** |

Both tractable, and the numbers **support** the protocol's claim that the sensitivity plan
needs no extra optimization. Add them to the Phase-A disk/memory and compute estimate
(Mi4).

## 7. Additional Phase-A gate this audit recommends

**A post-freeze T1 feasibility rescan**, on the existing Olhoff trajectories, after every
threshold is frozen and before production. Its only permitted outputs are "feasible" or
"infeasible, triggering a documented amendment". It may never retune a threshold.

Justification: this audit performed exactly that computation in minutes, from data that
already exists, and it changed the study. A protocol that can be shown infeasible before it
runs should be shown infeasible before it runs.

## 8. Revised sequence

The protocol's Phase A–E sequence is sound. Two insertions:

- **Before Phase A:** resolve C1 and C2 (human scientific decisions), apply the §3
  corrections, re-hash the protocol.
- **Within Phase A:** add the T1 feasibility rescan (§7) and the resource estimate (§6).

Phases B–E (smoke tests, the 240x30 integrity pilot, nine-mesh production, blinded offline
freeze, timing replays, independent post-run audit) need no change. The pilot's scope
restriction — engineering integrity only, never threshold selection or profile tuning — is
correct and should be enforced literally.

## 9. What would change this verdict

Resolution of C1 and C2 by a documented scientific decision, plus the §3 corrections, moves
this to **READY AFTER NARROW METHODOLOGICAL CORRECTIONS** and then to ready. No new
optimization runs are needed to get there — both blocking items are definitional, and the
evidence needed to decide them already exists in the repository.
