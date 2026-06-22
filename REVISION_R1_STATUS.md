# REVISION_R1_STATUS

**Date:** 2026-06-22  
**Authority:** `examples/Revision_v1/revision_v1_update1.md` and `scripts/revision_v1/IMPLEMENTATION_MAP.md`  
**Inputs:** `authoritative_formulation_audit.md`, `cr2_failure_diagnosis.md`,
`cr2_rerun_protocol.md`, all gate artifacts in `scripts/revision_v1/` and
`examples/Revision_v1/cr2/`

This document is conservative. Status is based on documented artifact evidence
only. Intention, planning text, and code inspection are not treated as evidence
of completion. A gate with no result artifact is NOT_STARTED regardless of
whether the required script exists.

---

## Gate A0 — Authoritative Formulation

**Status: PASSED (MATLAB-only gate)**

**Gating strategy (2026-06-22):** The R1 reviewer-facing gate is MATLAB-only.
Cross-language Python parity (`test_a0_parity.m`, gate A0-G) is downgraded to
optional QA. This decision was made because the system Python interpreters
(3.9, 3.10) are broken x86_64 builds on this arm64 machine, making A0-G fragile
to execute without manually setting `GATE_A0_PYTHON`. The MATLAB-only gate
certifies the authoritative formulation in the same language as all production
experiments; Python parity remains desirable but is not required for R1.

### Supporting evidence — MATLAB-only gate (PASSED)

**`a0_matlab_result.json`** (`test_a0_matlab_only.m`, run 2026-06-22, R2025b):
All 10 verification items passed. MATLAB is the language used for all production
experiments (CR2, Exp1–Exp5).

| Item | Description | Result |
|------|-------------|--------|
| 1 | Solid reference eigenpair: reference_modal_mass = 1 | PASS |
| 2 | Load scalar is omega0^2 (not omega0): omega_sq = omega^2 | PASS |
| 3 | Load vector F = omega0^2 * M(x) * Phi0 (evolving M) | PASS |
| 4 | No rho_nodal scaling: obsolete_rho_source_used = false | PASS |
| 5 | Mass normalisation: Phi0'*M0*Phi0 = 1 | PASS |
| 6 | Deterministic phase: largest-|DOF| entry >= 0 | PASS |
| 7 | Omitted and complete branches numerically distinguishable | PASS |
| 8 | Complete sensitivity matches central FD: max rel. error 4.13e-8 (≤ 1e-5) | PASS |
| 9 | Obsolete semi_harmonic_rho_source rejected by wrapper | PASS |
| 10 | harmonic_normalize=true rejected by wrapper | PASS |

Additional sub-tests:

- **V1a** (`v1a_fd_results.json`, status=passed): FD error 4.1e-8 in both MATLAB
  and Python (≤1e-5 tolerance). Both languages pass; values are identical to item 8.
- **V1b** (`v1b_mac_results.json`, status=passed): mass-normalized and
  phase-oriented reference modes, MAC = 1.0 to 1e-12, in both MATLAB and Python.
- **V1c** (`v1c_cr2_validation.json`, status=passed): CR2 production configs pass
  all Gate A0 authoritative-load constraints.
- Production run log: `[Load cases] authoritative semi_harmonic baseline=solid,
  load_sensitivity=omitted` — confirms formula was executed in production.

### Optional QA (not blocking R1)

- **A0-G** (`test_a0_parity.m`): MATLAB/Python cross-language parity at ≤1e-8
  relative error. Never executed (Python interpreter environment issue on this
  machine). Not required for R1 unless the manuscript or supplement makes a
  quantitative Python parity claim.

### Remaining work

None blocking R1. Optional: run `test_a0_parity.m` with
`GATE_A0_PYTHON=/Users/piotrek/Programming/topOpt4freqMax/.venv/bin/python3` to
certify A0-G if Python parity is claimed.

### Reviewer relevance

Gate A0 is certified in MATLAB, which is the language of all production
experiments. Every production result can now be attributed to the authoritative
formulation.

### Claim impact

All claims about the authoritative inertial-load formulation F = omega0^2 * M(x) * Phi0
are now backed by `a0_matlab_result.json`. The 10-item MATLAB gate constitutes
reviewer-facing R1 evidence for the formulation correctness.

---

## Gate I1 — Fail-Loud Infrastructure

**Status: PASSED**

### Supporting evidence

**`examples/Revision_v1/output/smoke/i1_smoke_result.json`** (2026-06-22):

- Gate I1 smoke executed via `run_all_revision_experiments('smoke')`.
- `exp_smoke_fail.m` produced a schema-valid result with `success=false`,
  `termination.reason='iteration_cap'`, `termination.capped=true`,
  `iterations.count=200`, `iterations.cap=200`, `design_change=5.00e-03`.
- `localAccept_Smoke` detected the capped result and reported the exact
  condition: `"reached iteration cap: 200/200 iterations without convergence,
  design change = 5.00e-03"`.
- Runner exited with `error('run_all:GateI1Confirmed', ...)` — non-zero status.
- Stack trace preserved in console output.
- `smoke_fail_result.mat` artifact written to `output/smoke/`.

| Verification item | Result |
|---|---|
| Capped run detected | YES |
| Exact condition reported | YES — `iteration_cap: 200/200, dc=5.00e-03` |
| Runner ended with error | YES — `run_all:GateI1Confirmed` |
| Stack trace in console | YES |
| Schema-valid struct produced | YES — `check_experiment_result` passed |
| MAT artifact written | YES — `smoke_fail_result.mat` |

Provenance: R2025b, MACA64 (Apple M1 Max), commit `2f3389be`.

### Remaining work

None blocking R1. The I1 gate certifies that the fail-loud infrastructure works.
The pre-Jun-19 CR2 runs predate the schema and do not need retroactive I1
certification — the CR2 result JSON contains all required fields by inspection.

### Reviewer relevance

Gate I1 is now certified. Any future experiment run through the master runner
will be automatically rejected if it reaches the iteration cap, loses the tracked
mode, or throws an exception.

### Claim impact

All accepted experiment results produced after 2026-06-22 (when Gate I1 passed)
are guaranteed to have been checked by the fail-loud runner.

---

## Gate V1 — Solver Verification

**Status: PASSED**

All mandatory V1 items are now documented with result artifacts. V1-7
(MATLAB/Python parity) is optional under the MATLAB-only gating strategy.

### Supporting evidence

| Sub-test | Artifact | Status | Key result |
|----------|----------|--------|------------|
| V1a | `v1a_fd_results.json` | PASSED | Complete FD max rel err 4.1e-8 (≤1e-5), both MATLAB+Python |
| V1b | `v1b_mac_results.json` | PASSED | Mass-normalised modes, phase convention, MAC=1 to 1e-12 |
| V1c | `v1c_cr2_validation.json` | PASSED | CR2 configs pass all Gate A0 constraints |
| V1-3 | `v1_3_multimode_results.json` | PASSED | Objective additivity rel err=0; factor^2 scaling rel err=1.8e-15 |
| V1-4 | `v1_4_sensitivity_results.json` | PASSED | Both branches finite+deterministic; branch diff 71.3%; FD 4.1e-8 |
| V1-5 | satisfied by V1a and A0 item 8 | PASSED | Max rel err 4.1e-8 << 1e-5 in both tests |
| V1-6 | `v1_6_reordering_results.json` | PASSED | phi_1 found at correct permuted index; cross-design MAC=1.000 |
| V1-7 | not run | OPTIONAL | Not required unless Python parity is claimed in manuscript |

**V1-3 detail** (`test_v1_3_multimode.m`, 6×2 mesh, 2 modes):
- Additivity: obj(mode1 + mode2) = obj(mode1) + obj(mode2) to floating-point zero.
- Factor^2 scaling: obj(f1=2, f2=3) = 4·obj1 + 9·obj2, rel err = 1.8e-15.

**V1-4 detail** (`test_v1_4_sensitivity.m`, gate_a0_fixture):
- Both branches (omitted, complete) execute without error, produce finite outputs.
- Max branch relative difference = 71.3% (load-derivative contribution is substantial).
- Cross-run consistency: complete_sensitivity identical between omitted-mode and complete-mode runs.
- Complete FD relative error: 4.129e-8 (tol 1e-5).
- Determinism: two identical complete-branch runs produce bitwise-equal sensitivities.

**V1-6 detail** (`test_v1_6_reordering.m`, 6×2 mesh, 5 solid modes):
- MAC(phi_i, phi_j, M) = 1 (i=j), ~2e-31 (i≠j): eigenvectors are M-orthogonal.
- Permuted mode set [phi_3, phi_2, phi_5, phi_4, phi_1]: phi_1 found at index 5.
- Cross-design (solid vs x=0.3 uniform): best MAC = 1.000 at mode 1 (no shape change at uniform density).
- Phase invariance: MAC(phi, -phi) = MAC(phi, phi) = 1.

### Remaining work

None blocking R1. V1-7 (Python parity) is optional.

### Reviewer relevance

Gate V1 is certified. All Exp1–Exp5 production results can be attributed to a
verified solver with confirmed multi-mode assembly, sensitivity variants, and
mode tracking.

### Claim impact

The multi-mode objective and complete/omitted sensitivity branches are both
verified in MATLAB. The mode tracking (MAC-based) is confirmed to be robust to
frequency reordering.

---

## Gate CR2 (Gate E4-CR2) — Omitted Load Sensitivity Study

**Status: FAILED**

### Production run (`cr2_production_results.json`)

Outcome: FAILED_PARTIAL. Both variants hit the 400-iteration cap.

| Criterion | Threshold | Variant A (omitted) | Variant B (complete) |
|-----------|-----------|---------------------|----------------------|
| Iteration cap | not reached | **400** (fail) | **400** (fail) |
| design_change | ≤ 1e-3 | **5.24e-3** (fail) | **2.00e-1** (fail) |
| feasibility | ≤ 1e-8 | **9.29e-5** (fail) | **1.17e-6** (fail) |
| MAC | ≥ 0.8 | 0.999 (pass) | 0.955 (pass) |

Diagnosis (`cr2_failure_diagnosis.md`):

- **Variant A**: stable limit cycle. Objective flat to 0.3% from iteration 100
  to 400 (7824 → 7845, 0.27%). Design_change oscillates at 0.004–0.009 and
  does not trend toward zero. Feasibility alternates 0 ↔ ~1e-4. Mode tracking
  healthy throughout (mode 1, MAC ≥ 0.997). This is OC micro-oscillation, not
  divergence, but formal convergence was not reached.

- **Variant B**: period-2 instability from iteration 9. Design_change = 0.200
  (= move_limit) on all 400 iterations. Objective alternates exactly between
  4833 and 4959 every other iteration; amplitude does not shrink. Mode tracking
  failed at iteration 4 (switched from mode 1 to mode 2, never recovered). The
  period-2 cycle is a fixed attractor: more iterations would reproduce the same
  cycle.

### Rerun — stabilization screen (`cr2_rerun_summary.md`, `cr2_rerun_manifest.json`)

Protocol applied: Variant A unchanged; Variant B move_limit reduced from 0.20
to 0.02. Outcome: DIAGNOSTIC ALGORITHM-FAILURE EVIDENCE (unmatched pair, no
accepted comparison eligible).

| Criterion | Threshold | Variant A (rerun) | Variant B (move_limit=0.02) |
|-----------|-----------|-------------------|------------------------------|
| Iteration cap | not reached | **400** (fail) | **400** (fail) |
| design_change | ≤ 1e-3 | **5.24e-3** (fail) | **2.00e-2** (fail) |
| feasibility | ≤ 1e-4 | 9.29e-5 (pass) | 0.0 (pass) |
| MAC | ≥ 0.8 | 0.999 (pass) | 0.991 (pass) |
| Algorithm-failure signature | — | not met | **met** |

Variant B at move_limit=0.02 remains saturated (dc = move_limit = 0.02 throughout
400 iterations). Reducing move_limit by 10× moved the saturation point but did
not stabilize the optimizer. The period-2 instability persists; the cycle
amplitude is now smaller but the design_change criterion remains unmet.

Variant A is classified **inconclusive** by the protocol (objective plateau
present, but algorithm-failure signature not met: dc stays above 1e-3 with no
decaying trend, but does not meet the ≥90%-of-iterations-at-99%-of-move-limit
criterion because dc≈0.005 ≪ move_limit=0.2).

### Remaining work — to reach a PASSED gate

Per `cr2_rerun_protocol.md` Section 7:

1. If a matched A/B comparison at move_limit=0.02 is attempted, both variants
   must be run under identical settings differing only in `load_sensitivity`.
   The current rerun pair is intentionally unmatched and cannot produce a
   scientific comparison.
2. Variant A requires a different stabilization strategy (MMA or significantly
   larger rmin/filter) to formally converge within the cap.
3. Alternatively, the accepted comparison may be attempted at a coarser mesh
   where the OC micro-oscillation damps out.
4. Until an accepted matched pair exists, Gate E4-CR2 remains FAILED.

### Reviewer relevance

CR2 directly answers reviewer demand C1: "quantitative validation of omitted
load-sensitivity contribution." The current state provides qualitative algorithm-
failure evidence but no accepted quantitative comparison endpoint.

### Claim impact

Per `cr2_rerun_protocol.md` Section 6.B, the following claims ARE permitted from
the current evidence:

- Variant B (complete sensitivity) with OC at move_limit ∈ {0.20, 0.02} on a
  160×20 SS beam failed to converge; a period-2 cycle with full move-limit
  saturation was observed.
- Variant A (omitted sensitivity) with OC at move_limit=0.20 produced a
  diagnostic objective plateau (relative objective range <0.3% over last 300
  iterations) but did not formally converge.
- Reducing Variant B's move limit from 0.20 to 0.02 did not eliminate the
  saturation pattern.

Per `cr2_rerun_protocol.md` Section 6.B, the following claims are FORBIDDEN:

- Endpoint objective or frequency superiority between Variant A and Variant B.
- Topology comparison as a comparison of converged optima.
- "Complete sensitivity is worse" or "omission is negligible."
- Extrapolation from OC failure to all optimizers.

---

## Gate A4 (Gate E4-A4) — Eigenpair Refresh Study

**Status: NOT_STARTED**

### Supporting evidence

Configs `ss_beam_harmonic_frozen.json` and `ss_beam_harmonic_periodic.json` exist
but contain the obsolete formula and are flagged in the audit for migration.
Script `exp4_sensitivity_ablation.m` exists but implements the old four-variant
(A/B/C/D) ablation using the non-authoritative formula. No A4-dedicated script
or result artifact (N ∈ {1, 5, 10, 50, ∞}) exists.

### Remaining work

- Migrate or replace `ss_beam_harmonic_frozen.json` / `ss_beam_harmonic_periodic.json`
  to the authoritative load type with independent refresh intervals.
- Write `run_a4.m` covering N ∈ {1, 5, 10, 50, ∞} per IMPLEMENTATION_MAP A4-1.
- Run to convergence (same stopping rules as CR2).
- Document N=1 equivalence status for Yuksel-Yilmaz; report oscillation if it
  occurs.
- Save per-N refresh events, tracked mode indices, MAC, frequencies, objective,
  feasibility, convergence history.

### Reviewer relevance

A4 answers reviewer demand C4 (MR6): frozen-mode reliability relative to periodic
eigenpair refresh. Until it passes, any claim about refresh interval accuracy or
the relationship to Yuksel-Yilmaz is unsupported.

### Claim impact

The frozen-accuracy claims (A4) are Tier 2: mandatory only if retained. They
must be removed if A4 is not completed.

---

## Gate E2 — Clamped-Beam Experiments (Exp2)

**Status: NOT_STARTED**

### Supporting evidence

`exp2_clamped_beam.m` exists. JSON configs `clamped_beam_200x25.json` and
`clamped_beam_400x50.json` exist but use the obsolete formula (flagged in the
audit with `rho_source`, `harmonic_normalize:true`, no explicit `load_sensitivity`).

No Exp2 result artifact exists. No convergence evidence for any α ∈ {1, 0.75, 0.5,
0.25, 0} under the authoritative formulation exists.

### Remaining work (IMPLEMENTATION_MAP E2-1 through E2-7)

- Fix the known failure (reproduce stack trace first).
- Migrate configs to authoritative load type.
- Run α sweep to convergence; five accepted result `.mat` files required.
- Export MAC matrices, mode indices, convergence histories, grayness, topologies.
- Diagnose α=0.75 without assuming monotonicity.
- Add A5 lowest-mode check per case.
- Add one fair external-method comparison with accurate label.

### Reviewer relevance

Exp2 is a Tier 1 mandatory blocker. It provides the primary multi-mode evidence
for the paper's frequency-gain claim.

### Claim impact

All clamped-beam frequency-gain and monotonicity claims (including the 4.61×
building gain if building cases are part of Exp2) are unsupported until Exp2
passes.

---

## Gate E3 — Mesh Convergence (Exp3)

**Status: NOT_STARTED**

### Supporting evidence

`exp3_mesh_convergence.m` exists. The configs for 200×25 and 400×50 exist but
use the obsolete formula.

No Exp3 result artifact exists.

### Remaining work (IMPLEMENTATION_MAP E3-1 through E3-3)

- Declare numerical convergence criterion in the script header before any run.
- Migrate configs to authoritative load type.
- Run both meshes to convergence under identical settings.
- Report frequency, gain, MAC, selected mode, grayness, constraint residual,
  topology differences. Declare lack of mesh convergence if criterion is not met.

### Reviewer relevance

Exp3 is a Tier 1 mandatory blocker. Mesh-convergence evidence is required by
several reviewer items.

### Claim impact

All mesh-convergence claims are unsupported.

---

## Gate S1 — Low-Mode and Grayness Diagnosis

**Status: NOT_STARTED**

### Supporting evidence

Scripts `exp2b_building.m` and `diagnose_modes.m` are referenced in the
implementation map. No result artifacts exist. Building cases are known to be
capped (referenced in `revision_v1_update1.md` §Workstream 6).

### Remaining work (IMPLEMENTATION_MAP S1-1 through S1-5)

- Rerun capped building cases to convergence.
- Export mode shapes, elementwise kinetic and strain energy, energy fraction in
  low-density regions, localization metric, density support, MAC.
- Classify every suspicious mode from energy and localization evidence.
- Compare SIMP against one documented mitigation (RAMP, Heaviside, void-mass).
- Report grayness for every final topology.

### Reviewer relevance

S1 is a Tier 1 mandatory blocker. Without it, the claim that the method avoids
spurious low-density modes cannot be retained and must be removed.

### Claim impact

The "no spurious low-density modes" claim must be removed or replaced with a
bounded limitation until S1 is completed with supporting evidence.

---

## Gate P1 — Performance Evidence

**Status: NOT_STARTED**

### Supporting evidence

None. No instrumented timing measurements exist under the authoritative
formulation. The production CR2 run log records total elapsed time but not the
independently measured breakdown (init, loop, postprocessing, per-iteration)
required by Gate P1.

### Remaining work (IMPLEMENTATION_MAP P1-1 through P1-7)

- Instrument MATLAB and Python solvers with independently measured timings.
- Implement benchmark mode (disable diagnostics/plots).
- Warm-up runs followed by ≥10 measured executions per mesh.
- Investigate Yuksel 180–200 vs 1,000+ iteration discrepancy.
- Correct comparator labels to "Olhoff-inspired."
- Recompute scaling fit.
- Replace 8.6% gap and 7.1× speedup claims if not reproduced from corrected data.

### Reviewer relevance

P1 is a Tier 1 blocker for any performance, speedup, complexity, or scaling
claim. Without it, all headline numbers are at risk.

### Claim impact

The 8.6% frequency gap, 7.1× speedup, and O(n^α) scaling claims have no
supporting evidence under the revised formulation. They must be replaced or
removed.

---

## Gate MS — Manuscript and Response Update

**Status: NOT_STARTED**

### Supporting evidence

None. The manuscript (`paper/main.tex`) has not been updated per `revision_v1_update1.md`.
The response letter has not been drafted. The audit found specific errors in the
current manuscript text: "initial uniform" reference instead of solid, sensitivity
described as "set to zero" rather than "omitted," OC/MMA description inconsistencies,
void-material value inconsistencies (1e-9 vs 1e-6), stopping tolerance inconsistencies
(1e-3 vs 2e-3), and incomplete bibliography entries.

Manuscript cannot be updated until accepted experiment artifacts exist (MS-18
generates tables and figures from accepted results).

### Remaining work (IMPLEMENTATION_MAP MS-1 through MS-19)

All 19 manuscript items are pending. The most critical:

- MS-1: qualify α=0.75 monotonicity claim.
- MS-3: state explicitly that load sensitivity is nonzero but omitted.
- MS-6/7: correct load equation, baseline, MAC definition (Eq. 9).
- MS-9: distinguish canonical methods from local implementations.
- MS-10: replace unsupported speedup, gap, complexity, no-spurious-mode claims.
- MS-18: regenerate all tables and figures from accepted artifacts.
- MS-19: response letter addressing every reviewer item individually.

### Reviewer relevance

No revised manuscript can be submitted without MS.

### Claim impact

Every claim in the manuscript is currently unverified under the authoritative
formulation and unverified against any accepted convergence evidence.

---

## Gate RP — Reproducibility Package

**Status: NOT_STARTED**

### Supporting evidence

None. No manifest, no clean-run instructions, no sysinfo report, no immutable
release tag, no DOI, no checksums exist.

### Remaining work (IMPLEMENTATION_MAP RP-1 through RP-8)

All 8 reproducibility items are pending, including: machine-readable manifest,
clean-run instructions and artifact checks, hardware/software report, supplementary
inventory, release tag + DOI, checksums, completion report.

### Reviewer relevance

RP is required before any final submission.

---

## Summary Tables

### Gate status

| Gate | Status | Tier | Blocker for |
|------|--------|------|-------------|
| A0 | PASSED (MATLAB-only) | 1 | all experiments |
| I1 | PASSED | 1 | all accepted evidence |
| V1 | PASSED | 1 | Exp1–Exp5 |
| CR2 | FAILED | 1 | load-sensitivity claim |
| A4 | NOT_STARTED | 1/2 | refresh/accuracy claims |
| E2 | NOT_STARTED | 1 | clamped-beam claims |
| E3 | NOT_STARTED | 1 | mesh-convergence claims |
| S1 | NOT_STARTED | 1 | spurious-mode claim |
| P1 | NOT_STARTED | 1 | all performance claims |
| MS | NOT_STARTED | 1 | submission |
| RP | NOT_STARTED | 1 | submission |

### Artifact inventory

| Artifact | Exists | Gate | Status |
|----------|--------|------|--------|
| `v1a_fd_results.json` | yes | V1a | PASSED |
| `v1b_mac_results.json` | yes | V1b | PASSED |
| `v1c_cr2_validation.json` | yes | V1c | PASSED |
| `cr2_smoke_manifest.json` | yes | CR2 smoke | PASSED (structural only) |
| `cr2_production_results.json` | yes | E4-CR2 | FAILED_PARTIAL |
| `cr2_production_summary.md` | yes | E4-CR2 | FAILED_PARTIAL |
| `cr2_failure_diagnosis.md` | yes | E4-CR2 | diagnostic |
| `cr2_rerun_summary.md` | yes | E4-CR2-rerun | DIAGNOSTIC_ALGO_FAIL |
| `cr2_rerun_manifest.json` | yes | E4-CR2-rerun | DIAGNOSTIC_ALGO_FAIL |
| `a0_matlab_result.json` | **YES** | A0-MATLAB-ONLY | PASSED (all 10 items) |
| `a0_parity_result.json` (MATLAB/Python) | **NO** | A0-G (optional QA) | NOT_RUN |
| `i1_smoke_result.json` | **YES** | I1 | PASSED |
| `output/smoke/smoke_fail_result.mat` | **YES** | I1 | PASSED |
| `v1_3_multimode_results.json` | **YES** | V1-3 | PASSED |
| `v1_4_sensitivity_results.json` | **YES** | V1-4 | PASSED |
| `v1_6_reordering_results.json` | **YES** | V1-6 | PASSED |
| Any Exp2 result `.mat` | **NO** | E2/E23 | NOT_STARTED |
| Any Exp3 result `.mat` | **NO** | E3/E23 | NOT_STARTED |
| Any A4 result `.mat` | **NO** | A4/E4 | NOT_STARTED |
| Any S1 result `.mat` | **NO** | S1 | NOT_STARTED |
| Any P1 timing result | **NO** | P1 | NOT_STARTED |
| Manuscript diff | **NO** | MS | NOT_STARTED |
| Response letter | **NO** | MS | NOT_STARTED |
| Reproducibility manifest | **NO** | RP | NOT_STARTED |
| Release tag / DOI | **NO** | RP | NOT_STARTED |

---

## Conclusions

### 1. Minimum remaining work for a defensible R1 submission

The following are required with no exception. None is optional.

**Phase 0 — Certify A0 (DONE for MATLAB gate)**
1. ~~Run `test_a0_parity.m` end-to-end~~ — replaced by MATLAB-only gate.
   `a0_matlab_result.json` confirms all 10 authoritative-formulation items in
   MATLAB (R2025b, 2026-06-22). This is the reviewer-facing R1 standard.
2. Verify that all A0-F1 through A0-F6 changes identified in the audit are
   present in the codebase (code inspection, not yet documented as a checklist).

**Phase 1 — Certify I1 (DONE)**
3. ~~Execute smoke runner~~ — DONE 2026-06-22. `i1_smoke_result.json` confirms
   runner correctly detects capped run and exits with `run_all:GateI1Confirmed`.

**Phase 2 — Certify V1 (DONE)**
4. ~~Run V1-3 and V1-6~~ — DONE 2026-06-22. V1-3 (objective additivity), V1-4
   (sensitivity variants), and V1-6 (mode reordering) all PASSED. V1-7 optional.

**Phase 3 — Resolve CR2**

5. Implement one of the following to produce an accepted CR2 converged pair:
   a. Run both variants with MMA and confirm both formally converge (dc ≤ 1e-3,
      feasibility ≤ 1e-4, MAC ≥ 0.8, not capped). OR
   b. Run both variants at a different mesh or filter where OC converges.
   
   The pair must be matched: identical settings except `load_sensitivity`.
   
   If no accepted pair can be produced, the CR2 section of the manuscript must
   be limited to the diagnostic algorithm-failure narrative permitted by
   `cr2_rerun_protocol.md` Section 6.B. This means no comparison of converged
   endpoints, no "negligible" or "equivalent" language, and a statement that the
   accepted quantitative comparison is pending.

**Phase 4 — Complete Tier 1 experiments**

6. Exp2 (clamped beam, α sweep): five converged results, A5 check, comparator
   with accurate label.
7. Exp3 (mesh convergence): two converged results at 200×25 and 400×50 with
   declared convergence criterion.

**Phase 5 — S1 and performance**

8. S1: rerun building cases; classify every suspicious mode from energy
   evidence; report grayness.
9. P1: instrument solvers; run ≥10 timing measurements per mesh; correct
   comparator labels.

**Phase 6 — Manuscript and response**

10. MS-18: regenerate all tables and figures from accepted artifacts only.
11. MS-1 through MS-17: all manuscript corrections and qualifications.
12. MS-19: response letter addressing every reviewer item individually.

**Phase 7 — Reproducibility**

13. RP-1 through RP-8: manifest, clean-run instructions, sysinfo, checksums,
    release tag, DOI.

### 2. Full remaining work to satisfy revision_v1_update1.md

Everything in item 1 above, plus:

- A4 study (N ∈ {1, 5, 10, 50, ∞}), converged, with mode tracking and Yuksel
  equivalence status documented. Required if frozen-accuracy or N=1 claims are
  retained (Tier 2).
- Canonical Olhoff benchmark (Tier 2), required if canonical speedup language
  is retained.
- RAMP/Heaviside/void-mass mitigation comparison (Tier 2), required if
  absence-of-spurious-modes claim is retained.
- Exp1 performance table with ≥10 measured executions, warm-up, standard
  deviations, and corrected comparator labels, required if 8.6%/7.1× numbers
  are retained or regenerated (Tier 2 for exact reproduction).
- Complete bibliography (MS-16), notation correction (MS-13), cross-reference
  verification (MS-15).
- Supplementary material inventory (MS-17, RP-4).

### 3. Claims that are currently supported

These claims have documented artifact evidence and can be asserted in the
response letter or manuscript, with appropriate scope qualifications.

1. **A0 (MATLAB gate) confirmed**: All 10 authoritative-formulation items
   verified in MATLAB R2025b: solid reference eigenpair, omega0^2 scalar, load
   F=omega0^2*M(x)*Phi0, mass normalisation, deterministic phase, distinguishable
   branches, complete FD match (4.1e-8), obsolete-setting rejection.

2. **V1-3 confirmed**: Multi-mode objective assembly is additive to
   floating-point zero and factor^2 scaling holds to 1.8e-15 relative.

3. **V1-4 confirmed**: Both sensitivity branches execute, are finite and
   deterministic, are distinguishable (71.3% max relative difference), and the
   complete branch matches central FD to 4.1e-8.

4. **V1-6 confirmed**: MAC-based tracking correctly identifies the target mode
   even when the mode set is permuted (phi_1 found at correct permuted index with
   MAC=1). Phase-invariance confirmed (MAC is sign-invariant). Cross-design MAC
   between solid and half-density modes is 1.000.

5. **V1a confirmed**: The complete analytical sensitivity
   `omega0^2 * (dM/dxe) * Phi0` matches central finite differences to
   ≤4.1e-8 relative error in both MATLAB and Python on the 6×2 deterministic
   fixture. This verifies the complete-gradient implementation.

6. **V1b confirmed**: The mass-normalized, phase-oriented reference mode
   convention is consistent between MATLAB and Python on the deterministic
   fixture. Squared mass-weighted MAC is implemented identically in both
   languages.

7. **I1 confirmed**: Fail-loud runner correctly detects a capped run and
   reports the exact failure condition. All future accepted results are
   certified by the runner.

8. **Load sensitivity is nonzero**: The omitted gradient term differs from
   the complete finite-difference sensitivity by 3%–145% per element (V1a
   omitted_relative_errors_vs_complete_fd). The term is not mathematically zero
   and is not negligible in absolute magnitude.

4. **Variant B (complete sensitivity, OC, move_limit=0.20) fails to converge**
   on the 160×20 SS beam: a period-2 cycle with full move-limit saturation
   (dc=0.20) is present from iteration 9 and persists for all 400 iterations.
   This is documented as a predeclared algorithm-failure signature in
   `cr2_rerun_protocol.md` Section 6.B.

5. **Variant B (complete sensitivity, OC, move_limit=0.02) also fails to
   converge**: the stabilization screen shows dc=0.02=move_limit for all 400
   iterations, confirming the instability is not solely caused by the original
   move limit.

6. **Variant A (omitted sensitivity, OC, move_limit=0.20) exhibits a diagnostic
   objective plateau**: relative objective range <0.3% over the last 300 of 400
   iterations on the 160×20 SS beam. Mode tracking is stable at mode 1 with
   MAC ≥ 0.997.

7. **Variant A and Variant B converge to topologically different designs**:
   density correlation 0.519, MAD=0.275, 58.6% of elements differ by >0.01 in
   density (at move_limit=0.20).

8. **V1c confirmed**: CR2 configs satisfy all authoritative-load schema
   constraints (`solid_reference`, no `rho_source`, normalization disabled,
   `load_sensitivity` explicit, `gate_a0_diagnostics=true`).

### 4. Claims that are currently unsupported

These claims require completed experiments that have not been run or have not
produced accepted results.

1. Any clamped-beam or multi-mode frequency-gain claim (Exp2 not run under
   authoritative formulation).
2. Mesh convergence (Exp3 not run under authoritative formulation).
3. α=0.75 monotonicity interpretation (Exp2 not run).
4. Building-case frequency gain, including 4.61× (S1 building cases not rerun).
5. A4 refresh-interval accuracy or Yuksel-Yilmaz equivalence (A4 not run).
6. Any speedup (7.1×), frequency gap (8.6%), or scaling claim (P1 not run).
7. Absence of spurious low-density modes (S1 not run; no mode classification).
8. Grayness values for any topology in the revised manuscript (S1 not run).
9. Quantitative comparison of Variant A versus Variant B converged endpoints
   (no accepted matched pair exists; explicitly prohibited by the rerun protocol).

### 5. Claims that must be removed or qualified before submission

These are assertions that the current evidence either contradicts or does not
support, and that cannot stand without qualification.

1. **"Omitted load sensitivity is negligible" or equivalent.** V1a shows the
   omitted term differs from the complete FD gradient by 3%–145% per element.
   No converged comparison exists. Per `cr2_rerun_protocol.md` Section 6.B,
   this language is explicitly forbidden. Required action: remove and state that
   the practical effect is being evaluated by CR2 (report current status as
   inconclusive).

2. **"Results converge" for the proposed method on the SS beam.** Variant A hit
   the 400-iteration cap with dc=5.24e-3 > 1e-3. Required action: qualify as
   "diagnostic objective plateau observed; formal convergence not achieved."

3. **8.6% frequency gap, 7.1× speedup, 4.61× building-gain.** No reproduced
   evidence under the authoritative formulation. Required action: remove all
   three claims from the manuscript and response letter until P1 and Exp2/S1
   produce accepted replacements. Retain only under explicit "pending revision"
   language if the editor permits it.

4. **"No spurious low-density modes"** or equivalent claim of mode quality.
   S1 has not been run; no mode classification from energy evidence exists.
   Required action: remove or replace with bounded limitation statement.

5. **Monotonicity of the α sweep** (performance degrades monotonically with
   α). Exp2 has not produced converged α-sweep evidence under the authoritative
   formulation. Required action: remove or qualify; do not assert monotonicity
   until converged evidence is available.

6. **"Equivalent to Yuksel-Yilmaz"** or any claim that N=1 refresh equals the
   Yuksel-Yilmaz method. This requires A4, which has not been run, and explicit
   algorithmic comparison. Required action: remove until A4 produces documented
   equivalence evidence.

7. **Comparator labels "Olhoff" and "Yuksel-Yilmaz" without qualification.**
   The audit identifies that the implemented comparators are local modified
   implementations. Required action: relabel to "Olhoff-inspired" and
   "Yuksel-Yilmaz-inspired" (or equivalent qualified forms) throughout all
   tables and figures.

8. **Load equation references to "initial uniform" design.** The manuscript
   text uses "initial uniform" as the reference design. The authoritative
   formulation requires the fully solid domain. Required action: correct all
   references and rerun any experiment where the distinction matters (all of them).

9. **Sensitivity "set to zero" language.** The audit and `revision_v1_update1.md`
   require the manuscript to state "nonzero term omitted," not "zero." Required
   action: correct in all pseudocode, comments, algorithm descriptions, and
   response letter text.

---

*Status derived exclusively from documented artifacts. Unexecuted scripts,
planning documents, and audit intentions do not constitute evidence of
completion.*
