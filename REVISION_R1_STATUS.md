# REVISION_R1_STATUS

**Date:** 2026-06-30
**Authority:** `examples/Revision_v1/revision_v1_update1.md` and `scripts/revision_v1/IMPLEMENTATION_MAP.md`

This document is conservative. Status is based on documented artifact evidence
only. Intention, planning text, and code inspection are not treated as evidence
of completion. A gate with no result artifact is NOT_STARTED regardless of
whether the required script exists.

---

## Gate A0 — Authoritative Formulation

**Status: PASSED (MATLAB-only gate)**

**Gating strategy:** R1 reviewer-facing gate is MATLAB-only. Python parity (A0-G)
is downgraded to optional QA because the system Python interpreters (3.9, 3.10)
are broken x86_64 builds on this arm64 machine. MATLAB is the language of all
production experiments.

### Supporting artifacts

| Artifact | Path | Status |
|---|---|---|
| Gate A0 MATLAB result | `scripts/revision_v1/a0_matlab_result.json` | PASSED (all 10 items) |
| FD verification | `scripts/revision_v1/v1a_fd_results.json` | PASSED |
| Mode normalization | `scripts/revision_v1/v1b_mac_results.json` | PASSED |
| Config validation | `scripts/revision_v1/v1c_cr2_validation.json` | PASSED |

### Accepted evidence

All 10 MATLAB gate items passed (R2025b, 2026-06-22):
solid reference eigenpair, omega0^2 scalar, load F=omega0^2·M(x)·Phi0,
no rho_nodal scaling, mass normalization Phi0'M0Phi0=1, deterministic phase,
distinguishable branches, complete FD match (max rel err 4.13e-8 ≤ 1e-5),
obsolete rho_source rejection, harmonic_normalize rejection.

### Rejected evidence

Python parity (A0-G): not run due to broken interpreter environment. Not
required for R1.

### Claim impact

All claims about the authoritative inertial-load formulation F=omega0^2·M(x)·Phi0
are backed by `a0_matlab_result.json`. Every production experiment (CR2, Exp2,
Exp3, S1) used configs verified by V1c to satisfy these constraints.

### Reviewer-response impact

Gate A0 in MATLAB constitutes reviewer-facing R1 evidence for formulation
correctness. Response letter may cite `a0_matlab_result.json` (10-item table)
and `v1a_fd_results.json` (FD verification) as the primary formulation evidence.

---

## Gate I1 — Fail-Loud Infrastructure

**Status: PASSED**

### Supporting artifacts

| Artifact | Path | Status |
|---|---|---|
| I1 smoke result | `examples/Revision_v1/output/smoke/i1_smoke_result.json` | PASSED |
| Smoke MAT file | `examples/Revision_v1/output/smoke/smoke_fail_result.mat` | EXISTS |

### Accepted evidence

- Smoke experiment reached 200-iteration cap with dc=5.00e-3.
- Runner detected the capped result and exited with `run_all:GateI1Confirmed`.
- Exact condition reported; stack trace preserved.
- Provenance: R2025b, MACA64 (Apple M1 Max), commit 2f3389be.

### Rejected evidence

None — gate passed cleanly.

### Claim impact

All accepted experiment results produced after 2026-06-22 are certified by the
fail-loud runner. Capped, mode-invalid, and exception outcomes are automatically
rejected with documented failure conditions.

### Reviewer-response impact

Gate I1 certifies the result-acceptance infrastructure. No individual experiment
result citation in the response letter requires separate qualification of the
acceptance mechanism.

---

## Gate V1 — Solver Verification

**Status: PASSED**

All mandatory V1 items documented with result artifacts. V1-7 (Python parity)
optional under the MATLAB-only gating strategy.

### Supporting artifacts

| Sub-test | Artifact | Status |
|---|---|---|
| V1a | `scripts/revision_v1/v1a_fd_results.json` | PASSED |
| V1b | `scripts/revision_v1/v1b_mac_results.json` | PASSED |
| V1c | `scripts/revision_v1/v1c_cr2_validation.json` | PASSED |
| V1-3 | `scripts/revision_v1/v1_3_multimode_results.json` | PASSED |
| V1-4 | `scripts/revision_v1/v1_4_sensitivity_results.json` | PASSED |
| V1-6 | `scripts/revision_v1/v1_6_reordering_results.json` | PASSED |
| V1-7 | not run | OPTIONAL |

### Accepted evidence

- **V1a**: Complete analytical sensitivity matches central FD to 4.1e-8 (tol 1e-5) in MATLAB and Python.
- **V1b**: Mass-normalized, phase-oriented modes consistent; MAC=1 to 1e-12.
- **V1c**: All CR2 configs satisfy Gate A0 authoritative-load schema constraints.
- **V1-3**: Multi-mode objective: additivity to floating-point zero; factor^2 scaling to rel err 1.8e-15.
- **V1-4**: Both branches finite and deterministic; branch diff 71.3%; complete FD err 4.1e-8.
- **V1-6**: MAC-based tracking finds permuted phi_1 at correct index with MAC=1; cross-design MAC=1.000; phase-invariant.

### Rejected evidence

V1-7 (Python parity): not run.

### Claim impact

Multi-mode objective assembly and complete/omitted sensitivity branches are
both verified in MATLAB. Mode tracking is robust to frequency reordering.
The load-sensitivity term (71.3% branch difference at V1-4) is mathematically
non-negligible — this refutes any "zero" or "negligible" characterization.

### Reviewer-response impact

V1-4 branch difference (71.3%) must be disclosed in the response letter.
The claim that load sensitivity is "set to zero" must be replaced by
"nonzero term omitted" in all manuscript text. V1-3 supports the multi-mode
formulation description.

---

## Gate CR2 (Gate E4-CR2) — Omitted Load Sensitivity Study

**Status: FAILED / DIAGNOSTIC**

### Supporting artifacts

| Artifact | Path | Status |
|---|---|---|
| Production results | `scripts/revision_v1/cr2_smoke_results.json` | FAILED_PARTIAL |
| Production summary | `scripts/revision_v1/cr2_smoke_summary.md` | FAILED_PARTIAL |
| Failure diagnosis | (cr2_failure_diagnosis.md) | DIAGNOSTIC |
| Rerun manifest | `scripts/revision_v1/cr2_rerun_manifest.json` | DIAGNOSTIC_ALGO_FAIL |

### Accepted evidence

Algorithm-failure signatures documented:
- **Variant A (omitted, move_limit=0.20)**: stable limit cycle; objective plateau
  (relative range <0.3% over last 300 of 400 iterations); dc oscillates 0.004–0.009;
  MAC stable ≥ 0.997. Not formally converged (dc > 1e-3).
- **Variant B (complete, move_limit=0.20)**: period-2 cycle from iteration 9;
  dc=0.200=move_limit on all 400 iterations; objective alternates 4833↔4959 every step.
- **Variant B (move_limit=0.02 rerun)**: still saturated; dc=0.020=move_limit
  throughout 400 iterations. Instability is not solely caused by original move limit.
- Topology correlation at move_limit=0.20: 0.519, MAD=0.275.

### Rejected evidence

**No accepted matched comparison pair exists.** All endpoints are unconverged.
Per `cr2_rerun_protocol.md` Section 6.B, the following are explicitly forbidden:
endpoint objective/frequency comparison between Variant A and B; topology
comparison as converged optima; "complete sensitivity is worse" or "omission
is negligible"; extrapolation from OC failure to all optimizers.

### Claim impact

CR2 provides qualitative algorithm-failure evidence only. No quantitative
comparison of omitted vs complete sensitivity at converged endpoints is available.
The claim that load sensitivity is negligible cannot be supported and is
forbidden by the rerun protocol.

### Reviewer-response impact

Reviewer demand C1 (quantitative validation of omitted load sensitivity) is
NOT fulfilled. Response letter must state: (i) the omitted term is mathematically
non-negligible (V1-4: 71.3% branch difference), (ii) Variant B (complete, OC)
failed to converge due to a period-2 instability, (iii) no accepted comparison
pair exists at this time, and (iv) the practical convergence impact is an open
question requiring MMA-based or coarser-mesh experiments.

---

## Gate A4 (Gate E4-A4) — Eigenpair Refresh Study

**Status: NOT_STARTED**

### Supporting artifacts

None. No A4-dedicated result artifact exists.

### Accepted evidence

None.

### Rejected evidence

Configs `ss_beam_harmonic_frozen.json` and `ss_beam_harmonic_periodic.json`
exist but use the obsolete formula and are not valid for A4.

### Claim impact

All frozen-accuracy and Yuksel-Yilmaz equivalence claims are Tier 2: mandatory
only if retained. Must be removed unless A4 is completed.

### Reviewer-response impact

Any claim about N=1 refresh equivalence to Yuksel-Yilmaz must be removed.

---

## Gate E2 — Clamped-Beam Experiments (Exp2)

**Status: PARTIAL — 1 of 5 α cases genuinely accepted**

### Supporting artifacts

| α | Artifact | Classification | Iterations | MAC | ω₁ (rad/s) | Grayness |
|---|---|---|---|---|---|---|
| 1.00 | `exp2_authoritative_alpha_1_00_result.json` | **accepted** | 1052 | 0.993 | 141.79 | 8.6% |
| 0.75 | `exp2_authoritative_alpha_0_75_result.json` | accepted¹ | 1 | 1.000 | 145.56 | ~100% |
| 0.50 | `exp2_authoritative_alpha_0_50_result.json` | **mode invalid** | 1241 | 0.748 | 2.98 | 4.5% |
| 0.25 | `exp2_authoritative_alpha_0_25_result.json` | **mode invalid** | 2000 | 0.794 | 2.59 | 8.4% |
| 0.00 | `exp2_authoritative_alpha_0_00_result.json` | accepted¹ | 1 | 1.000 | 145.52 | ~100% |

¹ Formally accepted by the runner but scientifically trivial: converged after 1
iteration with grayness ≈ 99.99% (uniform topology — no optimization occurred).
Design_change on the first step fell below 1e-3 without any topology evolution.
These are pre-convergence artifacts of the uniform initial design, not optimized
results, and must not be reported as Exp2 outcomes.

**Root cause of α=0.50/0.25 rejection:** MAC to solid reference mode 1 below
0.8 threshold; tracked-mode frequency drops to ≈3 rad/s (physical reference
is ≈290 rad/s) — consistent with a localized spurious low-density mode taking
over the lowest-frequency position. Identical pathology to Exp3 400×50.

### Accepted evidence

Single scientifically meaningful result: **α=1.00 on 200×25 mesh**.
- omega_1 = 141.79 rad/s (22.57 Hz), converged in 1052 of 2000 iterations.
- MAC to reference mode 1 = 0.993; A5 check: tracked mode is the lowest mode.
- Grayness = 8.6% (nontrivial optimized topology).
- All acceptance criteria met: dc=1.66e-4 ≤ 1e-3, feasibility=0, MAC ≥ 0.8.
- Timing: 108.0 s total, 0.100 s/iter, 249 MB peak.

### Rejected evidence

α=0.75 and α=0.00 formally accepted but trivial (1 iteration, uniform topology):
not usable as Exp2 scientific results.
α=0.50 and α=0.25 mode-invalid: localized spurious mode suppressed the
physical response. Not usable.
No multi-mode sweep comparison is available.

### Claim impact

- **Supportable:** The proposed method produces a converged, accepted single-mode
  (α=1) result on the 200×25 clamped beam with omega_1=141.79 rad/s.
- **Unsupported:** Multi-mode targeting (α ≠ 1, α ≠ 0), monotonicity of the
  α sweep, any frequency-gain claim across α, and the two-mode design narrative.

### Reviewer-response impact

The α-sweep is not complete. The response letter must state that only α=1.00
produced a valid converged result; α=0.50 and α=0.25 encountered the same
localized-mode pathology observed in Exp3. The multi-mode claim and the
monotonicity claim cannot be made. The single accepted result can be presented
as a preliminary demonstration of the method on a single case.

---

## Gate E3 — Mesh Convergence (Exp3)

**Status: FAILED / INCONCLUSIVE**

### Supporting artifacts

| Mesh | Artifact | Classification | Iterations | MAC | ω₁ (rad/s) | Topology corr. |
|---|---|---|---|---|---|---|
| 200×25 | `exp3_authoritative_200x25_result.json` | **accepted** | 1052 | 0.993 | 141.79 | — |
| 400×50 | `exp3_authoritative_400x50_result.json` | **mode invalid** | 1750 | 0.786 | 64.39 | −0.088 |
| Forensic audit | `exp3_fundamental_forensic_audit.md` | inconclusive | — | — | — | — |

Topology differences (200×25 vs 400×50, after upsampling):
MAD=0.519, RMSD=0.665, correlation=−0.088, Jaccard(ρ≥0.5)=0.300.

### Accepted evidence

200×25 result is accepted (same as Exp2 α=1.00 result: identical problem,
identical convergence). The forensic audit (2026-06-29) confirmed: geometry
is identical, BCs refine correctly, material/SIMP/filter/optimizer settings
match, reference loads agree within discretization error. No setup inconsistency
was found to explain the discrepancy.

The S1 baseline diagnostic (`s1_exp3_400x50_mode_summary.json`) on the 400×50
final design found: 8 localized low-density modes, 2 ambiguous modes, 0 physical
global modes among the first 10 modes. Modes 2–10 classified "localized
low-density mode" by energy-fraction criteria. Mode 1 classified "ambiguous"
(displacement localized but strain not clearly in low-density regions; dominates
the support-connected component but with low MAC to solid reference = 0.786).

### Rejected evidence

400×50 result is mode-invalid. Cannot serve as mesh-convergence evidence. The
current Exp3 is classified **inconclusive**: the physical setup is correct but
the cause of divergence cannot be definitively attributed to asymptotic
mesh-non-convergence vs localized-mode pathology without further diagnosis.

The topology correlation of −0.088 (designs are anti-correlated) and frequency
drop from 141.79 to 64.39 rad/s indicate the two meshes reached completely
different local optima or mode families, not a consistent mesh refinement.

### Claim impact

- **Unsupported:** Mesh convergence of the proposed method. No claim about
  consistent behavior across mesh refinements can be made.
- **Supportable as diagnostic:** The 400×50 run reveals a localized-mode
  pathology consistent with the well-known SIMP low-density mode problem.
  This is a limitation to be reported, not a feature.

### Reviewer-response impact

Reviewer demand for mesh-convergence evidence (multiple reviewer items) is NOT
fulfilled. The response letter must state: (i) Exp3 400×50 failed the MAC gate,
(ii) S1 diagnosis confirmed localized low-density modes in the 400×50 final
design, (iii) no mesh-convergence claim can be made without resolving the
localized-mode pathology, and (iv) this is a known limitation of SIMP at fine
meshes without length-scale enforcement.

---

## Gate S1 — Low-Mode and Grayness Diagnosis

**Status: PARTIAL — diagnosis complete, mitigation insufficient**

### Supporting artifacts

| Study | Artifact | Key finding |
|---|---|---|
| Baseline diagnostic (Exp3 400×50, pmass=1) | `s1_exp3_400x50_mode_summary.json` | 8/10 modes localized, 0 physical global |
| pmass=6 mitigation (400×50) | `s1_mitigation_400x50_mode_summary.json` | 1/10 physical, 9/10 localized; run accepted |
| Eq.4b hypothesis test (400×50) | `eq4b_study.json`, `eq4b_validation_result.json` | Run capped (2000 iters, dc=0.107); Eq.4b formula validated |
| Mitigation result | `s1_mitigation_400x50_result.json` | accepted, omega_1=131.93 rad/s, MAC=0.974 |

### Accepted evidence

**Baseline (pmass=1, Exp3 400×50 final design):**
- Modes 2–10: all classified "localized low-density mode" by strain-energy fraction.
  Strain fraction in low-density region (ρ<0.05): 0.999, 1.000, 0.992, 0.964,
  0.998, 0.998, 1.000, 1.000, 0.999 for modes 2–10 respectively.
- Mode 1: "ambiguous" — kinetic energy on support-connected component (97.2%)
  but MAC to solid reference = 0.786 and strain highly localized (top-1% fraction=0.565).
- Component count: 126 solid components at ρ≥0.5 (highly fragmented topology).

**pmass=6 mitigation (accepted result):**
- Mode 1 recovered as "physical global mode": kinetic fraction on support-connected
  component = 93.7%, strain = 97.4%, MAC=0.974, omega_1=131.93 rad/s.
- Modes 2–10: still "localized low-density mode" (9 out of 9 remaining).
  Low-density strain fraction ≥ 0.547 for all 9 modes.
- pmass=6 rescues mode 1 but does NOT eliminate the localized-mode family.
- Run formally accepted: 1579 iterations, dc=5.16e-4, feasibility=0.

**Eq.4b hypothesis test:**
- Validated: Du & Olhoff 2007 Eq.4b C1 mass formula matches authoritative
  formula to machine precision (max_abs difference = 0) above threshold=0.1.
- Run result: REJECTED (capped, 2000 iterations, dc=0.107 >> 1e-3, MAC=0.924).
- omega_1=77.01 rad/s (still well below 200×25 reference of 141.79 rad/s).
- Eq.4b does NOT resolve the capping problem on the 400×50 mesh.

### Rejected evidence

**No mitigation tested produced a clean spectrum.** Even under pmass=6,
9 of the first 10 modes at the accepted 400×50 design remain localized low-density.
The pmass=6 result accepted by the runner is therefore not a clean demonstration
of the method without spurious modes — it is a demonstration that mode 1 can be
recovered while the surrounding spectrum remains polluted.

Eq.4b mitigation: run capped; result rejected.

### Claim impact

- **"No spurious low-density modes"**: this claim is CONTRADICTED by the
  documented evidence. Even the accepted pmass=6 run has 9 localized modes
  in the spectrum below and near the tracked mode.
- **"The method avoids spurious modes"**: same — MUST be removed.
- **Reformulable as limitation:** "SIMP with linear mass interpolation (pmass=1)
  produces localized low-density modes at fine mesh resolution (400×50); aggressive
  mass penalization (pmass=6) partially mitigates this for the tracked mode but
  does not eliminate spurious modes from the surrounding spectrum. This is a known
  SIMP limitation; length-scale enforcement (Heaviside projection) or void-mass
  penalization is required for a clean spectrum."

### Reviewer-response impact

The no-spurious-modes claim must be removed and replaced by a limitation
statement citing the S1 evidence. The pmass=6 result can be shown as an
incomplete mitigation: it rescues the tracked mode but not the spectrum.
Heaviside projection or void-mass penalization is required for a complete fix
and has not been tested.

---

## Gate NB — OlhoffApproachExact Numerical Behaviour Freeze

**Status: PASSED / FROZEN**

### Supporting artifacts

| Study | Artifact | Key finding |
|---|---|---|
| Freeze memo | `NUMERICAL_BEHAVIOR_FREEZE.md` | Production settings frozen |
| Phase 1 | `phase1_inner300_summary.md` | `inner_max_iter=300` mostly solved inner convergence but did not remove cycle |
| Phase 2 | `phase2_asymptote_persistence_summary.md` | persistent asymptotes had negligible convergence effect |
| Phase 3 | `phase3_outermove005_summary.md` | `outer_move=0.05` strongly reduced cycle but capped |
| Phase 4 | `phase4_outermove002_summary.md` | `outer_move=0.02` converged at iteration 24; S1 found 0/10 localized modes |

### Accepted evidence

The numerical-behaviour investigation was restricted to MATLAB and
`OlhoffApproachExact`. It did not modify `ourApproach`, manuscript files,
mass/stiffness interpolation, objective, sensitivities, update ordering,
convergence tolerances, `alpha`, `inner_max_iter` after Phase 1, or the MMA
algorithm.

Phase outcomes:
- **Phase 1:** `inner_max_iter: 30 -> 300` reduced inner cap hits to 1.5%
  but left the outer two-cycle intact (`final_design_change=0.0914339`).
- **Phase 2:** preserving MMA asymptote state had negligible effect
  (`final_design_change=0.0921098`, decision C).
- **Phase 3:** `outer_move: 0.20 -> 0.05` strongly reduced the oscillation
  (`final_design_change=0.0206388`, beta parity gap 18.50 -> 6.03) but still
  capped at 400 iterations.
- **Phase 4:** `outer_move: 0.05 -> 0.02` converged at iteration 24 with
  `final_design_change=1.0179e-4 < 1e-3`; S1 found 0/10 localized low-density
  modes.

### Frozen settings

Frozen for the final `OlhoffApproachExact` experimental campaign:
`inner_max_iter=300`, `outer_move=0.02`, `alpha=0.5`,
`persistent_mma_state=true`, `mass_mode='du2007_c1'`, `penal=3`,
`rmin_elem=2.5`, `inner_tol=1e-4`, `outer_tol=1e-3`,
`mult_tol=1e-3`, `acceptance_check=false`, `move_lim=Inf`.

This is classified as numerical stabilization of paper-ambiguous move-limit
choices, not an algorithmic change: FE assembly, eigenproblem, generalized
sensitivities, interpolation laws, objective, multiplicity handling, update
ordering, and MMA update rules remain unchanged.

### Claim impact

Supportable for `OlhoffApproachExact`: the verified implementation can be run
with a documented, frozen numerical-stabilization setting (`outer_move=0.02`)
that removes the previously observed outer two-cycle in the CC 80x10 diagnostic
pilot.

This gate does **not** change the evidence status of `ourApproach` Exp2/Exp3,
the 400x50 mesh-convergence failure, or the S1 findings for the original Exp3
fine-mesh result.

### Reviewer-response impact

The final response can state that the Olhoff-style reference implementation
required an explicit move-limit stabilization and that the selected value was
chosen by a controlled one-parameter numerical-behaviour investigation. It
should be described as a numerical implementation detail where the paper is
ambiguous, not as a new optimization algorithm.

---

## Gate P1 — Performance Evidence

**Status: NOT_STARTED**

### Supporting artifacts

None. No instrumented timing measurements exist under the authoritative formulation.

### Accepted evidence

None.

### Rejected evidence

Table 3 timing values in `paper/reviews/algorithms_comparison.tex` are
placeholder estimates copied from manuscript Table 1, not from clean
benchmark runs. The stated scaling exponents (1.30, 1.25, 1.35) do not
match the log-log fit of the current table data (actual fit ≈ 1.07).

### Claim impact

The 8.6% frequency gap, 7.1× speedup, 4.61× building gain, and O(N_e^1.3)
scaling claims have no evidence base under the revised formulation.

### Reviewer-response impact

All performance headline claims must be removed or marked "pending regeneration"
until P1 produces accepted timing data.

---

## Gate MS — Manuscript and Response Update

**Status: NOT_STARTED**

No manuscript or response letter updates have been made.

---

## Gate RP — Reproducibility Package

**Status: NOT_STARTED**

No reproducibility artifacts exist.

---

## Summary Tables

### Gate status

| Gate | Status | Tier | Blocker for |
|------|--------|------|-------------|
| A0 | PASSED (MATLAB-only) | 1 | all experiments |
| I1 | PASSED | 1 | all accepted evidence |
| V1 | PASSED | 1 | Exp1–Exp5 |
| CR2 | FAILED / DIAGNOSTIC | 1 | load-sensitivity claim |
| A4 | NOT_STARTED | 1/2 | refresh/accuracy claims |
| E2 | PARTIAL (1/5 accepted) | 1 | clamped-beam claims |
| E3 | FAILED / INCONCLUSIVE | 1 | mesh-convergence claims |
| S1 | PARTIAL / INSUFFICIENT | 1 | spurious-mode claim |
| NB | PASSED / FROZEN | 1 | OlhoffApproachExact production settings |
| P1 | NOT_STARTED | 1 | all performance claims |
| MS | NOT_STARTED | 1 | submission |
| RP | NOT_STARTED | 1 | submission |

### Artifact inventory

| Artifact | Exists | Gate | Status |
|----------|--------|------|--------|
| `v1a_fd_results.json` | YES | V1a | PASSED |
| `v1b_mac_results.json` | YES | V1b | PASSED |
| `v1c_cr2_validation.json` | YES | V1c | PASSED |
| `a0_matlab_result.json` | YES | A0-MATLAB | PASSED |
| `a0_parity_result.json` | NO | A0-G | NOT_RUN (optional) |
| `i1_smoke_result.json` | YES | I1 | PASSED |
| `output/smoke/smoke_fail_result.mat` | YES | I1 | PASSED |
| `v1_3_multimode_results.json` | YES | V1-3 | PASSED |
| `v1_4_sensitivity_results.json` | YES | V1-4 | PASSED |
| `v1_6_reordering_results.json` | YES | V1-6 | PASSED |
| `cr2_smoke_results.json` | YES | CR2 smoke | DIAGNOSTIC |
| `cr2_rerun_manifest.json` | YES | CR2 rerun | DIAGNOSTIC_ALGO_FAIL |
| Exp2 α=1.00 result | YES | E2 | ACCEPTED |
| Exp2 α=0.75/0.00 result | YES | E2 | TRIVIAL (not usable) |
| Exp2 α=0.50/0.25 result | YES | E2 | REJECTED (mode invalid) |
| Exp3 200×25 result | YES | E3 | ACCEPTED |
| Exp3 400×50 result | YES | E3 | REJECTED (mode invalid) |
| Exp3 forensic audit | YES | E3 | INCONCLUSIVE |
| S1 baseline mode summary | YES | S1 | 0 physical global, 8 localized |
| S1 pmass=6 mode summary | YES | S1 | 1 physical, 9 localized |
| Eq.4b validation result | YES | S1 | REJECTED (capped) |
| Numerical behaviour freeze memo | YES | NB | FROZEN |
| Phase 1 inner300 result | YES | NB | DIAGNOSTIC |
| Phase 2 asymptote persistence result | YES | NB | DIAGNOSTIC |
| Phase 3 outermove005 result | YES | NB | DIAGNOSTIC |
| Phase 4 outermove002 result | YES | NB | PASSED |
| Any A4 result | NO | A4 | NOT_STARTED |
| Any P1 timing result | NO | P1 | NOT_STARTED |
| Manuscript diff | NO | MS | NOT_STARTED |
| Response letter | NO | MS | NOT_STARTED |
| Reproducibility manifest | NO | RP | NOT_STARTED |

---

## Conclusions

### Claims currently supported

1. **Authoritative formulation verified (A0):** F=omega0^2·M(x)·Phi0 with solid
   reference eigenpair, mass normalization, deterministic phase, and no rho_nodal
   scaling — confirmed in MATLAB R2025b by 10-item gate.
2. **Load sensitivity term is non-negligible (V1-4):** 71.3% max relative
   difference between omitted and complete branches at the first-iteration design.
3. **Multi-mode objective is additive (V1-3):** obj(case1+case2) = obj1+obj2 to
   floating-point zero; factor^2 scaling confirmed to 1.8e-15.
4. **MAC-based mode tracking is robust (V1-6):** correctly tracks permuted modes,
   phase-invariant, cross-design MAC=1.000 at uniform density.
5. **Single-mode proposed method converges on 200×25 clamped beam (Exp2 α=1.00):**
   omega_1=141.79 rad/s, 1052 iterations, MAC=0.993, grayness=8.6%.
6. **Localized low-density modes confirmed in 400×50 final design (S1 baseline):**
   8 of 10 modes localized, 0 physical global. This documents the SIMP low-density
   mode problem quantitatively.
7. **pmass=6 partially mitigates the problem (S1 mitigation):** mode 1 recovered
   as physical global (MAC=0.974, omega_1=131.93 rad/s), but 9 of 10 remaining
   modes still localized.
8. **Variant B (complete sensitivity, OC) fails to converge (CR2):** period-2
   cycle documented; Variant A exhibits stable plateau; no accepted comparison
   pair exists.
9. **OlhoffApproachExact numerical settings are frozen (NB):** controlled
   Phases 1-4 identify `outer_move=0.02`, `inner_max_iter=300`, `alpha=0.5`,
   and persistent MMA asymptotes as the frozen production settings. Phase 4
   converged at iteration 24 and S1 found 0/10 localized low-density modes.

### Claims that must be removed before submission

1. **"No spurious low-density modes"** — contradicted by S1 baseline (8/10 localized)
   and S1 mitigation (9/10 localized under pmass=6).
2. **"Omitted load sensitivity is negligible"** — contradicted by V1-4 (71.3%
   branch difference) and forbidden by CR2 rerun protocol.
3. **Multi-mode performance (α sweep monotonicity, two-mode design quality)**
   — only α=1.00 produced a non-trivial accepted result; α=0.50/0.25 are
   mode-invalid; α=0.75/0.00 are trivial.
4. **Mesh convergence** — Exp3 400×50 is mode-invalid; convergence is not
   demonstrated at fine mesh.
5. **7.1× speedup, 8.6% frequency gap, 4.61× building gain** — no accepted
   evidence under authoritative formulation; P1 not run.
6. **O(N_e^1.3) scaling with stated precision** — Table 3 values are placeholders;
   actual log-log fit of current data gives ≈1.07; P1 not run.
7. **"Equivalent to Yuksel-Yilmaz"** (N=1 refresh claim) — A4 not run.
8. **"Converges" for the proposed method** applied broadly — only α=1.00 on
   200×25 is demonstrated; 400×50 fails.
9. **Comparator labels "Olhoff" and "Yuksel-Yilmaz" without qualification** —
   local implementations; must be relabeled as "Olhoff-inspired" etc.

### Claims reformulable as limitations

1. **Spurious modes:** "Under SIMP with linear mass interpolation (pmass=1),
   localized low-density modes appear in the spectrum at fine resolution (400×50).
   Aggressive mass penalization (pmass=6) recovers the tracked mode (MAC=0.974)
   but does not eliminate spurious modes from the adjacent spectrum. Length-scale
   enforcement (Heaviside projection) or void-mass penalization is required for
   a clean spectrum and is left to future work."
2. **Mesh convergence:** "The method converges on a 200×25 mesh but produces
   a mode-invalid result on the 400×50 refinement due to localized-mode
   pathology. Mesh convergence cannot be demonstrated without resolving the
   length-scale issue."
3. **Load sensitivity:** "The omitted load-sensitivity term differs from the
   complete term by up to 71.3% at the initial design (V1-4). Its practical
   effect on converged results could not be quantified: Variant B (complete,
   OC optimizer) failed to converge due to a period-2 instability, and no
   accepted matched comparison pair is available."
4. **Performance claims:** "Timing measurements were made under conditions
   that include diagnostic overhead; clean benchmark data is required before
   speedup and scaling exponents can be stated."

### Minimum required work before R1 submission

1. **CR2:** Produce an accepted matched pair (MMA or coarser mesh) or restrict
   all language to the diagnostic-failure narrative permitted by the rerun protocol.
2. **Exp2:** Either resolve the localized-mode pathology for α<1 (Heaviside
   projection + void-mass) or restrict all multi-mode claims to the single
   accepted α=1 result.
3. **Exp3:** Either resolve the localized-mode pathology for the 400×50 mesh
   or replace the mesh-convergence claim with an explicit limitation.
4. **S1:** Either demonstrate a mitigation strategy that produces a clean
   spectrum (0 localized modes) or limit the no-spurious-mode claim entirely.
5. **P1:** Run ≥10 benchmark timing measurements per mesh to produce defensible
   scaling exponents and speedup values.
6. **NB:** No additional numerical-behaviour tuning is required for
   `OlhoffApproachExact`; use the frozen settings in `NUMERICAL_BEHAVIOR_FREEZE.md`.
7. **MS + RP:** All manuscript and response letter work remains to be done.

*Status derived exclusively from documented artifacts. Unexecuted scripts,
planning documents, and audit intentions do not constitute evidence of completion.*
