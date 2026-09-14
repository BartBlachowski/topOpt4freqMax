# AUDIT_PREREGISTRATION — c480_socp_causal_run

Frozen 2026-09-13, before any treatment code was written or executed and before
any treatment state existed. Repo HEAD `013cc48451d33bed61c5c4eea174bbd898d548a2`,
branch `benchmark-methodology-r2`. Its SHA-256 is recorded in
`evaluations/preregistration_sha256.txt` immediately after writing, and is
re-verified at finalization. No threshold, bar, rule, endpoint definition or
verdict mapping below may change after the treatment run begins. Any deviation
discovered later is disclosed in `PROVENANCE.md`, never silently corrected.

## 0. The experiment, in one sentence

ONE 480×60 topology-optimization run, identical in every scientific and
algorithmic respect to the authoritative retained three-rung C480 canary,
except that every Du–Olhoff sub-problem (25) is solved by an exact, independently
certified second-order-cone program instead of the production repeated-MMA
reconstruction (`innerLoop` → `mmasub`).

## 1. Control (not rerun)

| item | expected value (from repository evidence) |
|---|---|
| study | `diagnostics/three_rung_canary_preflight` |
| trajectory | `analysis/OlhoffCurrent/evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat` |
| trajectory SHA-256 | `a87546bc391cdc683def34a9f27678884528032f6b140e156349d2e74135ab9b` |
| initial ρ₀ SHA-256 (0.5·ones(28800,1), little-endian float64) | `8b5a00afc231e5bed8136c572955a0892a1b1b91bca392bccbc6de8dab063b07` |
| final ρ₃₈₆ SHA-256 | `0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60` |
| config hash (`olhoffcurrent_config_hash`) | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` |
| `+impl` tree hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| status / outer | CONVERGED / 386 |
| stage starts / descents | 1, 309, 348 / two, both `stageExhaustion` |
| S1 / S2 / S3 declarations | 308 (B) / 347 (B) / 386 (B, terminal) |
| levels / final move | [0.04 0.02 0.01] / 0.01 |
| ω₁ / ω₂ (final analysis) | 163.93225938567002 / 185.21032464590309 |
| M_nd / gray / mid | 26.34156302529312 % / 0.28729166666666667 / 0.11861111111111111 |
| broad-core fraction / area / max depth | 0.1301388888888889 / 1.041111111111111 / 0.36666666666666664 |
| inner MMA total | 7300 |

`C480_CONTROL_EVIDENCE_PASS` requires ALL of: trajectory hash match; ρ₀ and ρ₃₈₆
hash match (ρ₃₈₆ = RHO(:,386) and = record `rho_sha256`); stored `cfg` hashes to
the config hash above AND a fresh `cp_config(480,60)` resolution hashes to it;
stored `meta.implTree` equals the tree hash AND a fresh
`olhoffcurrent_source_manifest('Verify',true)` verifies to it; levels are exactly
[0.04 0.02 0.01]; `move.continuation.signal` and `stop.rule` are both
`stageExhaustion` (not `boundVariable`/`designChange`: not legacy four-rung, not
beta continuation, not old production stopping); hist has 386 columns; stage
lengths 308/39/39; controller events above reproduce from `hist`/`exh`; ω, M_nd,
gray, mid in the record reproduce from RHO(:,386) and the record to ≤1e-12
relative; broad-core metrics reproduce `gray_kkt_forensic_audit/evaluations/
geometry.json` ["480"] exactly with the copied geometry routine; Σ nInner = 7300;
the run CSV/record hashes equal those listed in the canary's `FINAL_SHA256.txt`.
Any failure ⇒ `C480_CONTROL_EVIDENCE_FAIL` ⇒ STOP (no treatment).

## 2. Treatment construction

### 2.1 Configuration
The treatment `cfg` is the control's stored `cfg` struct loaded from the control
trajectory, unmodified. Required: `isequal(cfg_treatment, cfg_control)` and
identical config hash, and equal to a fresh `cp_config(480,60)` hash. The single
scientific factor is carried OUTSIDE `cfg`, in a treatment descriptor
`treat.innerSolver`, because the canonical schema has no inner-SOCP field and
production code may not be modified:

    control   : innerSolver = repeatedMMA   (production innerLoop, published mmasub)
    treatment : innerSolver = exactSOCP     (this preregistration, §3–§4)

`optimizer.inner.{type,variant,tolerance,minIterations,maxIterations}` remain in
`cfg` unchanged and are inert under `exactSOCP` (declared, not deleted).
Allowed non-scientific differences: output paths, logging, telemetry, certificate
records, checkpointing, audit hooks.

### 2.2 Driver
Production `+impl/architecture/olhoffSolve.m` is NOT modified. The study uses
`scripts/cs_olhoffSolveSOCP.m`, a verbatim copy whose ONLY permitted departures
are: (a) function name and a second argument `treat`; (b) the inner-solver
dispatch: when `treat.innerSolver=='exactSOCP'` the call to `innerLoop` is
replaced by the certified SOCP adapter; when `'repeatedMMA'` the production
dispatch executes verbatim; (c) fail-closed termination on adapter rejection
(no update applied at that iteration); (d) NaN/Inf fail-closed check; (e)
telemetry capture, checkpoint save/resume, a preflight early-stop hook
(`treat.stopAfter`), and an iteration-1 identity assertion. Every numeric
expression of the outer loop (FE, eigensolve, multiplicity, gradients, offsets,
filter, move controller, update/clamp, exhaustion detector, stop logic, final
analysis) must be character-identical to production. `SINGLE_FACTOR_DIFF.md`
carries the unified source diff and `evaluations/single_factor_diff.json` the
machine-readable config comparison over all 81 schema rows.

### 2.3 Single-factor preflight (`C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS`)
Requires ALL:
- P1 config identity (§2.1).
- P2 source-diff audit: every diff hunk is in the permitted categories of §2.2.
- P3 control replay identity: the study driver in `repeatedMMA` mode with
  `stopAfter=3` reproduces the control's DRHO(:,1:3), RHO(:,1:3),
  hist.omega(:,1:3), hist.beta(1:3), hist.nInner(1:3) BITWISE. (Three
  outer iterations of the control formulation; not a scientific run, outputs
  used only for this equality.)
- P4 adapter known-answer test: the adapter applied to the hash-verified frozen
  C480 problem (`frozen_problem25_reference/evaluations/frozen_ctx.mat`,
  SHA-256 `084f219c…4667`) is eligible, certified (§4), and agrees with the
  certified oracle `conic_reference.mat` (`7503189a…4833`): |bs−bsO| ≤ 1e-8,
  gain recovery ≥ 0.999, d2 ≤ 0.01, dinf ≤ 0.1.
- P5 fail-closed tests: the adapter REJECTS without solving (i) a ctx with N=3,
  (ii) inconsistent offsets, (iii) asymmetric F, (iv) offDiag=false,
  (v) a volFun present; the certificate REJECTS a deliberately infeasible x and a
  deliberately suboptimal feasible x.
- P6 software smoke test on a TOY mesh 48×6 (NE=288), `exactSOCP`, `stopAfter=6`,
  plus a checkpoint/resume test (3 + resume 3 must equal 6 straight, bitwise).
  This is software verification on a different, trivially small problem; its
  outputs are excluded from every scientific analysis and no C480 treatment
  state is generated by it.
- P7 `+impl` source manifest verifies to `edbfe47e…52cb` immediately before launch.
Any failure ⇒ `C480_SOCP_SINGLE_FACTOR_PREFLIGHT_FAIL` ⇒ STOP. Software defects
found in P2–P6 may be fixed before launch (no scientific state exists yet); every
fix is logged in `PROVENANCE.md`; no threshold in this document may change.

## 3. SOCP eligibility contract (checked before EVERY solve)

Supported class: N = 2, consistent diagonal offsets, full off-diagonal coupling,
affine next-mode row, linear volume row, fixed coefficients. Checks:

| id | check | bar |
|---|---|---|
| E1 | N == 2, numel(lam)==2, J = n+N = 3 ≤ Jcalc | exact |
| E2 | F is NE×2×2, all finite; F(:,1,2) and F(:,2,1) bitwise equal | exact |
| E3 | offsets present; dOff == lam − lam(1) bitwise; dOff(1)==0; dOff(2) ≥ 0 | exact |
| E4 | \|(lam_j − dOff_j) − lam_1\| ≤ 4·eps·lam_1 (row-1 dominance) | 4 eps |
| E5 | 0 < lam(1) ≤ lam(2) ≤ lamJ, all finite; fJJ NE×1 finite | exact |
| E6 | offDiag == true (SOC route; the offDiag=false LP route is NOT preregistered) | exact |
| E7 | no volFun; projection disabled; filter.type=='sensitivity'; inner variable 'increment' (production `innerLoop` path, not `innerLoopLP`/`innerLoopRho`) | exact |
| E8 | box: lo = max(rhomin−rho,−move) ≤ 0 ≤ hi = min(1−rho,move); move > 0; bs box [0,5] | exact |
| E9 | affine rows: production row values (character mirror of `innerLoop`, `fp_problem.evalProd`) vs `Alin·x−blin` for next-mode and volume rows at x=[0;1] and at the returned x | ≤ 1e-12 |
| E10 | SOC equivalence: production row 1 (`deltaLambda`) vs cone residual at x=[0;1] and at the returned x | ≤ 1e-10 |
| E11 | redundancy: production row 2 ≤ row 1 + 1e-12 at the returned x | 1e-12 |
| E12 | gradient: production row-1 gradient vs closed-form cone gradient at returned x (skipped only in the apex case §4.3) | rel ∞-norm ≤ 1e-9 |

N = 1: not supported in this experiment (the configuration fixes N = 2 via
`multiplicity.method='subspace'`, `subspaceSize=2`; no LP path is preregistered).
N ≠ 2, or any E1–E8 failure ⇒ termination `SOCP_UNSUPPORTED_CASE_HIT` before
solving. E9–E12 failure at the returned point ⇒ `SOCP_EQUIVALENCE_FAILURE`.
`multJ` (next mode itself near-multiple) is RECORDED, not a rejection: the posed
row (25b) is affine regardless, production treats it identically ("log only"),
and multiplicity treatment may not be altered.
No fallback to MMA. No skipped iteration. No row dropped. No approximation.

## 4. Solve and certification (every outer iteration)

### 4.1 Formulation
Exactly `fp_problem` (byte-identical copy, SHA-256 `5ae495fe…a94f`):
variables x = [drho; bs], min −bs, one SOC ‖A_c x − b_c‖ ≤ d_cᵀx − γ_c,
Alin x ≤ blin (next mode, volume), xmin ≤ x ≤ xmax. Solved with
`coneprog` in x-coordinates, OptimalityTolerance 1e-10, ConstraintTolerance
1e-10, MaxIterations 500, Display off.

### 4.2 Attempt cascade (preregistered)
Attempt 1: LinearSolver `'augmented'`. Attempt 2 (only if attempt 1 is not
certified): LinearSolver `'schur'` (the configuration of the certified oracle).
Both solve the identical conic problem. If neither is certified ⇒ termination
`SOCP_CERTIFICATE_FAILURE`; no update applied. Exit flags are recorded and never
used for acceptance: an imperfect exit flag (e.g. −7) is accepted iff the
independent certificate below passes.

Returned x: raw box violation must be ≤ 1e-9, else the attempt fails; x is then
clipped to [xmin, xmax] and every check below is evaluated on the clipped x,
which is the accepted increment.

### 4.3 Certificate
Dual candidates, all dual-feasible by construction (μ ≥ ‖p‖, ν ≥ 0), each giving a
valid weak-duality lower bound
D(p,μ,ν) = Σᵢ min(qᵢ xminᵢ, qᵢ xmaxᵢ) − pᵀb_c + μγ_c − νᵀblin,
q = f + A_cᵀp − μ d_c + Alinᵀν:
(i) coneprog's returned duals with p = μ·w, w = s/‖s‖; (ii) complementary-slackness
recovery: nonnegative least squares for (μ,ν) on the interior-variable
stationarity rows (|x−bound| > 1e-6·width) plus the bs row, p = μ·w;
(iii) `fp_dualbound` general 5-variable maximization; (iv) `fp_dualbound`
aligned candidate (byte-identical copy, SHA-256 `f643384c…db74`).
x is CERTIFIED iff at least one candidate satisfies ALL of:

| id | quantity | bar |
|---|---|---|
| C1 | max production row value (rows 1–4, `innerLoop` arithmetic) | ≤ 1e-8 |
| C2 | box violation after clip / raw before clip | = 0 / ≤ 1e-9 |
| C3 | weak-duality gap −bs − D | −1e-8 ≤ gap ≤ 1e-8 |
| C4 | multipliers: μ ≥ ‖p‖, ν ≥ 0, implied box multipliers ≥ 0 | exact |
| C5 | row complementarity max\|μ_row·c_row\| with rows [μ;0;ν] | ≤ 1e-6 |
| C6 | box complementarity max(ξ(x−xmin), η(xmax−x))/(sRow0·move), ξ=max(q,0), η=max(−q,0) | ≤ 1e-4 |
| C7 | \|bs stationarity\| \|q_end\| | ≤ 1e-6 |
| C8 | stationarity RMS/sRow0 and max/sRow0 with implied box multipliers (`fp_kkt`, byte-identical copy `7a632985…19a3bf`) | ≤ 1e-6 / ≤ 1e-5 |

sRow0 = RMS(F11/lamref). In the smooth case q uses the production row gradient
(`fp_kkt`); in the APEX case (2‖s‖ ≤ 1e-9, predicted e₂−e₁ ≤ 1e-9·lamref)
q uses the conic subgradient from the candidate's p (generalized conic KKT),
E12 is skipped, and candidates (i)/(ii) are unavailable. `fp_kkt`'s own verdict
string is recorded but not used (its box-complementarity bar 1e-6 is stricter
than the 1e-8 gap and was calibrated on a single schur solve). The C6 bar is
fixed here at 1e-4 BEFORE any treatment solve.

### 4.4 Cross-solver diagnostic (never gates acceptance)
At outer iterations 1, 50, 100, 150, … the other linear solver also solves the
same problem; d2, dinf, |Δbs|, sign/bound agreement are recorded. Its time is
recorded separately and excluded from inner-solver cost.

## 5. Telemetry (every outer iteration)

N, dOff, lam, lamJ, lamref, bs, beta, predicted increments deltaLambda(F,drho,dOff)
and fJJᵀdrho, predicted gain beta−lam₁, max|drho|, fractions at lower density
bound / lower move bound / upper move bound / upper density bound / interior
(tolerance 1e-6·box width; move-limited vs density-limited classified as in
`fp_problem`), primal residual (C1), dual feasibility, row and box
complementarity, stationarity, primal objective, best dual bound, gap, every
candidate's gap, attempt count, exit flag / iterations / message per attempt,
eligibility record, apex flag, assembly/solve/certificate/eligibility times,
cross-solver diagnostic when run. Plus production `hist`, `res.diag`, `res.log`,
`res.exhaustion`, full RHO/DRHO trajectory, cv_export CSV (same 55-column schema
as the control), and the supplement CSV.

## 6. Primary and secondary endpoints

Definitions are the prior audits', by copied code: M_nd = 100·mean(4ρ(1−ρ));
gray 0.1<ρ<0.9; mid 0.4≤ρ≤0.6; broad core = gray ∧ EDT depth > 0.06 (physical,
h = 1/nely, domain 8×1, `gray_kkt_forensic_audit/scripts/geometry.py::fields`);
gray area, max depth, depth/R (R = 0.06), 4- and 8-connected gray components,
largest component area, quantiles, histograms.

**Primary** (treatment T vs control C at final design): M_nd, gray fraction, mid
fraction, broad-core fraction, broad-core area, max gray depth, and ω₁ (final analysis).
**Secondary**: λ₁, λ₂, ω₂, gap12, next-mode gap (ω₃−ω₂)/ω₂, volume, iterations,
stage lengths, controller events, physical and filtered stationarity (§9), topology
distances (§10), model realization (§7), move/reversal (§8), cost.

## 7. Model realization (Part 5/13 definitions)

For accepted step k: λ₁ₖ = ω₁(k)² from `hist.omega(1,k)` (pre-update FE at ρ_{k−1});
pred_k = beta_k − λ₁ₖ; act_k = λ₁,ₖ₊₁ − λ₁ₖ with λ₁,ₙ₊₁ from the final analysis;
eligible iff pred_k > 1e-7·λ₁ₖ; r_k = act_k/pred_k. Also: predicted
λ₁ from deltaLambda (dlam₁), absolute/relative model error, sign agreement,
cumulative predicted/realized, per stage. Two-cycle/reversal from §8.

`OUTER_MODEL_REALIZATION_*` (treatment only; control computed identically from
its retained hist.beta as comparator, not as verdict input):
- INCONCLUSIVE: < 20 eligible steps, or telemetry missing.
- PROBLEM_EXPOSED if ANY: median r < 0.25; fraction act<0 among eligible > 0.30;
  Σact/Σpred < 0.25; any stage with ≥ 10 eligible steps has median r < 0 or
  fraction act<0 > 0.50; any stage has fraction of steps with cos(dₖ,dₖ₋₁) < −0.5
  greater than 0.50.
- HEALTHY if ALL: 0.5 ≤ median r ≤ 2.0; fraction act<0 ≤ 0.10; Σact/Σpred ≥ 0.5;
  every stage with ≥ 10 eligible steps has median r ≥ 0.5 and fraction act<0 ≤ 0.25;
  every stage has fraction cos < −0.5 ≤ 0.25.
- MARGINAL otherwise.

## 8. Move / reversal telemetry definitions

Per k ≥ 2: cos(drhoₖ, drhoₖ₋₁); ‖drhoₖ+drhoₖ₋₁‖ and its ratio to
‖drhoₖ‖+‖drhoₖ₋₁‖; element sign-reversal fraction (drhoₖ·drhoₖ₋₁ < 0 among
|drho| > 1e-9 in both); per k ≥ 3 recurrence ‖ρₖ−ρₖ₋₂‖/‖ρₖ−ρₖ₋₁‖. Classes at
ρₖ₋₁: void ρ≤0.1, gray shell = gray ∖ mid, gray core = mid, solid ρ≥0.9. Gray
full-move fraction = gray elements at a ±move bound. Same quantities for the
control from its DRHO/RHO.

## 9. Stationarity at endpoints (no optimization)

At control ρ₃₈₆ and treatment final ρ: native FE/eigs/genGrad/filter kernels
exactly as `gray_kkt_forensic_audit/scripts/frozen_evaluate.m` (spectral part only,
no FD); raw gRaw, filtered gFiltered; `stationarity.py::kkt` copied verbatim
(all-free dual fit, gray-fit dual, bound tolerance 1e-7, scale = raw interior RMS of
the SAME design); residual by class. Control recomputation must reproduce the
retained `spectral_480.mat` (gRaw, gFiltered, lam) to ≤ 1e-12 relative, else the
stationarity comparison is reported INCONCLUSIVE. Material change: ratio T/C of
gray-class RMS (gray-fit dual) ≤ 0.5 (improved) or ≥ 2.0 (worsened); otherwise
similar. Reported for physical (raw) and filtered residuals separately.

## 10. Topology comparison

‖ρ_T−ρ_C‖₂, ‖·‖∞, RMS, Pearson correlation, threshold-0.5 solid overlap
(Jaccard, agreement fraction), gray-mask Jaccard, broad-core Jaccard, material
relocation fraction = Σ|ρ_T−ρ_C| / (2Σρ_C). Same physical coordinates.

## 11. Causal grayness verdict (mechanical, exactly one)

Ratios ρ_M = M_T/M_C, ρ_G = gray_T/gray_C, ρ_mid = mid_T/mid_C,
ρ_B = broad_T/broad_C, R_ω = ω₁_T/ω₁_C.
- G_major: ρ_M ≤ 0.50 ∧ ρ_B ≤ 0.25 ∧ ρ_G ≤ 0.50 ∧ ρ_mid ≤ 0.50.
- G_material: ρ_M ≤ 0.80 ∧ ρ_B ≤ 0.80.
- OBJ_OK: R_ω ≥ 0.99; OBJ_COLLAPSE: R_ω < 0.95.
- Terminal-window instability TWI (last 20 iterations): range M_nd > 1.0 pp, or
  range gray > 0.01, or range broad > 0.01, or range ω₁/ω₁_final > 0.5 %.

Precedence:
1. **E** `C480_SOCP_CAUSAL_RESULT_INCONCLUSIVE` if fail-closed termination
   (unsupported / equivalence / certificate / numerical), evidence identity failure,
   production modified, or required telemetry missing.
2. **D** `EXACT_INNER_SOLVE_EXPOSES_OUTER_GLOBALIZATION_FAILURE` if CAP_HIT
   (1600, no terminal declaration), or TWI, or OBJ_COLLAPSE, or
   ω₁_T < 0.95·max_k ω₁(k) along the treatment.
3. **A** `INNER_SOLVER_MAJOR_CAUSE_OF_C480_GRAYNESS` if G_major ∧ OBJ_OK.
4. **B** `INNER_SOLVER_PARTIAL_CAUSE_OF_C480_GRAYNESS` if G_material (and not A).
5. **C** `INNER_SOLVER_NOT_PRIMARY_CAUSE_OF_C480_GRAYNESS` otherwise.

## 12. Other gates (mechanical)

- `C480_FULL_RUN_SOCP_COVERAGE_PASS` iff every executed outer iteration was
  eligible (N=2) and produced a certified accepted solution, and the run ended by
  the frozen controller or the cap; else FAIL.
- `SOCP_REMAINS_VALID_INNER_SOLVER_CANDIDATE` iff coverage PASS ∧ realization ∈
  {HEALTHY, MARGINAL} ∧ causal ∈ {A,B,C}. `SOCP_INNER_SOLVER_CANDIDATE_WEAKENED`
  iff coverage PASS ∧ (realization PROBLEM_EXPOSED ∨ causal D).
  `SOCP_INNER_SOLVER_CANDIDATE_REJECTED` iff coverage FAIL. (INCONCLUSIVE
  realization with coverage PASS and causal A/B/C ⇒ WEAKENED.)
- `FILTER_FORMULATION_STUDY_NOW_JUSTIFIED` iff causal ∈ {B, C}; else STILL_DEFERRED.
- `PERFORMANCE_CAMPAIGN_RECONSIDERATION_JUSTIFIED` iff causal A ∧ realization
  HEALTHY ∧ coverage PASS; else STILL_BLOCKED. `PERFORMANCE_CAMPAIGN_CAN_RESUME`
  is never issued by this study.

## 13. Stop conditions and fail-closed behavior

Terminate the treatment immediately, apply no update at that iteration, save all
evidence, and record the termination if: E1–E8 fail (`SOCP_UNSUPPORTED_CASE_HIT`);
E9–E12 fail (`SOCP_EQUIVALENCE_FAILURE`); no certified attempt
(`SOCP_CERTIFICATE_FAILURE`); NaN/Inf in ω, λ, F, fJJ, drho or ρ
(`NUMERICAL_FAILURE`); iteration-1 FE differs from the control's hist.omega(:,1)
bitwise (`EVIDENCE_IDENTITY_FAILURE`). Production-tree change detected at launch or
at finalization ⇒ evidence flagged, causal verdict E. Natural ends: frozen terminal
declaration (CONVERGED) or cap 1600 (CAP_HIT). No wall-clock cap (it would be a
new stopping rule).

## 14. No-rerun, repair and resume policy

One launch. No rerun, retune, move change, controller change, filter change,
certificate-threshold change or formulation change after launch. A code defect
discovered after any C480 treatment iteration has executed may NOT be repaired and
continued: stop and report. Process interruption not caused by any check above
(host crash, killed process) permits resumption from the most recent automatic
checkpoint (every 25 iterations) with unmodified code; the checkpoint mechanism
must have passed P6; any resumption is disclosed. Analysis scripts (post-run,
read-only) may be fixed freely; they cannot change the trajectory.

## 15. Cost

Recorded, not a benchmark: wall, Σ tEig, Σ tGrad, Σ tInner (eligibility,
assembly, solve per attempt, certificate), cross-solver diagnostic time separately,
per-outer means, inner share; control from its retained hist/record. Host is shared;
timings are indicative.
