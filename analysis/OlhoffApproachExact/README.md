# OlhoffApproachExact — Olhoff & Du (2014) reference implementation

Target paper: `references/Olhoff2014_Structural.pdf` — N. Olhoff, J. Du,
*Structural Topology Optimization with Respect to Eigenfrequencies of Vibration*,
CISM Courses and Lectures, Springer 2014, pp. 275–297,
DOI 10.1007/978-3-7091-1643-2_11.

**Status (2026-07-30):** active reimplementation against Olhoff & Du (2014),
following [PLAN_Olhoff2014_exact.md](PLAN_Olhoff2014_exact.md). This supersedes
the archived note `OlhoffApproachExact.txt`, which targeted Du & Olhoff (2007)
and recorded the earlier, abandoned campaign. That note's ARCHIVE STATUS block
still applies to the **experiment artifacts** under `experiments/` that predate
2026-07-30 (`basin_exit_forensics_*`, `disconnected_local_mode_audit_*`,
`globalization_stabilization_*`, `missing_regularization_diagnostics_*`,
`persistent_mma_after_globalization_*`, `stabilization_basin_retention_*`,
`faithful_reconstruction/`, `mesh_resolution_campaign/`): those are diagnostic
evidence of the unsuccessful earlier reproduction, their solver was never
committed and cannot be regenerated, and they must not be used for
reviewer-facing comparisons.

## What is implemented

The computational procedure of the paper's Fig. 1, exactly:

```
0. rho <- 0.5 uniform;  choose n
1. assemble K(rho), M(rho) from RAW rho;  solve (1b), orthonormalize per (1c)
   detect multiplicity N of omega_n   (and R of omega_{n-1} for the gap problem)
2. generalized gradients f_sk (Eq. 13); usual gradients (Eq. 4/5) when N = 1
3. inner loop: solve subproblem (19) -- or (20) for the gap -- for Delta_rho,
   with eigenpairs, f_sk, N, R held FIXED
4. rho := rho + Delta_rho;  stop when ||Delta_rho|| < eps
```

Both solvers the paper offers for step 3 are implemented and cross-validated:
`subproblem_lp.m` (exact) and `subproblem_mma.m` (MMA, the paper's other named
option). The exactness contract — what is **[E]**xplicit in Olhoff2014, what is
**[D]** imported from Du & Olhoff (2007) because the 2014 chapter defers it to a
paper not in `references/`, and what is **[R]**econstructed — is
[PLAN_Olhoff2014_exact.md §1](PLAN_Olhoff2014_exact.md).

### The one reconstructed parameter

For N = 1, subproblem (19) is a linear program — the paper says so itself
(§2.5 final paragraph). Its exact optimum over the full box of (19f) is a vertex
with essentially every `Delta_rho_e` at a bound, and taking that step collapses
`omega_1` (CC: 145.97 → 0.063, 99.88 % of variables at a bound; reproduced as a
*required* test, `verify/v_subproblem.m` V-I6). Sequential linear programming is
not defined without a move limit, and the paper's own LP reduction cites Krog &
Olhoff (1999). So `cfg.move` is a reconstruction of an omitted implementation
detail. It is the only such parameter, it is calibrated against the paper's own
Fig. 4 iteration history, and it is printed in every run's contract banner.

## Layout

```
Matlab/
  fe_q4_exact.m            Q4 plane-stress element, consistent mass
  assemble_KM_exact.m      K_e = rho^p Ke*, M_e = m(rho) Me*   (Eq. 5)
  mass_interp.m            olhoff2014_pow [E] | du2007_step/c0/c1 [D]
  build_filter.m           mesh-independent filter kernel
  apply_sensitivity_filter.m   Sigmund sensitivity filter [D4]
  build_supports_exact.m   Fig. 2 BCs; mid-height SS/CS pin, even-nely assert
  compute_elem_sensitivity.m   Eq. (4)/(5) simple sensitivity
  generalized_gradients.m  Eq. (13) f_sk, exactly symmetric, basis-covariant
  detect_multiplicity.m    N, upward, with Schmitt hysteresis
  detect_multiplicity_below.m  R, downward, for the gap problem
  subproblem_lp.m          EXACT (19)/(20) by cutting-plane LP over the LMI
  subproblem_mma.m         (19)/(20) by MMA, N smooth constraints
  subproblem_kkt.m         feasibility + true optimality gap vs the exact solver
  lumped_mass.m            design-independent concentrated mass (section 3.3)
  topopt_freq_exact.m      Fig. 1 main loop, telemetry, stop_reason
  olhoff2014_case.m        every published number, in one place
  run_olhoff_case.m        run + report + declared decision rule + plots
  run_ss_n1.m  run_cs_n1.m  run_cc_n1.m        section 3.1
  run_ss_n2.m  run_cs_n2.m  run_cc_n2.m        section 3.2
  run_cc_gap23.m                                section 3.3
  run_all_olhoff_2014.m    all of the above + comparison table
  verify/                  acceptance tests, see below
  legacy/                  the pre-2026-07-30 solver, kept for provenance only
experiments/
  step_calibration/        Phase 5 move-limit sweep against Fig. 4
  paper_examples/          Phase 7/8 outputs
  <older dirs>             archived, see the status note above
```

## Running

```matlab
cd analysis/OlhoffApproachExact/Matlab
addpath(pwd, 'verify', fullfile('..','..','..','tools','Matlab'))

% acceptance tests
v_forward_model        % Fig. 2 initial frequencies, support interpretation
v_sensitivities_fd     % all derivatives vs finite differences
v_multiplicity         % detection + hysteresis
v_basis_invariance     % cluster-basis independence (the acid test)
v_subproblem           % inner loop: LP vs MMA, and the required LP-vertex test

% one paper example
res = run_cc_n1;

% everything
T = run_all_olhoff_2014;
```

`run_*` prints the contract used, the computed value beside the published one,
and a PASS/PARTIAL/FAIL verdict against the rule declared in the plan before the
runs. Outputs (density, per-iteration history, plots, summary) land in
`experiments/paper_examples/<case>/`.

## Verified so far

| test | result |
|---|---|
| Fig. 2 initial frequencies at p = 3, 80×10 | SS 68.62 / CS 104.07 / CC 145.97 vs 68.7 / 104.1 / 146.1 (−0.11 / −0.03 / −0.09 %) |
| p = 1 gives exactly 2× (ω² ∝ 0.5^{p−1}) | 137.25 = 2 × 68.62 — published values are p = 3 penalized |
| SS support interpretation | mid-height pin 68.40 vs bottom-corner pin 95.50 at 160×20 — corner reading ruled out |
| odd-`nely` guard | raises `build_supports_exact:OddNely` |
| all derivatives vs central differences | err/scale ≤ 1e−7 (λ′), ≤ 2.4e−9 (off-diagonal f_sk); f_sk symmetric to the bit |
| cluster basis invariance, LP path | Δρ invariant to 2.5e−13 under a random orthogonal rotation of the cluster basis |
| cutting-plane optimum vs independent bisection, N = 2,3 | agree to 5.3e−11 |
| N = 1 reduction of the general-N path | β to 2.2e−16, Δρ exactly |
| required LP-vertex collapse (`move = Inf`) | ω₁ 145.97 → 0.0634, 99.88 % at a bound |

Note that the MMA path is basis-invariant only in the limit: the difference decays
1.4e−1 → 5.8e−5 as the inner budget grows 20 → 200 and floors near 1e−4, and its
`stop_reason` is honestly reported as `max_iter`. That is why `lp` is the default
solver.

## Out of scope

Sections 3.4–3.6 (3D plate examples, bi-material). Recorded as future work.
