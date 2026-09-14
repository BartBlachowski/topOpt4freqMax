# CONIC_REFERENCE — Part 7

## Solver and availability

MATLAB R2025b (25.2), Optimization Toolbox 25.2: `coneprog` available. No
package was installed. Scripts: `fp_coneprog.m` (the preregistered single
call), `fp_coneprog_sweep.m` (the disclosed sweep), `fp_reference.m` (the
selected, certified solve), `fp_dualbound.m` (independent certificate).

## The preregistered call, and what happened

`coneprog` with `OptimalityTolerance = ConstraintTolerance = 1e−10`,
`LinearSolver = 'auto'`, x-form. Result: **exitflag −7** after 26 iterations
("Search direction is small and current iterate is not within the specified
constraint and/or optimality tolerance"), `bs = 1.001792128843`, production
rows feasible to 7.8e−14, but the solver's *dual* iterate had stalled (its
reported dual infeasibility rose to 3e−2 over the last three iterations) and
the weak-duality gap formed from its own duals was 7.1e−4. That is not a
certificate.

## Disclosed deviation: solver sweep and an exact reparametrization

Twenty runs: `LinearSolver ∈ {auto, prodchol, schur, augmented, normal}` ×
`{x-form, t-form}` × `{1e−10, 1e−8}`. The t-form is the exact affine change
`x = xmin + W t`, `t ∈ [0,1]^nvar`, `W = diag(xmax − xmin)`; its constraint
values were checked against the x-form at the 15 test points (max difference
8.6e−15, 8.0e−15, 0). Every candidate was scored by the **independent maximized
dual bound** (next section), never by its exit flag alone.

| form | solver | tol | exitflag | iters | bs | max production row | certified gap |
|---|---|---|---|---|---|---|---|
| x | auto / augmented | 1e−10, 1e−8 | −7 | 26 | 1.001792128843 | 7.8e−14 | 2.2e−11 |
| x | prodchol | 1e−10, 1e−8 | −7 | 28 | 1.001792128847 | 7.7e−14 | 1.8e−11 |
| **x** | **schur / normal** | **1e−10, 1e−8** | **−7** | **27** | **1.001792128848** | **4.0e−14** | **6.4e−12** |
| t | auto / augmented | 1e−10 | −7 | 18 | 1.001792128846 | 2.7e−14 | 2.0e−11 |
| t | prodchol | 1e−10 | −7 | 21 | 1.001792128849 | 2.5e−14 | 1.2e−11 |
| t | schur / normal | 1e−10 | −7 | 19 | 1.001792128848 | −2.5e−15 | 1.7e−11 |
| t | any | 1e−8 | **1** | 10 | 1.001760356551 | 8.7e−9 | **3.2e−5** |

Two facts decide the selection. (i) The only runs that terminate with
exitflag 1 are the t-form runs at the loosened 1e−8 tolerance, and they stop
**early**: `bs` is 3.2e−5 below the others and their certified gap is 3.2e−5.
(ii) Every 1e−10 run, whatever its exit flag, reaches the same primal point to
~5e−12 in `bs` and is certified globally optimal to ≤ 2.2e−11 by the
independent bound. The preregistered "strong termination = exitflag 1" is
therefore **not met by any accurate run**, and the preregistration's own
global-certificate rule (duality gap ≤ 1e−8, independent of solver status)
is what certifies the reference. The prior audit's script
`fp_coneprog_sweep.m` selected the exitflag-1 candidate mechanically; that
selection was overridden in `fp_reference.m` in favour of the smallest certified
gap, and this override is the deviation being disclosed.

**Selected reference:** x-form, `LinearSolver = 'schur'`, tolerances 1e−10,
27 iterations, 53.6 s (single thread, shared host).

## The independent certificate (`fp_dualbound.m`)

For `‖p‖ ≤ μ`, `ν ≥ 0`, Cauchy–Schwarz gives `‖A_c x − b_c‖ ≥ (p/μ)ᵀ(A_c x − b_c)`,
so for every feasible `x`

```
fᵀx ≥ qᵀx − pᵀb_c + μγ_c − νᵀb_lin,      q = f + A_cᵀp − μ d_c + A_linᵀν,
```

and minimizing the affine right side over the box is closed-form:
`D(p, μ, ν) = Σ_i min(q_i·xmin_i, q_i·xmax_i) − pᵀb_c + μγ_c − νᵀb_lin`.
`D` is concave in `(p, μ, ν)`; it was maximized by a derivative-free simplex
search over five scalars from three starts, then polished with the cone
direction fixed to the exact subgradient `s/‖s‖` at the solution.

| | value |
|---|---|
| primal `fᵀx*` | −1.0017921288482 |
| best dual bound | −1.0017921288547 |
| **duality gap** | **6.4e−12** (3.6e−9 of the gain `bs − 1`) |
| aligned-direction bound / gap | −1.0017921288574 / 9.2e−12 |
| certificate multipliers | `μ = 1.0000000000`, `ν = [4.1e−17, 0.7291136447]`, `w = [−1.0000, −2.78e−5]` |
| cone dual feasibility | `‖p‖ = 1.0000 ≤ μ`, cone complementarity 0 |
| gap decomposition | box complementarity 9.2e−12, cone 0, linear −7.7e−15 |

Because the problem is certified convex (`CONVEXITY_CERTIFICATION.md`), every
feasible point has objective ≥ −1.0017921288574. The returned point is
feasible (rows ≤ 4.0e−14, box violation 0) with objective −1.0017921288482.
**No feasible `x` can beat the reference by more than 9.2e−12 in `bs`.** That is
the global certificate; it does not use the solver's status, iterates or duals.

`μ + ν₁ = 1` holds to 1e−16, as stationarity in `bs` requires (`q_bs = −4e−17`).

## What `coneprog` itself reported

`output.dualitygap = 2.9e−15`, `primalfeasibility = 1.0e−3`,
`dualfeasibility = 17.4` in its internal scaled measures at the last iterate —
the solver's own dual stalled while its primal was already converged. The
returned `lambda.soc = 0.581` and box multipliers reproduce stationarity only
to normalized RMS 0.42 (`REFERENCE_KKT.md`), which is why the certificate
multipliers were computed independently.

## Retained

`evaluations/conic_reference.mat`: `xRef` (28 801), certificate multipliers
`muRef, xsiRef, etaRef, qRef`, active masks, and the raw `lambda` structure
returned by `coneprog`. `evaluations/reference_solution.json`: everything
above plus the solution description. `drho_ref` was never added to any density.
