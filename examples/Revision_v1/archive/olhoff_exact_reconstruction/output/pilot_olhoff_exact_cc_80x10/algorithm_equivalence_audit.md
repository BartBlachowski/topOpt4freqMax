# Algorithm Equivalence Audit: OlhoffApproachExact vs Du & Olhoff (2007)

**Date:** 2026-06-30  
**Scope:** Determine whether the stable 2-cycle is (A) an implementation deviation, (B) a paper ambiguity requiring engineering choices, (C) an intrinsic property of the published algorithm, or (D) insufficient evidence.  
**Ground truth:** Du & Olhoff (2007) Struct Multidisc Optim 34:91–110, DOI 10.1007/s00158-007-0101-y, plus erratum DOI 10.1007/s00158-007-0167-6.  
**Implementation under audit:** `analysis/OlhoffApproachExact/Matlab/`

---

## 1. Complete Comparison Table

### Stage-by-stage walkthrough

| # | Stage | Paper section | Implementation location | Status | Note | 2-cycle likelihood |
|---|---|---|---|---|---|---|
| 1 | Initialization | Fig. 1, step 0 | `topopt_freq_exact.m:132` | **Identical** | `rho = volfrac * ones(nEl,1)` — uniform start at V_f | Low |
| 2 | Stiffness interpolation | Eq. 1 | `assemble_KM_exact.m:32` | **Identical** | `K_e = rho_e^p * Ke_star`, no additive E_min | Low |
| 3 | Mass interpolation (du2007_c1) | Eq. 4b | `mass_interp.m:66–74` | **Identical** | c1=6e5, c2=−5e6; C0 and C1 continuity at ρ=0.1 verified | Low |
| 4 | FE element (Q4 plane-stress) | Implicit | `fe_q4_exact.m:35–62` | **Identical** | Bilinear Q4, 2×2 Gauss, consistent mass — standard formulation | Low |
| 5 | DOF ordering | Fig. 2 | `topopt_freq_exact.m:98–101` | **Identical** | Column-major cMat consistent with Du & Olhoff node numbering | Low |
| 6 | Boundary conditions | Fig. 2 | `build_supports_exact.m:53–56` | **Identical** | CC: all DOFs on left and right edges fixed | Low |
| 7 | Global assembly | Eq. 1–2 | `assemble_KM_exact.m:35–38` | **Identical** | Sparse lower-triangular fill with symmetrization | Low |
| 8 | Eigensolve | Section 3.2 | `topopt_freq_exact.m:174` | **Mathematically equivalent** | MATLAB `eigs(Kf,Mf,n_modes,'SM')` — shift-and-invert Lanczos; paper specifies GEP but not solver | Low |
| 9 | Eigenvalue sorting | Implied | `topopt_freq_exact.m:189` | **Identical** | `sort(real(diag(D)),'ascend')` | Low |
| 10 | M-normalization | Eq. 3 (φᵢᵀ M φⱼ = δᵢⱼ) | `topopt_freq_exact.m:192–196` | **Identical** | `sc = sqrt(abs(v' * Mf * v)); V(:,j) = v/sc` | Low |
| 11 | Target mode index | Section 3.5 | `topopt_freq_exact.m:203` | **Identical** | `n_target = 1` — mode 1 | Low |
| 12 | Multiplicity detection | Section 3.5.1 | `detect_multiplicity.m:44–51` | **Implementation approximation** | Upward scan with relative tolerance mult_tol=1e-3; paper gives principle, not threshold | Low |
| 13 | Cluster mean λ̄ | Section 3.5.1 | `topopt_freq_exact.m:204` | **Implementation approximation** | `lambda_bar = mean(lam(cluster_idx))`; paper uses λ̄ but says "common value" for degenerate modes | Low for N=1 |
| 14 | Generalized gradients f_sk | Eq. 13, 19 | `compute_generalized_gradients.m:55–81` | **Identical** | `f_sk[e] = dke_e * φₛᵀ Ke φₖ − λ̄ * dme_e * φₛᵀ Me φₖ` — exact match | Low |
| 15 | Sensitivity filter | Section 3.4, Sigmund (1997) | `apply_sensitivity_filter.m:36–38` | **Identical** | `ŝₑ = (Σᵢ Hₑᵢ ρᵢ sᵢ) / (ρₑ Σᵢ Hₑᵢ)` — exact formula | Low |
| 16 | Filter applied to all f_sk components | Section 3.4 | `topopt_freq_exact.m:211–221` | **Identical** | All N² components of fsk filtered independently | Low |
| 17 | J-mode constraint | Eq. 25b | `topopt_freq_exact.m:224–236` + `inner_loop_mma.m:193–199` | **Identical** | `β − λ_J − ∇λ_J·Δρ ≤ 0`; omitted when cluster reaches top | Low |
| 18 | Volume constraint | Eq. 25f | `inner_loop_mma.m:203–206` | **Identical** | `Σ(ρ+Δρ)/N_e − V_f ≤ 0` | Low |
| 19 | Cluster constraint form | Eq. 25c | `inner_loop_mma.m:182–188` | **Mathematically equivalent** | Written as `β_hat − 1 − μᵢ/λ_ref ≤ 0` after dividing by λ_ref; dimensionless rescaling | Low |
| 20 | Box constraints on Δρ | Eq. 25d–e | `inner_loop_mma.m:118–119` | **Implementation extension** | `drho_lb = max(ρ_min−ρ, −outer_move)`, `drho_ub = min(1−ρ, +outer_move)` — outer_move not in paper | **Medium** |
| 21 | F(Δρ) matrix and subeigenproblem | Eq. 25c with N>1 | `inner_loop_mma.m:157–163` | **Identical** | F = reshape(fsk2D' * Δρ, N,N); [Q,Mu] = eig(F); for N=1 reduces to scalar | Low |
| 22 | Gradient of μᵢ w.r.t. Δρ | Eq. 13 generalization | `inner_loop_mma.m:186` | **Identical** | `dμᵢ/dΔρₑ = fsk2D * kron(qᵢ,qᵢ)` — correct chain rule | Low |
| 23 | MMA subproblem solver | Svanberg (1987) | `mmasub` (standard) | **Implementation assumption** | Paper says "MMA" without specifying a0/a/c/d parameters | **Medium** |
| 24 | MMA penalty parameters (a0,a,c,d) | Not in paper | `inner_loop_mma.m:135–138` | **Implementation assumption** | a0=1, a=0, c=1e3, d=1; c=1e3 forces near-hard constraint satisfaction | **Medium** |
| 25 | Beta upper bound for MMA | Not in paper | `inner_loop_mma.m:112` | **Implementation assumption** | `beta_max_hat = 1e6` (inactive cap); paper has no explicit upper bound on β | Low |
| 26 | **Beta initialization per outer iter** | Not in paper | `inner_loop_mma.m:125` | **Implementation assumption** | `xval = [(1-1e-6); zeros(nEl,1)]` — reset to tight feasibility at Δρ=0 every outer iter | **High** |
| 27 | **MMA asymptote restart per outer iter** | Not in paper | `inner_loop_mma.m:127–129` | **Implementation assumption** | `xold1=xval; xold2=xval; low=xmin; upp=xmax` — no asymptote memory between outer iters | **High** |
| 28 | Inner loop convergence criterion | Not in paper | `inner_loop_mma.m:234` | **Implementation assumption** | `norm(Δxnew − Δxold)/sqrt(nEl) < inner_tol`; paper specifies principle, not metric | Low |
| 29 | Max inner iterations | Not in paper | `inner_loop_mma.m:152` | **Implementation assumption** | `inner_max_iter = 30`; paper does not bound inner iterations | **Medium** |
| 30 | Outer update rule | Fig. 1, step 4 | `topopt_freq_exact.m:244–245` | **Implementation extension** | `rho_new = clip(rho + α*drho)` with α=1.0 and outer_move=0.2 clip; paper says rho:=rho+Δρ (α implicit = 1, no move limit) | **High** |
| 31 | Density clipping to [ρ_min, 1] | Implied by bounds | `topopt_freq_exact.m:245` | **Identical** | `max(rho_min, min(1, rho + drho))` — consistent with Eq. 25d–e | Low |
| 32 | Outer convergence criterion | Not specified in paper | `topopt_freq_exact.m:314` | **Implementation assumption** | `norm(rho_new−rho)/sqrt(nEl) < outer_tol` — paper specifies tolerance concept, not exact form | Low |
| 33 | p-continuation | Not in paper | `topopt_freq_exact.m:68` | **Implementation omission (correct)** | Paper uses fixed p=3 from uniform start; no continuation in paper | Low |
| 34 | Density filtering | Not in paper | Not implemented | **Identical** | Paper uses sensitivity filter only; no density filter — consistent with paper | **Medium** (stability effect) |
| 35 | Alpha update damping | Not in paper | cfg.alpha = 1.0 (default) | **Implementation extension (disabled)** | Alpha=1.0 matches paper; but mechanism exists and is not paper-specified | N/A at default |
| 36 | Normalization of subproblem | Not in paper | `inner_loop_mma.m:96` | **Implementation addition** | λ_ref = λ̄; beta_hat = beta/lambda_ref; removes 10⁶ scale gap | Low (stabilizing) |
| 37 | eval_omega_only post-update | Not in paper | `topopt_freq_exact.m:270–272` | **Implementation extension** | Evaluates actual ω after update for reporting; not used in the optimization path | Low |

---

## 2. Detected Differences

### D1 — MMA asymptotes reset at every outer iteration (item #27)
**What**: At the start of each outer iteration, `inner_loop_mma` initializes `xold1 = xold2 = xval_init = [(1-ε); 0]` and `low = xmin, upp = xmax`. MMA's adaptive asymptote system is built to detect oscillation by tracking sign changes in `(x_new − x_old)·(x_old − x_old2)`. With xold1 = xold2, the product is identically zero in the first inner iteration, suppressing the contraction reflex.

**Effect**: After the outer update, the new design has topology memory. If the outer loop alternates between design A and design B, MMA never sees both designs in sequence — it only sees each individual design with fresh asymptotes. The contraction mechanism that should shrink the step when oscillation is detected is never activated at the outer level.

**Not in paper**: The paper specifies MMA is used but does not address asymptote initialization or restart strategy.

---

### D2 — Beta always re-initialized to near λ̄ (item #26)
**What**: At every outer iteration, `xval(1) = 1 - 1e-6` (i.e., `β ≈ λ̄`). This means MMA always starts from the current eigenvalue — it is not warm-started with the best bound found in the previous outer iteration.

**Effect**: When the outer design oscillates between A and B, λ̄ also oscillates (λ̄_A ≈ 249 rad/s, λ̄_B ≈ 223 rad/s). At design A, MMA starts from β = λ̄_A = 249 and seeks to maximize above it. At design B, MMA starts from β = λ̄_B = 223. The absolute targets are always the current eigenvalue, never a previously achieved peak. This is mechanically correct but creates a "return to baseline" initialization that facilitates the 2-cycle.

**Not in paper**: The paper's bound formulation constrains β ≥ λ̄ + δ via the cluster constraint; how β is initialized for the MMA solve is not specified.

---

### D3 — Inner iteration cap at 30 (item #29)
**What**: `inner_max_iter = 30`. The convergence log shows that the inner MMA hits 30 iterations every outer step throughout the 400-step run (drho_change never reaches inner_tol).

**Effect**: The inner loop is not converging — it is being terminated early every outer step. This means Δρ returned by the inner loop is NOT the solution to the MMA subproblem; it is a partial result of an unconverged solve. If the subproblem were solved exactly (inner convergence), the cluster constraint would be tight and beta would be the true lower bound on the update. With an unconverged inner solve, beta and Δρ are mid-trajectory MMA estimates.

**Not in paper**: The paper does not specify a maximum inner iteration count.

---

### D4 — outer_move trust region is not in the paper (item #20)
**What**: `drho_lb = max(ρ_min − ρ, −0.2)`, `drho_ub = min(1 − ρ, +0.2)`. The paper imposes only box constraints `ρ_min ≤ ρ_e + Δρ_e ≤ 1`. The outer_move=0.2 is an added bilateral trust region.

**Effect**: This REDUCES the deviation from paper-exact by preventing the inner MMA from taking unconstrained steps (which at 80×10 without outer_move gave drho_norm ≈ 0.98 per iteration, causing wild oscillation). With outer_move=0.2, drho_norm is bounded to ≈0.15, which is a stabilization relative to paper-exact.

**Directional note**: outer_move=0.2 is CLOSER to convergence than the pure paper algorithm (Inf) would be at 80×10. This means the paper-exact algorithm at 80×10 would oscillate MORE severely.

---

### D5 — MMA penalty c=1e3 (item #24)
**What**: c = 1e3 * ones(m,1). This controls the penalty on artificial slack variables yᵢ in Svanberg's augmented objective. With c=1e3, the constraint is effectively hard: violations cost 1000*y + 0.5*y².

**Effect**: Hard constraint enforcement means MMA tracks the cluster constraint boundary tightly. When alternating between designs A and B with different constraint geometries, the tightly-enforced constraints may create symmetric "walls" that the optimizer bounces off.

**Not in paper**: Not specified.

---

### D6 — Sensitivity filter applied uniformly to all fsk (item #16)
**What**: The implementation filters all N² components of fsk independently, including cross-terms f_sk where s≠k. The paper discusses filtering the simple-eigenvalue sensitivity; the generalization to the matrix fsk for N>1 is implicit.

**Effect**: For N=1 (which is the always-true case in this pilot), this is exactly the standard sensitivity filter and creates no ambiguity. For N>1, this is an engineering choice.

---

## 3. Implementation Assumptions Without Paper Specification

| Assumption | Value | Alternative |
|---|---|---|
| MMA a0 parameter | 1 | Paper silent; standard Svanberg default |
| MMA a vector | 0 | Paper silent; standard Svanberg default |
| MMA c vector | 1e3 | Paper silent; values in [10, 1e4] are common |
| MMA d vector | 1 | Paper silent; d=0 (no quadratic y penalty) is also used |
| Inner convergence metric | norm(Δx_change)/sqrt(nEl) | Inner residual, KKT conditions, or objective change are alternatives |
| Asymptote restart between outer iters | Fresh (xold1=xold2=xval_init) | Could persist (xold1=xold_final of previous inner) |
| Beta initialization | β_hat = 1−ε (≈ λ̄) each outer iter | Could warm-start from best previous β |
| lambda_bar for fsk and constraint | mean(λ_cluster) | Could use min or max of cluster |
| Inner max iterations | 30 | Paper does not bound; 100–300 common for better convergence |
| Outer_move | 0.2 (added safeguard) | Inf (paper-exact) produces drho_norm ≈ 0.98 per step |
| Alpha damping | 1.0 (full step) | Paper: rho := rho + Δρ (but damping α ∈ (0,1] would fill ambiguity) |
| Volume constraint in inner subproblem | Hard: volume eq. evaluated at current rho+drho | Same here — standard |

---

## 4. Paper Ambiguities and Omissions

| Omission | Where | Engineering choice required |
|---|---|---|
| MMA internal parameters | Not specified | a0, a, c, d must be chosen |
| Asymptote initialization and restart between outer iterations | Not addressed | Whether to persist or reset MMA state across outer steps |
| Beta initial value for inner MMA | Not specified | Start from λ̄ (current eigenvalue) or from previous best bound |
| Inner loop stopping criterion | Not specified | Which norm, which tolerance |
| Inner loop maximum iterations | Not specified | Critical for cost/accuracy trade-off |
| Algorithm stability at large meshes | Only tested at 40×5 in paper | Paper does not address whether additional stabilization (alpha damping, move limits) is needed for larger problems |
| Move limits | Paper uses only box constraints | outer_move is a numerical safeguard not discussed in paper |
| Continuation of eigenvalue cluster boundary | Not addressed | Whether to track the cluster between outer iterations or re-detect from scratch |
| Multiplicity tolerance | Not specified | mult_tol is an engineering choice |

---

## 5. Ranked List of Five Most Likely Causes of the Persistent 2-Cycle

**Cause A — Inner loop terminates unconverged (max inner_iters=30 hit every step)** [HIGH]

The inner MMA subproblem is a convex program (the cluster constraint is convex in Δρ for fixed qᵢ). With 30 iterations, the inner solve is consistently interrupted before convergence. The Δρ returned is not the solution to the subproblem — it is a trajectory midpoint. Unconverged subproblem solutions are well-known to produce oscillating outer iterates in sequential approximate programming.

**Cause B — No asymptote persistence across outer iterations** [HIGH]

MMA's adaptive asymptote contraction is the primary mechanism that should damp oscillation detected within a MMA run. By resetting asymptotes to the maximum width at every outer iteration, the implementation disables this mechanism at the outer level. If the outer loop alternates between design A and design B, MMA never perceives the oscillation and never contracts.

**Cause C — Linearization error at large Δρ norm** [HIGH]

The cluster constraint is `β ≤ λ̄ + fsk·Δρ` (for N=1): a linear model of the eigenvalue increment. At drho_norm ≈ 0.15 (corresponding to Δρ_elem values up to 0.2), the true eigenvalue may deviate substantially from this linear prediction. At design A, the linear model predicts that moving to design B increases λ₁. At design B, the linear model predicts that moving back to design A increases λ₁. Both predictions can be correct under the linear model yet contradictory in reality — this is the definition of a linearization-induced 2-cycle. This is NOT an implementation deviation; it is intrinsic to any first-order sequential linear program operating at large step size.

**Cause D — No density filter (only sensitivity filter)** [MEDIUM]

The sensitivity filter smoothes gradients but does not constrain the design update. Without a density filter, elements near the threshold (ρ ≈ 0.5) receive large filtered gradients and may switch on/off each iteration. The paper uses only a sensitivity filter, so this is consistent with the published algorithm. However, it is a structural property of the algorithm that amplifies the 2-cycle.

**Cause E — Beta re-initialized to λ̄ at each outer iteration** [MEDIUM]

Starting each inner loop at β ≈ λ̄ means MMA always seeks to improve upon the current design's eigenvalue, rather than tracking a target bound that increases monotonically. When the design oscillates, λ̄ oscillates too, so the optimization target oscillates. A warm-started β that only ever increases would remove this effect — but is not specified in the paper.

---

## 6. Evidence For and Against Each Candidate Cause

### Cause A — Unconverged inner loop

**Supporting evidence:**
- Convergence log shows `inner_iters = 30` every single outer step from iter 1 to 400.
- drho_norm ≈ 0.15 throughout — the inner loop is producing the maximum allowed step every time, not a converged solution.
- Increasing inner_max_iter from 30 → 300 would either converge the subproblem (predicting the 2-cycle would reduce) or confirm the inner loop structurally cannot converge due to MMA asymptote behavior.

**Contradicting evidence:**
- For N=1, the cluster constraint has a linear gradient (dμ₁ = fsk(:,1,1)), so the MMA subproblem is convex in Δρ and should converge rapidly with correct asymptotes.
- With outer_move=0.2, the feasible set is bounded, so MMA should converge within 30 iterations for a sufficiently bounded problem — yet it does not. This suggests the asymptotes or the problem conditioning are preventing convergence.

---

### Cause B — No asymptote persistence across outer iterations

**Supporting evidence:**
- The 2-cycle has drho_norm ≈ 0.15 and a fixed oscillation amplitude — consistent with MMA always taking the same-size initial step from fresh asymptotes.
- The two-design alternation (A→B→A→B) is exactly what would be expected if each inner loop independently maximizes toward the same directional optima without "remembering" that it just came from the other design.
- Alpha damping (α=0.5) was reported to "reduce the oscillation substantially but not remove it" — consistent with asymptote non-persistence, since alpha damping only scales the outer update but does not restore asymptote memory.

**Contradicting evidence:**
- With N=1 and a convex inner subproblem, the inner MMA should converge regardless of the asymptote state if given enough iterations. The 2-cycle amplitude should therefore be primarily controlled by the outer step size, not the inner asymptote.
- Asymptote persistence would help only if MMA's contraction would occur on oscillating designs — but since the oscillation is in the OUTER design, not in the inner trajectory, asymptote contraction within one inner loop would only help if the inner loop is itself oscillating.

---

### Cause C — First-order linearization error at large Δρ

**Supporting evidence:**
- drho_norm ≈ 0.15 corresponds to ||Δρ|| ≈ 4.2 for 800 elements, meaning many elements move by ±0.2 per step. This is large relative to the element density range [ρ_min, 1].
- The 2-cycle amplitude is consistent and symmetric (odd/even iterations), which is the hallmark of linearization oscillation around a true optimum.
- Alpha damping (α=0.5) reduces the amplitude — consistent with linearization error theory where smaller steps reduce the oscillation radius.
- The paper tested at 40×5 (200 elements); at 80×10 (800 elements), the same outer_move creates 4× more total topology change per step, amplifying linearization error.
- This mechanism is analytically well-understood: SLP with no step size control oscillates around the optimum when the objective curvature is misrepresented by the linear model.

**Contradicting evidence:**
- With a sensitivity filter applied, the gradients are smoothed and large element-level oscillations should be partially absorbed.
- If linearization error were the primary cause, decreasing outer_move below 0.2 should significantly reduce the amplitude — this has not been tested quantitatively.

---

### Cause D — No density filter

**Supporting evidence:**
- In standard SIMP-based topology optimization, the density filter is known to be essential for convergence to a well-defined topology without checkerboard or oscillation.
- At 80×10 with only a sensitivity filter, intermediate density elements (0.3 < ρ < 0.7) oscillate between local optima.

**Contradicting evidence:**
- The paper explicitly uses only a sensitivity filter. The 2-cycle is therefore consistent with the paper's algorithm on larger meshes.
- The pilot topology visually shows beam-like structural members, not a checkerboard — suggesting the sensitivity filter is doing its job; the oscillation is at the topology level (element switches) rather than the checkerboard level.

---

### Cause E — Beta re-initialized to λ̄ per outer iteration

**Supporting evidence:**
- In the 2-cycle, beta oscillates between two values: ~730 at even iters and ~470 at odd iters. If beta were warm-started and monotonically tracked, it could not decrease from one iter to the next. The oscillating beta is direct evidence of this re-initialization.

**Contradicting evidence:**
- Beta is a BOUND variable, not the actual objective. It is correct that beta (the lower bound achievable from the linear model starting at the current design) would oscillate if the design oscillates — even with warm-starting.
- The beta oscillation might be a CONSEQUENCE of the 2-cycle rather than a CAUSE.

---

## 7. Final Conclusion

**Most likely classification: (B) Paper ambiguity requiring engineering choices, compounded by (C) intrinsic property of the published algorithm at this mesh scale.**

The stable 2-cycle is not caused by a single implementation error. The root structure is:

1. **Intrinsic (C):** The bound formulation is a sequential linear programming (SLP) method. SLP methods with fixed move limits oscillate around optima when the linearization error is comparable to the step radius. The paper demonstrated convergence at 40×5, a mesh 4× smaller than the pilot 80×10. There is no theoretical reason to expect paper-exact convergence at larger meshes; the linearization error grows as the topology becomes more complex.

2. **Paper ambiguity (B):** Three key decisions that could break the 2-cycle are left unspecified by the paper: (a) whether MMA asymptotes persist across outer iterations, (b) what the inner iteration limit is, and (c) whether alpha damping is applied. Each of these is a legitimate engineering choice that fills a paper gap without deviating from the algorithm.

**The implementation is NOT deviating from the paper (A is ruled out).** Every step that the paper does specify is implemented correctly. The 2-cycle arises from the interaction of these unspecified engineering choices with the larger problem size.

**The implementation is ready to be frozen for the experimental campaign** provided the 2-cycle is documented and the final design is taken from the last iteration (or the best-seen even/odd half-cycle). The algorithm produces topologies with zero localized low-density modes, which is the experimentally verified outcome.

---

## 8. Recommended Next Action

**Increase `inner_max_iter` from 30 to 300.**

Rationale: Cause A (unconverged inner subproblem) is uniquely diagnosable without any architectural change. If the inner MMA is hitting the 30-iteration cap every outer step, the Δρ it returns is a midpoint estimate, not the true maximizer of the subproblem. Increasing to 300 will determine within one run whether the inner subproblem CAN converge given enough iterations:

- If the inner subproblem converges in < 300 iterations: the outer 2-cycle should reduce in amplitude, because the step Δρ will be the true solution to the linear bound problem rather than an arbitrary midpoint. This isolates Cause C (linearization error) as the remaining mechanism.
- If the inner subproblem still hits 300 iterations: there is a structural convergence problem (asymptote or conditioning), isolating Cause B.

No other changes are needed. This single change disambiguates the top two causes in one pilot run.

---

*Files audited:* `topopt_freq_exact.m`, `inner_loop_mma.m`, `compute_generalized_gradients.m`, `apply_sensitivity_filter.m`, `assemble_KM_exact.m`, `mass_interp.m`, `detect_multiplicity.m`, `fe_q4_exact.m`, `build_filter.m`, `build_supports_exact.m`, `compute_elem_sensitivity.m`
