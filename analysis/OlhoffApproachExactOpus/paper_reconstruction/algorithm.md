# Paper-derived algorithm (Du & Olhoff 2007)

Derived only from the paper + erratum. This is the algorithm exactly as the paper prescribes it (Fig. 1 + Sections 2–3.5), written as pseudocode. Where the paper leaves a choice open, the gap is marked `‹UNSPECIFIED›` rather than filled in — filling it is an implementation decision, deferred to Phase 4. See [du_olhoff_2007_spec.md](../specification/du_olhoff_2007_spec.md) for the full equation-by-equation derivation and classification.

---

## Inputs
- Admissible design domain, FE mesh (`N_E` elements), boundary conditions, optional concentrated masses.
- Target order `n` (1 = fundamental). Problem type: `MAX_FREQ` (eq. 25) or `MAX_GAP` (eq. 26).
- Volume fraction `α` (=0.5 in benchmarks).
- SIMP: `p` (continued 1→3), `q` (=1), mass-model choice {4, 4a, 4b} with `r≈6`, thresholds `0.1`.
- Material(s): single (`E*,M*`) or bimaterial (`E*¹,M*¹,E*²,M*²` with stiffness eq.5 / mass eq.6).
- Filter radius `r_min` for Sigmund (1997) sensitivity filter ‹UNSPECIFIED in paper›.
- Tolerances: outer `ε`, inner-loop, multiplicity ‹all UNSPECIFIED in paper›.
- Lower bound `ρ̲ = 1e-3`.

## Precompute (mesh-fixed)
- Element matrices `K*_e`, `M*_e` for solid material (and `E*²,M*²` for bimaterial).
- Element volumes `V_e`, domain volume `V_0`, `V* = α V_0`.

## Procedure

```
ρ ← uniform 0.5 over all elements          # Step 0
p ← p_start (≈1)                           # continuation start ‹schedule UNSPECIFIED›

repeat   # ===== MAIN (OUTER) LOOP =====
    # ---- Step 1: analysis ----
    assemble K(ρ) = Σ ρ_e^p K*_e
    assemble M(ρ) = Σ massInterp(ρ_e) M*_e          # eq. 2 / 4 / 4a / 4b
    solve  K φ_j = ω_j² M φ_j  for j=1..(≥ n+N),  M-orthonormalize (eq. 7c)
    detect N = multiplicity(ω_n)                     # rel. tol ‹UNSPECIFIED›
    if MAX_GAP: detect R = multiplicity(ω_{n-1})
    J ← n + N                                        # eq. choice of J

    # ---- Step 2: gradients ----
    for the N members of ω_n  (and R members of ω_{n-1}, and the simple caps J, n-R-1):
        build generalized gradients f_sk  (eq. 19), filtered by Sigmund (1997)
        # if N=1 these reduce to ∇λ_n (eq. 14, 21)

    # ---- Step 3: INNER LOOP — sub-optimization for Δρ ----
    Δρ ← 0
    repeat
        solve by MMA:
            MAX_FREQ:  maximize β   s.t. (25b)(25c)(25d)(25e)(25f)
            MAX_GAP:   maximize β2-β1 s.t. (26b)..(26i)
            # unknowns: bound var(s) + Δρ_e ; ω_j,φ_j,f_sk,N,R FROZEN
            # Δ(ω_j²) are dependent (eigenvalues of subeigenproblem 25d/26f/26g)
            # OPTIONAL: add (22) f_skᵀΔρ=0 (s≠k) ⇒ problem becomes an LP
        update MMA asymptotes/move limits ‹params UNSPECIFIED›
    until Δρ converged ‹tol UNSPECIFIED›

    # ---- Step 4: update + outer convergence ----
    ρ ← clip(ρ + Δρ, ρ̲, 1)
    advance p toward 3 ‹schedule UNSPECIFIED›
until ‖Δρ‖ < ε ‹UNSPECIFIED›

return ρ   # optimum topology (expected multiple/bimodal at optimum)
```

## The single hardest-to-operationalize step (paper-honest)

The paper specifies the inner sub-problem in **two** forms:

1. **Nonlinear (default, "MMA"):** constraints (25c)/(26c)/(26d) bound the *dependent* increments `Δ(ω_j²)`, which are the eigenvalues of the algebraic subeigenvalue problem (25d/26f/26g). The paper does **not** state how this nonlinear det-equality is fed to MMA. This is the largest reconstruction ambiguity.

2. **Linear (explicit, "LP"):** if the off-diagonal vanishing constraints (22) `f_skᵀΔρ = 0 (s≠k)` are added, then `Δλ_j = f_jjᵀ Δρ` (eq. 23) — each multiple eigenvalue behaves like a simple one and (25)/(26) become **linear programs** (Krog & Olhoff 1999). This is the only fully-concrete recipe in the paper.

Any faithful "exact" reconstruction must declare which of these two it implements. The LP route is the one the paper makes unambiguous.
