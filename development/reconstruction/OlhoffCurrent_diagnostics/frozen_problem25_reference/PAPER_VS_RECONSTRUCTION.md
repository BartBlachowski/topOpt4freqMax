# PAPER_VS_RECONSTRUCTION — Part 14

Source: `references/Du2007_Topological.pdf` (Struct Multidisc Optim 34:91–110),
pp. 97–99, and the publisher's erratum (34:545). Extracted text in
`scripts/` is not retained; the page text was read for this task.

## What problem (25) specifies (p. 98)

```
(25a)  max_{β, Δρ₁ … Δρ_NE}  β
(25b)  β − [ω_J² + f_JJᵀ Δρ] ≤ 0,                  J = n + N
(25c)  β − [ω_j² + Δ(ω_j²)] ≤ 0,                    j = n, …, n+N−1
(25d)  det | f_skᵀ Δρ − δ_sk Δ(ω²) | = 0,           s, k = n, …, n+N−1   (erratum form, with Δ)
(25e)  Σ_e (ρ_e + Δρ_e) V_e − V* ≤ 0,               V* = αV₀
(25f)  0 < ρ_min ≤ ρ_e + Δρ_e ≤ 1
```

Text (§3.5.2): the only unknowns are β and the increments Δρ_e, "which play
the role as independent variables, and the dependent variables are the
increments Δ(ω_j²)". All other iterates — ρ_e, ω_j, the f_sk, N — "have been
determined in steps 1 and 2 of the main iteration loop and are kept fixed".
J is chosen as n + N, the closest eigenfrequency above the cluster, assumed
simple, so its constraint is the linearized (25b).

§3.5.3: (25d) "represent[s] an algebraic subeigenvalue problem with Δ(ω²) as
an eigenvalue"; the coupling "is nonlinear in general". If the off-diagonal
products are forced to zero the subproblem reduces to a linear program (Krog &
Olhoff 1999).

## What Fig. 1 specifies (p. 97)

0. initialization; 1. FE eigen-solution and detection of multiplicity N;
2. generalized gradients f_sk if N > 1; 3. "Iterative solution of
sub-optimization problem (25) (or (26)) for increments Δρ_e" with an inner loop
"Increments Δρ_e converged? No → (repeat)"; 4. `ρ_e := ρ_e + Δρ_e`; outer test
"ρ_e converged? i.e. Δρ < ε ?".

## What it says about MMA

§3.5.3, verbatim: "The suboptimization problems (25a–f) and (26a–i) … can be
solved by using a mathematical programming method. In this paper, the MMA
method (Svanberg 1987) has been used." §1 lists MMA among the methods that
solve bound formulations efficiently. That is the whole of it.

## What the paper does NOT specify

| item | paper | production realization (reconstruction) |
|---|---|---|
| a move limit on Δρ | **none** — (25f) bounds only the density | hard box ±0.01 on drho (stage 3 of a ladder [0.04, 0.02, 0.01]) |
| inner convergence criterion | "Increments Δρ_e converged?" only | relative step `max|Δx_step|/max|drho| < 0.05`, ≥ 5 and ≤ 500 sub-iterates |
| MMA asymptote handling, initial asymptotes, `asyinit/asyincr/asydecr`, `move = 0.5` of `mmasub` | not stated | Svanberg's published September-2007 constants |
| reset or persistence of MMA state between outer iterations | not stated | reset: `x = [0; 1]`, `low/upp = box` at every outer iteration |
| MMA auxiliary constants `a0, a, c, d` | not stated | `1, 0, 1000, 0` |
| coordinates and scaling | Δρ and β | `x = [drho; β/λ₁]`, spectral rows divided by λ₁, volume by Vtot |
| bounds on β | none printed | `0 ≤ bs ≤ 5` |
| iteration cap | none | `maxInner = 500` |
| multiplicity tolerance value | "predefined, very small tolerance", no value | fixed subspace N = 2, no classifier |
| (25d) when the cluster is not exactly multiple | (25d) assumes an N-fold eigenvalue (p. 96, eq. 17) | diagonal offsets `dOff = λ_j − λ₁` retained (class C) |
| gradient of Δ(ω_j²) with respect to Δρ | not stated | `Σ v_s v_k f_sk` |
| filter radius; filtering of the off-diagonal f_sk | Sigmund (1997) "applied to the sensitivities of the objective functions" (§1); radius not given for this example | radius 0.06·b, applied to every f_sk and f_JJ |
| SIMP p schedule | "normally … increasing from 1 to 3" (§2.1) | fixed p = 3 (production ruling, README) |

## Consequence for this task

The reference computed here is the solution of the **production** frozen
problem, which includes the reconstruction's move box. The paper's (25) has no
move box; its frozen subproblem, with the density box alone, is a different
(larger) feasible set and was not solved here (the task forbids changing the
move limit). Every statement about "the solution of problem (25)" in this
audit therefore means: problem (25) as production poses it, at the frozen
480 state, with the ±0.01 move box. The move box is a reconstruction choice and
`ACTIVE_SET_ANALYSIS.md` shows it is scientifically consequential: 99.94 % of
the reference variables sit on a bound.
