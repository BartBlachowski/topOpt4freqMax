# WP1 — Exact mathematical definitions of E1, E2, E3
READ_ONLY_AUDIT — NOT_NEW_OPTIMIZATION_EVIDENCE

Source of truth: `analysis/three_method_parametric_study/study_evaluate_design.m`,
function `solve_modes`, lines 39–58. All three evaluators share the same Q4 element
matrices, the same mesh, the same midheight pinned support, and the same `eigs` call for
the three smallest eigenvalues; they differ only in K(x) and M(x).

Let `z` be the element density, already clamped to [0,1] at line 5.

## E1 — Proposed

    K_e(z) = 1e7 * ( 1e-6 + (1 - 1e-6) * z^3 )
    M_e(z) = 1e-6 + (1 - 1e-6) * z

No branch. No floor beyond the additive 1e-6. **Continuous and C-infinity on [0,1].**

## E2 — Yuksel

    K_e(z) = 1e7 * ( 1e-9 + (1 - 1e-9) * z^3 )

    g(z)   = z^6   for z <= 0.1
           = z     for z >  0.1

    M_e(z) = 1e-9 + (1 - 1e-9) * g(z)

Branch condition `z <= 0.1` (closed below, so z = 0.1 takes the x^6 branch).
**K is continuous. M is DISCONTINUOUS at z = 0.1.**

## E3 — Olhoff

    z3     = max(z, 1e-3)
    K_e(z) = 1e7 * z3^3

    g(z3)  = z3^6  for z3 <= 0.1
           = z3    for z3 >  0.1

    M_e(z) = g(z3)

No additive void floor; the floor is the clamp `max(z,1e-3)`.
**K is continuous (the max clamp is continuous). M is DISCONTINUOUS at z = 0.1.**

## Branch-point evaluation

| function | left limit z→0.1- | value at z = 0.1 | right limit z→0.1+ | classification |
|---|---|---|---|---|
| E1 mass | 0.1 | 0.1 | 0.1 | continuous, C-inf |
| E1 stiffness | 1e-3·1e7 | same | same | continuous |
| **E2 mass g** | **1e-6** | **1e-6** | **0.1** | **discontinuous** |
| **E3 mass g** | **1e-6** | **1e-6** | **0.1** | **discontinuous** |
| E2/E3 stiffness | continuous | — | — | continuous |
| E3 clamp max(z,1e-3) at z=1e-3 | 1e-3 | 1e-3 | 1e-3 | C0, not C1 |

Absolute jump at 0.1: **9.9999e-02**. Multiplicative jump: **1.0e+05**.

## The two continuity-restoring variants that exist in this repository

From `Matlab/reproduction2007/fem/massScale.m` and
`analysis/OlhoffApproachExact/Matlab/mass_interp.m`:

| variant | low branch | g(0.1-) | g'(0.1-) | continuity |
|---|---|---|---|---|
| `'4'` / `du2007_step` (**the one E2/E3 use**) | z^6 | 1e-6 | 6e-5 | **discontinuous** |
| `'4a'` / `du2007_c0`, c0 = 1e5 | 1e5·z^6 | 0.100000 | 6 | C0 (slope jumps 6 → 1) |
| `'4b'` / `du2007_c1`, c1 = 6e5, c2 = −5e6 | c1 z^6 + c2 z^7 | 0.100000 | 1.000000 | C1 |
| `'lin'` | z | 0.1 | 1 | C-inf |

Verified numerically: `1e5 * 0.1^6 = 0.100000` exactly; `6e5·0.1^6 − 5e6·0.1^7 = 0.100000`
with slope `6·6e5·0.1^5 − 7·5e6·0.1^6 = 1.000000`.

## Consequence for the generalized eigenproblem

The evaluators solve `K(x) φ = λ M(x) φ` on the free DOFs. A finite jump in one element's
mass coefficient changes M by a finite amount on that element's 8 DOFs. By the Rayleigh
quotient `λ = φᵀKφ / φᵀMφ`, adding mass can only decrease λ (K unchanged), so single
storage — which pushes at-risk elements onto the heavier linear branch — can only lower
omega. The *magnitude* of the change is not the interpolation jump itself: it is weighted
by the affected elements' modal participation `φᵀM_eφ / φᵀMφ`. A jump in an element with
negligible participation in mode 1 produces a negligible change in omega_1. **No universal
eigenfrequency jump follows from the interpolation jump**; the observed effect must be and
was measured empirically (WP6, WP9, WP11).
