# Specification — Du & Olhoff (2007), "Topological design of freely vibrating continuum structures for maximum values of simple and multiple eigenfrequencies and frequency gaps"

**Source:** Struct Multidisc Optim (2007) 34:91–110, DOI 10.1007/s00158-007-0101-y
**Erratum:** Struct Multidisc Optim (2007) 34:545, DOI 10.1007/s00158-007-0167-6

**Status of this document:** Phase 1 deliverable. Derived *only* from the two PDFs above. No implementation code was consulted. Every item is classified as **[E] Explicitly specified**, **[I] Implied**, or **[N] Not specified**.

**Erratum applied:** The erratum inserts the increment symbol `Δ` between `δ_sk` and `(ω²)` in equations (25d), (26f), (26g) — i.e. the off-diagonal coupling term is `δ_sk · Δ(ω²)`, an *increment* of the squared eigenfrequency, not `δ_sk · (ω²)`. It also corrects the Fig. 2 caption initial frequencies to `ω⁰_{1a}=68.7, ω⁰_{1b}=104.1`. Both corrections are folded into the equations below.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `ρ_e` | element volumetric material density (design variable), one per finite element, `e = 1,…,N_E` |
| `N_E` | number of finite elements in admissible design domain |
| `p` | SIMP penalization power for **stiffness**, `p ≥ 1` |
| `q` | penalization power for **mass**, `q ≥ 1` (normally `q = 1`) |
| `r` | high mass-penalization power used in low-density regions (`≈ 6`) to suppress localized modes |
| `E*_e`, `M*_e`, `K*_e` | element elasticity matrix / mass matrix / stiffness matrix for **fully solid** material |
| `E*¹,E*²,M*¹,M*²` | element matrices for solid materials *1 (stiffer) and *2 (bimaterial case) |
| `K`, `M` | global stiffness, mass matrices |
| `ω_j` | j-th eigenfrequency; `λ_j = ω_j²` j-th eigenvalue |
| `φ_j` | j-th eigenvector (mode shape), M-orthonormalized |
| `J` | number of candidate eigenfrequencies/modes computed |
| `n` | target order of the eigenfrequency being maximized (`n=1` ⇒ fundamental) |
| `N` | multiplicity of `ω_n` at current design |
| `R` | multiplicity of `ω_{n-1}` (only relevant for the gap problem) |
| `α` | prescribed volume fraction, `V* = α·V_0` |
| `V_0` | volume of admissible design domain; `V_e` element volume |
| `V*` | available volume of (solid / stiffer-material) material |
| `ρ̲` (rho-underbar) | small positive lower bound on `ρ_e`, set to `10⁻³` |
| `β, β_1, β_2` | scalar bound variables of the bound formulation |
| `Δρ_e` | increment of design variable in one main iteration |
| `f_sk` | generalized gradient vector (eq. 19) |
| `γ_0` | Lagrange multiplier of the volume constraint |
| `ε` | convergence tolerance on `‖Δρ‖` |

---

## 1. Material interpolation (Section 2)

### 1.1 Single-material SIMP — [E]
Stiffness (eq. 1):
```
E_e(ρ_e) = ρ_e^p · E*_e ,        0 ≤ ρ_e ≤ 1 ,  p ≥ 1
```
Mass (eq. 2):
```
M_e(ρ_e) = ρ_e^q · M*_e ,        q ≥ 1   (normally q = 1)
```
Global assembly (eq. 3):
```
K = Σ_{e=1}^{N_E} ρ_e^p · K*_e
M = Σ_{e=1}^{N_E} ρ_e^q · M*_e
```
**[E]** `E_e(0)=0`, `E_e(1)=E*_e`. A pure 0–1 design is physically correct under this model.
**[E]** `p` is "normally assigned values increasing from 1 to 3 during the optimization process" — i.e. **continuation on `p` from 1 → 3**. The *schedule* of the continuation (step size, iterations per level) is **[N]**.
**[E]** `q = 1` is the normal choice.

### 1.2 Localized-eigenmode suppression (Section 2.2) — [E] (mechanism), [N] (which variant the benchmarks used)
With `p≈3`, `q=1`, low-density regions (`ρ_e ≤ 0.1`) have stiffness (`ρ^3`) much smaller than mass (`ρ^1`), producing **spurious localized eigenmodes with very low eigenfrequencies**. Following Tcherniak (2002) (with a modification to avoid numerical singularity), the **mass** interpolation is altered in low-density regions while the **stiffness** power is kept at `p≈3`.

Base modified mass model (eq. 4) — discontinuous at `ρ_e = 0.1`:
```
M_e(ρ_e) = ρ_e   · M*_e ,   ρ_e > 0.1
         = ρ_e^r · M*_e ,   ρ_e ≤ 0.1 ,   r ≈ 6
```
`C⁰`-continuous variant (eq. 4a):
```
M_e(ρ_e) = ρ_e        · M*_e ,   ρ_e > 0.1
         = c_0·ρ_e^6   · M*_e ,   ρ_e ≤ 0.1 ,   c_0 = 10^5
```
`C¹`-continuous variant (eq. 4b):
```
M_e(ρ_e) = ρ_e                  · M*_e ,   ρ_e > 0.1
         = (c_1·ρ_e^6 + c_2·ρ_e^7)·M*_e ,   ρ_e ≤ 0.1 ,
           c_1 = 6×10^5 ,  c_2 = −5×10^6
```
(Continuity checked: at `ρ=0.1`, all three lower branches → `0.1·M*` for 4a/4b; eq. 4 jumps to `10⁻⁶·M*`. Derivative of 4b lower branch at `ρ=0.1` equals 1, matching the upper branch.)

**[E]** The authors state they tried all three (4),(4a),(4b) and found **only negligible differences** in final results because low-density regions contribute very little to the first several eigenfrequencies and all intermediate densities approach 0/1 anyway.
**[N]** Which of (4)/(4a)/(4b) was used for any *specific* benchmark figure. Threshold `0.1` is **[E]**; the value `r≈6` is **[E]** (approximate).

### 1.3 Bimaterial SIMP (Section 2.3) — [E]
Stiffness (eq. 5):
```
E_e(ρ_e) = ρ_e^p · E*¹_e + (1 − ρ_e^p) · E*²_e
```
with material *1 the stiffer one; `p = 3` was used (gave distinctive 0–1 designs).
Mass (eq. 6) — simple linear:
```
M_e(ρ_e) = ρ_e · M*¹_e + (1 − ρ_e) · M*²_e
```
**[E]** `ρ_e=1` ⇒ element is fully material *1; `ρ_e=0` ⇒ fully material *2.
**[E]** For bimaterial, `V*` is the total volume of stiffer material *1 (`Σ ρ_e V_e`); material *2 volume = `V_0 − V*`. In figures, material *1 = black, material *2 = gray.

### 1.4 Sensitivity filter — [E] (existence), [N] (parameters)
**[E]** "the mesh-independent filter developed by Sigmund (1997) … has been applied to the sensitivities of the objective functions in the computational models in the paper." (Section 2 introduction.) Purpose stated: prevent checkerboards and mesh dependence.
**[N]** Filter radius `r_min`, filter weighting kernel, and whether the filter is applied to all examples or only some. Whether the filter is applied to constraint/eigenfrequency sensitivities (the text says "objective functions") vs. all sensitivities is **ambiguous**.

---

## 2. Generalized eigenvalue problem (Section 3.1) — [E]

```
K φ_j = ω_j² M φ_j ,        j = 1,…,J            (7b)
φ_jᵀ M φ_k = δ_jk ,          j ≥ k, k,j=1,…,J     (7c)   ← M-orthonormalization
0 < ω_1 ≤ ω_2 ≤ … ≤ ω_J                          (8)
```
**[E]** `K`, `M` symmetric, positive definite (guaranteed by `ρ̲ > 0`).
**[E]** Eigenvectors are M-orthonormalized (Kronecker δ).
**[E]** All eigenfrequencies real and ordered ascending.
**[N]** The numerical eigensolver (e.g. subspace iteration, Lanczos/ARPACK, shift-invert). The paper only says "FE analysis". **[N]** Whether a shift is used.

---

## 3. Optimization variables — [E]

| Variable | Where | Notes |
|---|---|---|
| `ρ_e`, `e=1,…,N_E` | all problems | element densities, bound `ρ̲ ≤ ρ_e ≤ 1` |
| `β` | (15), (25) | scalar lower bound on the `n`-th (and higher) eigenfrequency, **maximized** |
| `β_1, β_2` | (16), (26) | scalar bounds; objective is `β_2 − β_1` (gap) |
| `Δρ_e` | sub-problems (25),(26) | per-iteration increments (inner-loop unknowns) |
| `Δ(ω_j²)` | (25),(26) | *dependent* increments of multiple eigenvalues (eigenvalues of the subeigenvalue problem 25d/26f/26g) |

**[E]** In the sub-optimization problems the **only true unknowns are the bound variables (`β`, or `β_1,β_2`) and the increments `Δρ_e`**; the `Δ(ω_j²)` are *dependent* quantities obtained from the algebraic subeigenvalue problem. `K, M, ω_j, φ_j, f_sk, N, R` are all held *fixed* during the inner loop (computed in steps 1–2 of the main loop).

---

## 4. Constraints — [E]

Common constraints `7(b–e)`:
```
K φ_j = ω_j² M φ_j ,   j=1,…,J            (7b)
φ_jᵀ M φ_k = δ_jk                         (7c)
Σ_e ρ_e V_e − V* ≤ 0 ,   V* = α V_0       (7d)   ← single global volume constraint
0 < ρ̲ ≤ ρ_e ≤ 1                          (7e)   ← box constraints,  ρ̲ = 10⁻³
```
**[E]** `α = 0.5` (50 %) in all reported volume-fraction examples.
**[E]** There is **no connectivity, symmetry, perimeter, or minimum-length-scale constraint** in the formulation (only the Sigmund density-filter on sensitivities). ⇒ The formulation *permits disconnected topologies* (see Phase 2 analysis).

---

## 5. Three optimization problems (Sections 3.1, 3.3)

### 5.1 Max fundamental eigenfrequency — max–min form (eq. 7a–e) — [E]
```
max_{ρ}  min_{j=1..J} { ω_j² }
s.t.  7(b–e)
```

### 5.2 Max n-th eigenfrequency — bound form (eq. 15a–c) — [E]
```
max_{β, ρ}  { β }
s.t.  β − ω_j² ≤ 0 ,   j = n, n+1, …, J     (15b)
      7(b–e)                                 (15c)
```
For `n=1` this reduces to (7a–e). Removing `β_1` and (16c) from (16) below reduces it to (15).

### 5.3 Max gap between n-th and (n−1)-st eigenfrequencies — bound form (eq. 16a–d) — [E]
```
max_{β_1, β_2, ρ}  { β_2 − β_1 }
s.t.  β_2 − ω_j² ≤ 0 ,   j = n, n+1, …, J    (16b)
      ω_j² − β_1 ≤ 0 ,   j = 1, …, n−1        (16c)
      7(b–e)                                  (16d)
```
**[E]** `n > 1` for the gap problem. `β_2` lower-bounds the upper group (`≥ n`); `β_1` upper-bounds the lower group (`≤ n−1`); maximizing `β_2 − β_1` pushes them apart.
**[E]** The bound formulations (15),(16) are **differentiable in all variables** if `β`(or `β_1,β_2`) are treated as design variables alongside `ρ_e, ω_j, φ_j` (the "SAND" view) — but that is a very large problem, so the paper instead uses the **nested** form: for fixed `ρ`, solve the eigenproblem; then the problems (15),(16) are *non-differentiable* in `ρ` alone because the `ω_j(ρ)` can be multiple. This non-differentiability motivates Sections 3.4–3.5.

---

## 6. Sensitivity analysis

### 6.1 Unimodal eigenfrequency (Section 3.2) — [E]
If `ω_j` is simple (`ω_{j-1} < ω_j < ω_{j+1}`), `φ_j` unique up to sign, and
```
(λ_j)'_{ρe} = φ_jᵀ ( K'_{ρe} − λ_j M'_{ρe} ) φ_j            (10)
```
With model (3):
```
(λ_j)'_{ρe} = φ_jᵀ ( p ρ_e^{p-1} K*_e − λ_j q ρ_e^{q-1} M*_e ) φ_j     (11)
```
Optimality condition (Lagrange, side constraints ignored) (eq. 12):
```
(λ_j)'_{ρe} − γ_0 V_e = 0 ,   γ_0 ≥ 0
```
Linearized increment (eqs. 13–14):
```
Δλ_j = ∇λ_jᵀ Δρ ,
∇λ_j = { φ_jᵀ(K'_{ρ1} − λ_j M'_{ρ1})φ_j , … , φ_jᵀ(K'_{ρ_{NE}} − λ_j M'_{ρ_{NE}})φ_j }ᵀ
```
**[E]** Solvable by OC (fixed-point, Cheng & Olhoff 1982) **or** MMA. The paper's computational procedure (Section 3.5) uses **MMA** (Svanberg 1987).

### 6.2 Multiple eigenfrequency (Section 3.4) — [E]
For an `N`-fold eigenvalue `λ̃ = λ_j = ω_j²`, `j = n,…,n+N−1`, the increments `Δλ_j` are the **eigenvalues of an N-dimensional algebraic subeigenvalue problem** (eq. 18, erratum-corrected):
```
det[ f_skᵀ Δρ − δ_sk Δλ ] = 0 ,   s,k = n,…,n+N−1            (18)
```
with generalized gradient vectors (eq. 19):
```
f_sk = { φ_sᵀ(K'_{ρ1} − λ̃ M'_{ρ1})φ_k , … , φ_sᵀ(K'_{ρ_{NE}} − λ̃ M'_{ρ_{NE}})φ_k }ᵀ ,
        s,k = n,…,n+N−1
```
**[E]** `f_sk = f_ks` (symmetry of K, M). Each `f_sk` is an `N_E`-vector; `f_skᵀΔρ` is a scalar.
**[E]** The `Δλ_j` are in general **nonlinear** functions of the increment direction `Δρ`; multiple eigenvalues do **not** admit a usual linearization.

**Special case N=1 (3.4.1):** (18)→ `f_nnᵀΔρ − Δλ_n = 0` (eq. 20), and `f_nn = ∇λ_n` (eq. 21) — recovers the simple case.

**Special case — vanishing off-diagonal terms (3.4.2) — [E]:** If the increment satisfies
```
f_skᵀ Δρ = 0 ,   s ≠ k ,   s,k = n,…,n+N−1          (22)
```
then
```
Δλ_j = f_jjᵀ Δρ ,   j = n,…,n+N−1                    (23)
```
i.e. each multiple eigenvalue behaves exactly like a simple one (sensitivity formulas identical). This is the basis for the **optional LP reduction** (Section 7.4).

---

## 7. Computational procedure (Section 3.5, Fig. 1)

### 7.1 Choice of J — [E]
`J = n + N` — "the order of the closest eigenfrequency that is **greater than** the multiple eigenfrequency `ω_j, j=n,…,n+N−1`." `J` must be large enough to capture all members of a possible N-fold eigenfrequency `ω_n = … = ω_{n+N-1}`. `ω_J` (`= ω_{n+N}`) is assumed **simple**, so its bound constraint can be linearized with the gradient `f_JJ`.
**[I]** `J` is updated each main iteration because `N` (and possibly the ordering) can change between iterations.

### 7.2 Multiplicity detection — [E] (rule), [N] (tolerance value)
`N` = multiplicity of `ω_n`, `R` = multiplicity of `ω_{n-1}` (gap problem only). Two eigenfrequencies are treated as "multiple" if their **relative difference is within a predefined, very small tolerance**.
**[N]** The numerical value of that tolerance.

### 7.3 Main (outer) loop — [E]
```
Step 0.  Problem initialization. Choose n; initialize design variables ρ_e
         (benchmarks: uniform ρ_e = 0.5).
Step 1.  Solve generalized eigenproblem (7b,c) by FE for ω_j, φ_j, j=1..J.
         Detect multiplicity N of ω_n (and R of ω_{n-1} for the gap problem).
Step 2.  Compute generalized gradients f_sk (eq. 19) if N>1 (and R>1);
         else usual gradients ∇λ (eq. 14) if N=1 (and R=1).
Step 3.  INNER LOOP: solve sub-optimization problem (25) [n-th freq]
         or (26) [gap] for increments Δρ_e, with K,M,ω_j,φ_j,f_sk,N,R FIXED.
         Iterate inner loop until Δρ_e converged.
Step 4.  Update:  ρ_e := ρ_e + Δρ_e.
         Convergence check: is ‖Δρ‖ < ε ?
            yes → STOP (optimum topology obtained)
            no  → return to Step 1.
```
**[E]** Outer-loop convergence criterion: `‖Δρ‖ < ε`.
**[N]** Numerical value of `ε`. **[N]** Maximum outer iteration count. (Figs. 4,7,9,13,15,17,18 show ~50–100 iterations.)

### 7.4 Inner-loop sub-optimization problems

#### (A) Max n-th eigenfrequency (eq. 25a–f, erratum-corrected) — [E]
```
max_{β, Δρ_1,…,Δρ_{NE}}  { β }                                    (25a)
s.t.
  β − [ ω_j² + f_jjᵀ Δρ ] ≤ 0 ,            j = J = n+N            (25b)
  β − [ ω_j² + Δ(ω_j²) ] ≤ 0 ,             j = n,…,n+N−1          (25c)
  det[ f_skᵀ Δρ − δ_sk Δ(ω²) ] = 0 ,       s,k = n,…,n+N−1        (25d)
  Σ_e (ρ_e + Δρ_e) V_e − V* ≤ 0 ,          V* = α V_0             (25e)
  0 < ρ̲ ≤ ρ_e + Δρ_e ≤ 1 ,                e = 1,…,N_E            (25f)
```
- (25b): the **upper** simple eigenfrequency `ω_J` is linearized with its gradient `f_JJ` (it is simple) — this prevents the maximized group from overshooting and switching order with mode `J`.
- (25c): the `N` members of the multiple eigenfrequency are bounded using their *dependent* increments `Δ(ω_j²)`.
- (25d): algebraic subeigenvalue problem coupling `Δρ` and the `Δ(ω_j²)`; nonlinear in general; furnishes `N` eigenvalues `Δ(ω_j²)`.
- (25e): incremented volume constraint. (25f): incremented box constraints.

#### (B) Max gap between n-th and (n−1)-st (eq. 26a–i, erratum-corrected) — [E]
```
max_{β_1, β_2, Δρ}  { β_2 − β_1 }                                  (26a)
s.t.
  β_2 − [ ω_j² + f_jjᵀ Δρ ] ≤ 0 ,          j = J = n+N            (26b)
  β_2 − [ ω_j² + Δ(ω_j²) ] ≤ 0 ,           j = n,…,n+N−1          (26c)
  [ ω_j² + Δ(ω_j²) ] − β_1 ≤ 0 ,           j = n−R,…,n−1   (R≤n−1) (26d)
  [ ω_j² + f_jjᵀ Δρ ] − β_1 ≤ 0 ,          j = n−R−1     (if R≤n−2)(26e)
  det[ f_skᵀ Δρ − δ_sk Δ(ω²) ] = 0 ,       s,k = n,…,n+N−1        (26f)
  det[ f_skᵀ Δρ − δ_sk Δ(ω²) ] = 0 ,       s,k = n−R,…,n−1        (26g)
  Σ_e (ρ_e + Δρ_e) V_e − V* ≤ 0 ,          V* = α V_0             (26h)
  0 < ρ̲ ≤ ρ_e + Δρ_e ≤ 1 ,                e = 1,…,N_E            (26i)
```
- The **upper** group (`≥ n`, possible N-fold) is treated as in (25b–d).
- The **lower** group (`≤ n−1`, possible R-fold) is bounded *from above* by `β_1`: (26d) for the R members, (26e) for the next simple one below them (`n−R−1`), (26g) is the lower-group subeigenvalue problem.
- (26h),(26i): incremented volume and box constraints.

#### Inner-loop solver — [E]
**[E]** "The sub-optimization problems (25a–f) and (26a–i) … can be solved by a mathematical programming method. In this paper, the **MMA** method (Svanberg 1987) has been used."
**[E] (optional)** If the additional constraints (22) `f_skᵀΔρ = 0, s≠k` are imposed (forcing vanishing off-diagonal terms), then `Δλ_j = Δ(ω_j²)` are linear in `Δρ`, and (25),(26) **reduce to linear programming problems** (Krog & Olhoff 1999), solvable by an LP algorithm. The paper presents this as an alternative, *not* the default; the default is MMA on the nonlinear det-coupled problems.
**[N]** MMA move limits / asymptote update parameters. **[N]** Inner-loop convergence tolerance & max inner iterations.

---

## 8. Benchmark setups (Sections 4, 5)

### 8.1 Single-material 2D beam-like structures (Section 4)
**[E]** Plane-stress elements. Design domain `a = 8`, `b = 1` (aspect 8:1). Material: `E = 10⁷`, `ν = 0.3`, `ρ_m = 1` (SI). Volume fraction `α = 50 %`. Initial design: uniform `ρ = 0.5`.
Three boundary-condition cases (Fig. 2):
- **a** — simply supported at both ends.
- **b** — one end clamped, other simply supported.
- **c** — clamped at both ends.

**[N]** Mesh resolution (number of elements `N_E`) is **not stated numerically**; must be inferred from figures. **[N]** Element type details (4-node Q4 assumed but not stated), thickness value (taken `=1`? not stated). **[N]** Exact support node sets.

Reported results:
| Example | Initial ω⁰ | Optimum ω^opt | Increase | Multiplicity |
|---|---|---|---|---|
| 4.1a Max ω_1, SS | 68.7 | **174.7** | +154 % | bimodal |
| 4.1b Max ω_1, C–SS | 104.1 | **288.7** | +177 % | bimodal |
| 4.1c Max ω_1, C–C | 146.1 | **456.4** | +212 % | bimodal |
| 4.3a Max ω_2, SS | — | **598.3** | — | bimodal |
| 4.3b Max ω_2, C–SS | — | **732.8** | — | bimodal |
| 4.3c Max ω_2, C–C | — | **849.0** | — | bimodal |

**[E]** For 4.1a the first two modes are simple-support beam-type modes at `ω=174.7`; the third mode (general 2D) is at `ω=284.9` (Fig. 5).
**[E]** All optimum fundamental frequencies are **bimodal** — the fundamental frequency starts simple and coalesces with the 2nd during optimization (Fig. 4).

### 8.2 Comparison vs. mean-eigenvalue approach (Section 4.2) — [E]
Mean eigenvalue (eq. 27):
```
λ* = ( Σ_{j=n}^{n+L-1} 1/ω_j² )^{-1}
```
With `n=1, L=3`. Result (Fig. 6): `ω_1 = 161.7` (simple), **lower** than the present method's bimodal `174.7`. ⇒ present method superior; mean-eigenvalue avoids the 1–2 coalescence and yields thin truss members.

### 8.3 Max gap, 2D beam with concentrated mass (Section 4.4) — [E]
Clamped beam (case c geometry) + concentrated nonstructural mass `m_c = ½ m_b` (`m_b` = mass of all structural material) at the **midpoint of the lower edge**. Maximize gap `ω_3 − ω_2`. Result (Fig. 9): gap = **810** (548 % larger than initial). 3rd & 4th form a bimodal that finally coalesces with 5th (trimodal).

### 8.4 Plate-like 3D structures (Section 5)
**[E]** 8-node 3D continuum elements with **Wilson-incompatible displacement modes**. Quadratic plates `a = 20, b = 20, t = 1`. Material: `E = 10¹¹`, `ν = 0.3`, `ρ_m = 7800` (SI). `α = 50 %`. Initial uniform `ρ = 0.5`.

5.1 — Three BC + mass cases (Fig. 10):
- **a** SS at four corners, `m_c = m_0/3` at center.
- **b** four edges clamped, `m_c = m_0/10` at center.
- **c** one edge clamped (others free), `m_c = m_0/10` at midpoint of opposite edge.

| Example | Initial ω⁰ | Optimum ω^opt | Increase | Mode |
|---|---|---|---|---|
| 5.1a Max ω_1 | 8.1 | **16.4** | +101 % | unimodal |
| 5.1b Max ω_1 | 31.1 | **65.4** | +111 % | unimodal |
| 5.1c Max ω_1 | 3.5 | **9.7** | +179 % | unimodal |
| 5.1 SS@4corners+center Max ω_1 | 24.6 (bimodal) | **60.3** | — | bimodal |
| 5.2a Max ω_2 | — | **46.0** | — | trimodal |
| 5.2b Max ω_2 | — | **155.4** | — | bimodal |
| 5.2c Max ω_2 | — | **39.8** | — | bimodal |

5.2/5.3 — **Bimaterial:** stiffer *1 = same as single-material black; weaker *2 has `E*² = 0.1·E*¹`, `M*² = 0.1·M*¹`. Volume fraction of *1 = 50 %.
- Fig. 16/17: max 4th/5th/6th eigenfrequency of clamped bimaterial plate w/ center mass: `ω^opt_{4b}=243.8` (bimodal), `ω^opt_{5b}=249.7` (unimodal), `ω^opt_{6b}=353.2` (bimodal).
- Fig. 18: max gap `ω_3 − ω_2`, bimaterial quadratic plate (Fig. 10a, SS@4 corners): initial symmetric design has `ω_3 − ω_2 = 0` ⇒ **infinitely large relative increase**.
- Fig. 19: max gap `ω_3 − ω_2`, Fig. 10c geometry: `(ω_3−ω_2)_opt = 31.7`; `ω_1=4.23, ω_2=18.4, ω_3=50.1`.

**[N]** 3D mesh resolution, support node sets, value of `m_0`, exact mass-attachment node.

---

## 9. Stopping / convergence criteria — summary

| Criterion | Value | Class |
|---|---|---|
| Outer loop: `‖Δρ‖ < ε` | tolerance `ε` not given | rule [E], value [N] |
| Inner loop: `Δρ_e` converged | not quantified | rule [E], value [N] |
| Multiplicity tolerance (relative) | "very small", value not given | rule [E], value [N] |
| Eigenvalue lower bound `ρ̲` | `10⁻³` | [E] |
| Penalization continuation `p: 1→3` | schedule not given | rule [E], schedule [N] |

---

## 10. Assumptions (stated by the paper) — [E]

1. Linear elasticity, **no damping**.
2. `K`, `M` symmetric positive definite (ensured by `ρ̲ > 0`).
3. Eigenvectors M-orthonormalized.
4. `J` large enough to capture all members of the targeted multiple eigenfrequency; `ω_J = ω_{n+N}` is simple.
5. The `n−R−1`-th (and `J`-th) eigenfrequency just outside the multiple groups is simple ⇒ its bound constraint is linearized (eqs. 25b/26b/26e).
6. Within the inner loop, `K, M, ω_j, φ_j, f_sk, N, R` are frozen at the values from steps 1–2.
7. Low-density regions contribute negligibly to the first several eigenfrequencies (justifies indifference among mass models 4/4a/4b).

---

## 11. Ambiguities & undocumented details (consolidated) — [N] unless noted

1. **Mesh resolution** for every benchmark (2D and 3D) — not given numerically.
2. **Filter radius** `r_min` and kernel of the Sigmund (1997) filter — not given. Also ambiguous whether the filter is applied to objective sensitivities only or to all sensitivities, and whether eigenfrequency (constraint) sensitivities are filtered.
3. **Which mass model** (4 / 4a / 4b) and the exact value of `r` per example — not pinned (`r≈6`).
4. **Penalization continuation schedule** (`p` from 1→3): step sizes, iterations per level, whether `q` is ever continued — not given (`q=1` stated as normal).
5. **MMA parameters**: move limits, asymptote adaptation, initial asymptotes — not given.
6. **Convergence tolerances**: outer `ε`, inner-loop tolerance, multiplicity tolerance — not given numerically.
7. **Eigensolver**: type, shift, number of requested modes beyond `J` — not given.
8. **Mode-tracking between outer iterations** (how modes are matched across iterations so `N`, ordering stay consistent) — not described.
9. **Sub-problem (25d/26f/26g) handling in MMA**: how the *nonlinear det subeigenvalue equality* is supplied to MMA (as N coupled inequality pairs? via eigen-decomposition each MMA sub-iteration?) is **not operationalized** in the paper — only the LP reduction (via constraint 22) is fully concrete. This is the single biggest reproduction ambiguity (see comparisons in Phase 3).
10. **Sign/normalization convention** for eigenvectors entering `f_sk` when the eigenvalue is multiple (eigenvectors non-unique within the invariant subspace) — the subeigenvalue problem (18) is invariant to the choice, but a concrete numerical implementation must pick a basis; not discussed.
11. **Initial design** stated as uniform `ρ=0.5` for benchmarks; whether any symmetry is imposed/broken (e.g. perturbation to trigger bimodality) — not stated.
12. **Thickness / plane-stress thickness** for 2D, and `m_0` for 3D — not given.
13. **Boundary condition node sets** (which nodes are "simply supported" / "clamped" on a continuum mesh) — not given.

---

## 12. Implied items (reasoning) — [I]

- **[I]** `J` is recomputed each outer iteration (since `N` may change). The paper says `J=n+N` and that `N` is detected each step ⇒ `J` is iteration-dependent.
- **[I]** The number of computed modes must be ≥ `J = n+N` (and for the gap problem also cover `n−R−1`), so at least `n+N` modes per FE solve.
- **[I]** Because MMA is used and `p` is continued and the start is symmetric uniform `ρ=0.5`, the procedure is **path-dependent** (see Phase 2).
- **[I]** Q4 (4-node bilinear) elements for the 2D plane-stress case (standard; only "plane stress elements" stated).
- **[I]** Volume constraint is active at the optimum (material is the scarce resource being placed to stiffen the structure), so `Σ ρ_e V_e = α V_0` at convergence.
- **[I]** The "increase" percentages in the tables are `(ω^opt − ω⁰)/ω⁰`.

---

## 13. Cross-reference of every numbered equation

| Eq | Role |
|---|---|
| (1) | SIMP stiffness interpolation |
| (2) | SIMP mass interpolation (power `q`) |
| (3) | Global K, M assembly |
| (4),(4a),(4b) | Modified mass interpolation in low-density regions (localized-mode suppression) |
| (5) | Bimaterial stiffness interpolation |
| (6) | Bimaterial mass interpolation (linear) |
| (7a–e) | Max fundamental freq (max–min) + constraints |
| (8) | Eigenfrequency ordering |
| (9) | Differentiated eigenproblem |
| (10) | Unimodal eigenvalue sensitivity (general) |
| (11) | Unimodal eigenvalue sensitivity (SIMP model 3) |
| (12) | Optimality condition (Lagrange) |
| (13),(14) | Linearized eigenvalue increment + gradient vector |
| (15a–c) | Bound formulation, max n-th freq |
| (16a–d) | Bound formulation, max gap |
| (17) | Definition of N-fold multiple eigenvalue |
| (18) | Subeigenvalue problem for multiple-eigenvalue increments [erratum] |
| (19) | Generalized gradient vectors f_sk |
| (20),(21) | N=1 reduction of (18),(19) |
| (22),(23),(24) | Vanishing off-diagonal terms ⇒ simple-like increments |
| (25a–f) | Inner sub-problem: max n-th freq [25d erratum] |
| (26a–i) | Inner sub-problem: max gap [26f,26g erratum] |
| (27) | Mean-eigenvalue objective (comparison method, not the present method) |

---

## 14. One-paragraph algorithm summary (the "exact" reconstruction target)

Given a target order `n`, volume fraction `α`, SIMP powers `p,q`, and a uniform initial `ρ=0.5`: **outer loop** — (1) assemble `K(ρ), M(ρ)` with SIMP (stiffness `ρ^p`, mass `ρ^q` with low-density modification 4/4a/4b), solve `Kφ=ω²Mφ` for the lowest `J=n+N` M-orthonormal modes, and detect the multiplicity `N` of `ω_n` (and `R` of `ω_{n-1}` for gaps); (2) build either the simple gradients `∇λ` (eq. 14, if `N=1`) or the generalized gradients `f_sk` (eq. 19, if `N>1`); (3) **inner loop** — solve by MMA the sub-optimization problem (25) [max `ω_n`] or (26) [max gap], whose unknowns are the bound variable(s) and the density increments `Δρ_e`, subject to bound constraints on the (possibly multiple) eigenfrequency increments — where the increments of multiple eigenvalues come from the algebraic subeigenvalue problem (25d/26f/26g) [or, if off-diagonal terms are forced to vanish via (22), the whole sub-problem linearizes to an LP]; iterate the inner loop to convergence in `Δρ`; (4) update `ρ := ρ + Δρ` (continuing `p` toward 3), and repeat the outer loop until `‖Δρ‖ < ε`. The defining design feature, repeatedly emphasized, is that the optimum is **almost always a multiple (bimodal/trimodal) eigenfrequency**, and the machinery (bound formulation + subeigenvalue increments + the `J=n+N` linearized cap) exists precisely to handle that multiplicity and to prevent mode-order switching.
