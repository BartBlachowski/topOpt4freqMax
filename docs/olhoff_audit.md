# Olhoff & Du (2014) — Algorithm Audit

**Paper:** N. Olhoff & J. Du, "Structural Topology Optimization with Respect to Eigenfrequencies of Vibration," Chapter 11 in *Topology Optimization in Structural and Continuum Mechanics*, CISM 2014. DOI 10.1007/978-3-7091-1643-2_11.

**MATLAB implementation:** `OlhoffApproach/Matlab/topFreqOptimization_MMA.m`

---

## 1. Problem Formulation

### 1.1 Objective and constraints

| Paper | Code | Status |
|-------|------|--------|
| Maximise min{ωⱼ²} via bound variable β (Eq. 9a–9c, Sec. 2.3) | Auxiliary variable `Eb` = β / `lambda_ref` plays the role of β. Objective `f0 = -Eb + grayness_penalty` (line 308). Eigen-constraints `fval(j) = (Eb - λⱼ/lambda_ref)` (lines 323–328). | **DIFFERENT** — paper's pure `max{β}` objective gains an extra *grayness penalty* term (lines 298–316). This term (∝ mean(4·x·(1-x))) is not present in Eq. 9a or 19a; it was added to push discrete solutions during Heaviside continuation. The physics content of the constraint set is preserved. |
| Volume constraint: Σρₑ Vₑ − V* ≤ 0, one-sided inequality (Eq. 1d) | Two-sided equality enforced via `g_up = mean(xPhys) − volfrac ≤ 0` and `g_low = volfrac − mean(xPhys) ≤ 0` (lines 330–339). | **DIFFERENT** — paper uses a single inequality; code uses a pair to force equality. Functionally produces the same optimal volume fraction at convergence. |
| J eigenvalue constraints β − ωⱼ² ≤ 0, j = n,…,J (Eq. 9b, 19b–19c) | `fval(1:J)` with `J = cfg.J` modes (lines 321–328). | **MATCH** — all J smallest eigenvalues are constrained. |
| Bound variable β ∈ ℝ (Eq. 9a) | `Eb` ∈ [`cfg.Eb_min`, `cfg.Eb_max`] with initial value `cfg.Eb0` (lines 137–140). Scaling: `lambda_ref = cfg.lambda_ref` (line 133). | **MATCH** in structure; scaling by `lambda_ref` is a numerical conditioning choice not discussed in the paper. |

### 1.2 Material interpolation

| Paper | Code | Status |
|-------|------|--------|
| SIMP stiffness: Eₑ = Emin + ρₑᵖ(E₀−Emin) (Eq. 5) | `Ee = Emin + (xPhys.^penal)*(E0-Emin)` (line 209); sensitivity `penal*(E0-Emin)*(xPhys.^(penal-1)).*…` (line 294). | **MATCH** |
| Linear mass: ρₑ = ρmin + ρₑ(ρ₀−ρmin) (Eq. 5, exponent q=1) | `re = rho_min + xPhys*(rho0-rho_min)` (line 210); sensitivity `lam_sorted(j)*(rho0-rho_min).*…` (line 295). | **MATCH** — q = 1 (linear) is physically correct and consistent with the paper's single-material model. |
| Design variable range 0 ≤ ρ̲ ≤ ρₑ ≤ 1 (Eq. 1e) | `xmin(1:nEl) = 1e-3`, `xmax(1:nEl) = 1` (lines 137–138). | **MATCH** — ρ̲ = 1 × 10⁻³ (void elements are not fully empty to prevent singular K). |

### 1.3 Finite element model

| Paper | Code | Status |
|-------|------|--------|
| 2D plane-stress Q4 elements (Sec. 3.1, beam examples) | `q4_rect_KeM_planeStress` (lines 658–702). 2×2 Gauss quadrature, consistent mass matrix. | **MATCH** |
| Consistent mass matrix (Eq. 1b) | `Me = M0/rho0` assembled as a consistent mass from shape-function integrals (lines 692–699). | **MATCH** |

### 1.4 Boundary conditions

| Paper | Code | Status |
|-------|------|--------|
| Fig. 2(a) — simply-supported (SS): triangular pin supports drawn at mid-height of both vertical edges | `buildSupports("SS")`: pins at mid-height nodes of left and right edges (`leftMid`, `rightMid`), DOFs u,v fixed (lines 643–646). | **MATCH** — comment in code explicitly references "Figure 2(a)". |
| Fig. 2(b) — clamped–simply (CS): left edge fully clamped, right edge has pin at mid-height | `buildSupports("CS")`: all DOFs on left column + u,v of `rightMid` (lines 638–641). | **MATCH** |
| Fig. 2(c) — clamped–clamped (CC): both edges fully clamped | `buildSupports("CC")`: all DOFs on both left and right columns (lines 632–634). | **MATCH** |
| CF (cantilever) — not in beam examples but a natural extension | `buildSupports("CF")`: left edge fully clamped (lines 628–629). | **EXTRA** — not a benchmark case in the paper but a valid extension. |

---

## 2. Outer Loop (Fig. 1, Sec. 2.5)

The paper describes the outer loop as Steps 1 → 2 → 3 → 4 → convergence check → repeat.

```
for it = 1 : maxiter                          % outer loop, line 187
```

### Step 0 — Initialise design variables (Fig. 1)

| Paper | Code | Status |
|-------|------|--------|
| "Define value of n and initialise design variables ρₑ" (Fig. 1 Step 0) | `xval = [volfrac*ones(nEl,1); cfg.Eb0]` (lines 140–141). Uniform initial density equal to `volfrac`. | **MATCH** |

### Step 1 — Solve generalised eigenvalue problem (Fig. 1; Eq. 1b,c)

| Paper | Code | Status |
|-------|------|--------|
| Solve Kφⱼ = ωⱼ²Mφⱼ for J modes (Eq. 1b) | `eigs(Kf,Mf,J,'SM',optsSM)` (line 222) with residual check and retry (lines 226–260). | **MATCH** |
| M-orthonormalise eigenvectors (Eq. 1c): φⱼᵀMφₖ = δⱼₖ | `v = v/sqrt(v'*(Mf*v))` inside the sensitivity loop (line 290). | **MATCH** |
| Order eigenvalues 0 < ω₁ ≤ ω₂ ≤ … ≤ ωⱼ (Eq. 2) | `sort(real(diag(D_low)))` (lines 223–224). | **MATCH** |
| Detect multiplicity N of ωₙ (and R of ωₙ₊₁) (Fig. 1 Step 1) | **Not implemented.** No code detects whether the active eigenvalue is multiple. | **MISSING** — N is implicitly assumed to be 1 throughout. This is a known limitation: the code applies simple-eigenvalue sensitivities even if two eigenvalues coalesce (which frequently happens at the optimum; paper Figs. 3, 4). |

### Step 2 — Sensitivity analysis (Fig. 1; Eq. 4/5/8/13)

| Paper | Code | Status |
|-------|------|--------|
| Simple eigenvalue sensitivity (Eq. 4/5): ∂λⱼ/∂ρₑ = φⱼᵀ(K'ₑ − λⱼM'ₑ)φⱼ | `dlam(:,j) = penal*(E0-Emin)*(xPhys.^(penal-1)).*sum((pe*Ke).*pe,2) - lam_sorted(j)*(rho0-rho_min).*sum((pe*Me).*pe,2)` (lines 288–296). Computed for all j = 1,…,J. | **MATCH** — exact implementation of Eq. 4/5 for the SIMP+linear-mass model. |
| Generalised gradient vectors t_sk for N-fold eigenvalue (Eq. 12/13, Sec. 2.4) | **Not implemented.** `dlam(:,j)` uses only the diagonal j==k terms; off-diagonal coupling terms (φₛᵀ(K'ₑ−λ̄M'ₑ)φₖ for s≠k) are never formed. | **MISSING** — consequence: when eigenvalues coalesce the sub-eigenvalue problem (Eq. 12) is not solved; MMA receives incorrect gradients for the multiple-eigenvalue case. In practice the optimizer still converges to near-optimal bimodal topologies, but convergence is not guaranteed to be monotone near coalescence. |
| Chain rule through Heaviside projection: dλ/dx = dλ/dxPhys · dH/dxTilde (not in paper) | `reshape(-dlam(:,j)/lambda_ref,nely,nelx).*dH` then `bwd(g)` (lines 324–326). | **EXTRA** — correct chain rule for the Heaviside + PDE-filter pipeline not in the paper. |

### Step 3 — Iterative solution of optimisation sub-problem (Fig. 1; Eq. 19)

| Paper | Code | Status |
|-------|------|--------|
| "Iterative solution of optimization sub-problem (19) for increments Δρₑ" — an **inner loop** until Δρₑ converged (Fig. 1, inner branch) | **Single** call to `mmasub` per outer iteration (line 354). No inner convergence loop. | **DIFFERENT** — paper explicitly shows an inner loop (see the "Increments Δρₑ converged?" diamond in Fig. 1). The current code exits the sub-problem after one MMA step. Task 1 adds the inner loop scaffold (`cfg.inner.enabled`). |
| Sub-problem solved by MMA (Svanberg 1987) (Sec. 2.5, last paragraph) | `mmasub.m` + `subsolv.m` — Svanberg's September 2007 reference implementation. | **MATCH** |
| Sub-problem (19b): β − [ωⱼ² + t_jjᵀΔρ] ≤ 0 for j = J = n+N (simple neighbour) (Eq. 19b) | `fval(j) = (Eb - lam_sorted(j)/lambda_ref)` (line 323). All J modes contribute separate constraints; no distinction between 19b (simple upper neighbour) and 19c (multiple modes). | **DIFFERENT** — because N is always assumed 1, the code treats all J modes identically rather than using the two-category structure of Eq. 19b vs 19c. |
| Move limit on Δρₑ embedded in MMA asymptotes (Eq. 19f implicit in MMA) | Additional hard move limit: `xnew(1:nEl) = min(max(xnew(1:nEl), xval-move_lim), xval+move_lim)` (lines 374–376). | **EXTRA** — explicit trust-region clamp on top of the MMA asymptote move, not in the paper. Improves robustness during β continuation. |

### Step 4 — Update design variables (Fig. 1)

| Paper | Code | Status |
|-------|------|--------|
| ρₑ := ρₑ + Δρₑ after inner loop converges (Fig. 1 Step 4) | `xold2=xold1; xold1=xval; xval=xnew` (line 392). Boundary-box and Eb-damp clamps applied first (lines 362–390). | **MATCH** in spirit; clamping is a robustness addition. |

### Outer convergence check (Fig. 1)

| Paper | Code | Status |
|-------|------|--------|
| "ρₑ converged? i.e., ||Δρ|| < ε?" (Fig. 1, outer check) | Composite check: `rel_change_obj < cfg.conv_tol && change_x < cfg.conv_tol && grayness < 0.05` (lines 441–449). Requires `grayness < 0.05` (near-discrete) and polishing iterations at max β. | **DIFFERENT** — paper uses a single norm criterion on Δρ. Code adds relative-objective change and grayness conditions. The `change_x` term (`norm(Δρ)/sqrt(nEl)`) is equivalent to the paper's ||Δρ||/√N criterion, but the grayness gate is extra. |

---

## 3. Features in the Code Not Present in the Paper

These are practical additions made during implementation; they do not alter the underlying mathematical problem.

| Feature | Code location | Rationale |
|---------|---------------|-----------|
| Heaviside projection with β continuation | Lines 93–115, helper `heavisideProjection` (lines 596–600) | Improves discreteness; absent from the 2014 paper which assumes crisp densities. |
| Adaptive β schedule and "safe" iterations after β jump | Lines 195–198, 415–432 | Prevents numerical instability during continuation. |
| Grayness penalty in objective | Lines 298–316 | Forces discrete solutions during early β stages. |
| Guard against objective drop (extra `evalModes` call) | Lines 380–390 | Halves move limit if trial ω₁ drops >1 %; costs one extra eigensolution per outer iter. |
| `lambda_ref` scaling of Eb | Lines 133, 323–328 | Numerical conditioning; not in paper. |
| Eb variable as extra MMA degree of freedom | Lines 134, 137–140 | Adds β as a MMA variable instead of separately updating it; this is a reformulation equivalent to the paper's SAND (Simultaneous Analysis and Design) approach. |
| Retry eigensolver with tighter tolerance | Lines 236–259 | Robustness; absent from paper. |

---

## 4. Ambiguities and Assumptions

1. **Support location for SS/CS (Fig. 2(a),(b))**: the paper's Fig. 2 drawings show triangular pin-symbols at the mid-height of the vertical edges. The code places DOF constraints at the **exact mid-height node** (`midIdx = round(nely/2)+1`). For odd `nely` this is exact; for even `nely` it is off by half an element. *Assumption*: mid-height-node pinning.

2. **Which modes count as "the n-th eigenfrequency"**: the paper indexes eigenfrequencies from 1 (fundamental). In the code `J = cfg.J` modes are constrained (default J=3) and the **first** (`lam_sorted(1)`) is the objective. This corresponds to n=1, maximising the fundamental eigenfrequency (Eq. 9 / Sec. 3.1).

3. **Sensitivity with projection**: the paper's Eq. 4/5 gives ∂λ/∂ρₑ (physical density). The code applies the chain rule through Heaviside projection and density filter analytically (lines 311, 324–326). The paper does not discuss filtering/projection because it operates on crisp densities; the extension is standard and correct.

4. **Eigenvalue coalescence in practice**: the paper's examples (Figs. 3, 4) show the optimum is consistently **bimodal** (N=2). Because the code uses simple-eigenvalue sensitivities, the gradient near coalescence is only a subgradient of the true min-eigenvalue function. The code compensates partly through the J simultaneous eigenvalue constraints and through the bound-variable Eb, which naturally introduces a max-min structure.

---

## 5. Summary Table

| Algorithm element | Paper ref | File : lines | Status |
|---|---|---|---|
| Bound formulation (max β) | Eq. 9a–9c, Sec. 2.3 | lines 133–148, 321–328 | **MATCH** (scaled) |
| J eigenvalue constraints | Eq. 9b, 19b–c | lines 321–328 | **MATCH** |
| Volume constraint | Eq. 1d | lines 330–339 | **DIFFERENT** (two-sided) |
| SIMP stiffness interpolation | Eq. 5 | lines 209, 294 | **MATCH** |
| Linear mass interpolation | Eq. 5 (q=1) | lines 210, 295 | **MATCH** |
| Eigenvalue ordering | Eq. 2 | lines 223–224 | **MATCH** |
| Eigenvalue solve | Eq. 1b, Fig. 1 Step 1 | lines 221–261 | **MATCH** |
| M-orthonormalization | Eq. 1c | line 290 | **MATCH** |
| Multiplicity detection (N, R) | Fig. 1 Step 1, Sec. 2.5 | — | **MISSING** |
| Simple eigenvalue sensitivity | Eq. 4/5, Sec. 2.2 | lines 288–296 | **MATCH** |
| Generalised gradients (multiple eig) | Eq. 12/13, Sec. 2.4 | — | **MISSING** |
| Inner loop (sub-problem iterations) | Fig. 1 Step 3 (inner branch) | — (scaffold added in Task 1) | **DIFFERENT** → scaffold |
| MMA sub-problem solver | Sec. 2.5, Svanberg 1987 | `mmasub.m`, `subsolv.m` | **MATCH** |
| Outer convergence criterion | Fig. 1 outer check | lines 438–450 | **DIFFERENT** (augmented) |
| Design update | Fig. 1 Step 4 | line 392 | **MATCH** |
| Q4 plane-stress element | Sec. 3.1 | lines 658–702 | **MATCH** |
| Boundary conditions SS/CS/CC | Fig. 2(a–c) | lines 602–656 | **MATCH** |
| Heaviside projection + β continuation | — | lines 93–115, 596–600 | **EXTRA** |
| Grayness penalty | — | lines 298–316 | **EXTRA** |
| Move-limit clamp | — | lines 370–378 | **EXTRA** |
| Guard against ω drop | — | lines 380–390 | **EXTRA** |
