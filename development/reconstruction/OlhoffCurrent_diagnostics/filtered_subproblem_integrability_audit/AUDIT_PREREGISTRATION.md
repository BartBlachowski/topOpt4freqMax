# AUDIT_PREREGISTRATION — filtered_subproblem_integrability_audit

Frozen before any evaluation beyond file-identity checks. No tolerance, bar or
direction below may be changed after seeing results.

## 0. Scope and locks

ZERO optimization runs. ZERO accepted density updates. ZERO continuation. ZERO
controller transitions. No production file is modified. Every solve of the
frozen inner subproblem is labelled **FROZEN-SUBPROBLEM CERTIFICATION** and its
`drho` is never applied to any density.

Primary mesh: **480×60**. 400 and 800 are used only if a specific secondary
check is named below. This is not a cross-mesh study.

## 1. Authoritative state

The 480×60 three-rung canary endpoint, identical to the state the
`gray_kkt_forensic_audit` used:

| | |
|---|---|
| study | `diagnostics/three_rung_canary_preflight` |
| trajectory | `evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat` |
| ρ SHA-256 | `0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60` |
| config hash | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` |
| `+impl` tree | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| final outer | 386 |

Two densities are distinguished throughout and never conflated:

* **ρ₃₈₅** — `RHO(:,385)`, the density the **final MMA subproblem was built at**;
* **ρ₃₈₆** — `RHO(:,386)` = `res.rho`, the authoritative frozen endpoint, the
  state the previous audit's KKT analysis used.

`FROZEN_480_STATE_IDENTITY_PASS` requires: ρ₃₈₆ hash matches the record above,
config hash matches, `+impl` tree matches, outer = 386, stage = 3, move = 0.01,
N = 2, `multJ` = 0 at 386. Any mismatch ⇒ FAIL ⇒ STOP.

## 2. The subproblem to certify

The inner problem solved at outer iteration 386, built at ρ₃₈₅, with
`move = 0.01`, `lamref = λ₁(ρ₃₈₅)`, `Vtot = 0.5·NE`, `offDiag = true`, N = 2,
J = 3, `minInner = 5`, `maxInner = 500`, `tolInner = 0.05`, MMA constants
`a0 = 1, a = 0, c = 1000, d = 0`.

Variables `x = [drho(NE); bs]`, `nvar = NE+1`.
`xmin = [max(ρmin−ρ₃₈₅, −move); 0]`, `xmax = [min(1−ρ₃₈₅, +move); 5]`.
Objective `f0 = −bs`. Constraints, m = N+2 = 4:
rows 1..N `bs − (λ_j + Δλ_j)/lamref`; row N+1 `bs − (λ_J + f_JJ'drho)/lamref`;
row N+2 `(Σ(ρ+drho) − Vtot)/Vtot`.

**Reproduction requirement.** The reconstruction is accepted only if re-running
the frozen subproblem returns `drho` **bitwise equal** to `DRHO(:,386)`. If it
does not, the reconstruction is reported as failed and Part 3 is INCONCLUSIVE.

## 3. Dual provenance — three classes, never merged

* **retained exact dual** — none is expected; `innerLoop` discards `mmasub`'s
  duals with `~`. Recorded as absent if so.
* **reconstructed dual** — captured by re-running the identical `mmasub` calls
  on the identical inputs and keeping `lam, xsi, eta, mu, zet, s`. Exact for
  the MMA convex subproblem.
* **best-fit dual** — nonnegative least-squares fit, used only as an optimistic
  bound and always labelled as such.

## 4. Inner-KKT metrics (Part 3)

Two levels, reported separately.

**Level 1 — the last MMA convex subproblem.** Its own KKT: primal feasibility,
`lam ≥ 0`, `xsi,eta ≥ 0`, complementarity `lam_i·f̃_i`, `xsi_j(x_j−xmin_j)`,
`eta_j(xmax_j−x_j)`, and stationarity of the separable approximation.

**Level 2 — the true nonlinear inner problem (25) at the returned `drho`.**
L = f0 + Σ λ_i f_i on the box. Projected stationarity residual r_e: `∂L/∂x_e`
where interior, `min(∂L/∂x_e, 0)` at a lower bound, `max(∂L/∂x_e, 0)` at an
upper bound. Bound tolerance `1e-12` relative to the box width.

Normalizers, fixed now: drho block by `s_row = RMS_e(|ddlam(e,1)|/lamref)`;
`bs` block by 1.

**Verdict bars (Level 2, production terminal iterate):**

* `FINAL_INNER_MMA_KKT_PASS` — reproduction bitwise exact **and**
  `max fval ≤ 1e-8` **and** `min λ ≥ −1e-12` **and**
  `max|λ_i·f_i| ≤ 1e-6` **and** normalized projected stationarity RMS ≤ `1e-3`.
* `FINAL_INNER_MMA_KKT_FAIL` — normalized stationarity RMS ≥ `1e-1`, or primal
  infeasibility ≥ `1e-4`, or a negative dual beyond `−1e-8`.
* `FINAL_INNER_MMA_KKT_INCONCLUSIVE` — otherwise, or reproduction not exact.

**Corroborating certification re-solve.** The identical frozen subproblem is
re-solved with `tolInner = 1e-10` and `maxInner = 500`, `minInner = 5`
unchanged. Only the stopping point differs; objective, constraints, bounds, F,
λ, move and MMA constants are identical. Its `drho` is discarded. Reported as a
separate row, never as the production result.

## 5. Gradient objects (Part 4)

* `g_phys(ρ) = F(:,1,1)` from `genGrad` **before** filtering = ∂λ₁/∂ρ.
* `g_filt(ρ) = applyFilter(flt, ρ, g_phys)`, i.e. exactly
  `g_filt = (H·(ρ∘g_phys)) ./ (Hs∘max(1e-3,ρ))`.
  Since the solver clamps ρ ≥ ρmin = 1e-3, `max(1e-3,ρ) = ρ` on the admissible
  set, so `g_filt = A(ρ)·g_phys` with `A(ρ) = diag(1/(Hs∘ρ))·H·diag(ρ)`.
  The ρ-dependence of A is carried in **every** derivative test below.
* `grad F_reg` — the hypothetical scalar potential, never assumed to exist.

At `drho = 0` the active spectral row's gradient is exactly `−g_filt/lamref`
(`deltaLambda` returns the identity basis at A = 0), so `g_filt` is the field
that actually reaches MMA.

## 6. Jacobian symmetry test (Part 5)

`J = ∂g/∂ρ` probed by central differences, never assembled:
`Jv ≈ [g(ρ+δv) − g(ρ−δv)]/(2δ)`, with `‖u‖₂ = ‖v‖₂ = 1`.

Statistic per pair and per δ:
`r = |uᵀJv − vᵀJu| / max(|uᵀJv|, |vᵀJu|)`, plus the absolute
`a = |uᵀJv − vᵀJu|`.

`δ ∈ {1e-3, 3e-4, 1e-4, 3e-5}`, all four evaluated for every pair.

**Direction families, deterministic and fixed now** (all normalized; any
direction whose perturbation would leave `[ρmin, 1]` at the largest δ is
zeroed on the offending elements and renormalized, and that is recorded):

* `D1` low-frequency: `cos(πx/L)` and `cos(2πx/L)` element fields.
* `D2` gray-core localized: two disjoint Gaussian bumps inside the broad gray
  core (core = connected gray region as defined by the previous audit).
* `D3` interface: a bump centred on the element of maximal `|∇ρ|`.
* `D4` random: `randn` under `rng(0)` and `rng(1)`, MATLAB default generator.
* `D5` volume-neutral: `D1`, `D2`, `D4` members with their mean removed.

**Ten preregistered pairs:** (D1a,D1b), (D2a,D2b), (D1a,D2a), (D3,D2a),
(D4a,D4b), (D1a,D4a), (D2a,D4a), (D5-D1a,D5-D1b), (D5-D2a,D5-D4a), (D3,D4b).

Positive control: the identical test on `g_phys`. λ₁ is a scalar function, so
its gradient must be conservative wherever λ₁ is simple and smooth; the control
**must** pass or the whole test is void (stop condition).

## 7. Closed-loop line integrals (Part 6)

Rectangle `ρ → ρ+au → ρ+au+bv → ρ+bv → ρ`, each edge integrated by 8-point
Gauss–Legendre on `g·dρ`. Amplitudes `a = b ∈ {1e-3, 3e-4, 1e-4}`.
Both orientations. Same ten pairs as §6, and the same positive control on
`g_phys`.

Normalization: `|∮| / (a·b·max(|uᵀJv|,|vᵀJu|))` — by Stokes this tends to the
relative asymmetry of §6 for a small rectangle. Raw `|∮|` and its scaling with
`a` are also reported: a genuine curl gives `|∮| ∝ a²`.

## 8. Mixed partials (Part 7)

Element set S, deterministic: 8 broad-gray-core, 8 gray-shell, 6 solid-like
(ρ > 0.9), 6 void-like (ρ < 0.1), each chosen by rank order of a fixed
criterion, plus the 4 nearest neighbours of the first core element. For each
`j ∈ S` the full column `J(:,j)` is taken by central FD on ρ_j at
`δ_e = 1e-5`, giving the `|S|×|S|` submatrix. Reported: `J_ij − J_ji` and
`|J_ij − J_ji| / max(|J_ij|,|J_ji|)`, for both `g_phys` and `g_filt`.

## 9. Analytic prediction to be tested (Part 8)

With `g = g_phys`, `Hess = ∂²λ₁/∂ρ²`:

```
J_filt = A·D_{g/ρ} + A·Hess − diag(g_filt/ρ)
```

The last term is diagonal, hence symmetric. The first term has the **closed
form** skew part

```
skew₁(e,j) = ½·H_ej·[ g_j/(Hs_e·ρ_e) − g_e/(Hs_j·ρ_j) ]
```

computable with no finite differences. Preregistered test: for element pairs
with `H_ej ≠ 0`, the measured `J_ej − J_je` must agree with
`2·skew₁(e,j) + [A·Hess − (A·Hess)ᵀ]_ej`; and for pairs with `H_ej = 0` the
measured asymmetry must come entirely from the `A·Hess` term. Agreement is
reported, not assumed.

## 10. Volume constraint (Part 9)

The volume row gradient is `1/Vtot` on every element, independent of ρ. Adding
a constant vector field changes no Jacobian, hence no curl. This is proved in
`FILTER_OPERATOR_ANALYSIS.md` and verified numerically on one pair.

## 11. Multiplicity control (Part 10)

Every perturbed evaluation records ω₁…ω₅ and `gap12`. A sample is **excluded**
from integrability conclusions if `gap12 < 0.05` (the multiplicity tolerance)
or if mode ordering changes. At ρ₃₈₆ `gap12 = 0.1298`, so exclusions are not
expected; the count is reported either way. If more than 5 % of samples are
excluded at any δ, that δ is dropped and recorded as dropped.

## 12. Verdict rules

`FILTERED_FIELD_LOCALLY_NONCONSERVATIVE` requires **all** of:
1. median `r` over the ten pairs ≥ `0.05` at the two smallest δ;
2. `r` does **not** vanish under refinement — median `r` at the smallest δ is
   ≥ 0.5× the median at the largest δ;
3. positive control on `g_phys` gives median `r` ≤ `0.01` at the same δ;
4. median normalized loop integral ≥ `0.05`, with exact sign flip under
   orientation reversal and `|∮| ∝ a²` within a factor 2 over the amplitude
   range.

`FILTERED_FIELD_LOCALLY_CONSERVATIVE` requires median `r` ≤ `0.01` at the
smallest δ, decreasing with δ, and median normalized loop integral ≤ `0.01`.

Otherwise `FILTERED_FIELD_INTEGRABILITY_INCONCLUSIVE`.

`EFFECTIVE_SCALAR_OBJECTIVE_IDENTIFIED` requires a derivation or numerical
demonstration that `g_filt = ∇F` for an explicitly written `F`. Convenience is
not evidence; absent such a demonstration the verdict is
`EFFECTIVE_SCALAR_OBJECTIVE_NOT_IDENTIFIED`.

## 13. Stop conditions

Stop if: the 480 state cannot be identified; the final subproblem cannot be
reproduced bitwise; the `g_phys` positive control fails; FD estimates do not
converge enough to interpret; or perturbations cross mode-switching boundaries
uncontrollably.

## 14. Forbidden

No projection, no p-continuation, no filter change, no other mesh as a primary
case, no production repair, no density update, and no recommendation of
projection or p-continuation unless this audit produces new evidence
specifically supporting it.
