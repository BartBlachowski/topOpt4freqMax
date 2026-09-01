# OlhoffRegularized — implementation and natural-convergence audit

Audit of `analysis/OlhoffRegularized/`, focused on whether

```matlab
formulation = 'olhoff'
optimizer   = 'mma'
```

reaches **native** `CONVERGED` — under its own stopping logic, before
`max_outer_iterations` — at scientifically meaningful resolution, and whether
that survives mesh refinement.

`CAP_HIT` is not counted as success anywhere in this report.
`GLOBALIZATION_STALLED` is not counted as convergence anywhere in this report.

---

## 0. Provenance

| item | value |
|---|---|
| repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| branch | `benchmark-methodology-r2` |
| HEAD at audit start | `665eae1ef10490ecc83dc41b0e017f721118efb2` |
| working tree at audit start | dirty — `analysis/OlhoffRegularized/**` and `README.md` already modified; `docs/complexity*` and one `analysis/iteration_efficiency_final/runs/smoke` directory untracked |
| MATLAB | R2025b, `25.2.0.3042426` Update 1, maca64 |
| toolboxes used | Optimization Toolbox (`linprog`, `dual-simplex-highs`) |
| host | Darwin 25.5.0, 10 physical cores, 68.7 GB |
| implementation under audit | `analysis/OlhoffRegularized/Matlab/` (note: the files sit under `Matlab/`, not directly under `analysis/OlhoffRegularized/` as the brief anticipated) |
| frozen primitives | `Matlab/reproduction2007/{algo,fem,filter,mma,runner}` |

**The frozen historical reproduction was not modified.** `git status
Matlab/reproduction2007` is empty at the end of the audit; the primitives the
regularized code consumes are unchanged
(`genGrad.m` `f28581d4…`, `deltaLambda.m` `d17af5f9…`, `eigSolve.m` `f514e828…`,
`mmasub.m` `e18c9602…`). No frozen campaign evidence was touched and the
nine-resolution performance campaign was not re-run.

Audit-only material is isolated under `analysis/OlhoffRegularized/audit/`
(`scripts/`, `results/`, `logs/`). Nothing under `audit/` is production code.

### Run naming

| prefix | meaning |
|---|---|
| `ss…` | **as-shipped**: the implementation exactly as found at audit start |
| `C_…` | intermediate: certificate correction only, harness still passing `stationarity_tol = 0.02` (kept because it exhibits defect **CV-4** in its pure form) |
| `Rpre1b_…` | corrected, but made **before** defect CV-1b was found; retained as that correction's before/after evidence |
| `R_…` | **corrected**: every correction in force, defaults only |
| `DIAG_…` | deliberate diagnostic deviations, labelled in the saved metadata |

Only `ss…` and `R_…` rows are results. `C_…`, `Rpre1b_…` and `DIAG_…` are
evidence about the corrections and are labelled as such wherever they appear.

### Problem and mesh policy

All scientific runs are the simply-supported 8×1 beam unless stated, `volfrac`
0.5, `p` 3, `move` 0.005, mass law `4b`, density filter, `eigs` shift-invert.
The filter radius is held at a **constant physical value 0.075** (7.5 % of the
beam height) across meshes — 1.5 elements at 160×20, 2.25 at 240×30, 3.0 at
320×40 — because a mesh-refinement study at constant *element* radius refines
the problem, not just the discretisation. 1.5 elements at 160×20 is the
repository's own 160×20 setting (`run_regularized_fixed_pinned.m`). No mesh was
tuned separately.

Meshes below 160×20 appear only in `run_olhoff_regularized_tests.m`, which is
software mechanics (dispatch, accounting, controller forcing, classification)
and is never cited as evidence about convergence quality.

---

## 1. Executive verdict

> ### PARTIALLY VERIFIED

**As shipped, the answer to the final scientific question is NO.** The
implementation's `CONVERGED` status was reachable, and reached, at designs that
still possess a reproducible feasible physical ascent direction. Convergence was
produced by the move-limit controller shrinking its own steps until three
step-size proxies fell below tolerance, while the one scale-invariant guard was
set loose enough not to object. That is fixed-work termination wearing a
stationarity label.

**After four evidence-driven corrections, the answer is YES for the LP route
and — with the qualification recorded in §11 — for the MMA route**, with an
independently reproduced certificate and physical fixed-step verification at
160×20, 240×30 and 320×40.

The corrections make convergence strictly **harder**, or leave it unchanged;
none of them can manufacture a `CONVERGED`. No tolerance was loosened, no cap was raised to
obtain a stop, and no parameter was tuned per mesh.

---

## 2. Component verdicts

| component | verdict | basis |
|---|---|---|
| algorithmic implementation (FE, filter, gradients, volume, accounting) | **VERIFIED** | independent rebuild reproduces ρ and ω **bit-identically** (max abs diff `0.000e+00`) at every terminal design; filtered-volume chain rule is exact, not linearised; volume residual ≤ 5.5e-12 |
| MMA inner solve | **PARTIALLY VERIFIED** | never exhausts its cap (95–115 of 500 iterations) and stops on its own declared test — but returns **2.3–3.6 % below the exact optimum of the same subproblem** (§6.3). Under-solving biases the stationarity slope *downward*, i.e. towards false stationarity |
| globalization (trial eigensolve, accept/reject, trust adaptation) | **VERIFIED** | acceptance evaluates the physical trial design; accepted ratios 0.93–1.01; rejection-driven trust adaptation behaves correctly, and on its own (with the ceiling controller disabled) drives the criticality measure two orders of magnitude lower than the shipped controller allows |
| move-ceiling controller | **NOT VERIFIED as shipped / VERIFIED after correction** | as shipped it contracted the ceiling 10 accepted updates after the criticality measure first dipped below a loose threshold and froze the design there (defects **CV-2**, **CV-3**); at 240×30 and 320×40 the shipped tolerance split drove the ceiling onto `move_min` and produced a guaranteed `GLOBALIZATION_STALLED` (**CV-4**) |
| native convergence criterion | **NOT VERIFIED as shipped / VERIFIED after correction** | as shipped, three of its four conditions are step-size proxies the controller can zero out, and the fourth was calibrated 4.7× looser than the objective tolerance the same test declares (§6.2) |
| independent stationarity | **VERIFIED as a method; result is design-dependent** | the WP2 certificate reproduces the production certificate to 4 significant figures and is confirmed by physical fixed-step eigensolves with `actual/predicted → 0.999` as `t → 0` |
| mesh robustness | **VERIFIED for the corrected criterion on the LP route; see §11 for MMA** | corrected `olhoff/lp` converges naturally at 160×20 / 240×30 / 320×40 in 574 / 506 / 480 outer iterations |

---

## 3. Results

`max feasible ascent` is the **independent** WP2 quantity: the largest relative
gain in λ₁ that any feasible direction can deliver in one step of the reference
radius 5e-3, computed by a separate cutting-plane solver from a separately
rebuilt model, at the *exact* multiplicity. It is directly comparable with
`objective_tol = 1e-5`. "Independent stationarity certified?" means that number
is at or below `objective_tol` **and** the physical fixed-step check confirms the
first-order model (`actual/predicted → 1` as `t → 0`).

### 3.1 As shipped

| Mesh | Route | Status | Outer | Accepted | CAP? | Native CONVERGED? | Indep. stationarity certified? | max feasible ascent | ω₁ |
|---|---|---|---|---|---|---|---|---|---|
| 160×20 | Olhoff + LP | CONVERGED | 426 | 426 | no | yes | **NO** | 4.67e-05 (4.7× `objective_tol`) | 161.475960 |
| 160×20 | Olhoff + MMA | *(§9.1)* | | | | | | | |
| 160×20 | KS + LP | CONVERGED | 595 | 595 | no | yes | **NO** | 3.37e-05 at the clustered N=2, **2.93e-04 (29× `objective_tol`) at the true N=1** | 161.419478 |
| 160×20 | KS + MMA | *(§9.1)* | | | | | | | |
| 240×30 | Olhoff + MMA | *(§9.1)* | | | | | | | |
| 320×40 | Olhoff + MMA | *(§9.1)* | | | | | | | |

### 3.2 After correction

| Mesh | Route | Status | Outer | Accepted | CAP? | Native CONVERGED? | Indep. stationarity certified? | max feasible ascent | ω₁ |
|---|---|---|---|---|---|---|---|---|---|
| 160×20 | Olhoff + LP | CONVERGED | 574 | 574 | no | yes | **YES** | 8.17e-06 | 161.619285 |
| 240×30 | Olhoff + LP | CONVERGED | 506 | 506 | no | yes | **YES** | 6.49e-06 | 163.001140 |
| 320×40 | Olhoff + LP | CONVERGED | 480 | 480 | no | yes | **YES** | 7.33e-06 | 162.272761 |
| 160×20 | KS + LP | **CAP_HIT** | 3000 | 2999 | yes | no | n/a — correctly refused | 2.64e-04 | 161.493510 |
| 160×20 | Olhoff + MMA | *(§9.2)* | | | | | | | |
| 240×30 | Olhoff + MMA | *(§9.2)* | | | | | | | |
| 320×40 | Olhoff + MMA | *(§9.2)* | | | | | | | |

### 3.3 Diagnostic runs (deliberate deviations, not results)

| tag | deviation | Status | Outer | ω₁ | terminal criticality slope |
|---|---|---|---|---|---|
| `DIAG_nocontract_olhoff_lp` | move-ceiling contraction disabled (`progress_tolerance = 0`) | CAP_HIT @3000 | 3000 | **161.884840** | 8.12e-04 (min over run 8.09e-05) |
| `DIAG_tightstat_olhoff_lp` | `stationarity_tol = 2e-3` only | CONVERGED | 574 | 161.619285 | 1.638e-03 |
| `C_ss240x30_olhoff_lp` | certificate in force, `stationarity_tol = 0.02` still passed by the harness | **GLOBALIZATION_STALLED** | 589 | 162.814309 | Inf (all trials failed at `move_min`) |
| `C_ss320x40_olhoff_lp` | same | **GLOBALIZATION_STALLED** | 571 | 162.093030 | Inf |

---

## 4. WP0 — implementation trace

### 4.1 The `olhoff`/`mma` path, end to end

`topopt_olhoff_regularized.m`

1. installs the frozen path via `repro2007_paths()` and asserts implementation
   identity (`repro2007_assert_identity`) — a run that starts has proved which
   implementation it is executing;
2. seeds `cfg` from `repro2007_config('fig3a_best')`, then overrides
   `massInterp → '4b'` (C¹, so a trial step of any size has a meaningful
   prediction; the historical Eq. (4) law is discontinuous at ρ = 0.1 and is
   warned about), `filterMode → 'density'`, and the iteration limits;
3. `localModel` builds the mesh, `edofMat`, `iK/jK`, `K0/M0` and the boundary
   DOFs, including the cantilever concentrated mass added on the *reduced* free
   DOFs after assembly;
4. per outer iteration: `localModes` → `assemble2D` + `eigSolve('eigs')` for
   `Jcalc = n + Nmax = 5` modes, M-orthonormalised;
5. `localMultiplicity(w,n,Nmax,tolMult=0.05)` → `N`, `J = n+N`;
6. `localOlhoffGradients` → `genGrad` for the cluster with `λ̃ = mean(λ_cluster)`
   and for the simple mode `J`, then the **density-filter chain rule**
   `H' (f ./ Hs)`;
7. trial loop (≤ 8): `localOlhoffMma` solves problem (25) with
   `offDiag = true`, i.e. the full Eq. (25d) sub-eigenvalue model through
   `deltaLambda`, over the box `[max(ρmin−x, −trust), min(1−x, trust)]` with the
   filtered volume row;
8. the trial design is **physically evaluated**: `xTrial → ρTrial = H xTrial ./ Hs
   → assemble → eigensolve → λ₁(trial)`;
9. `ratio = actual/predicted`; accept iff `predicted > 0 ∧ actual ≥ 0 ∧ ratio ≥ 0.10`;
10. trust grows only on an accurate boundary step (`ratio ≥ 0.75 ∧ dxInf ≥ 0.8·trust`)
    and only up to `moveCeiling`; shrinks on `ratio < 0.25` or rejection;
11. the persistent move ceiling contracts on the stage-exhaustion gate;
12. native convergence test; `CAP_HIT` is assigned only after the loop, so it can
    never collide with `CONVERGED`.

### 4.2 Verified

* **Off-diagonal terms are live for `olhoff`/`mma` and only there.**
  `cfg.offDiag = strcmp(optimizer,'mma') && strcmp(formulation,'olhoff')`;
  `localOlhoffMma` calls `deltaLambda(ctx.F, drho)` on every inner iterate when
  set, and `localValidate` hard-errors if the Eq. (16)/(22) LP route is asked for
  `offDiag`. Now asserted per route in the regression suite (group 1).
  At `N = 1` the coupled model degenerates to the diagonal one by construction —
  the routes are therefore only genuinely distinct from the first bimodal
  iteration (outer ≈ 111 at 160×20).
* **Volume feasibility is represented exactly, not linearised.** With the
  density filter, `Σρ = (1./Hs)' H x` is *linear in the design variable*, so
  `volumeWeights = H'(1./Hs)` is not a first-order approximation: the LP row is
  exact. Measured terminal volume residual `Σρ − volfrac·NE`: 2.3e-13 (160×20),
  2.3e-12 (240×30), 5.5e-12 (320×40); `mean(ρ) = 0.5000000000` in every run.
* **Filtering and its chain rule are consistent.** `ρ = H x ./ Hs`,
  `dλ/dx = H'(dλ/dρ ./ Hs)`, and the same map is applied to `f_sk`, to `f_JJ`
  and to the volume row. The independent rebuild in §6.1 reproduces ρ from the
  saved design variable with max abs difference `0.000e+00`.
* **Trial acceptance evaluates the physical trial design** (step 8 above), not a
  surrogate.
* **Accounting.** `rejected_trials = trial_total − accepted_updates`;
  `trial_total ≤ maxOuter · max_trial_steps`; inner iterations are summed over
  trials. Asserted in the suite.
* **`trust ≤ moveCeiling` always**, and **`moveCeiling` is monotone
  non-increasing** — asserted over the full history in the suite, and confirmed
  over every scientific run.
* **No `CAP_HIT` path can become `CONVERGED`.** Asserted structurally
  (`status == CAP_HIT ⇒ max(convergenceCount) < persistence`).
* `slowProgressCount` is computed, logged and **never read** — dead telemetry,
  harmless.

### 4.3 Defect F-1 (fixed) — runner overwrote the caller's route

`run_regularized_fixed_pinned.m` advertised an externally selectable optimizer
and then discarded it two lines later:

```matlab
 5: if ~exist('optimizer','var')||isempty(optimizer),optimizer="lp";end   % advertised
 …
 9: bcType="fixedPinned";
10: optimizer="mma";                                                       % overwrote it
```

`optimizer = "lp"; run_regularized_fixed_pinned` silently ran MMA. Fixed by
deleting line 10 and moving the runner's *actual* policy (MMA) into the default
on line 5, so the effective numerical policy is unchanged and the override now
works. Regression: suite group 11 statically rejects any unconditional
assignment to `optimizer`/`formulation` in any runner and requires all three to
keep advertising the override.

---

## 5. WP1 — audit of the native convergence criterion

### 5.1 What the shipped implementation computes

Per outer iteration *k*, with `λ = λ_n(ρ_k)` the current first eigenvalue:

```
predicted      = β* − λ                          β* = optimum of the local subproblem
                                                      over the box of radius `trust`
predScale      = max(|λ|, 1)
predSlope      = max(predicted, 0) / (predScale · max(trust, eps))
dxInf          = ‖ρ_{k+1} − ρ_k‖_∞
dxRms          = rms(ρ_{k+1} − ρ_k)
relObjective   = |λ_n(ρ_{k+1}) − λ| / predScale
stationarityOK = predSlope ≤ stationarity_tol                     (shipped: 2e-2)

convergenceCount++  iff   accepted ∧ dxInf ≤ 1e-4 ∧ dxRms ≤ 1e-5
                                   ∧ relObjective ≤ 1e-5 ∧ stationarityOK
                    or    ¬accepted ∧ trust ≤ move_min ∧ stationarityOK
                    else reset to 0

CONVERGED           iff   convergenceCount ≥ persistence           (shipped: 20)
```

Because β is the maximised lower bound on the whole set
{λ_n … λ_{n+N−1}, λ_J}, `predicted` is the local model's best attainable increase
of the *minimum* eigenvalue — the right thing to differentiate.

### 5.2 The eight questions

**1. Is it dimensionless?** Yes. `predicted` and `predScale` both carry the units
of λ; `trust` is a density increment, dimensionless. `predSlope` is a pure
number.

**2. Does it remain meaningful as trust → 0?** Yes. Over a box of radius *r*
intersected with the design bounds and the volume half-space, the local optimum
is positively homogeneous of degree 1 in *r* once *r* is small enough that the
bounds no longer clip, so `predicted(r) = r·χ + o(r)` and `predSlope → χ/λ`, a
**non-zero constant at a non-stationary point**. Measured at the 160×20
as-shipped terminal design: `predSlope = 9.340266e-3` at r = 5e-3 and
`9.340269e-3` at r = 1.953e-5 — six significant figures over a 256× change of
radius.

**3. Can shrinking trust manufacture apparent stationarity?** *The measure*: no
— (2) is exactly the property that prevents it, and it is confirmed numerically.
*The algorithm as shipped*: **yes, indirectly**, because the other three
convergence conditions are step-size proxies and the controller shrinks steps
deliberately (see 8 and §7). The correction removes even the indirect route by
evaluating the certificate at a fixed reference radius.

**4. Can `max(predicted,0)` hide model failure?** Yes, structurally. `drho = 0`
is always feasible with `β = λ_n`, so a *solved* subproblem cannot return
`predicted < 0`; a negative value means the local solve failed. The clip maps
that to `predSlope = 0` — "perfectly stationary" — which is the worst available
interpretation. The LP route is protected by `st.lpFlag == 1`; the MMA route is
not, because `mmasub` penalises constraints (c = 1000) rather than enforcing
them, so its returned `bs` is neither guaranteed feasible nor optimal.
Corrected by counting and logging every occurrence
(`info.iterations.negative_predictions`). **Measured incidence across every run
in this audit: 0.** The hazard is real but was not triggered.

**5. Is it a legitimate first-order stationarity measure?** Yes, subject to 7.
It is the ∞-norm-scaled criticality measure

```
χ(x; r) = max { D λ_n(x)[d] : ‖d‖_∞ ≤ r,  ρmin ≤ x+d ≤ 1,  w'd ≤ V* − V(x) }
```

normalised by `λ·r`. `χ = 0` iff no feasible first-order ascent exists, which is
precisely first-order stationarity for `max λ_n` over the bound-and-volume set.

**6. How does the volume constraint enter?** As one linear row inside the same
subproblem, `w'd ≤ V* − V(x)` with `w = H'(1./Hs)`. Because the filtered volume
is *linear* in the design variable, this row is exact rather than linearised.
At every terminal design the volume is at the cap (residual ≤ 5.5e-12), so the
row is active and the certificate measures constrained ascent, not free ascent.
Independently reproduced.

**7. At a multiple eigenvalue, is the right sub-eigenvalue problem represented?**
Two distinct issues, one benign here and one a genuine defect.

*(a) Within a degenerate cluster.* `D λ_min[d] = λ_min(A(d))` with
`A_sk = f_sk'd`. The MMA route models this exactly (`deltaLambda`); the LP route
imposes Eq. (22) `f_sk'd = 0` and maximises `min_j f_jj'd`, a restriction whose
optimum is a lower bound. **Measured**: at every terminal design examined the
unrestricted cutting-plane optimum and the Eq. (22)-restricted LP optimum agree
to twelve digits, and the optimal `A(d)` is diagonal with equal diagonal entries
to ≈1e-13 (e.g. `[50.2546, 1.1e-13; 1.1e-13, 50.2546]`). The restriction is not
binding at these designs; full off-diagonal information is *available* on the MMA
route but does not change the answer.

*(b) Which modes form the cluster.* As shipped the certificate inherits
`tolMult = 0.05`, a 5 % tolerance chosen — rightly — to make the *step* model
robust. For a *stationarity certificate* it is wrong: as `t → 0` only the exactly
degenerate modes constrain `D λ₁`, and folding a strictly separated mode into the
cluster adds constraints that do not exist in the limit, **understating**
criticality. Measured on the `ks/lp` terminal design
(ω₁ = 161.4195, ω₂ = 162.4845, gap 0.66 %):

| cluster assumed | χ/(λ·r) at r = 5e-3 | best relative λ₁ gain in one move-limit step | physical `actual/predicted` at t = 1e-5 |
|---|---|---|---|
| N = 2 (shipped, `tolMult = 0.05`) | 6.74e-3 — **passes** the shipped 2e-2 | 3.37e-05 | 0.935 |
| N = 1 (exact, gap ≫ 1e-6) | **5.88e-2** — 2.9× **above** the shipped tolerance | 2.93e-04 | **1.0000** |

The physical fixed-step check decides in favour of N = 1. This is defect
**CV-1**.

**8. Could convergence be declared simply because accepted steps become tiny?**
**Yes — and that is exactly what the shipped implementation does.** Three of the
four conditions (`dxInf`, `dxRms`, `relObjective`) are monotone in the step size,
and `dxInf ≤ trust` identically, so the move-ceiling controller can satisfy all
three by contracting the ceiling. The only guard that survives step shrinkage is
`predSlope ≤ stationarity_tol`, and at 2e-2 it did not object. §7 shows the
trajectory.

---

## 6. WP2 — independent stationarity certificate

### 6.1 Method

`audit/scripts/audit_stationarity.m` never reads the production convergence
boolean. It:

* **rebuilds the model from the problem definition** (`audit_model.m`,
  `audit_assemble.m`) rather than calling the production `localModel`, which is a
  private nested function — and then *checks* the rebuild: ρ from the saved
  design variable and ω from the rebuilt model match the production values with
  max abs difference `0.000e+00` in every case;
* **re-solves the eigenproblem twice**, once with the production start vector and
  once with a different start vector and a doubled Krylov basis; agreement
  1.5e-15 – 4.7e-15 relative;
* **determines multiplicity independently** at three tolerances (1e-3, 1e-2,
  5e-2) and reports the raw gaps;
* **builds the first-order model independently** — `genGrad` with
  `λ̃ = mean(λ_cluster)`, then the filter chain rule — for N = 1, 2, 3 so that a
  wrong multiplicity interpretation cannot pass unnoticed;
* **solves the directional-ascent problem with its own solver**
  (`audit_ascent.m`): Kelley cutting planes on the sub-eigenvector `v` with an LP
  master,

  ```
  max_d  t    s.t.  t ≤ Σ_sk v_s v_k (f_sk' d)  for each generated cut v
                    w' d ≤ V* − V(x)
                    max(ρmin−x, −r) ≤ d ≤ min(1−x, r)
  ```

  separating by `λ_min(A(d))`. It uses only `linprog` and `eig` on an N×N
  matrix — no `deltaLambda`, no MMA, no production stopping logic. The reported
  rate is the **true** `λ_min(A(d))` at the returned `d`, not the master bound;
* **runs a control** (`audit_ascent_lp.m`) that imposes Eq. (22) instead, so the
  effect of the off-diagonal information is measured rather than assumed;
* **normalisation, stated explicitly**: `d` ranges over the design box
  translated to the origin (`x + d ∈ [ρmin,1]^NE`) intersected with an ∞-ball of
  radius `r` and the volume half-space. Because that set is convex and contains
  0, the path `x + t·d` is feasible for every `t ∈ [0,1]`, and the volume stays
  feasible since `w'(t d) = t (w'd) ≤ 0`. Rates are reported at
  r ∈ {1 (whole box), 5e-3 (the move limit), final trust};
* **verifies physically**: for `t ∈ {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5}`
  it forms `x + t·d`, filters, assembles, and re-solves the eigenproblem,
  reporting predicted Δλ, **actual Δλ₁ on the ordered spectrum**, actual Δω₁,
  the ratio, `mean(ρ)`, and the multiplicity after the step. The ordered-spectrum
  reading is what makes an incorrect multiplicity interpretation detectable.

### 6.2 Result — the shipped `CONVERGED` designs are not stationary

`ss160x20_olhoff_lp`, status `CONVERGED` at outer 426, ω₁ = 161.475960:

* exact multiplicity N = 2 (relative gap 7.5e-11) — the production clustering
  agrees here;
* maximum feasible ascent, unrestricted, at the move limit r = 5e-3:
  **Δλ = 1.2177**, i.e. **4.67e-05 relative** — *4.7× the `objective_tol = 1e-5`
  that the same convergence test declares*;
* physical fixed-step verification along that direction:

| t | predicted Δλ | actual Δλ₁ | actual Δω₁ | actual/predicted |
|---|---|---|---|---|
| 1e-2 | 5.025459e-01 | 3.581791e-01 | +1.10907e-03 | 0.7127 |
| 3e-3 | 1.507638e-01 | 1.378288e-01 | +4.26778e-04 | 0.9142 |
| 1e-3 | 5.025459e-02 | 4.881883e-02 | +1.51164e-04 | 0.9714 |
| 1e-4 | 5.025459e-03 | 5.011198e-03 | +1.55169e-05 | 0.9972 |
| 1e-5 | 5.025459e-04 | 5.024738e-04 | +1.55588e-06 | **0.9999** |

`mean(ρ) = 0.500000000` at every step — the ascent is feasible.

**A design labelled `CONVERGED` still possesses a reproducible feasible physical
ascent direction. Native convergence is therefore not certified for the shipped
criterion.** The residual is small in absolute terms (Δω₁ ≈ +1.1e-3 rad/s at
t = 1e-2, 7e-6 relative), but it is real, reproducible, and above the tolerance
the criterion itself declares.

The same holds for `ss160x20_ks_lp` (§5.2 question 7b), where the residual is an
order of magnitude larger: 2.93e-04 relative, `actual/predicted = 1.0000`.

### 6.3 Controls that make the certificate trustworthy

* **Wrong multiplicity is caught.** At the truly degenerate `olhoff/lp` design,
  assuming N = 1 predicts Δλ = 1.518e+03 but the physical step *lowers* λ₁
  (`actual/predicted = −1.18`, stable across five decades of t). Assuming N = 3
  likewise gives a spurious ascent that is physically a descent (−1.59). Only the
  correct N = 2 model reproduces the physics. Conversely at the near-degenerate
  `ks/lp` design only N = 1 reproduces it.
* **The off-diagonal information is measured, not assumed.** Unrestricted vs.
  Eq. (22)-restricted rates agree to twelve digits at every terminal design.
* **The eigensolve is cross-checked** with an independent start vector.
* **The production certificate was validated against this one** after the
  correction, on four terminal designs, from separate code paths, separate model
  builds and different solvers (production: Eq. (22) LP; audit: cutting plane):

| design | production `certificateSlope` | independent WP2 | agreement |
|---|---|---|---|
| `R_ss160x20_olhoff_lp` (N = 2) | 1.634559e-03 | 1.634828e-03 | 4 s.f. |
| `R_ss240x30_olhoff_lp` (N = 2) | 1.298430e-03 | 1.298572e-03 | 4 s.f. |
| `R_ss320x40_olhoff_lp` (N = 2) | 1.464139e-03 | 1.465001e-03 | 4 s.f. |
| `R_ss160x20_ks_lp` (N = 1) | 5.281016e-02 | 5.281018e-02 | **7 s.f.** |

  The N = 2 rows differ in the fourth figure because the production certificate
  additionally carries the Eq. (25b) constraint on the simple mode `J`, which the
  audit's directional problem omits; at N = 1 the two problems coincide exactly
  and so do the answers.

---

## 7. WP4 — why convergence was not reached, diagnosed from the trajectory

The classification below uses the causes listed in the audit brief.

### 7.1 The shipped trajectory, `ss160x20_olhoff_lp`

| outer | ω₁ | trust | ceiling | ratio | predSlope | dxInf | relObj | convCount |
|---|---|---|---|---|---|---|---|---|
| 1 | 69.156 | 5.00e-03 | 5.00e-03 | 1.007 | 4.42e+00 | 5.00e-03 | 2.23e-02 | 0 |
| 101 | 145.850 | 5.00e-03 | 5.00e-03 | 0.984 | 7.00e-01 | 5.00e-03 | 3.45e-03 | 0 |
| 201 | 160.153 | 5.00e-03 | 5.00e-03 | 0.991 | 9.90e-02 | 4.67e-03 | 4.91e-04 | 0 |
| 251 | 161.087 | 5.00e-03 | 5.00e-03 | 0.963 | 2.26e-02 | 4.24e-03 | 1.09e-04 | 0 |
| **277** | — | 5.00e-03 | 5.00e-03 | — | **first value ≤ 2e-2** | — | — | 0 |
| **286** | 161.373 | 5.00e-03 | 5.00e-03 | — | 1.83e-02 | — | — | **first ceiling contraction** |
| 351 | 161.468 | 3.13e-04 | 3.13e-04 | 0.995 | 1.03e-02 | 2.46e-04 | 3.21e-06 | 0 |
| 401 | 161.475 | 7.81e-05 | 7.81e-05 | 0.999 | 9.67e-03 | 5.08e-05 | 7.55e-07 | rising |
| 426 | 161.476 | 3.91e-05 | 3.91e-05 | 0.999 | 9.35e-03 | 2.54e-05 | 3.65e-07 | **20 → CONVERGED** |

Ceiling contractions fired at outer 286, 306, 326, 346, 366, 386, 406, 426 —
**exactly every 20 accepted updates**, i.e. as fast as the dwell counter allows,
from the moment the criticality measure first dipped below the threshold.

Reading the causes from the brief against this evidence:

* **(A) material topology change** — no: `dxInf` 2.5e-05 at the end.
* **(B) objective still improving materially** — no: `relObj` 3.6e-07.
* **(C) inner MMA fails** — not applicable on the LP route; see §7.3 for MMA.
* **(D) trust pinned at its ceiling** — **yes, by design**: `move_max` defaults
  to `move`, so trust starts at the ceiling and can never grow. Every accepted
  step during the ascent phase saturates the box (`dxInf ≈ 0.996·trust`). This
  is disclosed in the README and is not itself a defect, but it means the *only*
  mechanism that can produce small steps is the ceiling controller.
* **(E) ceiling fails to contract** — no, the opposite.
* **(F) ceiling contracts too slowly** — no, the opposite: **it contracts too
  eagerly.**
* **(G) internally inconsistent thresholds** — **yes**: the run stops at a state
  whose own local model predicts a step gaining 4.67e-05 relative, against an
  `objective_tol` of 1e-05 in the same test.
* **(H) predicted-slope stationarity remains large** — **yes, and it plateaus**:
  9.35e-3 at termination, never approaching zero.
* **(I) multiplicity switching** — no: N = 2 stably from outer 111.
* **(J) perpetual small accepted steps** — **yes**, and they are what satisfies
  the stopping test.
* **(K) filter-driven motion** — no evidence.
* **(L) objective converged but state-change test cannot** — the inverse.

### 7.2 The controlled experiment that isolates the cause

`DIAG_nocontract_olhoff_lp` — identical in every respect except that the
stage-exhaustion contraction is disabled (`progress_tolerance = 0`), cap raised
to 3000 purely so the trajectory can be observed:

| | shipped (contraction on) | contraction disabled |
|---|---|---|
| status | CONVERGED @426 | CAP_HIT @3000 |
| ω₁ | 161.475960 | **161.884840** |
| terminal criticality slope | 9.35e-03 | 8.12e-04 (minimum over the run **8.09e-05**) |
| terminal trust | 3.91e-05 (ceiling-driven) | 3.13e-04 (**rejection-driven**, ceiling untouched at 5e-3) |
| ceiling contractions | 8 | 0 |

Two things follow.

1. **The move-ceiling controller is the proximate cause of premature
   termination.** Left alone, the ordinary rejection-driven trust adaptation
   reduces the criticality measure by two orders of magnitude and finds a
   *better* design. The controller freezes it 115× higher.
2. **The certificate was self-fulfilling.** Because the same threshold gated both
   the ceiling contraction and the convergence declaration, the controller shut
   the optimizer down as soon as the threshold was first met — so the terminal
   criticality is always found just *under* the threshold, whatever the threshold
   is. Confirmed by re-running with a 10× tighter threshold
   (`DIAG_tightstat_olhoff_lp`): terminal slope 1.638e-3, again just under its
   2e-3 threshold. Two thresholds, two plateaus, each immediately below its own
   gate.

That is the mechanism by which "convergence" was manufactured, and it is what
made this look like a passing stationarity test rather than fixed-work
termination.

3. **The criticality measure does not go to zero on this problem.** With the
   controller disabled it falls to ~8e-5 by outer ~1600 and then *rises* again,
   wandering in a band of roughly 1e-4 – 3e-3 for the remaining 1400 iterations
   while ω₁ gains a further 0.01 rad/s. The terminal region is a shallow,
   non-smooth bimodal ridge, not a point where exact stationarity is attained.
   Any honest stopping rule for this problem is therefore a *tolerance* statement,
   not an assertion of exact stationarity — which is precisely why the tolerance
   must be tied to something meaningful.

### 7.3 The MMA inner solve

Over the 23 pre-contraction outer iterations of `R_ss160x20_olhoff_mma` where
`trust == certificate_radius == 5e-3` and `N = 1` — so the nested MMA and the
certificate LP are solving *the same* convex subproblem — the two optima can be
compared directly:

| | value |
|---|---|
| inner iterations per outer | 95 – 115 (median 103), cap 500, **cap hit 0 times** |
| inner solve reported converged | every accepted trial |
| MMA optimum / exact LP optimum | 0.964 – 0.977, **median 0.972** |

So: the nested MMA **does** converge on its own declared criterion and never
merely exhausts its cap — but it lands **2.3 – 3.6 % short of the exact optimum
of its own subproblem**. The direction of that error matters: under-solving
*understates* `predicted`, hence understates `predSlope`, hence biases the
stopping test **towards false stationarity**. This is why the corrected
certificate takes `max(certSlope, predSlope)` rather than trusting the route's
own model slope.

---

## 8. Corrections

All three make convergence strictly harder. None can produce a `CONVERGED` that
the shipped code would have refused.

### CV-1 — the certificate used the step model's clustering tolerance

*Mathematics.* The directional derivative of the ordered `λ_n` as `t → 0` is the
smallest eigenvalue of the sub-eigenvalue matrix over the **exactly** degenerate
cluster. A strictly separated mode folded into the cluster contributes a
constraint that does not exist in the limit, so the maximin value is a lower
bound on the true criticality — the certificate under-reports.

*Before.* `ks/lp` 160×20 terminal design: certificate 6.74e-3 (N = 2 by
`tolMult = 0.05`) → passes 2e-2 → `CONVERGED` at outer 595. True N = 1
criticality 5.86e-2; physical `actual/predicted = 1.0000` at t = 1e-5. (The
shipped run reported 6.79e-3 for its own slope at its terminal trust radius; the
values quoted here are the independent audit's, evaluated at r = 5e-3.)

*After.* A separate `certificate_mult_tol` (default 1e-6) seeds the certificate's
cluster. The same run now reports certificate 5.281e-2 throughout and correctly
**refuses** to converge (`CAP_HIT` at 3000). Regression: suite group 9 forces
`tol_mult = 3.0` so the step model clusters four modes and asserts
`certificateN == 1`.

### CV-1b — …but a strictly exact cluster over-reports near a degeneracy

*Found by running the correction on a second problem*, which is why WP8 is worth
doing before declaring a fix good.

*Symptom.* On the fixed–pinned 160×20 beam the first two modes converge to a
gap of 3.9e-6 relative — small, but above `certificate_mult_tol`. The
strictly-exact rule therefore used N = 1, and the resulting certificate ran
7–27× above the route's own model slope (outer 500: route 1.09e-02 vs
certificate 2.96e-01; outer 1000: 8.81e-03 vs 6.29e-02), blocking both
contraction and convergence for over a thousand iterations.

*Mathematics.* The certificate is a **finite-step** statement — "no feasible step
of the reference radius gains more than `objective_tol`". At that radius the
N = 1 model predicted a gain of 60 in λ against a gap to λ₂ of 0.53: it was
predicting λ₁ rising 113× past λ₂, which is not a model of the *ordered* λ₁ over
that step, because λ₂ would have become the minimum and its coupling to λ₁ is
then first-order. Asymptotic exactness (`t → 0`) is the wrong criterion for a
finite-radius certificate.

*After.* The cluster is a **fixed point**: seeded at the exact multiplicity, then
grown while

```
λ_{n+Nc} − λ_n  <  predicted gain of the current cluster
```

capped at `Nmax`. This introduces no new constant, collapses to the exact
multiplicity when the gaps are large relative to the attainable gain (which is
the `ks/lp` case CV-1 was about), and grows to the sub-eigenvalue model when they
are not (the fixed–pinned case). Regression: suite group 9 asserts the fixed
point — `predicted gain ≤ gap to the first excluded mode`, unless the cluster has
reached `Nmax` or the LP failed — over whole runs, on two boundary conditions and
at reference radii three orders of magnitude apart.

*Cost of the discovery.* Every corrected run was repeated. The pre-CV-1b results
are retained under `audit/results/Rpre1b_*` and are not cited as results.

### CV-2 — the certificate was evaluated at the controller's own trust radius

*Mathematics.* A stopping test must not be a function of the variable the
controller is free to drive to zero. `predSlope` is scale-invariant in exact
arithmetic, so this is not fatal on its own — but combined with CV-3 it is what
lets the controller close the loop on itself.

*After.* `localCertificate` evaluates at a fixed reference radius
`certificate_radius` (default `move_max`), computed once per outer iteration and
never following `trust` or `moveCeiling`. Regression: suite group 8 asserts
radius-scale invariance to 1e-6 relative across a 10× radius change, and asserts
`certificateRelativeGain == certificateSlope · certRadius` over a whole run in
which the trust radius varies.

### CV-3 — the stationarity threshold was uncalibrated

*Mathematics.* `predSlope · certificate_radius` **is** the local model's best
*relative* gain in λ_n for one step of the reference radius. The only threshold
consistent with the objective tolerance the same test declares is therefore

```
stationarity_tol = objective_tol / certificate_radius      = 1e-5/5e-3 = 2e-3
```

which reads: *no feasible step within the configured move limit improves λ_n by
more than `objective_tol`.* This is a derivation from tolerances the
configuration already contains, not a new tuned number.

*Before.* 2e-2 — admitting a terminal design whose own model predicts a
4.67e-05 relative gain, 4.7× `objective_tol`, independently confirmed physically.

*After.* Derived default; an explicitly looser request is clamped, with a
warning, for **both** the convergence declaration and the controller.
Regression: suite group 7.

### CV-4 — the contraction gate and the convergence gate could disagree

*Discovered from evidence.* The first correction attempt left the caller's
`stationarity_tol` in force for the ceiling controller while clamping only the
convergence declaration. At 240×30 and 320×40 the result was a **guaranteed
stall**: the controller contracted 16 times on the loose 2e-2, collapsing the
ceiling onto `move_min = 1e-7`, at which radius the LP subproblem became
numerically unsolvable — every trial rejected, `predSlope = Inf`,
`GLOBALIZATION_STALLED` at outer 589 and 571 with the convergence test never once
satisfiable (`convergenceCount = 0` for the entire run).

*After.* One clamped tolerance for both. The same 240×30 configuration now
converges naturally at outer 506. Regression: suite group 7 asserts both the
clamp and, behaviourally, that a loose request produces **zero** contractions at
a point the convergence test rejects.

### F-1 — runner discarded the caller's optimizer

See §4.3.

### Files changed

| file | change |
|---|---|
| `Matlab/topopt_olhoff_regularized.m` | `localCertificate` (new), `localFilterGradients` (extracted), derived + clamped tolerance, `statMeasure = max(certSlope, predSlope)` used by every stationarity decision, negative-prediction counter, five new history fields, one new verbose column |
| `Matlab/run_regularized_fixed_pinned.m` | removed the unconditional `optimizer="mma"`; default moved to the guard line (policy unchanged) |
| `Matlab/run_regularized_{simply_supported,fixed_pinned,cantilever}.m` | removed the explicit `'stationarity_tol',.02` so the derived default applies |
| `Matlab/run_olhoff_regularized_tests.m` | 1 group → 11 groups (§12) |
| `README.md` | documents the certificate, the derivation of the threshold, and the audit |

Pre-correction copies of every file the audit touched are kept under
`audit/results/*.BEFORE`.

---
