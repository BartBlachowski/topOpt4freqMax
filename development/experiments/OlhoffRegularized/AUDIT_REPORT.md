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

**The final scientific question — does `formulation='olhoff', optimizer='mma'`
naturally converge to a defensible stationary topology before its iteration cap,
and does that survive mesh refinement — is answered NO, twice over, for two
different reasons.**

*As shipped*, it reports `CONVERGED` at 160×20 and the result is false. The
terminal state is a globalization stall — every trial rejected, trust on
`move_min` — that was relabelled as convergence because the nested MMA returned a
**negative** predicted improvement and `max(predicted,0)` reported that failure
as perfect stationarity (defect **CV-5**). The design sits **4.0 % below** the ω₁
the LP route reaches on the identical problem, and carries an independently
certified, physically confirmed feasible ascent worth **96×** the objective
tolerance the same test declares.

*After correction*, it honestly reports `CAP_HIT` at 160×20 **and** at 240×30 —
and that is the right answer, because the route is genuinely nowhere near
stationary after 1000 iterations. The cause is **not** the stopping rule: it is
that the mode pair never coalesces. The design parks at a mode gap of
4.3–4.8e-02, immediately beneath `tol_mult = 0.05`, at every mesh tested. The
deficit **grows** with refinement — residual ascent 86× the objective tolerance
at 160×20 and 260× at 240×30, with ω₁ falling from 4.0 % to 9.6 % below the LP
route — so mesh refinement makes this route worse, not better.

**All four routes produce a false `CONVERGED` as shipped** — Olhoff+LP, Olhoff+MMA,
KS+LP and KS+MMA — with residual ascents of 4.7×, 96×, 29× and 44× the objective
tolerance, every one confirmed by physical fixed-step eigensolves with
`actual/predicted → 1.0000`.

**What does work.** After five corrections, `formulation='olhoff', optimizer='lp'`
converges naturally and is independently certified at **160×20, 240×30 and
320×40** (574 / 506 / 480 outer iterations), and generalises to the fixed–pinned
beam (1567) and the cantilever with concentrated mass (2961) without any
per-problem tuning. That is a defensible, mesh-robust, general convergence
mechanism.

Every correction makes convergence strictly **harder**, or leaves it unchanged;
none can manufacture a `CONVERGED`. No tolerance was loosened, no cap raised to
obtain a stop, no parameter tuned per mesh, and no capped or stalled run
suppressed.

Three candidate mitigations for the MMA route were tested; two were refuted and
one only confirmed the mechanism without recovering the objective (§9.3.1). The
evidence points at the **off-diagonal Eq. (25d) coupling applied to a
near-but-not-exactly degenerate pair** — the three routes that reach ω₁ ≈ 161.6
are exactly the three that never form `f_sk` for s ≠ k. That is a statement about
the formulation, not the stopping rule, and this audit does not attempt to
correct it.

## 2. Component verdicts

| component | verdict | basis |
|---|---|---|
| algorithmic implementation (FE, filter, gradients, volume, accounting) | **VERIFIED** | independent rebuild reproduces ρ and ω **bit-identically** (max abs diff `0.000e+00`) at every terminal design; filtered-volume chain rule is exact, not linearised; volume residual ≤ 5.5e-12 |
| MMA inner solve | **NOT VERIFIED** | it never exhausts its cap (82–418 of 500 over 312 outer iterations) and stops on its own declared test, so it is not merely running out of budget — but it returns **2.3–3.6 % below the exact optimum of the same subproblem** (§7.3), and in the terminal region of the 160×20 run it returned a **negative** predicted improvement, which is impossible for a solved subproblem. Under-solving biases the stationarity slope *downward*, towards false stationarity; the negative case triggered defect **CV-5** in production |
| globalization (trial eigensolve, accept/reject, trust adaptation) | **VERIFIED** | acceptance evaluates the physical trial design; accepted ratios 0.93–1.01; rejection-driven trust adaptation behaves correctly, and on its own (with the ceiling controller disabled) drives the criticality measure two orders of magnitude lower than the shipped controller allows |
| move-ceiling controller | **NOT VERIFIED as shipped / VERIFIED after correction** | as shipped it contracted the ceiling 10 accepted updates after the criticality measure first dipped below a loose threshold and froze the design there (**CV-2**, **CV-3**). Separately, **CV-4** was found in the audit's own first correction attempt, not in the shipped code: leaving the caller's looser tolerance in force for the controller while clamping only the convergence test drove the ceiling onto `move_min` at 240×30 and 320×40 and produced a guaranteed `GLOBALIZATION_STALLED`. Both gates now use one clamped tolerance |
| native convergence criterion | **NOT VERIFIED as shipped / VERIFIED after correction** | as shipped, three of its four conditions are step-size proxies the controller can zero out; the fourth was calibrated 4.7× looser than the objective tolerance the same test declares (§6.2) **and could be satisfied outright by a failed local model** (CV-5), which is how the primary `olhoff/mma` route reached a false `CONVERGED` 4.0 % below the LP optimum |
| independent stationarity | **VERIFIED as a method; result is design-dependent** | the WP2 certificate reproduces the production certificate to 4 significant figures and is confirmed by physical fixed-step eigensolves with `actual/predicted → 0.999` as `t → 0` |
| mesh robustness | **VERIFIED for the corrected criterion on the LP route; §9.5 for MMA** | corrected `olhoff/lp` converges naturally at 160×20 / 240×30 / 320×40 in 574 / 506 / 480 outer iterations |

---

## 3. Results

`max feasible ascent` is the **independent** WP2 quantity: the largest relative
gain in λ₁ that any feasible direction can deliver in one step of the reference
radius 5e-3, computed by a separate cutting-plane solver from a separately
rebuilt model, at the *exact* multiplicity. It is directly comparable with
`objective_tol = 1e-5`. "Independent stationarity certified?" means that number
is at or below `objective_tol` **and** the physical fixed-step check confirms the
first-order model (`actual/predicted → 1` as `t → 0`).

### 3.1 As shipped — all four routes declare CONVERGED, none is certified

| Mesh | Route | Status | Outer | Accepted | CAP? | Native CONVERGED? | Indep. stationarity certified? | max feasible ascent | ω₁ |
|---|---|---|---|---|---|---|---|---|---|
| 160×20 | Olhoff + LP | CONVERGED | 426 | 426 | no | yes | **NO** (act/pred 0.9999) | 4.67e-05 — 4.7× `objective_tol` | 161.475960 |
| 160×20 | **Olhoff + MMA** | CONVERGED | 682 | 664 | no | yes | **NO** (act/pred 1.0000) | **9.61e-04 — 96×** | **155.277719** |
| 160×20 | KS + LP | CONVERGED | 595 | 595 | no | yes | **NO** (act/pred 1.0000) | 2.93e-04 — 29× | 161.419478 |
| 160×20 | KS + MMA | CONVERGED | 856 | 856 | no | yes | **NO** (act/pred 1.0000) | 4.39e-04 — 44× | 161.095079 |

Every one of the four terminal designs retains a feasible first-order ascent
direction that the independent audit reproduces and that fixed-*t* eigensolves
confirm, with `actual/predicted → 1.0000` as `t → 0` in every case. The primary
route is the worst: its `CONVERGED` design sits **4.0 % below** the ω₁ the LP
route reaches on the identical problem, and it got there through defect
**CV-5** — a failed local model read as perfect stationarity (§8).

The as-shipped mesh sweep for `olhoff/mma` at 240×30 and 320×40 was started and
then suspended in favour of the corrected runs; it is not reported. The
as-shipped defect is established at 160×20 on all four routes, and at all three
meshes on the LP route (§3.3 and §7).

### 3.2 After correction

| Mesh | Route | Status | Outer | Accepted | CAP? | Native CONVERGED? | Indep. stationarity certified? | max feasible ascent | ω₁ |
|---|---|---|---|---|---|---|---|---|---|
| 160×20 | Olhoff + LP | CONVERGED | 574 | 574 | no | yes | **YES** (act/pred 0.9993) | 8.17e-06 | 161.619285 |
| 240×30 | Olhoff + LP | CONVERGED | 506 | 506 | no | yes | **YES** (act/pred 0.9997) | 6.49e-06 | 163.001140 |
| 320×40 | Olhoff + LP | CONVERGED | 480 | 480 | no | yes | **YES** (act/pred 0.9990) | 7.33e-06 | 162.272761 |
| 160×20 | **Olhoff + MMA** | **CAP_HIT** | 1000 | 1000 | **yes** | **no** | n/a — correctly refused | 8.57e-04 — 86× | 156.051175 |
| 240×30 | Olhoff + MMA | **CAP_HIT** | 1000 | 1000 | **yes** | **no** | n/a — correctly refused | 2.60e-03 — 260× | 147.427773 |
| 320×40 | Olhoff + MMA | stopped @590 | — | — | — | — | not reached | — | 145.823 |
| 160×20 | KS + LP | **CAP_HIT** | 3000 | 2999 | **yes** | no | n/a — correctly refused | 2.64e-04 — 26× | 161.493510 |
| 160×20 | KS + MMA | **CAP_HIT** | 1000 | 1000 | **yes** | no | n/a — correctly refused | 3.32e-04 — 33× | **161.963354** |
| 160×20 | Fixed–pinned, Olhoff + LP | CONVERGED | 1567 | 1567 | no | yes | **YES** (act/pred 0.9936) | 2.35e-06 | 258.148376 |
| 120×80 | Cantilever + mass, Olhoff + LP | CONVERGED | 2961 | 2961 | no | yes | **YES** (act/pred 0.9999) | 9.01e-06 | 104.152802 |

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
**This is not hypothetical: it fired, in production, on the primary
`olhoff`/`mma` route, during this audit's own WP3 run** — see defect **CV-5**
in §8. From outer 665 the run sat with every trial rejected, trust at
`move_min`, and the nested MMA returning a negative predicted improvement; the
clip reported `predSlope = 0`, the *rejected* branch of the convergence test
began counting, and the run declared `CONVERGED` at ω₁ = 155.278 — 4.0 % below
what the LP route reaches on the same problem. Corrected by failing closed:
a negative predicted improvement is counted, logged, and sets the stationarity
measure to `+Inf`. `predicted == 0` is left alone, since that is a model
genuinely finding no ascent.

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

*Measured effect on these runs: none.* Every corrected run was repeated under the
fixed point, and each reproduced its pre-CV-1b result exactly — `olhoff/lp` at
160×20 converged at outer 574 with ω₁ = 161.619285 both times, at 240×30 at
outer 506 with 163.001140, `ks/lp` at 3000 with 161.493510, fixed–pinned at 1567
with 258.148376. On the fixed–pinned run the certificate value itself differed on
**1130 of 1567 iterations, by up to 108×** (outer 200: 1.01e-01 corrected vs
3.73e-01 before; outer 380: 1.63e-02 vs 1.04e-01), yet ω, trust and the move
ceiling are bit-identical throughout — because at every iteration where the two
disagreed, both were on the failing side of the threshold. The defect was real
and large; it simply never reached a decision here. It would reach one on any
design whose route slope sits below tolerance while a spurious certificate sits
above it, which is exactly what a near-degenerate converged design looks like.

The pre-CV-1b results are retained under `audit/results/Rpre1b_*` as this
correction's before/after evidence and are not cited as results.

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

### CV-5 — a failed local model was read as perfect stationarity

*Found in production, on the primary route, during the audit's own WP3 run.*
Until it fired, this defect was a structural argument (§5.2 question 4) with a
measured incidence of zero.

*Mathematics.* `drho = 0` with `beta = lambda_n` is feasible for problem (25), so
a **solved** subproblem cannot return `predicted < 0`. A negative value means the
local solve failed — and `mmasub` can produce one, because it penalises its
constraints (`c = 1000`) rather than enforcing them, so its returned `beta` is
neither guaranteed feasible nor optimal. The shipped code wrote

```matlab
predSlope = max(predicted,0)/(predScale*max(trust,eps));
```

so a failed model was reported as `predSlope = 0` — **perfect stationarity**, the
worst available reading of the evidence.

*What it did.* On the as-shipped `olhoff/mma` 160×20 run, from outer 665 onward:
every one of the 8 trial steps rejected, trust collapsed onto
`move_min = 1e-7`, the nested MMA returning a negative predicted improvement
(1576 inner iterations per outer, so the solver was working, not stalling), and

```
665   155.278 ... 1.00e-07  6.25e-04   8  1576   no   0   0.00e+00  0.00e+00
```

`ratio = 0`, `max|drho| = 0`, `predSlope = 0`. That state is the definition of
`GLOBALIZATION_STALLED` — minimum trust radius, no accepted step. The clip
turned it into `stationarityOK = true`, so the *rejected* branch

```matlab
if trust<=reg.moveMin*(1+1e-12) && stationarityOK
    convergenceCount=convergenceCount+1;
```

began counting, and 20 iterations later the run declared **`CONVERGED`**.

*Where.* ω₁ = 155.278 — **4.0 % below** the 161.619 the LP route reaches on the
same problem, and frozen since outer 654. Not a marginal miss.

*After.* Fail closed. A negative predicted improvement is counted, logged, and
sets the stationarity measure to `+Inf`:

```matlab
if predicted<0
    negativePredictions = negativePredictions+1;
    eventLog{end+1} = ...;
    predSlope = Inf;          % an unusable model is not evidence of stationarity
else
    predSlope = predicted/(predScale*max(trust,eps));
end
```

`predicted == 0` is left alone — that is a model genuinely finding no ascent, and
is legitimate. With the fix the same state classifies as
`GLOBALIZATION_STALLED`, which is what it is.

*Independently, the certificate already blocked it.* The corrected run on the
same mesh and route reports a certificate of 1.99e-01 in that region — a hundred
times the 2e-03 threshold — because the certificate is computed by its own LP
and does not inherit the MMA's failure. The two defences are independent, which
is the point: `statMeasure = max(certSlope, predSlope)`.

*Control.* Same problem, same route, same start. At outer 610 the corrected run
stood at ω₁ = 155.286, already past the value at which the as-shipped run had
frozen, and still advancing.

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

## 9. WP3 / WP5 — the primary MMA route

### 9.1 As shipped: a false CONVERGED, 4.0 % below the LP optimum

```
tag=ss160x20_olhoff_mma  status=CONVERGED  stop=persistent_stationarity_and_state_change
outer=682/1000  accepted=664  rejected=150  inner=134376  contractions=3
w1=155.277719  w2=162.162410  N=2  trust=1.000e-07  ceil=6.250e-04
dxInf=0  dxRms=0  relObj=0  slope=0.000e+00  wall=14340 s
```

The terminal state is not a stationary point; it is a stall wearing a
convergence label. From outer 665 onward, every one of the 8 trial steps is
rejected, the trust radius sits on `move_min = 1e-7`, and the nested MMA returns
`predicted = -2.3484e-03` — **negative**, on all 18 of those iterations. It is
not out of budget: 441 inner iterations at most per trial against a cap of 500,
**zero** per-trial cap hits. It converged on its own criterion and returned a
value a solved subproblem cannot produce. `max(predicted,0)` reported that as
`predSlope = 0`; the *rejected* branch of the stopping test began counting; 20
iterations later the run declared `CONVERGED`. This is defect **CV-5** (§8).

Independent WP2 audit of that terminal design:

| quantity | value |
|---|---|
| ω₁ / ω₂, exact multiplicity | 155.2777 / 162.1624, gap 4.43 % → **N = 1** |
| independent max feasible ascent, one step of the move limit | **9.61e-04 relative — 96× `objective_tol`** |
| physical fixed-*t*, t = 1e-5 | actual/predicted = **1.0000**, Δω₁ = +5.48e-05 |
| physical fixed-*t*, t = 1e-3 | 0.9958, Δω₁ = +5.46e-03 |
| physical fixed-*t*, t = 1e-2 | 0.9587, **Δω₁ = +0.0525** |
| physical fixed-*t*, t = 3e-2 | 0.8783, **Δω₁ = +0.144** |
| volume along the ascent | feasible at every step (`mean(ρ)` ≤ 0.4999996) |
| ω₁ vs `olhoff/lp` on the identical problem | 155.278 vs **161.619** — 4.0 % low |
| grayness | 0.3666, against 0.2621 for the LP terminal design |

**WP3 primary pass condition: FAILED.** The status is `CONVERGED` and
`outer < max_outer_iterations`, but the independent stationarity audit refutes
it decisively.

### 9.2 Corrected: an honest CAP_HIT

```
tag=R_ss160x20_olhoff_mma  status=CAP_HIT  stop=maximum_outer_iterations
outer=1000/1000  accepted=1000  rejected=0  inner=146332  contractions=0
w1=156.051175  trust=5.000e-03 (never left its ceiling)
dxInf=1.417e-03  dxRms=6.799e-05  relObj=2.082e-05
route slope=5.070e-03  certificate=1.722e-01  negative_predictions=0
```

Reported as **CAP_HIT**. The WP4 classification follows from the run's own
telemetry and is unambiguous — causes **(A)** and **(B)**:

* material topology change remains significant: `dxInf` = 1.4e-03, three orders
  of magnitude above `density_tol`;
* the objective is still improving materially: `relObj` = 2.08e-05, above the
  `objective_tol` = 1e-05 the test declares;
* nothing else is wrong: 0 rejected trials, 0 ceiling contractions, trust never
  left its ceiling, 0 negative predictions, 0 per-trial inner-cap hits.

The criterion is behaving correctly. The route is simply nowhere near stationary
after 1000 iterations. Independent WP2 agrees: max feasible ascent **8.57e-04
relative, 86× `objective_tol`**, `actual/predicted = 1.0000` at t = 1e-5 and
Δω₁ = +0.047 at t = 1e-2. Production certificate 1.7220e-01 against the
independent 1.7139e-01.

`negative_predictions = 0` for this run, so the CV-5 hardening — added after it
started — is provably a no-op here; the run is bit-identical under the hardened
code.

### 9.3 Why: the mode gap never closes

This is the substantive algorithmic finding of the audit, and it is **not** a
defect of the stopping rule.

| outer | `olhoff/lp` gap (ω₂−ω₁)/ω₁ | step N | `olhoff/mma` gap | step N |
|---|---|---|---|---|
| 100 | 8.02e-02 | 1 | 6.00e-02 | 1 |
| 250 | **1.17e-06** | 2 | 4.32e-02 | 2 |
| 400 | 1.08e-06 | 2 | 4.39e-02 | 2 |
| 550 | **6.19e-10** | 2 | 4.40e-02 | 2 |
| 850 | — (converged @574) | | 4.69e-02 | 2 |
| 1000 | — | | **4.84e-02** | 2 |

The LP route drives the pair to genuine coincidence and finds the bimodal
optimum. The MMA route parks at a gap of 4.3–4.8e-02 and never escapes — the gap
*increases* monotonically from outer 250 onward.

The mechanism: as soon as the gap falls below `tol_mult = 0.05` the step model
declares the pair a cluster, and `genGrad` then builds **every** generalized
gradient in that cluster — the diagonal `f_jj` included — with
`λ̃ = mean(λ₁, λ₂)`. That substitution is exact for a genuine degeneracy and
wrong for a 4.8 % gap. The coupled Eq. (25d) model is thereafter solving a
degenerate problem that is not the one in front of it, and the design settles
just under the clustering threshold: a self-sustaining pseudo-degeneracy. The LP
route escapes it because the Eq. (22) equality constraints force
`f₁₂'drho = 0`, which decouples the pair and lets λ₁ rise on its own.

This is cause **(I)** from the WP4 list — multiplicity handling preventing
stable convergence.

The same plateau appears at every mesh, which makes it a property of the
clustering tolerance rather than of a discretisation: the terminal gap is
4.84e-02 at 160×20, 4.44e-02 at 240×30 and 4.64e-02 at 320×40, all sitting
immediately beneath `tol_mult = 0.05`.

### 9.3.1 Three mitigations tested; two refuted, one partial

The diagnosis above names the clustered model. Three ways to escape it were
tested, and the report records what each actually did rather than what was hoped.

**(a) Tighten the clustering tolerance** (`DIAG_tolmult_mma_ss160x20`,
`tol_mult` 0.05 → 0.005). The gap does close, by an order of magnitude — 4.46e-03
against the baseline's 4.4e-02 — which confirms the mechanism. But ω₁ at outer
633 is 155.741 against the baseline's ~155.35 at the same point: a marginal gain,
nowhere near the LP route's 161.62. **Partial: it confirms the mechanism but does
not recover the objective.** And the gap simply re-parks just under the *new*
tolerance (1.6e-03 – 4.7e-03 against a 5e-03 gate) — the identical
threshold-plateau signature that CV-3 exhibits in the stopping rule, now in the
step model. Stopped at outer 739.

**(b) Tighten the inner tolerance** (`DIAG_tolinner_probe_ss160x20`,
`tol_inner` 1e-2 → 1e-3), aimed at the 2.8 % subproblem under-solve:

| | `tol_inner = 1e-2` (shipped) | `tol_inner = 1e-3` |
|---|---|---|
| MMA optimum / exact LP optimum | 0.972 | 0.991 |
| inner iterations per outer | 99–115 | 373–482 (cap 500) |

It halves the shortfall at **4× the cost**, saturating against
`max_inner_iterations`, and a 0.95 % shortfall still compounds over a thousand
iterations. **Refuted as a practical mitigation.** Answered in 8 outer
iterations; stopped there.

**(c) Per-mode λ̃ on the diagonal gradients** (`DIAG_permode_mma_ss160x20`,
`cluster_lambda = 'per_mode'`). Eq. (19) needs a common λ̃ only for the
off-diagonal `f_sk`; the diagonal `f_jj` belongs to one mode and its own λ_j is
the correct value. Since the KS route — which builds every gradient per mode —
reaches the LP optimum with the same optimizer, this looked like the carrier of
the deficit. It is not:

| outer | `per_mode` ω₁ / gap | `mean` ω₁ / gap |
|---|---|---|
| 110 | 142.240 / 4.5044e-02 | 142.238 / 4.5185e-02 |
| 120 | 143.992 / 4.4110e-02 | 144.006 / 4.4460e-02 |
| 140 | 146.045 / 4.3330e-02 | 146.172 / 4.3790e-02 |
| 160 | 147.414 / 4.2840e-02 | 147.529 / 4.3300e-02 |

The gap is ~1 % smaller and ω₁ is very slightly **lower**; both differences are
negligible and the trajectories run parallel. **Refuted**, on four matched
iterations spanning 50 outer steps past the cluster onset; stopped at outer 176. The
arithmetic says why, and should have been done before the run: swapping the
cluster mean for λ_j moves λ̃ by half the λ-gap, ≈4.5 %, and only inside the mass
term of `f_jj = p·ρ^(p−1)·φᵀK₀φ − λ̃·g'(ρ)·φᵀM₀φ` — a few per cent perturbation
that the maximin structure absorbs. The option is retained (default `'mean'`,
unchanged) with a regression test, because it is a legitimate control, not
because it helps.

**What the evidence actually points at.** The three routes that reach ω₁ ≈ 161.6
— `olhoff/lp`, `ks/lp`, `ks/mma` — are exactly the three that never use the
off-diagonal terms `f_sk` (s ≠ k): the LP routes force `f_sk'drho = 0` by
Eq. (22), and the KS routes never form them. The single route that stalls at
156.05 is the only one that feeds the full Eq. (25d) coupling into its
subproblem. On this evidence the deficit is carried by the off-diagonal
sub-eigenvalue coupling itself when it is applied to a pair that is near- but
not exactly degenerate — not by the diagonal λ̃, and not by the inner tolerance.

That is a statement about the **formulation**, not about the stopping rule, and
this audit does not attempt to correct it: the off-diagonal coupling is what
`formulation='olhoff', optimizer='mma'` *is*. Removing it would turn the primary
route into the LP route. It is recorded as the principal open item (§14).

It also explains the certificate plateau. Over the corrected run the certificate
falls 2.13e+00 → 4.18e-01 → 2.57e-01 → 1.72e-01 at outer 100 / 200 / 500 / 1000,
i.e. it has been flat within 5 % since outer 700. Geometric extrapolation of the
500→1000 decay puts the 2e-03 threshold at roughly outer 6600 — against **406**
for the LP route on the identical problem — and the flatness after 700 makes even
that optimistic. On the evidence, the corrected `olhoff/mma` route does not
approach stationarity on this problem at any tractable iteration count while
`tol_mult = 0.05`.

### 9.4 Cost

| | `olhoff/lp` | `olhoff/mma` |
|---|---|---|
| status @160×20 | CONVERGED @574 | CAP_HIT @1000 |
| ω₁ | **161.619285** | 156.051175 |
| certificate | 1.63e-03 (certified) | 1.72e-01 |
| local solves | 574 LP | 146 332 MMA iterations |
| wall clock | 90 s | 15 478 s — **172×** |

The nested MMA's extra cost buys neither a better terminal objective nor a
better stationarity certificate on this problem. Both effects have the same
origin: it solves its own subproblem to 2.3–3.6 % below the optimum (§7.3), and
it applies the degenerate-cluster model to a pair that is not degenerate (§9.3).


### 9.5 WP5 — mesh robustness

| mesh | route | status | outer | ω₁ | terminal mode gap | certificate | indep. residual ascent |
|---|---|---|---|---|---|---|---|
| 160×20 | Olhoff + LP | **CONVERGED** | 574 | 161.619285 | 5.53e-11 | 1.63e-03 | **8.17e-06** ✓ |
| 240×30 | Olhoff + LP | **CONVERGED** | 506 | 163.001140 | 4.69e-11 | 1.30e-03 | **6.49e-06** ✓ |
| 320×40 | Olhoff + LP | **CONVERGED** | 480 | 162.272761 | 1.06e-10 | 1.47e-03 | **7.33e-06** ✓ |
| 160×20 | Olhoff + MMA | CAP_HIT | 1000 | 156.051175 | 4.84e-02 | 1.72e-01 | 8.57e-04 |
| 240×30 | Olhoff + MMA | **CAP_HIT** | 1000 | 147.427773 | 4.46e-02 | 5.20e-01 | 2.60e-03 |
| 320×40 | Olhoff + MMA | stopped @590 | — | 145.823 | 4.64e-02 | 5.55e-01 | — |

**The corrected criterion is mesh robust on the LP route.** Natural convergence
at all three meshes in 574 / 506 / 480 outer iterations — note the count
*decreases* with refinement — each independently certified with a residual
ascent below `objective_tol` and confirmed by physical fixed-step eigensolves
(`act/pred` 0.9993 / 0.9997 / 0.9990). ω₁ agrees to 0.9 % across a 4× change in
element count at constant physical filter radius. Volume residual ≤ 5.5e-12 and
`mean(ρ) = 0.5000000000` throughout every run.

**The MMA route's failure is equally mesh robust, and is not a stopping-rule
defect.** The terminal mode gap parks at 4.84e-02 (160×20) and 4.46e-02
(240×30) — immediately beneath `tol_mult = 0.05` — with certificates of
1.72e-01 and 5.20e-01. The 240×30 run is a completed `CAP_HIT` at outer 1000
with ω₁ = 147.428, an independently certified residual ascent of **2.60e-03 —
260× `objective_tol`** (`act/pred = 1.0000` at t = 1e-5, Δω₁ = +0.133 at
t = 1e-2), and a grayness of 0.5666 against the LP route's 0.2365 at the same
mesh: the design is not merely short of stationarity, it is still substantially
grey. The 320×40 run was stopped at outer 590 once the same plateau (4.64e-02)
and certificate (5.55e-01) were established; its partial trajectory is retained
and it is **not** reported as a completed run.

Note the deficit *grows* with refinement: the residual ascent is 86× the
objective tolerance at 160×20 and 260× at 240×30, and ω₁ falls further behind the
LP route (156.05 vs 161.62 at 160×20; 147.43 vs 163.00 at 240×30 — 4.0 % then
9.6 % low). Refining the mesh makes the MMA route worse, not better.

A convergence rule that works at 160×20 and systematically fails at 240×30 or
320×40 would not be satisfactory. That is not what happens here: the rule is
uniform across meshes, and what varies is only whether the *route* can reach a
design the rule will certify.

## 10. WP6 — four-route control experiment at 160×20

Identical FE problem (simply supported 8×1, 160×20, `rmin` 1.5 elements,
`volfrac` 0.5, `p` 3, `move` 0.005), identical regularization controls. This is
diagnostic, not a contest.

### 10.1 Do the routes reach the same physical basin?

**The LP-solved routes do. The MMA-solved routes do not.** Pairwise terminal
filtered-density comparison, simply supported 160×20 only
(`audit/scripts/audit_compare.m`; PNGs beside each result directory):

| A | B | mean abs Δρ | Pearson r |
|---|---|---|---|
| `ss/olhoff/lp` | `ss/ks/lp` | 0.0131 | **0.99594** |
| `ss/olhoff/lp` | `R_ss/olhoff/lp` | 0.0149 | **0.99579** |
| `ss/olhoff/lp` | `DIAG_nocontract` | 0.0421 | **0.95772** |
| `R_ss/olhoff/lp` | `DIAG_nocontract` | 0.0303 | **0.96873** |
| `ss/olhoff/mma` | `R_ss/olhoff/mma` | 0.0121 | 0.99794 |
| `ss/ks/mma` | `R_ss/ks/mma` | 0.0262 | 0.98977 |
| `ss/olhoff/lp` | `ss/olhoff/mma` | 0.1236 | **0.85307** |
| `R_ss/olhoff/lp` | `R_ss/olhoff/mma` | 0.1293 | **0.83947** |
| `R_ss/olhoff/lp` | `R_ss/ks/mma` | 0.1429 | **0.80315** |
| `ss/olhoff/mma` | `ss/ks/mma` | 0.1357 | 0.85068 |

Every LP-family design — both formulations, both criteria, and the diagnostic
with the controller disabled — is the same topology to r ≥ 0.956. The
MMA-family designs sit at r ≈ 0.80–0.86 from them, and are visibly different.

Two different reasons, which must not be conflated:

* `olhoff/mma` is simply **unconverged**. Its ω₁ is 4.0 % low and its mode gap
  has not closed; it is behind on the same trajectory, not somewhere else. The
  0.998 correlation between its as-shipped and corrected runs confirms the two
  code versions track each other exactly — it is the *route* that lags.
* `ks/mma` is a **genuinely different design**. It attains ω₁ = 161.963, higher
  than the certified LP design's 161.619, at r = 0.803 from it, with 40× more
  residual feasible ascent. Optimising a smooth aggregate of the lowest modes
  selects a different point.

**Correction of record.** An earlier reading of this comparison, taken before the
MMA runs completed, covered only the LP-family designs and concluded that all
routes reach the same basin. That is wrong once MMA is included. What survives is
the narrower claim the audit actually needs, which is about the *criterion* and
not the route: on the LP route, the shipped criterion and the no-contraction
diagnostic reach the same topology (r = 0.958), so the shipped criterion stopped
early **on the same optimum** rather than finding a different one — which is what
rules out "it stopped early but found something equally good".

### 10.2 Does full off-diagonal multiple-eigenvalue information change the result?

**Not materially, at these designs.** At every terminal design examined, the
unrestricted directional-ascent optimum (cutting plane on the sub-eigenvector)
and the Eq. (22) equality-restricted optimum agree to twelve digits, and the
optimal `A(d)` comes out diagonal with equal diagonal entries to ≈1e-13. The
maximin direction naturally decouples the pair, so the Krog & Olhoff restriction
is not binding there.

The two routes are in any case only distinguishable from the first bimodal
iteration (outer ≈ 111 at 160×20): while `N = 1` the coupled Eq. (25d) model and
the diagonal model are the same model, and `deltaLambda` on a 1×1 matrix returns
`f₁₁'d`.

### 10.3 Does the extra cost of MMA buy anything?

At 160×20, corrected, on the same problem:

| route | status | ω₁ | certificate on λ₁ | indep. residual ascent | local solves | wall |
|---|---|---|---|---|---|---|
| Olhoff + LP | **CONVERGED @574** | 161.619285 | **1.63e-03** | **8.17e-06** | 574 LP | 90 s |
| Olhoff + MMA | CAP_HIT @1000 | 156.051175 | 1.72e-01 | 8.57e-04 | 146 332 MMA its | 15 478 s |
| KS + LP | CAP_HIT @3000 | 161.493510 | 5.28e-02 | 2.64e-04 | 3 013 LP | 455 s |
| KS + MMA | CAP_HIT @1000 | **161.963354** | 6.55e-02 | 3.32e-04 | 89 421 MMA its | 11 958 s |

**No.** On the genuine Olhoff formulation the nested MMA costs 172× the wall
time of the LP route and returns a worse objective and a certificate two orders
of magnitude worse.

The table also contains the audit's sharpest warning against reading stationarity
off an objective value: **`ks/mma` attains the highest ω₁ of any run — 0.21 %
above the certified LP design — while carrying 40× more residual feasible
ascent** (3.32e-04 vs 8.17e-06, physically confirmed at `act/pred = 0.9809`,
Δω₁ = +0.0182 at t = 1e-2). A design can be better on the objective and further
from stationarity at the same time. This is why the audit never treats a high
eigenfrequency or a plausible topology as evidence.

### 10.4 Does KS smoothing change the local optimum?

**It changes where the optimizer stops, not which basin it is in.** The KS
terminal design has ω₁ = 161.419, ω₂ = 162.484 — a 0.66 % gap, where the Olhoff
routes drive the pair to numerical coincidence (gap 5e-11). That is the expected
effect of maximising a smooth lower aggregate of the lowest modes rather than the
minimum itself: the aggregate is maximised slightly before the modes coalesce.

The consequence for this audit is sharp, and it appears on both KS routes. The
as-shipped `ks/lp` terminal design is `CONVERGED` for its own aggregate while
carrying a **2.93e-04 relative feasible ascent in the physical λ₁**, physically
confirmed (`actual/predicted = 1.0000` at t = 1e-5, Δω₁ = +2.58e-05; +0.0251
rad/s at t = 1e-2). The as-shipped `ks/mma` design is likewise `CONVERGED` with
4.39e-04 (44× `objective_tol`, `act/pred = 1.0000`).

The corrected criterion separates the two objectives explicitly. On `ks/mma` at
160×20 the route's own aggregate slope is a comfortable **3.37e-03** while the
certificate on the physical λ₁ reads **6.55e-02** — a factor of 19 apart, in the
same iteration. Optimising the aggregate is not optimising the eigenfrequency,
and the certificate is what says so. Both corrected KS runs consequently
`CAP_HIT` rather than certify.

**A KS `CONVERGED` is not an eigenfrequency-stationarity result.** That is the
concrete reason the KS route must never be reported as an Olhoff reproduction.

### 10.5 Volume feasibility, LP vs MMA

The LP route enforces the filtered-volume row exactly; `mmasub` treats it as a
`c = 1000` penalty, so it had to be measured rather than assumed
(`audit/scripts/audit_volume_mma.m`, 160×20, 12 outer iterations):

| route | `mean(ρ)` | worst drift | sign |
|---|---|---|---|
| `olhoff/lp` | 0.5000000000 over the whole run | ≤ 5.5e-12 (residual) | — |
| `olhoff/mma` | 0.4999994 – 0.4999995 | 7.60e-07 | always **below** the cap |

The nested MMA is therefore slightly conservative on volume and never
infeasible. Because the filtered volume is *linear* in the design variable,
neither route incurs a linearisation error here — the difference is purely the
penalty vs. the hard row.

## 11. WP8 — does the mechanism generalise beyond the simply supported beam?

| problem | mesh | route | status | outer | ω₁ | indep. residual ascent | physical act/pred |
|---|---|---|---|---|---|---|---|
| simply supported 8×1 | 160×20 | Olhoff + LP | CONVERGED | 574 | 161.619285 | 8.17e-06 | 0.9993 |
| **fixed–pinned 8×1** | 160×20 | Olhoff + LP | **CONVERGED** | 1567 | 258.148376 | **2.35e-06** | 0.9936 |
| **cantilever 15×10 + tip mass** | 120×80 | Olhoff + LP | **CONVERGED** | 2961 | 104.152802 | **9.01e-06** | 0.9999 |

**Yes.** The corrected criterion is a general mechanism, not a
simply-supported-beam-specific fix. Both additional problems converge naturally
and are independently certified below `objective_tol`, with the physical
fixed-step check confirming the first-order model. Nothing was tuned per problem:
every run used the same derived defaults.

Two observations worth recording:

* **The iteration count is strongly problem-dependent**: 574 / 1567 / 2961. The
  cantilever needed 2961 of a 3000 cap. Any frozen production cap must be set
  from the problem, not from a convention — and the fixed–pinned runner's
  existing 1600 would have been just sufficient for its own problem on the LP
  route (converged at 1567), which is closer to the edge than is comfortable.
* **The cantilever optimum is simple, not bimodal** (ω₁ = 104.15, ω₂ = 153.86,
  N = 1) — the concentrated tip mass breaks the degeneracy the beams develop.
  The certificate handles both regimes with the same code path, which is the
  self-consistent-cluster rule (CV-1b) doing its job.

The fixed–pinned problem on the **MMA** route was started (`R_cs160x20_olhoff_mma`,
cap 2000) and suspended at outer 607 in favour of the simply-supported scientific
runs; it is not reported. Its trajectory to that point shows the same clustered
plateau as the simply-supported case.

## 12. WP7 — regression tests

`run_olhoff_regularized_tests.m` grew from one group to eleven. Every case is a
toy-mesh **software-mechanics** test (dispatch, parsing, accounting, controller
forcing, classification); none is cited as evidence about convergence quality,
topology or mesh behaviour.

| # | test | guards |
|---|---|---|
| 1 | four routes execute; volume held; caps respected; trust ≤ ceiling; ceiling monotone; **`offDiag` live for `olhoff/mma` and only there**; LP route refuses `offDiag` | route dispatch, accounting, F-1-adjacent wiring |
| 2 | forced stage exhaustion contracts the persistent ceiling and logs it; post-update trust ≤ ceiling | controller mechanics |
| 3 | **a significant single-step improvement blocks contraction**: with `progress_tolerance` wide open the cumulative test always passes, so contraction can only come from the spike guard; the guard is calibrated from the run's own observed maximum single-step progress, asserted to block at 0 and to permit at 10× | cumulative-progress logic cannot be fooled by one significant improvement |
| 4 | **`CAP_HIT` can never satisfy the native-convergence classification**: status, stop reason, and `max(convergenceCount) < persistence` | new |
| 5 | **minimum trust without stationarity ⇒ `GLOBALIZATION_STALLED`** (forced by `min_inner = max_inner`, `tol_inner = 0`, `move_min = move`) | new |
| 6 | **convergence requires the declared persistence**: `persistence = 3` converges at exactly outer 3 with counter 3; the identical run capped at 5 with `persistence = 8` is `CAP_HIT` | new |
| 7 | the convergence tolerance is clamped to `objective_tol/certificate_radius`; the caller's looser request is recorded but **cannot reach either gate**; behaviourally, a loose request produces **zero** contractions at a non-stationary point (CV-4) | new |
| 8 | the certificate is **radius-scale invariant** (1e-6 relative across a 10× radius change) and is evaluated at the fixed reference radius over a run in which the trust radius moves (CV-2) | new |
| 9 | the certificate uses the **exact** multiplicity: with `tol_mult = 3.0` the step model clusters, `certificateN == 1` (CV-1) | new |
| 10 | cantilever concentrated-mass path executes with a positive tip mass | retained |
| 11 | **every runner honours an externally selected route**: no unconditional assignment to `optimizer`/`formulation`, and all three still advertise the override (F-1) | new |
| 12 | **a negative predicted improvement is never read as stationarity** (CV-5): asserted structurally over every recorded iteration of every route — a finite stationarity slope may not coexist with `predicted < 0` — and the counter exists | new |
| 13 | **`cluster_lambda` dispatches, validates, and actually changes the diagonal gradients** when the cluster is not degenerate; an invalid value is rejected | new |

Result: `OLHOFF_REGULARIZED_TESTS_PASS groups=13`.

---

## 13. Scientific identity

Three things must not be conflated, and this report does not conflate them.

| | what it is | what may be claimed |
|---|---|---|
| **Historical reproduction** — `Matlab/reproduction2007/` | the frozen, fixed-work Du–Olhoff (2007) clean-room reproduction, 1600 outer iterations, no stopping test | paper reproduction. Untouched by this audit; `git status` on it is empty and the four primitives the regularized code consumes are byte-identical to their audited state |
| **Regularized Olhoff formulation** — `formulation='olhoff'` | the genuine multiple-eigenvalue Olhoff local model (Eq. (16)/(22) LP, or the full Eq. (25d) sub-eigenvalue problem through nested MMA) plus a trust-region acceptance test, a persistent move-limit continuation and a stopping certificate. The local subproblem is Olhoff's; **the globalization and the stopping rule are disclosed numerical extensions and are not in the paper** | a globalized Olhoff formulation. **Not** a paper-literal reproduction, and its iteration counts are not comparable with the 1600-iteration historical run |
| **KS-inspired alternative** — `formulation='ks'` | the spectral objective replaced by a smooth Kreisselmeier–Steinhauser lower aggregate of the lowest modes | an Olhoff-*inspired* regularization. **Never** an Olhoff reproduction. This audit additionally shows it optimises a different function: its terminal design is `CONVERGED` for its own aggregate while carrying a 2.93e-04 relative feasible ascent in the *physical* λ₁ |

The regularized implementation reuses the frozen FE, filter, generalized-gradient,
LP and MMA primitives, as permitted, through `repro2007_paths()` with an
identity assertion. The audit's own WP2 code deliberately does **not** reuse the
production model builder or its stopping logic.

---

## 14. Remaining uncertainties

1. **The principal open item: why the nested-MMA route cannot close the mode
   gap.** The three routes that reach ω₁ ≈ 161.6 — `olhoff/lp`, `ks/lp`,
   `ks/mma` — are exactly the three that never form the off-diagonal `f_sk`
   (s ≠ k): the LP routes force `f_sk'drho = 0` by Eq. (22), the KS routes never
   build them. The one route that stalls at 156.05 is the only one that feeds the
   full Eq. (25d) coupling into its subproblem, and `ks/mma` proves the nested
   MMA optimizer itself is not at fault. Three mitigations were tested (§9.3.1):
   tightening `tol_mult` confirms the mechanism but recovers little objective;
   tightening `tol_inner` is not viable (0.972 → 0.991 at 4× cost, saturating
   the inner cap); per-mode λ̃ on the diagonal gradients is refuted outright.
   Diagnosing the off-diagonal coupling itself was out of scope, because removing
   it *is* removing the primary route.
2. **The certificate is a lower bound for an exactly degenerate cluster of size
   ≥ 2.** In production it uses the Eq. (22) equality-restricted LP, whose
   feasible set is contained in the true one. Measured at every terminal design
   here the restriction is not binding (twelve-digit agreement with the
   unrestricted cutting plane), and the value is combined by `max()` with the
   route's own slope — which on the MMA route *is* the unrestricted model. This
   is empirical, not proved for all designs.
3. **The threshold is configuration-relative.** "No feasible step within `move`
   improves λ₁ by more than `objective_tol`" is well posed, but its strength
   depends on the user's `move` and `objective_tol`. Raising `move` weakens it.
   This is now explicit rather than hidden; it is not an absolute guarantee.
4. **Exact stationarity is not attained on this problem by any route.** §7.2
   shows the criticality measure falling to ~8e-05 and then *rising*, wandering
   in a 1e-04 – 3e-03 band for 1400 further iterations. Every `CONVERGED` here is
   a tolerance statement about a near-stationary design, and should be cited as
   such.
5. **The nested MMA under-solves its own subproblem by ~2.8 %** and this is only
   partly reducible (§9.3.1b). The corrections defend the *stopping test* against
   that bias but do not remove it from the *steps*.
6. **`move_max = move` pins the trust radius at its ceiling** through the entire
   ascent phase, so the trust-region machinery only ever contracts. A genuinely
   adaptive radius (`move_max > move`) was not exercised.
7. **Runs stopped once their outcome was determined**, and are labelled as such
   rather than reported as completed: `olhoff/mma` at 320×40 (outer 590),
   fixed–pinned `olhoff/mma` (outer 607), the as-shipped `olhoff/mma` mesh sweep
   at 240×30 and 320×40 (outer 114/125), the headroom diagnostic (outer 380), and
   the three mitigation runs. In each case the quantity that decides the
   classification — mode-gap plateau, certificate, or subproblem accuracy — had
   already stabilised.
8. **One filter-radius policy** (constant physical 0.075). Other radii, the
   sensitivity-filter modes `diag`/`all`/`none`, and the discontinuous mass law
   `4` are untested; the code warns about the last.
9. **`n = 1` only.** Optimising a higher mode (`cfg.n > 1`) was not exercised.
10. **The audit's raw evidence is not versioned.** The repository `.gitignore`
    excludes `*.log`, `*.csv`, `*.txt` and `*.mat`, so every trajectory, log and
    WP2 output under `audit/` is untracked: 3.4 MB of logs, 6.2 MB of trajectory
    CSVs, 0.8 MB of WP2 text output and 15 MB of `run.mat`. Only the scripts and
    this report are committed. The numbers quoted here are therefore reproducible
    (Appendix B) but not archived, and a `git clean` would destroy them. Deciding
    whether to add a scoped un-ignore for the ~10 MB of text evidence is a
    repository-policy question left to the maintainer.
11. **The 3000-iteration caps** on the corrected LP runs are diagnostic headroom,
    not a proposed production cap. §11 shows the natural convergence iteration is
    strongly problem-dependent (574 / 1567 / 2961), so a production cap must be
    set from the problem.

## 15. Recommendation

**Ready with qualification — for the LP route. Further work required for the MMA
route.**

### Use now

`formulation='olhoff', optimizer='lp'` with the corrected defaults. Natural
convergence at 160×20 / 240×30 / 320×40 and on two further problems, every
terminal design independently certified below `objective_tol` and confirmed by
physical fixed-step eigensolves. This is the route for production work.

### Do not use for a convergence claim

`formulation='olhoff', optimizer='mma'`. It does not reach a certifiable design
at any mesh tested, and the cause is in the formulation's off-diagonal coupling,
not in the stopping rule (§14.1). It costs ~172× the LP route's wall time for a
worse objective and a certificate two orders of magnitude worse. It remains
useful as a research object and as the control that exposed CV-5.

The KS routes are diagnostics only. `ks/mma` attains the highest ω₁ of any run
(161.963) and is **not** stationary for the physical λ₁ (residual ascent 33×
`objective_tol`) — a high eigenfrequency is not evidence.

### Conditions of use

1. **Cite `CONVERGED` as what it is**: *"no feasible step within the configured
   move limit improves λ₁ by more than `objective_tol`, sustained over
   `persistence` accepted updates"* — not "a stationary point". §7.2 shows the
   criticality measure does not reach zero on this problem.
2. **Do not raise `stationarity_tol`.** It is derived and a looser request is
   clamped with a warning. Raising it via `objective_tol` re-opens CV-3 exactly.
3. **Set `max_outer_iterations` from the problem.** Measured natural convergence
   is 574 (simply supported), 1567 (fixed–pinned) and 2961 (cantilever). The
   fixed–pinned runner's existing 1600 is closer to the edge than is comfortable.
4. **Do not compare iteration counts with `Matlab/reproduction2007`.** That is a
   fixed-work reproduction with no stopping test.
5. **Never quote a KS `CONVERGED` as an eigenfrequency-stationarity result.**
6. **Re-run `run_olhoff_regularized_tests` before trusting any change** to the
   controller, the certificate or the step model. Groups 4–9 and 11–13 exist
   specifically to prevent the regressions this audit found.

### Further work, in priority order

1. **Diagnose the off-diagonal Eq. (25d) coupling** (§14.1). This is the one
   thing standing between the primary route and a usable result, and the KS
   control has narrowed it to a single term. Suggested next experiment: at a
   recorded near-degenerate design, compare the step the coupled model produces
   with the step the Eq. (22)-restricted model produces, and check which one the
   *physical* trial eigensolve prefers — a fixed-design comparison costing
   minutes rather than another thousand-iteration run.
2. **Replace the Eq. (22)-restricted production certificate** with the
   cutting-plane certificate for exactly degenerate clusters of size ≥ 2
   (§14.2), at a cost of 3–5 extra LP solves per outer iteration.
3. **Reconsider `tol_mult = 0.05` for the step model.** It creates a
   self-sustaining pseudo-degeneracy: the design parks just beneath whatever
   value it is given (§9.3, §9.3.1a). The same threshold-plateau signature that
   CV-3 exhibits in the stopping rule.
4. **Exercise `move_max > move`** so the trust region can genuinely expand
   (§14.6).

## Appendix A — "same basin?" is a per-route question

Superseded by §10.1, which carries the full table and the correction. The short
form: every LP-family terminal design at 160×20 is the same topology to
r ≥ 0.956, including the diagnostic with the move-ceiling controller disabled;
the MMA-family designs sit at r ≈ 0.80–0.86 from them.

That distinction is why "the topology looks right" is not evidence about the
stopping test, and why this audit never uses it as evidence. `ks/mma` makes the
point sharpest: highest ω₁ of any run, a different topology from the certified
design, and 40× the residual ascent.

## Appendix B — reproducing the audit

```matlab
addpath analysis/OlhoffRegularized/Matlab
run_olhoff_regularized_tests                       % 11 groups, toy meshes

addpath analysis/OlhoffRegularized/audit/scripts
audit_run('R_ss160x20_olhoff_lp','simply',160,20,1.5,'olhoff','lp',3000)
audit_stationarity('R_ss160x20_olhoff_lp')         % WP2, independent
audit_wp5_table({'R_ss160x20_olhoff_lp'})
audit_main_table({'R_ss160x20_olhoff_lp'})
```

`audit_run` writes `audit/results/<tag>/{run.mat,trajectory.csv}` and one
`AUDITRESULT` line; `audit_stationarity` adds `stationarity.mat` and prints the
certificate and the physical fixed-step table. `audit/scripts/summarize.py`
collects every `AUDITRESULT` line in `audit/logs/` into one table.
