
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

