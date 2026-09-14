
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

