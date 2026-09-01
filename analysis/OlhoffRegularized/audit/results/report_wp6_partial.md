
## 10. WP6 — four-route control experiment at 160×20

Identical FE problem (simply supported 8×1, 160×20, `rmin` 1.5 elements,
`volfrac` 0.5, `p` 3, `move` 0.005), identical regularization controls. This is
diagnostic, not a contest.

### 10.1 Do the routes reach the same physical basin?

**Yes.** Pairwise terminal-density correlation on this mesh is ≥ 0.955 across
every route and every criterion, including the diagnostic run with the
controller disabled (Appendix A). Mean |Δρ| ≤ 0.047. The routes do not select
different local optima; they stop at different points along the same shallow
bimodal ridge. ω₁ ranges over 161.42 – 161.88 — a spread of 0.29 %.

This is the single most important control in the audit, because it is what
rules out the reading "the shipped criterion stopped early but found a different
and equally good optimum". It did not: it found the *same* optimum, less far
along.

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

### 10.3 Does KS smoothing change the local optimum?

**It changes where the optimizer stops, not which basin it is in.** The KS
terminal design has ω₁ = 161.419, ω₂ = 162.484 — a 0.66 % gap, where the Olhoff
routes drive the pair to numerical coincidence (gap 5e-11). That is the expected
effect of maximising a smooth lower aggregate of the lowest modes rather than the
minimum itself: the aggregate is maximised slightly before the modes coalesce.

The consequence for this audit is sharp. The KS terminal design is `CONVERGED`
for its own aggregate while carrying a **2.93e-04 relative feasible ascent in
the physical λ₁**, physically confirmed (`actual/predicted = 1.0000` at
t = 1e-5, Δω₁ = +2.58e-05; and +0.0251 rad/s at t = 1e-2). Under the corrected
criterion — which certifies λ₁, not the aggregate — the same run correctly
**refuses to converge** and reports `CAP_HIT` at 3000.

**A KS `CONVERGED` is not an eigenfrequency-stationarity result.** That is the
concrete reason the KS route must never be reported as an Olhoff reproduction.

### 10.4 Volume feasibility, LP vs MMA

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
