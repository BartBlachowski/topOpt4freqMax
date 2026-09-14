# INNER_MMA_KKT — Part 3

## Verdict

```
FINAL_INNER_MMA_KKT_FAIL
```

The final frozen subproblem does **not** satisfy its own first-order optimality
conditions, and — the decisive part — it does not do so at **any** iteration
count tested. This is a statement about the inner iteration's behaviour on this
subproblem, **not** a claim that published MMA contains a coding defect.

## Precondition: the reconstruction is exact

| check | result |
|---|---|
| `drho` vs retained `DRHO(:,386)` | **bitwise equal**, max abs difference **0** |
| `nInner` | 19 reproduced vs 19 recorded |
| β | 26873.74466704797 both, difference **0** |

So everything below is measured on the actual final subproblem.

## Dual provenance

**No exact retained dual exists**: `innerLoop` discards `mmasub`'s
`lam, xsi, eta, mu, zet, s` with `~`. The duals used here are **reconstructed** —
captured by re-running the identical `mmasub` calls on identical inputs — and
are exact for the MMA convex subproblem. No best-fit dual was needed.

## Primal feasibility — passes

| row | constraint | value | status |
|---|---|---|---|
| 1 | spectral, mode 1 | −4.71234e−06 | feasible, **active** |
| 2 | spectral, mode 2 | −0.276516 | feasible, slack |
| 3 | next mode J = 3 | −4.99559 | feasible, slack |
| 4 | volume | −3.09101e−06 | feasible, **active** |

Box violations: **0** on both sides. Move limit 0.01 respected;
max\|Δρ\| = 6.243e−04, i.e. 6.2 % of it.

## Dual feasibility — passes

`λ = [0.9972567734, 3.6167841e−07, 2.0018109e−08, 0.6324256598]`, all ≥ 0;
`min λ = 2.00e−08`. `min ξ = 1.58e−06`, `min η = 1.58e−06`, both ≥ 0.
`y ≈ 1.0e−10`, `z = 1.0e−07`, `ζ = 1` — consistent with `a = 0`, `a0 = 1` and
`λ ≪ c = 1000`, so the MMA problem reduces to (25) as written.

## Complementarity — passes

| quantity | value |
|---|---|
| max \|λᵢ·fᵢ\| | 4.699e−06 |
| max ξ·(x − xmin) | 7.403e−06 |
| max η·(xmax − x) | 1.036e−05 |

Active set {mode 1, volume} with duals 0.997 and 0.632; the two slack rows carry
duals at 1e−07 and 1e−08. Complementarity is clean at the 1e−05 level, which is
`mmasub`'s interior-point barrier floor.

## Stationarity — FAILS

### A disclosed preregistration error

`AUDIT_PREREGISTRATION.md` §4 defines the projected residual with a bound
classification tolerance of 1e−12 of the box width. `mmasub` solves by an
**interior-point** method and never puts a variable exactly on a bound, so at
that tolerance **0 of 28 800** variables are classified active and the statistic
silently omits `−ξ + η` from the Lagrangian gradient.

Both numbers are reported. The preregistered one is not altered.

| statistic | production (19 inner) | certification (500) | extended (5000) |
|---|---|---|---|
| preregistered projected, normalized RMS | **1.0060** | 1.0247 | 1.0299 |
| **exact MMA residual** (includes ξ, η), normalized RMS | **0.3597** | 0.5857 | 0.5966 |
| exact, normalized max | 2.9378 | 4.5811 | 5.1021 |

Bound-tolerance sweep (1e−12 … 1e−2): the projected statistic stays at 1.006
until 1e−2, where 1 584 variables are finally classified active and it drops to
0.2. Nothing in that range rescues it.

**Both statistics cross the preregistered FAIL bar of 0.1**, so the verdict does
not depend on the error.

Normalizer: `sRow = RMS(|ddlam(:,1)|/lamref) = 6.434693e−04`, as preregistered.

## The decisive evidence: more iterations make it worse

`FROZEN-SUBPROBLEM CERTIFICATION` — the identical subproblem, only the stopping
point changed, `drho` discarded and never applied.

| | production | certification | extended |
|---|---|---|---|
| `tolInner` | 0.05 | 1e−10 | 1e−12 |
| `maxInner` | 500 | 500 | 5000 |
| iterations taken | **19** | 500 (cap) | **5000** (cap) |
| converged | flag `true` | **no** | **no** |
| final relStep | 0.0412 | 0.003724 | **0.002077** |
| max\|Δρ\| | 6.243e−04 | 8.756e−03 | **9.787e−03** |
| …as fraction of the move limit | **6.2 %** | 87.6 % | **97.9 %** |
| exact KKT normalized RMS | 0.3597 | 0.5857 | **0.5966** |

Three things follow, and they matter more than the verdict label:

1. **The iteration does not converge.** After 5000 sub-iterates the relative step
   is 2.08e−03, still five orders from the 1e−12 target. It is **not monotone**:
   over the last 100 iterates it ranges [8.97e−04, 1.24e−02] with a mean of
   3.25e−03. It oscillates rather than settling. The minimum ever reached is
   4.24e−04.
2. **The KKT residual does not improve — it degrades**, 0.360 → 0.586 → 0.597.
   The failure is not under-convergence toward a good point.
3. **The subproblem's own solution is move-limit-bound.** The increment grows
   monotonically toward the bound: 6.2 % → 87.6 % → **97.9 %** of the move limit,
   a factor **15.7** larger than what production returns.

`figures/FIG_13_inner_convergence.*` shows both histories.

## What this means, stated carefully

Production's `tolInner = 0.05` is a **relative-step** test, not an optimality
test, and `innerLoop`'s own header says so — the paper gives no criterion and
this one is a declared reconstruction. Stopping at 19 sub-iterates returns an
increment that is 1/15.7 of what the subproblem actually asks for.

**The design's behaviour is therefore governed by where the inner loop is
truncated, not by the solution of the subproblem it poses.** Had the subproblem
been solved accurately, each outer iteration would move the design by essentially
the full move limit.

What this does **not** license:

* **Not** "MMA is broken." Plain `mmasub` carries no global-convergence
  guarantee when iterated as a fixed-point map without the GCMMA conservative
  outer loop; oscillation is a known possibility, not a defect. Its own convex
  subproblem is solved cleanly at every call — complementarity is at the
  interior-point floor throughout.
* **Not** an explanation of *why* the iteration oscillates. The filtered
  gradient it is handed is amplified ~30× in RMS in void (operator row sums to
  245), and the residual concentrates there (void 0.578 against gray-core
  0.013) — but establishing that as the cause would need a counterfactual this
  audit is forbidden to run.

## Residual by density class

| class | n | exact KKT residual, normalized RMS |
|---|---|---|
| void (ρ < 0.1) | 10 264 | **0.5776** |
| gray shell | 4 526 | 0.0160 |
| gray core | 3 748 | **0.0129** |
| solid (ρ > 0.9) | 10 262 | 0.1712 |

The filtered subproblem is **nearly stationary inside the gray regions** — which
is why the design stops moving there — and fails in void. This independently
reproduces the `gray_kkt_forensic_audit`'s finding that the filtered gray
residual (0.049) is far below the physical one (0.334), and localizes the
remaining inner-subproblem failure to the void region.

## Consequence for Part 11

The evidence places this in **Case B**: the final inner subproblem is not solved
to its own optimality conditions, so inner/local optimization accuracy has to be
resolved before the filtered field's non-conservativity can be assigned
responsibility for the endpoint. That the field *is* non-conservative is
established independently (`JACOBIAN_SYMMETRY.md`, `CLOSED_LOOP_INTEGRALS.md`)
and does not depend on this verdict — but the ordering of the two questions does.
