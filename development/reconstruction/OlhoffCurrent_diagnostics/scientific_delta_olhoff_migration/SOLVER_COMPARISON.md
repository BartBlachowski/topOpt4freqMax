# SOLVER_COMPARISON — Part 9 (inner problem-(25) solver)

## 1. Executed path on both sides

Both S480 (`duOlhoffAdaptivePedersen`) and C480 set `optimizer.inner.type = mma`,
`variant = published`, `variable = increment`, so `olhoffSolve` calls **`innerLoop(ctx)`** in both.
`innerLoop.m`, `deltaLambda.m`, `useMMA.m`, `mma_published/mmasub.m`, `mma_published/subsolv.m`
are **byte-identical** (SOURCE_TO_TARGET_FILE_MAP). The source's changes to `innerLoopRho.m`
(outer-history asymptotes) apply only to `variable = 'design'` and are not executed.

| item | source | target | identical? |
|---|---|---|---|
| variables | [Δρ (NE); β̂ = β/λ_ref] | same | yes |
| scaling | β̂ ∈ [0, 5], start 1 (β = λ_n); rows divided by λ_ref; volume by V_tot | same | yes |
| objective / rows | −β̂; (25c)×N via eigenvalues of diag(dOff)+A(Δρ) with gradients Σ v_js v_jk f_sk; (25b) f_JJ; (25e) linear | same | yes (bitwise rows at 9 states) |
| box | lo = max(ρ_min − ρ, −d), hi = min(1 − ρ, d) | same formula | code yes; **d is a per-element vector in the source, a scalar in the target** |
| retained MMA state | xold1, xold2, low, upp and iteration counter persist across the sub-iterates of one outer call; reset at each new outer call | same | yes |
| asymptote initialization | low/upp = x ∓ 0.5 (xmax − xmin) (published asyinit 0.5) | same | code yes; distances scale with each element's box |
| asymptote update | asyincr 1.2, asydecr 0.7, albefa 0.1, move 0.5 | same | yes |
| subsolv | epsimin 1e−7 | same | yes |
| stopping | it ≥ 5 and max|Δx_step| / max|x| < 0.05 | same | yes |
| max inner calls | 500 | same | yes |
| relinearization | none inside the loop (F, f_JJ, λ frozen per outer); A(Δρ) re-eigensolved each sub-iterate | same | yes |
| β handling | MMA variable, no controller authority | same (C480); production β drives the ladder outside the inner loop | inner: yes |
| multiplicity | N = 2 fixed, offsets and off-diagonals | same | yes |

## 2. Measured identity

At nine frozen states (ρ₀, C480 iterations 10/20/100/386, S480 final, M1 iterations 5/11/64) the
full inner solve with a common box 0.04 returns a **bitwise identical Δρ, β and nInner** from the
target `+impl` and from the source code under the target's material law. The offline solve also
reproduces in-run steps bitwise: target C480 iterations 1, 11, 21, 101 and source M1 iterations 1,
6, 12 (`evaluations/same_state_comparison.json`).

## 3. Does the source avoid the known repeated-MMA failure mechanism?

**No — it uses the same inner solver under a different box architecture.** Measured along the
trajectories (`evaluations/trajectory_analysis.json`):

| quantity | C480 (target) | M1 (source code, SIMP/4b) | S480 (source, retained hist) |
|---|---|---|---|
| inner sub-iterations per outer (median / mean) | 18 / 18.9 | 20 / 19.3 | 18 / 19.1 |
| non-converged inner loops | 0 | 0 | 0 |
| increments on a bound (1e−6 of box width), all iterations | 0 % | 0 % | n/a |
| elements at ≥ 99 % of their move box, first 10 iterations (median) | 11.6 % | 9.9 % | n/a |
| RMS(Δρ_e / d_e), first 10 iterations (median) | 0.57 | 0.49 | n/a |
| max|Δρ| / max box, whole run (median) | — | — | 0.78 |
| first step at ρ₀ | ‖Δρ‖₂ 4.27 with box 0.04 | ‖Δρ‖₂ 13.39 with box 0.10 (cos 0.960) | = M1 (bitwise prefix) |

The prior target findings — the exact problem-(25) optimum is almost fully bound-saturated, and
repeated MMA stops far from it (frozen_inner_solver_study; c480_socp_causal_run control: 0 % of
increments on a bound) — apply **unchanged** to the source: its MMA steps are equally unsaturated
relative to their boxes. What differs is the box: the source's per-element box starts 2.5× larger,
grows back to 0.10 for monotone elements and contracts to 0.002 for oscillating ones, so the same
attenuated MMA produces larger absolute steps where the design moves coherently and vanishing steps
where it chatters. Classification of that effect: D7 (outer box/controller), not D6.

Verdict eligibility (preregistration §10): the inner code and settings are identical on the executed
paths, so **SOURCE_SUCCESS_PRIMARILY_INNER_SOLVER is not eligible**, and "MMA accuracy/state" is
ranked IDENTICAL BETWEEN IMPLEMENTATIONS in CAUSAL_ATTRIBUTION.md. This also means the SOCP inner
solver candidate from the earlier target studies is neither supported nor refuted by the source's
success.
