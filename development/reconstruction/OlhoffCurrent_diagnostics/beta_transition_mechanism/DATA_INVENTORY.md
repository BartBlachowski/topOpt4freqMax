# DATA INVENTORY — what this audit used, and what does not exist

No optimisation was run. Everything below already existed at HEAD `cb6c0ea`.

## Scalar telemetry used

| mesh | production-ladder run | fixed-move counterfactual | iterations |
|---|---|---|---|
| 160x20 | `move_transition/runs/armP_160x20_iterations.csv` | `move_stop/runs/fixedmove_160x20_iterations.csv` | 600 / 400 |
| 320x40 | `move_transition/runs/armP_320x40_iterations.csv` | `move_stop/runs/fixedmove_320x40_iterations.csv` | 600 / 216 |
| 400x50 | `move_activity_400/runs/P400_400x50_iterations.csv` | `move_activity_400/runs/F400_400x50_iterations.csv` | 139 / 369 |

The `armP` runs are the production ladder carried past the production stop, which
is what makes the *post-descent* behaviour of the stall metric observable at the
two coarser meshes.

Fields present in all of them and used here: `beta`, `move`, `stage`,
`moveDescent`, `omega1`, `omega2`, `gap12`, `volume`, `Mnd_pct`, `gray_frac`,
`mid_frac`, `l2`, `rms`, `maxAbs`, `nInner`, `innerConv`, `multN`, `degen`.

## Raw element-level evidence used

`analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat`
(102.1 MB, `RHO`/`DRHO` 20000×369 double) and its P400 counterpart (36.6 MB,
20000×139), both **hash-verified through the evidence gate before use**. Used to
compute the bound-active population, which is not derivable from the CSVs.

## What does NOT exist, and was therefore not analysed

**Gradient statistics were never recorded.** The production recorder (`hist`)
carries `N, beta, cumInner, dBeta, degen, dxNorm2, dxOuter, dxPhys2, gap12,
innerConv, massLow, move, multJ, nInner, omega, pEvent, pPen, pStage, projBeta,
projEvent, projStage, stage, tEig, tGrad, tInner, tOuter, vol, volErr` — no norms
of `f_JJ`, `F`, or the volume gradient. Brief §12 (L1/L2/L∞/median/percentiles of
the gradients) is therefore **not answerable from existing evidence**, and no
figure reconstructs it.

This is not a blocking gap, for a reason that is worth stating rather than
assuming. Gradient magnitude enters beta *only* through `Delta lambda(drho)`, and
therefore only through the predicted gain `g = (beta − λ)/λ`. `g` **is** recorded
(it is a function of `beta` and `omega1`, both present), so the quantity that
matters for the audit is observable without the gradients themselves. Measured,
`g/move ≈ 2.0` identically at all three meshes over iterations 5–15, which is the
mesh-consistency check §12 was asking for. A new instrumentation run under §14
was therefore **not** authorised.

Also unavailable: MMA asymptote vectors (`low`/`upp`) and per-inner-iteration
beta traces, neither of which is retained. The asymptote *scale* is nonetheless
known analytically — `low/upp = x ∓ 0.5(xmax−xmin)` with `xmax−xmin = 2·move` at
MMA iterations 1–2 — so the move→asymptote coupling could be audited from code.

## Derived quantities defined in this audit

| symbol | definition | source |
|---|---|---|
| `rel(outer)` | `(mean β[−10:] − mean β[−20:−10]) / mean β[−20:−10]` | transcribed from `olh.move.limit` |
| `g` | `(beta − omega1²)/omega1²`, the predicted eigenvalue gain | `beta`, `omega1` |
| remaining evolution | `(M_nd(i) − M_nd(fixed-move end))/M_nd(i)` at `i` = last iteration at move 0.04 | identical to the definition frozen in `move_activity_400` |
| bound-active | `|drho_e| ≥ move·(1−1e−12)` | raw `DRHO` |

The remaining-evolution definition is deliberately unchanged from the previous
study so that 9.0% / 43.4% / 50.1% remain directly comparable.

## Validation

`scripts/btm_common.py`'s re-derived predicate reproduces **every actual
production descent at all three meshes** (`[79,90,101]`, `[130,141,152]`,
`[138]`), confirming the audit analyses the deployed rule.
