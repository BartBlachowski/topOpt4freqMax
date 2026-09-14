# STATE_IDENTITY — Part 1

## Verdict

```
FROZEN_480_PROBLEM25_STATE_PASS
```

All preregistered identity checks pass; zero blockers. Source:
`evaluations/state_identity.json`, produced by `scripts/fp_state.m`.

## The authoritative state

| | |
|---|---|
| study of origin | `diagnostics/three_rung_canary_preflight` (three-rung 480×60 canary) |
| trajectory read | `evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat` (directly, not a derived cache) |
| final outer | **386**, stage 3, move **0.01** |
| ρ₃₈₆ SHA-256 (authoritative endpoint) | `0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60` |
| ρ₃₈₅ SHA-256 (density the final subproblem was built at) | `9b1443e508a6e4ecb5288d30167c7258d837be127aa3a9a42d8cc092240f3ded` |
| Δρ₃₈₆ SHA-256 (production P19 increment) | `0b12cd7f9ae32decc6e95bf63e89fa7ca4530d13ec6dc7d815b618d46af8a9fd` |
| config hash | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` (recomputed = recorded) |
| `+impl` tree | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`, source manifest verifies |
| N (fixed subspace) / J | 2 / 3 |
| `multJ` at 386 / `nInner` at 386 / `innerConv` | 0 / 19 / 1 |
| controller | `stageExhaustion` for continuation and stop, ladder `[0.04 0.02 0.01]`, terminal branch B at 386 |

All three hashes match the values the two preceding audits recorded, so this
task continues on exactly the state they analysed.

## The subproblem is posed at ρ₃₈₅

`olhoffSolve` evaluates the FE problem at the density *before* the update. The
final inner subproblem was therefore built at ρ₃₈₅ and produced ρ₃₈₆. All
frozen quantities below are evaluated at ρ₃₈₅.

## Fresh FE evaluation reproduces the retained context bitwise

`fp_state` re-assembles, re-solves the eigenproblem and re-forms the
generalized gradients at ρ₃₈₅ through the production functions
(`assemble2D`, `eigSolve`, `olh.multi.detect`, `genGrad`, `applyFilter`) and
compares with the `ctx` the prior audit retained
(`filtered_subproblem_integrability_audit/evaluations/inner_kkt_state.mat`).

| quantity | fresh vs retained |
|---|---|
| `F` (28 800 × 2 × 2, filtered) | **bitwise equal** |
| `fJJ` | **bitwise equal** |
| `lam`, `lamJ`, `dOff`, `rho`, `move` | **bitwise equal** |
| `omega(1:5)` vs `hist.omega(:,386)` | max relative difference **0** |

The retained production increment `stP.xFinal(1:NE)` equals `DRHO(:,386)`
bitwise.

## Frozen scalar data of the subproblem

| symbol | value |
|---|---|
| ω₁…ω₅ at ρ₃₈₅ | 163.93172747916933, 185.2086604517607, 401.39842250345157, 559.4330251495452, 755.2308321724405 |
| λ₁ = `lamref` | **26873.611274304643** |
| λ₂ | 34302.247906335586 |
| λ_J = λ₃ | 161120.69358825943 |
| `dOff` | **[0, 7428.636632030943]** — PRESENT (not absent as the task prompt expected) |
| `dOff(2) − (λ₂ − λ₁)` | 0 (exact) |
| `gap12` | 0.129791427808232 |
| `offDiag` | true (full (25d) coupling) |
| ρmin, volfrac, Vtot | 1e−3, 0.5, 14 400 |
| ρ₃₈₅ range / mean | [0.0010125425191870787, 0.9998951138429818] / 0.4999984098511315 |
| `max(1e−3, ρ)` guard in `applyFilter` | inactive: 0 elements below 1e−3 |
| β recorded at 386 | 26873.74466704797 (= `bs_P19 · lamref`, reproduced exactly) |

Multiplicity configuration: `subspace`, size 2, tolerance 0.05,
`diagonalOffsets = true`, `offDiagonal = true`. Filter: `sensitivity`, applied
to `all` f_sk, physical radius 0.06 (3.6 elements). MMA: published Svanberg
2007, `a0 = 1, a = 0, c = 1000, d = 0`, `tolInner = 0.05`, `minInner = 5`,
`maxInner = 500`.

## Exact box bounds

`x = [drho; bs]`, `xmin = [max(ρmin − ρ₃₈₅, −move); 0]`,
`xmax = [min(1 − ρ₃₈₅, +move); 5]`.

| | count |
|---|---|
| lower bound set by the move limit (−0.01) | 19 360 |
| lower bound set by the density floor (ρmin − ρ, magnitude < 0.01) | 9 440 |
| upper bound set by the move limit (+0.01) | 19 100 |
| upper bound set by the density ceiling (1 − ρ, magnitude < 0.01) | 9 700 |
| box width min / max | 0.010012542519187078 / 0.02 |

Because ρ₃₈₅ never touches either density bound exactly, every element has a
two-sided box of positive width. The 9 440 floor-limited elements (ρ ≈ 0.001)
can move down by only ≈ 1.25e−5 to 1e−2; the 9 700 ceiling-limited ones
(ρ > 0.99) can move up by less than 0.01.

## Two prior states available without recomputation

| state | source | β | `nInner` |
|---|---|---|---|
| P19 (production) | `DRHO(:,386)` = `stP.xFinal` | 26873.74466704797 | 19 |
| M500 | prior audit `stC.xFinal` | 26892.489471331457 | 500 |

The prior audit did **not** retain its 5000-iteration state (`clear drhoL`);
M5000 is regenerated here by frozen replay (`MMA_TRAJECTORY_TO_REFERENCE.md`).
