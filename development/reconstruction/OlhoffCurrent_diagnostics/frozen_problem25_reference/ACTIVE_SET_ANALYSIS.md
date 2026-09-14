# ACTIVE_SET_ANALYSIS — Part 13

Source: `evaluations/reference_solution.json` (`active`, `active_sweep`,
`by_class`, `certificate_aligned`).

## Which mechanism sets the optimum

| constraint | status at the reference | multiplier |
|---|---|---|
| clustered minimum eigenvalue (row 1) | **active**, value 1.5e−14 | μ₁ = 1.0000 |
| second cluster row (row 2) | slack 0.278 (redundant) | 0 |
| next mode J = 3 (row 3) | slack 4.950 | 4e−17 |
| volume (row 4) | **active**, value 4.0e−14 | ν₂ = 0.7291 |
| box (move limit ∪ density bounds) | **99.94 % of variables on a bound** | ξ, η ≥ 0 with `max ξ = 7.55e−3`, `max η = 8.6e−4` |

Two general constraints are active — the minimum-eigenvalue cone and the
volume — and the solution is otherwise a **vertex of the box** with 16
interior coordinates (at 1e−6·width; 3 at 1e−3·width). The objective is set
by: the cone (μ₁ = 1 is forced by stationarity in `bs`, since the next-mode
row is inactive) trading against volume (ν₂ = 0.73) through the box.

## Fractions at each bound (tolerance 1e−6 × width)

| where | count | fraction of NE = 28 800 |
|---|---|---|
| at −move (lower bound set by the move limit) | 4 906 | 17.03 % |
| at ρ floor (lower bound set by ρmin − ρ, only for ρ < 0.011) | 9 404 | 32.65 % |
| at +move (upper bound set by the move limit) | 4 788 | 16.63 % |
| at ρ ceiling (upper bound set by 1 − ρ, only for ρ > 0.99) | 9 686 | 33.63 % |
| interior | 16 | 0.06 % |
| **any bound** | **28 784** | **99.94 %** |
| **a move-limited bound** | **9 694** | **33.66 %** |

## Move-bound dominated? Two readings, both reported

*Preregistered reading* (§8: "≥ 50 % of variables within 1e−6·width of a
move-limited bound"): **33.66 % ⇒ NOT move-bound dominated** by that rule.

*Structural reading*: of the 28 800 variables, 19 140 have at least one side
of their box set by the **density** bound rather than the move limit (ρ₃₈₅
within 0.01 of 0.001 or of 1), and the reference pushes 19 090 of those onto
that density bound. Among the remaining variables the move limit is the
binding side, and the reference puts essentially all of them on it. By
density class of ρ₃₈₅:

| class | n | −move | floor | +move | ceiling | interior | Σdrho |
|---|---|---|---|---|---|---|---|
| void (ρ < 0.1) | 10 266 | 404 | **9 404** | 454 | 0 | 4 | −8.79 |
| gray (0.1 ≤ ρ ≤ 0.9) | 8 272 | **4 268** | 0 | **3 992** | 0 | 12 | −2.70 |
| solid (ρ > 0.9) | 10 262 | 234 | 0 | 342 | **9 686** | 0 | +11.54 |

So the reference is **bound-saturated**: it is a bang-bang step in which every
gray element moves by the full ±0.01 (4 268 down, 3 992 up, 12 interior), the
void goes to the floor and the solid to the ceiling, with the volume balanced
(Σdrho = +0.0458 = the residual volume slack, 3.2e−6 of Vtot). The move limit
is the binding bound for every gray element. The preregistered 50 % rule
counts the void/solid elements, whose boxes are narrower than the move limit
on one side, as "not move-bound"; on the other side of their box (the +move
side for void, the −move side for solid) 404 + 454 + 234 + 342 = 1 434 of them
are also at the move limit.

## Consequence (stated, not acted on)

The hard move box is scientifically consequential for the frozen problem: with
99.94 % of variables on a bound, `max|drho_ref| = move` to 1e−10, and the
objective gain of the step bounded by the box, the solution of production's
problem (25) at this state is determined by the box geometry together with
the cone and volume rows. Whether the box should be different is not this
task's question; the move limit is not changed.

## Interior coordinates and primal non-uniqueness

Only 16 coordinates are interior at 1e−6, and only 22 reduced gradients are
below 1e−6 of the maximum. The optimal face therefore has dimension at most
~20 and the optimal `drho` is unique up to motions inside that face, which
change none of the five image functionals. The `fmincon` cross-check confirms
the objective and the image while landing at a slightly different primal
(`FMINCON_CROSSCHECK.md`).
