# WP14 — Why BASE_mma behaves where the legacy nested MMA failed
NESTED-MMA ROUTE AUDIT

The existing `OLHOFFEXACT_FAILURE_POSTMORTEM.md` establishes the legacy failure modes. This
document isolates which concrete differences plausibly explain the change. **"MMA was fixed"
is not the explanation** — the inner solver is the same published `mmasub`/`subsolv` pair.

| control | legacy nested MMA | BASE_mma | assessed contribution |
|---|---|---|---|
| **filter radius** | `rmin = 2.5` elements (0.125 physical at 160x20) | **1.2 elements** (0.06 physical), via the `rminPhys` override | **PRINCIPAL.** The controlled LP sweep at the same mesh gives coalescence only for 1.1–1.5 and none at 2.0+. At 2.5 the LP endpoint is `omega1 = 159.49`, gap 5.04%, one blurred component — the wrong basin. BASE_mma is inside the successful band. |
| move limit | literal path unrestricted (full density box); stabilized path 0.2 with damping 0.5 | fixed 0.01, never adapted, never rejected | **PRINCIPAL.** The legacy full-box step was shown independently destructive: an exact LP at the initial simple eigenvalue also selects a collapsing box vertex. A 0.01 cap makes the local linearisation predictive over its own step. |
| inner iteration cap | 30 | 300 | **CONTRIBUTORY.** At 30 the legacy solver met its declared test 0/120 times. Here `N=1` solves converge in 83–108, so a 30-cap would have truncated **every** inner solve. |
| inner stopping rule | absolute Euclidean, scaled by `sqrt(nEl)` | relative to the accumulated increment, `dx/max|xmma| < 0.01`, with `minInner = 5` | **CONTRIBUTORY.** The relative form is scale- and move-invariant. The post-mortem records that an absolute test silently degenerates once the move limit is small. |
| status consumption | logged, then **ignored**; cap-hit increments applied unconditionally | recorded per outer (`conv` column) and available; still applied | **PARTIAL.** BASE_mma also applies non-converged increments (114 of 752). What changed is that the increments are now bounded by a 0.01 move, so an unconverged one cannot be catastrophic. The status is *visible* rather than *consumed*. |
| asymptote handling | restarted each outer; persistence tested and made results worse | fresh `low`/`upp` per inner solve, `xold1 = xold2 = x` at entry | neutral — same as legacy default |
| `beta` scaling | upper bound `1e6`, enlarging the scale | scaled variable `bs` with `xmax = 5`, `beta = bs * lam_ref` | **CONTRIBUTORY.** Normalising by `lamref` keeps the objective and constraint rows O(1), which is what made `subsolv` well-conditioned; the legacy `1e6` bound did not. |
| damping | stabilized path applied `alpha = 0.5` | none | neutral here |
| multiplicity threshold | 0.1% relative frequency | 5% relative frequency | **MIXED — see below.** |
| off-diagonal treatment | full coupling | full coupling (`offDiag = 1`) | **unchanged** — full coupling is not the failure cause |
| outer acceptance | unconditional (legacy) / trust-ratio rejection (rebuilt) | unconditional | **CONTRIBUTORY.** The rebuilt July solver froze because trust rejection blocked every post-coalescence step. BASE_mma has no rejection layer, so it traverses coalescence. |

## The ranked explanation

1. **Filter radius 2.5 → 1.2.** Basin selection. Necessary.
2. **Bounded move (0.01) with no rejection layer.** Makes each local model predictive and
   still permits post-coalescence evolution — the two failure modes of the legacy and
   rebuilt solvers respectively.
3. **Workable inner budget and a scale-invariant stopping test.** 300 vs 30, relative vs
   absolute.
4. **`lamref` normalisation of the MMA subproblem.** Conditioning.

The mathematics is unchanged. Every difference is a numerical-realisation control, which is
exactly the post-mortem's conclusion restated with the MMA route now on the favourable side
of each control.

## The multiplicity threshold cuts both ways

Raising `tolMult` from 0.1% to 5% is what lets BASE_mma engage the multiple-mode model at
all — the legacy 0.1% rule would have classified almost none of this trajectory as `N = 2`.
But at 5% **without hysteresis** the classifier chatters: 29 `N` transitions and **14 returns
from `N = 2` to `N = 1`** between outer 56 and 231, switching up at a relative gap of
4.28–4.73% and back down at 5.00–5.09%. The design is repeatedly re-modelled across a
threshold it is sitting on.

That is a defect BASE_mma shares with, rather than inherits from, the legacy code: it is a
*different* manifestation of the same unresolved question about what "very small tolerance"
means in the paper. It is quantified in `MULTIPLICITY_TRANSITIONS.csv` and visible as the
sawtooth in `figures/fig1_spectrum_N_gap.png` and `figures/fig4_lp_vs_mma.png`.
