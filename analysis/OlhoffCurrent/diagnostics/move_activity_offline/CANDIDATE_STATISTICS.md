# CANDIDATE STATISTICS — offline characterisation

Brief sec. 9 asks for a conceptual comparison of statistic families; sec. 10 asks
for the max-vs-median tradeoff to be analysed explicitly; sec. 11 asks whether an
active *fraction* is the right invariant. This file answers those three together,
because the data make them one question.

Everything here is **descriptive**. No threshold is proposed for production, and
the diagnostic bins are bins, not candidate controller settings.

## The test each family has to pass

The two meshes give one genuinely decisive comparison, and it is available
because `fixedmove` is bitwise identical to `baseline` up to the descent:

| | 160x20, iteration 78 | 320x40, iteration 129 |
|---|---|---|
| the next iteration | production descends 0.04 -> 0.02 | production descends 0.04 -> 0.02 |
| `M_nd` there | 13.494 | 23.451 |
| `M_nd` if move 0.04 is held to the end | 12.275 | 13.282 |
| **evolution still to come** | **1.218 pts = 9.0%** | **10.168 pts = 43.4%** |
| verdict on the descent | roughly on time | **badly premature** |

A rule of the form *"descend when `S < theta`"* must therefore **permit** at
160x20/78 and **block** at 320x40/129. That requires

> **`S(160x20 @ 78) < S(320x40 @ 129)`.**

This is a necessary condition, it needs no threshold to be chosen, and it is the
sharpest test the surviving data support.

## Results

| family | statistic | `S` @160x20/78 | `S` @320x40/129 | ordering |
|---|---|---:|---:|---|
| **A. maximum** | `max(u)` | 0.99974 | 0.64707 | **INVERTED** |
| **B. high percentile** | P90…P99 | *not recoverable* | *not recoverable* | — |
| **C. active fraction** | `frac(u>0.25)` | 0.07188 | 0.00828 | **INVERTED** |
| | `frac(u>0.025)` | 0.22125 | 0.25578 | OK |
| | `frac(u>0.0221)` | 0.22500 | 0.26562 | OK |
| | `frac(u>0.0025)` | 0.30750 | 0.33797 | OK |
| **C'. active count** | `count(u>0.025)` | 708 | 3274 | OK (4.6x margin) |
| **D. RMS / global** | `RMS(u)` | 0.23134 | 0.05945 | **INVERTED** |
| **E. participation** | `N_eff` | 171.3 | 108.1 | **INVERTED** |
| | `phi_eff = N_eff/NE` | 0.05355 | 0.00844 | **INVERTED** |
| *reference* | `omega1` rel. range (W=10) | 1.494e-3 | 1.784e-3 | OK (1.19x — negligible) |

Two structural facts fall out.

**1. Everything driven by the loud tail is ordered backwards.** `max(u)`,
`RMS(u)`, `N_eff` and the high-threshold active fraction all read *quieter* at
320x40 — the mesh with 4.8x more relative evolution remaining. No threshold on
any of them can work, at any value, because the required inequality is violated
before a threshold is even chosen. This is not a calibration problem.

**2. The discriminating signal is widespread slow drift, not a few fast
elements.** At 320x40/129 only 0.83% of elements exceed `u = 0.25`, but 25.6%
exceed `u = 0.025`. At 160x20/78 the pattern inverts: 7.2% exceed 0.25 while
22.1% exceed 0.025. The premature descent happens while a quarter of the design
is still creeping — and the elements that creep are invisible to every
tail-weighted statistic.

## Sec. 10: the max-vs-median tradeoff, resolved by measurement

The brief asks that the opposite pathologies be analysed rather than assumed.
They are not symmetric here.

- **`max(u)` — one element vetoes.** Confirmed, and worse than "too sensitive".
  Its Spearman rank correlation with remaining `M_nd` evolution is **0.066** at
  160x20: it carries essentially *no* information about how much topology
  evolution is left. It sits at 0.9995 for the entire 400-iteration fixed-move
  run while `M_nd` falls from 99.5 to 12.3. And at both production descents at
  160x20 (iterations 79 and 90) `max(u)` was ~0.999 and ~0.996 — fully saturated —
  yet only 9.0% of evolution remained. So a saturated `max` does not even imply
  immaturity. This is the previous study's `r_rho` finding, quantified.

- **`N_eff` — the tempting middle, and why it fails.** `N_eff` is exactly
  computable and move-invariant by construction, which makes it the most
  attractive candidate on paper. It is nonetheless unsuitable: it is ordered
  backwards at the decisive comparison, and it does not collapse across meshes
  (best-fit exponent -0.20, residual 0.78 — see below). Its apparent
  threshold-window overlap (91.98..93.74, a **2% wide** band) is two noisy curves
  crossing, exactly the coincidence the brief warns against, not a margin.

- **The median pathology does not arise, for a mundane reason.** `P50(u)` is not
  recoverable. But the concern behind it — that a bulk statistic hides a small
  meaningful moving region — is testable and **does** apply to `RMS(u)`, which is
  bulk-weighted yet still inverted. The failure is not that RMS hides a moving
  front; it is that RMS measures *speed* when what predicts remaining evolution
  is *extent*.

## Sec. 11: is an active *fraction* the right invariant?

No — and neither is an active *count*. Both were tested directly.

Write the statistic as `activeCount / NE^alpha`. Dividing a count series by a
per-mesh constant rescales its admissible-threshold window exactly, so a single
threshold works at both meshes iff the count windows overlap after rescaling:

```
alpha in [ log_R(loB/hiA) , log_R(hiB/loA) ],    R = NE_320/NE_160 = 4
alpha = 0   constant COUNT        alpha = 0.5  interface length ~ sqrt(NE)
alpha = 1   constant FRACTION
```

A second, independent estimate matches the two fixed-move runs at equal `M_nd`
maturity and finds the `alpha` minimising the RMS log-discrepancy between meshes.

| threshold | window method, `alpha` (c_min 0.99, D 10) | collapse method, best `alpha` | residual | collapses? |
|---|---|---|---:|---|
| `|drho|>1e-4` | 0.710 .. 0.933 | +0.960 | 0.170 | yes |
| `|drho|>8.84e-4` | 0.554 .. 0.924 | +0.832 | 0.115 | yes |
| `|drho|>1e-3` | 0.513 .. 0.929 | +0.808 | 0.123 | yes |
| `|drho|>1e-2` | -2.071 .. -0.064 | -0.192 | 0.898 | **no** |
| `N_eff` | -0.360 .. 0.022 | -0.200 | 0.780 | **no** |

Both methods, independently, put `alpha` for the low-threshold active set at
roughly **0.7–0.93**, and both **exclude constant count (`alpha = 0`) outright**.
Constant fraction (`alpha = 1`) sits at the upper edge — admissible for the
`1e-4` threshold, marginal for the others. Interface-length scaling
(`alpha = 0.5`) sits at the lower edge and is excluded by the collapse fit.

So the honest answer to sec. 11 is: **the mature active set scales as roughly
`NE^0.8` — between an interface length and an area fraction, and cleanly
distinguishable from a constant number of elements.**

The load-bearing caveat: **this is one exponent fitted to two meshes.** The
quoted bands are fit sensitivity, not statistical confidence. Two points cannot
validate a power law, and the `1e-4`/`1e-3` estimates (0.96 vs 0.81) differ by
more than either band, which is itself a sign the single-exponent model is being
strained. A third mesh would settle it, and a third mesh requires an optimisation
run.

## Within-mesh vs across-mesh: the distinction that matters

Rank correlation against remaining `M_nd` evolution, on the fixed-move runs:

| statistic | Spearman, 160x20 | Spearman, 320x40 |
|---|---:|---:|
| `max(u)` | **0.066** | 0.644 |
| `RMS(u)` | 0.618 | 0.782 |
| `N_eff` | 0.619 | 0.743 |
| `count(|drho|>8.84e-4)` | **0.680** | **0.990** |
| `count(|drho|>1e-3)` | **0.687** | **0.987** |
| `omega1` rel. range (W=10) | 0.617 | 0.923 |

**Within a single mesh the low-threshold active count is an excellent maturity
indicator** (0.99 at 320x40). What fails is *transfer*: the value at which it
signals maturity moves with the mesh, and the exponent governing that move is
not pinned by two meshes. That is a much narrower deficiency than "no statistic
works", and it is what makes the next step an experiment rather than more
offline work.

## Family E: persistent active fraction

Brief sec. 9E asks for a statistic separating a tiny persistent saturated set
from widespread continued evolution. **True persistence is not computable** —
element identity did not survive, so a genuine persistent-set statistic cannot be
formed or tested offline (see `DATA_INVENTORY.md`).

What *can* be said is that the separation family E was meant to achieve is
already achieved by threshold placement alone: a low threshold sees the
widespread drift, a high threshold sees only the saturated set, and the two are
ordered oppositely across meshes. Whether adding temporal persistence of element
identity would improve on that is untested and untestable here.

A self-normalising variant that avoids `alpha` altogether was tested and
**fails**: `s(k) = count(k)/max_j count(j)` degenerates, because at iteration 1
essentially every element moves (peak = 3200/3200 and 12664/12800), so the peak
is `~NE` and `s` collapses to the area fraction — the `alpha = 1` case, whose
overlap is empty at `c_min >= 0.99`.

## Summary

| family | verdict |
|---|---|
| A. `max(u)` | **unsuitable.** Rank correlation 0.066 at 160x20; ordered backwards; saturated at both 160x20 descents where the design was in fact nearly mature. |
| B. high percentiles P90–P99 | **untestable.** Not recoverable from surviving data. Cannot be recommended or dismissed on evidence. |
| C. active fraction, **high** threshold (`u>0.25`) | **unsuitable.** Ordered backwards; does not collapse (`alpha ~ -0.19`). |
| C. active **count/fraction, low** threshold (`u ~ 0.0025–0.025`) | **the only survivor.** Correct ordering, 4.6x margin, Spearman 0.68/0.99, collapses across meshes under `NE^0.8`. Not yet calibratable. |
| D. `RMS(u)` | **unsuitable.** Ordered backwards. Measures speed, not extent. |
| E. persistent active set | **untestable.** Requires element identity. |
| — `N_eff` (participation number) | **unsuitable**, despite being exact and move-invariant. Ordered backwards; does not collapse; its only overlap window is 2% wide. |
