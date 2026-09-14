# DATA INVENTORY

Brief sec. 2 asks exactly what per-element, per-iteration information is
available. The answer determines how much of the requested analysis is possible,
so it is stated bluntly first.

## Headline: the per-element density history does not survive

**No run retains per-element data.** All three prior studies reconstructed a full
`RHO` matrix (`NE x nOuter`) in memory and saved it to a `.mat` file, but every
one of those files has been deleted, and `.mat` is git-ignored inside
`analysis/OlhoffCurrent/diagnostics/`, so none was ever committed.

| study | code that saved it | file | status |
|---|---|---|---|
| `move_stop` | `ms_run.m:131` `save(f,'out','cfg','RHO','-v7.3')` | `runs/{baseline,fixedmove}_{160x20,320x40}.mat` | **absent**; hash-recorded in `FINAL_SHA256.txt`, so its loss is provable |
| `admission_rule` | `ar_run.m:81` | `runs/unstopped_*.mat` | **absent**; never listed in the manifest |
| `move_transition` | `mt_run.m` `save(f,'out','cfg','tcfg','RHO','-v7.3')` | `runs/arm{P,U}_*.mat` | **absent**; never listed in the manifest |

A repo-wide search found no surviving copy. `analysis/OlhoffArchive.zip` holds an
unrelated older `OlhoffApproachExact` tree, not these runs.

Consequently the following are **not computable from recorded data** and are not
reported anywhere in this study:

- per-element `rho_e(k)`, `rho_e(k-1)`, `Delta rho_e`;
- any percentile of `u` other than the max (P50/P75/P90/P95/P97.5/P99);
- **element identity** — and therefore Jaccard overlap between iterations,
  persistence duration, birth/death rate, or whether the saturated set is the
  same elements or a rotating population (brief sec. 6);
- **spatial coordinates** — and therefore spatial maps of `rho`, `|Delta rho|` or
  `u_e`, clustering, boundary/interface association, or the location of the
  66 saturated elements (brief sec. 7).

Sections 6 and 7 of the brief, and required figures 5, 6 and 7, cannot be
answered or produced from recorded data. Nothing has been interpolated or
modelled to fill the gap; the figures at those numbers are explicitly-labelled
aggregate substitutes.

## What DOES survive

Ten per-iteration CSVs, all bit-identical to their recorded hashes. Every CSV
carries, per outer iteration, exactly two functionals of the increment
distribution:

| column | definition (from `mt_telemetry.m` / `ms_run.m`, cross-checked in `olhoffSolve.m`) |
|---|---|
| `maxAbs` | `max_e |drho_e(k)|` (`hist.dxOuter`) |
| `l2` | `||drho(k)||_2` (`hist.dxNorm2`) |
| `rms` | `l2/sqrt(NE)` — an **identity**, verified to 5.8e-15 relative; carries no information beyond `l2` |

plus `move`, `stage`, `moveDescent`, `Mnd_pct`, `gray_frac`, `mid_frac`,
`omega1`, `omega2`, `gap12`, `volume`, `beta`, and per-study extras.

**Only the `move_stop` runs additionally record exact active-set counts** —
`nActive_epsRMS`, `nActive_1e4`, `nActive_1e3`, `nActive_1e2`, i.e.
`sum(|drho_e| > tau)` for `tau = {8.83883e-4, 1e-4, 1e-3, 1e-2}`. These are the
only *measured* points of the increment distribution anywhere in the surviving
record, and they carry most of this study's weight.

## Per-run table

| run | study | mesh | NE | arm / move policy | iters | full density history | `Delta rho` exactly reconstructible | move history | spatial coords | exact active counts |
|---|---|---|---:|---|---:|---|---|---|---|---|
| `baseline_160x20` | move_stop | 160x20 | 3200 | production ladder | 1–91 | **no** | **no** | yes | **no** | **yes** (4 thresholds) |
| `baseline_320x40` | move_stop | 320x40 | 12800 | production ladder | 1–131 | **no** | **no** | yes | **no** | **yes** |
| `fixedmove_160x20` | move_stop | 160x20 | 3200 | fixed move 0.04 | 1–400 (cap) | **no** | **no** | yes | **no** | **yes** |
| `fixedmove_320x40` | move_stop | 320x40 | 12800 | fixed move 0.04 | 1–216 (converged) | **no** | **no** | yes | **no** | **yes** |
| `unstopped_160x20` | admission_rule | 160x20 | 3200 | production ladder, stop relaxed | 1–600 | **no** | **no** | yes | **no** | no |
| `unstopped_320x40` | admission_rule | 320x40 | 12800 | production ladder, stop relaxed | 1–600 | **no** | **no** | yes | **no** | no |
| `armP_160x20` | move_transition | 160x20 | 3200 | production ladder | 1–600 | **no** | **no** | yes | **no** | no |
| `armP_320x40` | move_transition | 320x40 | 12800 | production ladder | 1–600 | **no** | **no** | yes | **no** | no |
| `armU_160x20` | move_transition | 160x20 | 3200 | `max(u)<0.5` for 10 | 1–600 | **no** | **no** | yes | **no** | no |
| `armU_320x40` | move_transition | 320x40 | 12800 | `max(u)<0.5` for 10 | 1–600 | **no** | **no** | yes | **no** | no |

"Spatial coords" would in principle be reconstructible from the mesh dimensions
*if* per-element values existed to place on them; with no per-element values the
question is moot.

## Partial distributional information rescued from `move_transition/METRICS.json`

`mt_spatial.m` computed `quantile(u, [0.50 0.90 0.99 1.00])`, `fracAtBound`
(`u>=0.99`) and `fracAbove50` (`u>=0.50`) for **every** iteration, but
`mt_export.m` never wrote them to CSV. What was persisted is only:

- `spatial_at_stop` — those quantiles at the single stop iteration, per run;
- `spatial_median_fracAtBound` / `spatial_median_fracAbove50` — medians over the run;
- `transitions[].rrhoPrev10` — `r_rho` for the 10 iterations before each descent;
- `stages[]` — per-stage `rMed`/`rMin`/`rMax` and endpoint `M_nd`/`omega1`.

Four iterations' worth of four quantiles is not a per-iteration percentile
trajectory, so brief sec. 4's percentile family (P75…P99 vs iteration) remains
unanswerable. The `armU_160x20` figure quoted in the premise — 66/3200 elements
at the bound, `utilQuantiles = [7.85e-4, 0.0379, 0.996, 0.9993]` — survives only
as this single stop-iteration snapshot.

## The one exact reconstruction that IS available, and why it matters

Two facts make the surviving aggregates far more useful than they look:

1. **`N_eff` is exact and move-free.** The participation number
   `N_eff(k) = (l2(k)/maxAbs(k))^2 = (sum_e drho_e^2)/(max_e drho_e^2)` is the
   effective number of elements carrying the increment. The move cancels
   identically, so unlike `r_rho` it is invariant to the move level — the exact
   defect the previous study found in `r_rho` (its point 8).

2. **`frac(u >= t) <= RMS(u)^2/t^2` is an exact one-sided bound** (Markov on
   `u^2`). It *caps* the active fraction. It cannot lower-bound it beyond the
   trivial `1/NE`, because one element at `max(u)` with the rest arbitrarily
   small is consistent with any `(max, RMS)` pair. Measured against the exact
   counts (612 comparisons), the cap runs a **median 9.4x loose** (10th–90th
   percentile 3.8x–13.1x; range 1.05x–35.9x), so it is used only where a genuine
   upper limit is wanted, never as a stand-in measurement.

## Move-indexing convention (brief sec. 3)

Established by reading the solver, not assumed. In `olhoffSolve.m`, `mvNow` is
computed at line 267, passed as the box into the inner solve at line 298, and the
resulting increment's statistics are stored at the same index (lines 341–342,
373, 380) as the move itself (line 381). Therefore

> **`move(k)` governs the transition `rho(k-1) -> rho(k)`**, and
> `u_e(k) = |rho_e(k) - rho_e(k-1)| / move(k)`.

`move(k-1)` is **not** the right denominator. This matches `mt_spatial.m`
(`d = abs(RHO(:,k)-prev)/P.move(k)`) and the recorded `r_rho` column
(`P.ratio = h.dxOuter./h.move`), so no off-by-one correction is applied and none
is needed. Because production has `projection.enabled = false` (scope-locked in
`mt_run.m`), `olhoffSolve` takes its non-projection branch and `drho` is the
**physical** density increment, so `maxAbs`/`l2` are formed on the physical
field. Under projection this would not hold.

## Verified cross-checks

- `rms == l2/sqrt(NE)` to <= 5.8e-15 relative in all ten runs.
- All three production-ladder studies independently agree on the descent
  iterations: **79, 90, 101** at 160x20 and **130, 141, 152** at 320x40.
- `baseline` and `fixedmove` are **bitwise identical** (max abs difference
  **0.000e+00** across `Mnd`, `maxAbs`, `l2`) for iterations 1–78 at 160x20 and
  1–129 at 320x40 — i.e. up to the iteration before production's first descent.
  The fixed-move arm is therefore an **exact counterfactual continuation** of
  production at `move = 0.04`, which is what licenses every "how much evolution
  remained" statement in `REPORT.md`.
