# WP10 — Topology audit
NESTED-MMA ROUTE AUDIT

## Evidence limitation, stated first

**`BASE_mma_160x20` saved no density field.** `run_case.m` writes `results/<name>.mat` only
after `olhoffOpt` returns, and it never returned (WP11). There is no `res.rho`, no snapshot
history, and therefore no topology at outer 100, 250, 500, 752, and no "best near-bimodal
state" density. The requested per-state topology sequence **cannot be produced from this
run**.

What is available is `fm_mma_diag.mat`: the **same configuration** (verified field by field —
only `maxOuter` differs, 400 vs 800) whose trajectory is **bit-identical to BASE_mma over all
400 shared outer iterations** (`nInner`, `cumInner`, `N` exact; `omega1`, `max|drho|` to
printed precision). Its saved `res.rho` is therefore a legitimate, verified proxy for the
BASE_mma design **at outer 400**, and nothing later.

All diagnostics below use the frozen study machinery: exact-count volume-preserving
projection with increasing-index tie-break, four-neighbour connectivity, mid-height support
nodes, `A_sig = 0.01` (= 4 elements at 160x20).

## Results

| artifact | route | mesh | outer | omega1 | gap % | grayness | gray frac | comps | connected | detached | max detached (el) | hard gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fm_mma_diag` | **MMA** | 160x20 | 400 | 168.513 | 2.586 | 0.1442 | 0.1769 | 16 | yes | 15 | **14** | **FAIL** |
| `lprmin1.2` | LP | 160x20 | 1600 | 168.240 | 0.217 | 0.1321 | 0.1638 | 14 | yes | 13 | 4 | **FAIL** |
| `lprmin1.1` | LP | 160x20 | 1600 | 170.128 | 0.371 | 0.1062 | 0.1388 | 24 | yes | 23 | 4 | **FAIL** |
| `lprmin2.5` | LP | 160x20 | 1600 | 159.491 | 5.036 | 0.2819 | 0.3209 | 1 | yes | 0 | 0 | PASS |
| `lp240_rmin1.3` | LP | 240x30 | 1600 | 170.471 | 0.232 | 0.1032 | 0.1201 | 1 | yes | 0 | 0 | **PASS** |
| `FIG4_definitive` | LP | 240x30 | 400 | 170.745 | 2.531 | 0.1082 | 0.1250 | 1 | yes | 0 | 0 | PASS |

## Reading

**Is the MMA topology paper-like? Yes.** `figures/topology_comparison.png` renders all six
density fields on an identical grey scale. The MMA field is a symmetric, X-braced truss with
a clear central void — the same structural family as the accepted LP reproductions and
visibly the same family as Fig. 3a. It is *not* the blurred, non-coalesced `rmin = 2.5`
family. On visual grounds the MMA route reaches the correct basin, which corroborates the
spectral evidence.

**Does it pass the study's hard gate? No — and neither does the matched LP run.** At 160x20
both routes leave small detached islands (MMA 15, max 14 elements; LP 13, max 4 elements)
against `a_sig = 4`. The gate is passed only by the 240x30 clean BEST result, which is a
single component. So island formation at 160x20 is a **mesh/maturity property of this
benchmark at this resolution, not a defect of the MMA route specifically**. It would be
wrong to reject MMA on this basis alone.

**But MMA is measurably worse on every topology metric available:** more components (16 vs
14), a larger worst island (14 vs 4 elements — 3.5× the significance threshold vs exactly at
it), and higher grayness (0.1442 vs 0.1321, and 0.1769 vs 0.1638 gray fraction). The
comparison is not maturity-matched — MMA at outer 400 against LP at outer 1600 — so this is
suggestive rather than conclusive.

**The frequency comparison does not rescue it.** MMA reaches `omega1 = 168.513` against LP's
168.240 (+0.16%), but its eigengap is 2.586% against LP's 0.217% — an order of magnitude
worse on the property that defines the Fig. 3a benchmark. `omega3` also differs materially
(343.4 vs 286.0), indicating a genuinely different higher-mode structure, not just a
different convergence depth.

## What could not be audited

- topology at outers 500 and 752 (no artifact);
- the "best near-bimodal state" (BASE_mma never reaches a persistent near-bimodal state —
  only 2 of 752 outers have a relative gap below 1%, and none below 0.5%);
- support-connected modal-energy audit (not saved by either route);
- comparison against Fig. 3a beyond the visual family match, which is the same standard the
  existing clean-room work applies.
