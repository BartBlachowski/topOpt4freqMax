# BASELINES — frozen production references (Phase 8)

**No production run was made for this study.** Every number below is read from an
existing frozen study's tracked `METRICS.json` and per-iteration CSV, or from the
one surviving production trajectory. Machine-readable: `evidence/baselines.json`.

---

## 1. Sources

| mesh | scalar record | per-iteration CSV | final density field |
|---|---|---|---|
| 160×20 | `diagnostics/move_stop/METRICS.json` → `baseline_160x20` | `move_stop/runs/baseline_160x20_iterations.csv` | **UNAVAILABLE** — raw `.mat` lost |
| 320×40 | `diagnostics/move_stop/METRICS.json` → `baseline_320x40` | `move_stop/runs/baseline_320x40_iterations.csv` | **UNAVAILABLE** — raw `.mat` lost |
| 400×50 | `diagnostics/move_activity_400/METRICS.json` → `P400` | `move_activity_400/runs/P400_400x50_iterations.csv` | `evidence/move_activity_400/P400_400x50_trajectory.mat` — present, declared, hash-valid |

All three baselines are the production preset with **no overrides** beyond mesh,
cap and recorder; `move_stop` additionally asserts field-for-field that its
baseline arm equals what `olhoffcurrent_config` produces, so "baseline" really is
production.

## 2. The frozen table

| | 160×20 | 320×40 | 400×50 |
|---|---|---|---|
| terminal status | `NATIVE_CONVERGED` | `NATIVE_CONVERGED` | `CONVERGED` |
| outer iterations | **91** | **131** | **139** |
| inner (MMA) iterations | 2241 | 2614 | 2918 |
| wall time [s] | 125.48 | 387.96 | 541.0 |
| move transitions | **79** (.04→.02), **90** (.02→.01) | **130** (.04→.02) | **138** (.04→.02) |
| final move / stage | 0.01 / 3 | 0.02 / 2 | 0.02 / 2 |
| **reached move = 0.005** | **no** | **no** | **no** |
| ω₁ | 169.49522702153845 | 165.95078925220545 | 162.882615630062 |
| ω₂ | 171.9599959181592 | 183.76971226824654 | 175.403921349379 |
| relative gap (ω₂−ω₁)/ω₁ | 0.01476553890041266 | 0.10674152016769037 | 0.0768731866865097 |
| volume | 0.49999900877797954 | 0.49999913861781514 | 0.499999149662958 |
| `M_nd` [%] | 13.402499279526742 | 23.359567952074254 | 32.3283256233213 |
| gray fraction (0.1–0.9) | 0.149375 | 0.26375 | 0.3476 |
| mid-density fraction (0.4–0.6) | 0.025 | 0.095625 | 0.1882 |
| outer tolerance ε | 0.05 | 0.1 | 0.125 |
| **β stall first fires** | **79** | **130** | **138** |
| convergence event | 91 | 131 | 139 |
| final ρ hash | **unavailable** | **unavailable** | available (see `evidence/baselines.json`) |
| MATLAB | 25.2.0.3042426 (Update 1) | 25.2.0.3042426 (Update 1) | 25.2.0.2998904 |

## 3. What the baselines show about production, before any candidate result

Two facts, read straight off the table, and both are the reason this study exists.

**Production descends at exactly the β-stall iteration.** 79, 130, 138 — the
first β stall and the first descent are the same iteration on all three meshes.
The move ladder is doing nothing but following β.

**Production then stops within 1–12 iterations, and never reaches the bottom of
its own ladder.** 160×20 stops at 91 having reached 0.01; 320×40 at 131 having
reached 0.02; 400×50 at 139 having reached 0.02. The declared ladder
`[0.04 0.02 0.01 0.005]` is never traversed. The mechanism is arithmetic rather
than physical: halving the move halves the achievable `‖Δρ‖`, which is compared
against a move-independent ε, so a descent mechanically manufactures the
convergence signal one or two iterations later.

That is the behaviour the intervention is aimed at, and it is recorded here
before any candidate number was produced.

## 4. Unavailable fields

The 160×20 and 320×40 production **final density vectors** do not exist on this
machine (`PROVENANCE.md` §5). Therefore, for those two meshes:

* no density-field distance production↔candidate;
* no production topology image;
* no final-ρ hash.

They are marked unavailable in every table and figure. Regenerating them would
require a production rerun, which is not authorized and is not needed: every
scalar the comparison rests on survives in the tracked `METRICS.json` and CSVs.
