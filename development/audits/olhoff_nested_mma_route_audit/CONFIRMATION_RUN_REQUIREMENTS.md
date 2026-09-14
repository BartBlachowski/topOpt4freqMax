# WP20 — Minimum confirmation experiment
NESTED-MMA ROUTE AUDIT — SPECIFICATION ONLY, NOT EXECUTED, NOT AUTHORIZED

## What does NOT need confirming

Reproducibility of the trajectory is **already established** and needs no new run.
`fm_mma_diag` and `BASE_mma_160x20` are two independent executions of the same configuration
and their inner-iteration counts, cumulative counts and multiplicity classifications are
**bit-identical over all 400 shared outer iterations**. The nested-MMA route is
deterministic and reproducible on this machine.

The route decision itself also does not require a new run: it is settled by the cost
asymmetry and the campaign-impact analysis, neither of which a confirmation run would change.

## What a confirmation run would be *for*

Two distinct purposes, which should not be conflated:

### (A) To make the MMA evidence citable as a secondary variant — REQUIRED

The multiplicity-cost result (`N=1` 93.4 vs `N=2` 147.8 mean sub-iterates, p = 1.1e-10) is
worth reporting, but its sole primary artifact is a log from a session that died at outer
752 of a declared 800, with no saved result. That is not a citable artifact.

Minimum experiment:

| parameter | value | reason |
|---|---|---|
| configuration | **identical to BASE_mma**, unchanged | no parameter may be altered to improve the result |
| mesh | 160x20 | the mesh the evidence is on |
| `maxOuter` | 800 as declared | complete the run it was meant to be |
| filter | `rminPhys = 0.06` (→ `rminEl = 1.2`) | unchanged; but the echo defect should be fixed so the log states 1.2 |
| saved fields | `res` struct with `rho`, full `hist` (already recorded), **plus a density snapshot history** | the topology sequence WP10 could not audit |
| inner detail | per-outer inner summaries already exist; add `st.dxHist` / `st.relHist` retention for a handful of representative outers | WP8 could not be answered: inner histories are computed but discarded |
| determinism | `threads = 1`, fixed `v0` in `eigSolve` | already the case |
| stop status | explicit reason recorded | BASE_mma has none |

This is **observational logging plus completion**, not an algorithmic change. Estimated cost
from measured timing: ~17.8 s per outer × 800 ≈ **4 hours** at 160x20.

### (B) To overturn the route decision — WOULD REQUIRE MORE, AND IS NOT RECOMMENDED

The one genuine fairness gap in this audit is that the routes are **not move-matched**: LP ran
at `move = 0.005` for 1600 outers, MMA at `move = 0.010` for 752. No LP run exists at
`rmin = 1.2` **and** `move = 0.010`, and no MMA run exists at `move = 0.005`. BASE_mma's
cumulative move budget (7.52) is comparable to LP's (8.00), which is why the gap comparison
is reported as reasonably fair — but it is not a controlled match.

If someone wished to argue that MMA would reach LP's 0.217% eigengap under matched
conditions, the test would be an MMA run at `move = 0.005`, `maxOuter = 1600`, everything
else unchanged. Measured cost: ~17.8 s × 1600 ≈ **8 hours** at 160x20 alone.

**This audit does not request that run**, because even a favourable outcome would not change
the recommendation: the 400×-per-outer cost makes MMA infeasible as the principal route for a
nine-mesh campaign regardless of the eigengap it achieves. The run would refine the
scientific comparison, not the engineering decision.

### Second mesh

**Not required for route selection.** A second mesh would be required only to claim that the
multiplicity-cost result generalises beyond 160x20. If that claim is to be made, one
additional mesh (240x30) at the same configuration would suffice — estimated ~4× the 160x20
cost per outer. **Do not request all nine meshes.**

## Explicitly not authorized by this audit

No run has been launched. No parameter has been changed. No campaign has been started.
