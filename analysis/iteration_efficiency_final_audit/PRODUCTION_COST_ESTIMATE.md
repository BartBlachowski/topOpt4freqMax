# Production cost estimate

All figures below rest on **measured** timings on this machine (MATLAB R2025b, arm64, `maxNumCompThreads(1)`), not on prior documentation. Treat totals as order-of-magnitude (±50%).

## 1. Measured unit costs

Candidate-C evaluation (3 models, adaptive from 3 modes, no binary diagnostic) and the exact-count topology gate, per state:

| mesh | Ne | evaluator (s) | topology (s) | offline / state (s) | LP native / outer (s) |
|---|---:|---:|---:|---:|---:|
| 160×20 | 3 200 | 0.114 | 0.0088 | 0.123 | 0.184 |
| 240×30 | 7 200 | 0.222 | 0.0159 | 0.238 | 0.148 |
| 320×40 | 12 800 | 0.389 | 0.0161 | 0.405 | 0.249 |
| 480×60 | 28 800 | 0.894 | 0.0335 | 0.928 | 0.597 |

Fitted exponents: evaluator `p = 0.941`, topology `p = 0.572`, LP native `p = 0.560`.

Sanity checks against a real trajectory: over 40 sampled states of the frozen 480×60 Olhoff run the evaluator averaged 0.892 s (median 0.857, p95 1.003, max 1.835) — within 1% of the single-state figure. Only 5% of states escalate the adaptive search; the worst observed case (k=194, ordinal 13, schedule 3→6→12→24) cost 2.92 s against 0.907 s for a non-escalating state at the same mesh, a 3.2× premium. A ~10% campaign-level escalation premium is applied below.

Other methods at 320×40: Proposed 0.154 s/iteration (no eigensolve in the OC update), Yuksel 0.058 s/iteration averaged over both stages — roughly 0.52× and 0.20× the LP per-outer cost.

The extrapolated LP cost at 800×100 is 0.83 s/outer, but the frozen campaign recorded a **median 1.817 s/outer at 800×100** (`tEig` 1.368 s, ~75%). The native figures below are scaled by 2.2× to honour that anchor.

## 2. Principal campaign (Proposed / Yuksel / Olhoff-LP, nine meshes)

Work per (method, mesh) cell:

- reference trajectory: 3 200 native updates
- measurement trajectory: `B_meas` updates — 3 200 for Olhoff (B0 = 3 200), ≈ 2 200 for Proposed (B0 = 900) and Yuksel (B0 = 2 000) at the observed `b_ref ≈ 2 100`
- offline evaluation: every reference and measurement state
- timing replays: 6 distinct horizons (3 q × {enter, cert}) × 4 runs (1 warm-up + 3 retained). At the 96×12 anchor the six horizons sum to 2 279 updates → ≈ 9 100 native updates per cell

| component | estimate |
|---|---|
| optimization runs | 27 reference + 27 measurement + ~648 timing replays = **~702 runs** |
| native optimization — Olhoff-LP | ~40 h |
| native optimization — Proposed | ~20 h |
| native optimization — Yuksel | ~8 h |
| **native subtotal** | **~67 h** |
| offline Candidate-C + topology (3 methods × ~6 400 states × 9 meshes, +10% escalation) | **~54 h** |
| **principal campaign total** | **~120 h ≈ 5 days serial single-threaded** |

Storage:

| item | estimate |
|---|---|
| Olhoff double trajectories (Σ Ne = 307 200 × 3 201 × 8 B, reference + measurement) | ~16 GB |
| Proposed / Yuksel observer files (pre-allocated at `H + 2100 = 5 300` columns, 2 methods × 2 passes) | ~52 GB |
| extracted trajectories, per-cell `.mat`, tables, figures, topologies | ~15 GB |
| **total** | **~80–110 GB** |

Worth noting: `install_observer` pre-allocates `nElements × (H+2100)` doubles regardless of how many states are recorded, so the observer files dominate storage. At 800×100 a single observer file is ~3.4 GB.

## 3. Optional MMA secondary campaign

Measured at 160×20: nested MMA **12.58 s/outer** against LP **0.184 s/outer** — a **68× per-outer cost ratio**, with 101–108 inner MMA iterations per outer update (all converged, none at the 300 cap).

The good news is that `innerLoop.m` holds `F`, `fJJ` and `lam` fixed, so inner iterations are `mmasub`/`subsolv` solves and do **not** re-solve the eigenproblem. The bad news is that a `mmasub` call on `Ne+1` variables still scales with the mesh, on top of the same eigensolve LP pays.

| scope | estimate |
|---|---|
| nine meshes, reference + measurement | order **10³ h (months)** — **not practical** |
| three coarse meshes (160×20, 240×30, 320×40) — the contract's `minimum_valid_meshes = 3`, enough for a scaling fit | **~150–200 h** |
| 160×20 only — inner-work statistics, no scaling fit | **~25 h** |

**Recommendation.** Do not make the principal campaign contingent on MMA. The frozen protocol already marks MMA `optional_secondary_method` with `separate_rows: true`, and `method_plan` / the selector keep the routes fully independent, so `olhoffVariant='lp'` delivers the complete principal comparison on its own. Run the principal nine-mesh campaign under `'lp'`, then decide MMA scope separately: three coarse meshes if an MMA scaling exponent is wanted, or 160×20 alone if only the inner-work characterisation is needed. Report whichever subset is run as an explicitly partial, paper-native, uncontrolled comparison.

Note also finding **F-07**: the LP route uses a mesh-dependent filter radius of 1.3 **elements** while the MMA route uses a mesh-independent 0.06 **physical** — so an MMA mesh sweep does not refine the same continuum problem the LP sweep does. The manifest's `fairness_note` currently mentions only the move limit.

## 4. Practicality verdict

The principal nine-mesh campaign is **practical**: roughly five days of unattended serial compute and ~100 GB. The optional MMA nine-mesh campaign is **not**, and must not gate it.
