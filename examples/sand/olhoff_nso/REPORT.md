# Null Space Optimizer as a replacement for MMA in the Du–Olhoff loop — A/B at 160×20, 240×30, 320×40 (2026-10-09)

**Thesis under test.** The Null Space Optimizer (NSO, Feppon et al.), as used for the SAND
heat-sink problem in `tools/SAND/ex13_heat_SAND.py`, could beneficially replace the MMA
optimizer in the Du & Olhoff (2007) nested loop of `analysis/Olhoff`.

**Verdict: NOT SUPPORTED at 160×20.** NSO is a cheaper *iteration* (0.08–0.18 s vs 1.9 s)
but a far slower *optimizer* for this problem: with the production iteration cap it lands
2–10 % below the production ω₁ with gray designs; with three times the budget it gets within
0.9 % at the same wall time; started from the production optimum it does not improve on it.
The property that made NSO attractive in the SAND paper — thousands of sparse local
constraints — does not exist here (4 dense eigenvalue rows, 1 volume row).

## Set-up

* FE model: Python mirror of `+impl/fem` (`olhoff_fe.py`), verified against the MATLAB
  production run: identical initial ω = 68.39859221 / 253.38506687 / 420.76715164 / 514.09815806.
* Production formulation (results/olhoff_ref_160x20_cfg.json): Pedersen stiffness p = 3,
  linear mass, SS mid-height pins, axial restraint both ends, Sigmund sensitivity filter
  R = 0.06 (1.2 el), V = 0.5, ρ₀ = 0.5, ρ_min = 1e-3, 4 eigenvalues carried, N fixed at 2,
  adaptive per-element move box from 0.1, stop ‖Δρ‖₂ < 0.05, cap 400.
* Arms (all single-threaded; MATLAB R2025b, Python 3.13 / scipy 1.17):
  1. **Production**: MATLAB nested MMA with the (25d) sub-eigenvalue coupling (reference).
  2. **NSO**: `nlspace_solve` on x = (ρ, β), J = −β, h = β − λ_j/λ_ref (j = 1..4), box and
     volume as inequalities, Euclidean or Helmholtz metric, PIQP; dt ∈ {0.01…0.2};
     normalization for 50 iterations (paper) or always (fixed max step); optional
     Krog–Olhoff equality f₁₂ᵀξ = 0 on the N = 2 subspace.
  3. **Single-loop MMA** (package port of Svanberg's mmasub) on the same class, move 0.05/0.1/0.2.
* Full table: results/comparison.md; figures results/fig_histories.png, fig_designs.png.

## Results

| arm | iters | ω₁ | ω₂−ω₁ [%] | M_nd [%] | s/iter | wall [s] |
|---|---|---|---|---|---|---|
| Production nested MMA (MATLAB) | 121, converged | **169.21** | 0.70 | **11.5** | 1.92 | 232 |
| single-loop MMA, move 0.2 | 400, cap | 168.36 | 0.41 | 13.2 | 0.83 | 332 |
| NSO dt 0.02, paper normalization | 400, cap | 153.55 | 0.90 | 53.5 | 0.08 | 31 |
| NSO dt 0.05, fixed max step | 400, cap | 167.88 | 9.41 | 20.1 | 0.11 | 44 |
| NSO dt 0.2, fixed max step | 400, cap | 165.13 | 7.55 | 15.7 | 0.13 | 50 |
| NSO dt 0.05, fixed max step, 1200 it | 1200, cap | 167.64 | 1.86 | 13.1 | 0.14 | 166 |
| NSO dt 0.02 warm-started at the production optimum | 400 | 169.18 | 0.12 | 11.6 | 0.18 | 73 |
| NSO dt 0.05 warm-started at the production optimum | 400 | 168.78 | 1.35 | 11.9 | 0.16 | 65 |

Time to first reach ω₁ ≥ 165: production 30 iterations / 46 s; best NSO (dt 0.2) 161 iterations
/ 20 s; NSO dt 0.05 293 iterations / 32 s. Time to reach ω₁ ≥ 168: production 45 it / 78 s;
no NSO arm within 400 iterations.

## Reading

1. **Per-iteration cost.** 98 % of the production wall time is the nested MMA inner loop
   (227 of 232 s; the 121 eigen-solves cost 4.4 s). NSO's QPs are negligible here because
   only 5 constraints plus the near-active bounds enter them. So NSO *does* remove the
   dominant cost of an Olhoff iteration.
2. **Progress per iteration is much worse.** NSO is a projected gradient flow in the
   ρ-metric with an ∞-norm step cap: the eigenvalue sensitivity is concentrated in a few
   elements, so only those move at the cap and the bulk creeps. MMA's separable convex
   approximation moves every variable to its own asymptotes. Net effect: the production
   solver converges at 121 iterations; NSO is still gray (M_nd 20–53 %) at 400.
3. **Bimodality.** The Du–Olhoff optimum is bimodal (ω₁ ≈ ω₂); the (25d) coupling in the
   inner loop handles that. NSO sees only the diagonal gradients f_jj and treats the two
   eigenvalue constraints as independent inequalities; its 400-iteration designs keep a
   5–15 % gap, which closes only with 1200 iterations. Adding the Krog–Olhoff equality
   f₁₂ᵀξ = 0 (the "LP route") did not help (ω₁ 164.8 / 159.6).
4. **Stationarity.** Warm-started at the production optimum with a shrinking step, NSO
   stays there (169.18, M_nd 11.6 %): the production optimum is a KKT point for NSO's
   formulation too, and NSO finds nothing better. With a fixed step it drifts down (168.78).
5. **The SAND argument does not transfer.** NSO's benefit in the heat-sink paper is sparse
   Jacobians of n local constraints. The Olhoff problem has 5 dense rows; the "nested loop"
   exists for the multiple-eigenvalue model (25d), not for constraint count. Replacing MMA
   inside `innerLoop.m` would put a gradient flow where a convex-approximation solver is the
   right tool; replacing the whole nested loop gives the numbers above.

## Follow-up 1 — NSO as predictor, production Du–Olhoff as corrector (160×20)

Hypothesis (user): NSO reaches ω₁ ≥ 165 in ~20 s where production needs ~46 s, so
ρ₀ →(NSO)→ ρ_coarse →(Du–Olhoff + nested MMA)→ ρ* could cut the total below 232 s.

Protocol: NSO (fixed max step, ∞-norm normalised) stopped at the first iterate with
ω₁ ≥ 150 / 160 / 165; that exact density handed to the production solver as `design.initial`
(sandbox copy `olhoffSolveWarm.m`, one line changed to accept a vector); production run to its
native stop ‖Δρ‖₂ < 0.05 with the unchanged adaptive move box starting at 0.1.

| chain | NSO it / s | ω₁ handed over | corrector outer it | corrector **inner MMA it** | corrector s | **total s** | final ω₁ | gap % | M_nd % |
|---|---|---|---|---|---|---|---|---|---|
| cold production (baseline, clean rerun) | – | 68.4 | 121 | 2369 | 228 | **228** | 169.21 | 0.70 | 11.5 |
| NSO dt 0.05 → ω₁ ≥ 165 → production | 294 / 27 | 165.0 | 102 | 2038 | 205 ᶜ | **231** | 168.68 | 0.52 | 11.8 |
| NSO dt 0.2 → ω₁ ≥ 165 → production (clean rerun) | 162 / 17 | 164.7 | 106 | 2379 | 252 | **269** | 168.55 | 0.41 | 12.0 |
| NSO dt 0.2 → ω₁ ≥ 160 → production | 74 / 6 | 160.7 | 224 | 5247 | 552 ᶜ | **558** | 168.30 | 0.36 | 12.5 |
| NSO dt 0.2 → ω₁ ≥ 150 → production | 31 / 2 | 150.9 | 128 | 2759 | 293 ᶜ | **296** | 168.81 | 0.72 | 12.1 |

ᶜ measured while other jobs ran; the clean rerun of the ω₁ ≥ 165 / dt 0.2 case moved from 257 s
to 252 s, so the contamination is ~2 %.

**Result: the predictor–corrector chain does not beat the cold start.** Best case ties (231 vs
228 s) with ω₁ 0.3 % lower; the others are 18–145 % slower. Why:

* The production cost is **inner MMA iterations × 0.097 s**, not outer iterations. The warm
  start changes the outer count by −16 % at best but the inner count by −14 % … +120 %. A gray
  NSO design (M_nd 30–55 %) with a 7–14 % ω₂−ω₁ gap is not "close" for the nested MMA: the
  (25d) coupling and the adaptive move box have to re-do the discretisation and the mode
  coalescence from scratch, and the move box restarts at 0.1 regardless of the start.
* The ω₁ ≥ 165 state NSO hands over is reached by production itself at outer iteration 30
  (46 s); production's first 30 iterations are **not** the expensive part — the expensive part
  is the last 70 iterations of discretisation at ~2 s each, which the handover does not shorten.
* All chains end 0.2–0.5 % below the cold ω₁: the warm start changes which local optimum is
  reached, not favourably.

## Follow-up 2 — scalability, 160×20 / 240×30 / 320×40

Same arms (production vs NSO fixed step dt 0.05 / 0.2, cap 400). Full table results/scaling.md.

| mesh | arm | iters | ω₁ | gap % | M_nd % | wall s | s / iter | it (s) to 98 % of production ω₁ |
|---|---|---|---|---|---|---|---|---|
| 160×20 | production | 121 ✓ | 169.19 | 0.7 | 11.5 | 232 | 1.92 | 32 (50) |
| 160×20 | NSO dt 0.2 | 400 cap | 165.13 | 7.6 | 15.7 | 50 | 0.13 | 197 (25) |
| 240×30 | production | 111 ✓ | 167.35 | 11.8 | 12.3 | 373 | 3.36 | 29 (57) |
| 240×30 | NSO dt 0.2 | 400 cap | 165.64 | 2.8 | 14.8 | 185 | 0.46 | 297 (137) |
| 320×40 | production | 101 ✓ | 165.86 | 17.7 | 14.1 | 453 | 4.49 | 39 (150) |
| 320×40 | NSO dt 0.2 | 400 cap | 165.42 | 3.0 | 15.3 | 524 | 1.31 | 245 (321) |

* **The per-iteration ratio T_MMA/T_NSO shrinks with n**: 15 → 7.3 → 3.4. Production's
  outer iteration grows ≈ n^0.6 (inner MMA, 97–98 % of its time); NSO's grows ≈ n^1.4 — its
  eigen-solve is 0.04 → 0.11 → 0.31 s but the QP/active-set part (2·n bound constraints) is
  0.09 → 0.35 → 1.0 s. NSO's time to 98 % of the production ω₁ is 25 → 137 → 321 s
  against production's 50 → 57 → 150 s: the early-phase speed advantage at 160×20 is gone by 240×30.
* **Quality converges with n.** At 320×40 NSO's capped design is within 0.3 % of production
  (165.42 vs 165.86) and nearly bimodal (gap 3 %), whereas production's own 320×40 optimum has a
  17.7 % gap — the production solver's loss of bimodality under refinement (see memory notes on
  fine-mesh behaviour) is not shared by NSO. That is an observation about the two optimisers'
  paths, not a speed argument.

## Verdict after the follow-ups

* NSO is not a drop-in replacement for MMA in the Du–Olhoff loop (160×20, confirmed at 240×30/320×40).
* NSO as a fast predictor does not shorten the production run: the costly phase is the nested
  MMA's discretisation tail, which a gray warm start does not remove (best case tie, typical +18 %).
* Scalability runs the wrong way for NSO: its per-iteration cost grows faster than the nested
  MMA's, so the 160×20 speed ratio is the most favourable case, not a lower bound.
* What remains genuinely attractive is cheap: 97–98 % of production time is the inner MMA at
  ~0.1 s per sub-iteration for 3 201 variables. A faster MMA sub-solver (vectorised `subsolv`,
  or a dual solver exploiting m = 4) is the lever, not a different optimiser.

## What would make a fair follow-up (not done)

* A metric that mimics MMA's per-variable scaling (e.g. diagonal of the asymptote
  curvature) instead of Euclidean/Helmholtz; this is where Feppon's tunable metric could help.
* A fixed-iteration wall-time comparison with a fast MMA (the MATLAB inner loop at
  0.1 s per MMA sub-iteration is itself far from optimal).

## Files

`olhoff_fe.py` (FE mirror), `olhoff_problem.py` (Optimizable), `run_olhoff_nso_vs_mma.py`
(runner), `compare_olhoff_nso.py` (table/figures), `results/olhoff_ref_160x20.{mat,log,json}`
(MATLAB reference, produced by `run_olhoff_ref.m`; warm starts go through the sandbox copy `olhoffSolveWarm.m`), `results/*.json|_rho.npy` (Python arms), `results/comparison_<mesh>.{md,csv}`, `results/scaling.md`, `results/predictor_corrector_160x20.md`, `predictor_corrector_table.py`, `scaling_table.py`.
