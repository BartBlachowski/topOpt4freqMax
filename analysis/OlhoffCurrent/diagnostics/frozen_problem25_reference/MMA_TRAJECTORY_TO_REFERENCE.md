# MMA_TRAJECTORY_TO_REFERENCE — Part 12

FROZEN-SUBPROBLEM REPLAY: `scripts/fp_replay.m`, an audit-only
character-for-character mirror of `innerLoop.m` with the stop tolerance set
to 1e−12 and checkpoints recorded (every iteration 1–50, every 10 to 100,
every 50 to 1 000, every 100 to 5 000; 113 checkpoints with `x`, MMA duals
`lam, xsi, eta`, `y, z`). 5 000 sub-iterations, 5 984 s wall (single thread,
shared host). No density updated. `evaluations/mma_replay.mat`,
`mma_replay_summary.json`, `mma_comparison.json`.

## Fidelity

| check | result |
|---|---|
| iterate 19 vs production `DRHO(:,386)` | **bitwise equal** |
| iterate 500 vs prior audit `stC.xFinal` | **bitwise equal** |
| max `y` / max `z` over 5 000 iterations | 5.9e−10 / 5.9e−7 (MMA auxiliaries inactive: the MMA problem is (25)) |
| converged (relStep < 1e−12) | no; final relStep 2.08e−3, min 4.2e−4 |

The replay is the production iteration, only run longer.

## Checkpoint table (selected)

| iter | `bs` | G | dist₂/‖ref‖ | cosine | max\|drho\|/move | KKT (MMA duals) | relStep |
|---|---|---|---|---|---|---|---|
| 1 | 0.9999904 | 1.0054 | 0.999 | 0.467 | 0.007 | 0.020 | 1.0 |
| 3 | 0.9995951 | 1.2259 | 0.998 | 0.229 | 0.019 | 0.365 | 0.47 |
| 10 | 0.9998943 | 1.0590 | 0.997 | 0.496 | 0.037 | 0.329 | 8.8e−2 |
| **19** (production stop) | **1.0000050** | **0.9972** | **0.995** | **0.669** | **0.062** | **0.360** | 4.1e−2 |
| 50 | 1.0002184 | 0.8781 | 0.984 | 0.639 | 0.153 | 0.141 | 1.9e−2 |
| 100 | 1.0003663 | 0.7956 | 0.970 | 0.626 | 0.297 | 0.224 | 9.6e−3 |
| 200 | 1.0005110 | 0.7149 | 0.943 | 0.633 | 0.586 | 0.447 | 6.5e−3 |
| 500 | 1.0007025 | 0.6080 | 0.921 | 0.598 | 0.876 | 0.586 | 3.7e−3 |
| 1000 | 1.0008037 | 0.5515 | 0.909 | 0.595 | 0.962 | 0.425 | 2.9e−3 |
| 2000 | 1.0008869 | 0.5051 | 0.887 | 0.632 | 0.959 | 0.498 | 6.4e−3 |
| 3000 | 1.0008252 | 0.5396 | 0.867 | 0.664 | 0.972 | 0.720 | 2.6e−3 |
| 4000 | 1.0005771 | 0.6780 | 0.849 | 0.686 | 0.973 | 0.160 | 5.3e−3 |
| 5000 | 1.0009624 | 0.4630 | 0.830 | 0.707 | 0.979 | 0.597 | 2.1e−3 |

The first ten sub-iterates go *below* the starting objective (G > 1: the
first MMA steps lower the predicted eigenvalue) before climbing.

## Classification (preregistration §7)

| criterion | rule | measured | met |
|---|---|---|---|
| approaching in objective | G₅₀₀₀ ≤ 0.1 and G₅₀₀₀ < 0.5·G₁₉ | 0.463; 0.463 < 0.499 | **no** (first clause) |
| approaching in design | dist₅₀₀₀ < 0.5·dist₁₉ and ≤ 0.1 | 0.830 vs 0.497 | **no** |
| moving away | dist₅₀₀₀ > 1.1·dist₁₉ | 0.830 vs 1.094 | no |
| orbiting | late range ≥ 0.2·mean and \|Spearman\| < 0.5 | range 0.079 vs 0.174; Spearman −1.00 | no |

⇒ **E, inconclusive** by the preregistered rules. Describing what the data
show without the rules: over 1 000–5 000 the distance to the reference
decreases **monotonically** (Spearman −1.00) but slowly, from 0.909 to 0.830;
the objective gap **oscillates** between G = 0.44 and 0.85 in the last 1 000
iterations (mean 0.53, minimum 0.4425 at iteration 4997) with no trend to
zero; the cosine rises from 0.59 to 0.71. The sequence drifts toward the
reference in direction and sign pattern (94 % sign agreement at 5 000) while
its magnitude stays a quarter of the reference's and its objective retains
roughly half the achievable gain. It is neither converging to the reference
nor orbiting it at fixed distance: it is a slow, non-monotone-in-objective
drift that would need far more than 5 000 sub-iterations to reach the
bang-bang vertex, if it reaches it at all.

## Why `max|drho|/move ≈ 0.98` was misleading

The prior audit reported the 5 000-state as "97.9 % of the move limit". That
is the largest single element. The reference uses the full move on 9 694
elements and a bound on 28 784; M5000 has **no** element within 1e−6·width of
a bound, 112 gray elements within 10 % of ±move, and a median gray increment
of 15 % of the reference's. The repeated-MMA sequence approaches the box from
the inside at the interior-point rate of its subsolver, one asymptote update
at a time.

## Figures

`FIG_06_mma_maxdrho_vs_iter`, `FIG_07_objective_gap_vs_iter`,
`FIG_08_distance_vs_iter`, `FIG_09_kkt_vs_iter`.
