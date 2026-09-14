# PERFORMANCE_AUDIT (Part 16)

## Measurement context

- **Machine and process.** Mac Studio, Apple M1 Max, 64 GB RAM, macOS 26.6.2, MATLAB R2025b. One `-batch` process with `maxNumCompThreads(1)`, run under `caffeinate`. No other agent computation ran during the campaign; the user's idle desktop MATLAB session stayed open.
- **Timers** are the runner's own, and the tap matched them bit for bit:
  - total = caller-side `tic/toc` around `olhoffSolve`;
  - tOuter, tEig (FE assembly plus `eigs`), tGrad and tInner (nested MMA) come from `res.hist`.
- **Excluded from every timer:** the hooks, the E1/E2/E3 evaluator, export and plotting.
- **Timing accounting.** The runner identity T_total = T_outer\inner + T_inner + T_overhead holds with a residual of **0 s** at all nine meshes. The independent cross-check against the solver's self-reported wall time is ≤ 6.1e-4 s.
- **Elapsed time.** Launch to exit took 01:29:50 → 05:34:54, i.e. 14 704 s. The nine solves account for 14 636.8 s. The remaining ~67 s cover MATLAB start, preflight, warm-up, evaluator and export.

## A. Total cost

| mesh | NE | DOFs (total) | outer | inner | **total wall [s]** | total wall [h:mm:ss] | eigensolves (outer+1) |
|---|---|---|---|---|---|---|---|
| 160x20 | 3 200 | 6 762 | 121 | 2 369 | 223.9 | 0:03:44 | 122 |
| 240x30 | 7 200 | 14 942 | 111 | 2 077 | 354.9 | 0:05:55 | 112 |
| 320x40 | 12 800 | 26 322 | 101 | 2 001 | 427.2 | 0:07:07 | 102 |
| 400x50 | 20 000 | 40 902 | 93 | 1 913 | 570.1 | 0:09:30 | 94 |
| 480x60 | 28 800 | 58 682 | 112 | 2 137 | 981.4 | 0:16:21 | 113 |
| 560x70 | 39 200 | 79 662 | 130 | 2 371 | 1 425.0 | 0:23:45 | 131 |
| 640x80 | 51 200 | 103 842 | 156 | 2 868 | 2 215.8 | 0:36:56 | 157 |
| 720x90 | 64 800 | 131 222 | 204 | 3 752 | 3 537.5 | 0:58:58 | 205 |
| 800x100 | 80 000 | 161 802 | 246 | 4 650 | 4 901.0 | 1:21:41 | 247 |

The outer count is **non-monotone** in the mesh. It falls from 121 to 93 over 160x20 → 400x50 and rises to 246 at 800x100. Total wall time therefore mixes two effects: cost per outer iteration and the number of outer iterations.

## B. Cost per outer iteration

| mesh | wall / outer [s] | mean tOuter [s] | median tOuter [s] | first-20 mean / median tOuter [s] | last-20 mean / median tOuter [s] | eig / outer [s] | eig share of wall | inner / outer [s] | inner per MMA iteration [s] | inner share of wall | outer excl. inner / outer [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 160x20 | 1.851 | 1.850 | 1.888 | 0.858 / 0.837 | 1.814 / 1.279 | 0.0324 | 1.8 % | 1.815 | 0.0927 | 98.1 % | 0.035 |
| 240x30 | 3.198 | 3.197 | 3.789 | 1.098 / 1.011 | 3.803 / 3.808 | 0.0626 | 2.0 % | 3.130 | 0.1673 | 97.9 % | 0.067 |
| 320x40 | 4.230 | 4.228 | 4.254 | 1.949 / 1.825 | 5.270 / 5.244 | 0.1086 | 2.6 % | 4.113 | 0.2076 | 97.2 % | 0.116 |
| 400x50 | 6.130 | 6.128 | 6.823 | 2.848 / 2.793 | 8.427 / 8.481 | 0.1764 | 2.9 % | 5.942 | 0.2888 | 96.9 % | 0.186 |
| 480x60 | 8.762 | 8.759 | 9.397 | 3.489 / 3.364 | 11.917 / 11.913 | 0.2725 | 3.1 % | 8.473 | 0.4441 | 96.7 % | 0.287 |
| 560x70 | 10.962 | 10.958 | 10.631 | 4.188 / 4.314 | 16.372 / 16.509 | 0.3737 | 3.4 % | 10.565 | 0.5793 | 96.4 % | 0.393 |
| 640x80 | 14.204 | 14.200 | 13.998 | 5.233 / 5.193 | 22.432 / 22.270 | 0.4939 | 3.5 % | 13.679 | 0.7441 | 96.3 % | 0.521 |
| 720x90 | 17.341 | 17.336 | 18.020 | 6.943 / 6.917 | 24.325 / 24.270 | 0.8925 | 5.1 % | 16.407 | 0.8921 | 94.6 % | 0.928 |
| 800x100 | 19.923 | 19.918 | 21.730 | 7.868 / 8.316 | 27.855 / 27.886 | 1.1143 | 5.6 % | 18.759 | 0.9924 | 94.2 % | 1.159 |

Observations. These are descriptive; nothing was re-run to explain them.

1. **The nested MMA dominates:** 94–98 % of wall time. The FE assembly plus `eigs` is 1.8–5.6 %, and its share grows with mesh.
2. **Cost per outer iteration rises during a run.** At every mesh the last-20 tOuter is 2.1–4.3× the first-20 tOuter, while the MMA iterations per outer step fall slightly (e.g. 800x100: 19.8 → 17.1). The **cost per MMA iteration therefore grows 2.6× (160x20) to 4.9× (640x80) within a run**, estimated as first-20 versus last-20 tOuter per MMA iteration. This matches the per-element box shrinking over the run (mean box 0.1 → 0.02–0.03; 0.008 at 160x20). The cause was not diagnosed here.
3. Mean and median tOuter differ by up to 18 % (240x30: 3.20 against 3.79) because of that within-run growth. Either one alone understates the drift.

## Scaling with problem size

The fits below are descriptive. No asymptotic law is claimed.

**Method.** Ordinary least squares of log(y) on log(x) over **all nine points**. No point is excluded, and all nine rows are NATIVE_CONVERGED (none censored). Sensitivity is the exponent range when each point is dropped in turn (leave-one-out). The fits are in `METRICS.json` `fits`; the runner's own `confbench_scaling_fit` gives the same numbers for the quantities it fits.

| quantity y | p vs NE | R² | leave-one-out p range | point whose removal moves p most | p vs total DOFs |
|---|---|---|---|---|---|
| total wall | **0.961** | 0.920 | 0.888 – 1.140 | 160x20 (→ 1.140); 800x100 (→ 0.888) | 0.974 |
| wall / outer | **0.755** | 0.992 | 0.745 – 0.797 | 160x20 (→ 0.797); 320x40 (→ 0.745) | 0.765 |
| median tOuter | 0.742 | 0.986 | 0.725 – 0.764 | 240x30 (→ 0.764); 800x100 (→ 0.725) | 0.752 |
| eig time / outer | **1.101** | 0.984 | 1.064 – 1.193 | 160x20 (→ 1.193); 800x100 (→ 1.064) | 1.116 |
| inner time / outer | 0.743 | 0.993 | 0.734 – 0.783 | 160x20 (→ 0.783); 320x40 (→ 0.734) | 0.753 |
| inner time / MMA iteration | 0.761 | 0.986 | 0.750 – 0.804 | 160x20 (→ 0.804); 320x40 (→ 0.750) | 0.771 |
| outer iterations | 0.206 | 0.453 | 0.140 – 0.343 | 160x20 (→ 0.343); 800x100 (→ 0.140) | 0.209 |

**How to read these fits:**

- **Total wall is poorly described by one power law.** The residuals of log T are +0.39 at 160x20, −0.43 at 400x50 and +0.39 at 800x100, a systematic U shape driven by the non-monotone outer count. The exponent ranges from 0.89 to 1.14 depending on a single point. The runner's fixed-exponent fit T = C·NE^1.5 (R² 0.990 on T) is dominated by the largest meshes and is not a better law, only a different weighting.
- **Per-outer cost is well described over this range**, with p ≈ 0.75 and R² 0.99. An exponent below 1 over 3 200–80 000 elements means that size-independent per-iteration overhead still matters at small meshes. It should not be extrapolated.
- **The eigen-analysis per outer iteration grows slightly superlinearly** (p ≈ 1.10), as expected for sparse shift-invert `eigs`. It is a small share of the cost.
- **DOFs versus elements.** DOFs = 2(nelx+1)(nely+1) is not exactly proportional to NE, so exponents against DOFs are 0.003–0.015 higher. The two reads are equivalent.
- **Figures:** `figures/total_wall_vs_size.png`, `figures/time_per_outer_vs_size.png`, `figures/outer_iterations_vs_mesh.png`. The runner's own figures are `table1_complexity_fit*.png` in the output root.

## Environment comparison, not a performance claim

The upstream sweep produced the **same designs bit for bit** (SOURCE_SWEEP_COMPARISON.md) but recorded 1.26–2.26× longer wall times (406 s → 224 s at 160x20; 6 169 s → 4 901 s at 800x100). The arithmetic is identical, so the difference is environmental: machine load and execution context at the time of the old sweep. The wall times in this campaign are the ones to cite for the production baseline, and they apply to this machine and configuration only.
