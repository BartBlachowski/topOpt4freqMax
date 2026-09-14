# TERMINATION_AUDIT (Part 13)

```
PEDERSEN_ADAPTIVE_CROSS_MESH_TERMINATION_PASS
```

The rule is PREREGISTRATION.md §10.4. Endpoints use the implementation's own vocabulary:

- `olhoffcurrent_run` / runner status: `NATIVE_CONVERGED | CAP_HIT | SOLVER_FAILURE | RUN_ERROR | UNRECOGNIZED_STOP`;
- solver `res.status`: `CONVERGED | CAP_HIT`.

**The terminal criterion.** It is the design-change rule of `olhoffSolve.m` at `stop.rule = designChange`: ‖Δρ‖₂ < ε with ε = 0.05·√(NE/3200). No guard applies, so settledMove, boxInactiveFraction, ladder and maxDesignChange are all off. It is **not** an objective-change test, so no objective-change quantity is part of the stop logic. That makes ε a constant per-element RMS change of 8.84e-4 at every mesh.

## 1. Endpoints

| mesh | runner status | solver status | log line | outer / cap | final ‖Δρ‖₂ | ε | ‖Δρ‖₂/ε | final max\|Δρ\| | largest box (ceiling 0.1 / floor 0.002) | mean box | nested MMA not converged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 160x20 | NATIVE_CONVERGED | CONVERGED | converged at outer iteration 121 | 121 / 400 | 4.745e-2 | 0.050 | 0.949 | 0.0136 | 0.1 | 0.0084 | 0 |
| 240x30 | NATIVE_CONVERGED | CONVERGED | … 111 | 111 / 400 | 7.456e-2 | 0.075 | 0.994 | 0.0101 | 0.1 | 0.0223 | 0 |
| 320x40 | NATIVE_CONVERGED | CONVERGED | … 101 | 101 / 400 | 9.959e-2 | 0.100 | 0.996 | 0.0087 | 0.1 | 0.0278 | 0 |
| 400x50 | NATIVE_CONVERGED | CONVERGED | … 93 | 93 / 400 | 1.245e-1 | 0.125 | 0.996 | 0.0071 | 0.1 | 0.0267 | 0 |
| 480x60 | NATIVE_CONVERGED | CONVERGED | … 112 | 112 / 400 | 1.484e-1 | 0.150 | 0.989 | 0.0073 | 0.1 | 0.0278 | 0 |
| 560x70 | NATIVE_CONVERGED | CONVERGED | … 130 | 130 / 400 | 1.735e-1 | 0.175 | 0.991 | 0.0076 | 0.1 | 0.0292 | 0 |
| 640x80 | NATIVE_CONVERGED | CONVERGED | … 156 | 156 / 400 | 1.985e-1 | 0.200 | 0.993 | 0.0075 | 0.1 | 0.0292 | 0 |
| 720x90 | NATIVE_CONVERGED | CONVERGED | … 204 | 204 / 400 | 2.237e-1 | 0.225 | 0.994 | 0.0072 | 0.1 | 0.0329 | 0 |
| 800x100 | NATIVE_CONVERGED | CONVERGED | … 246 | 246 / 400 | 2.483e-1 | 0.250 | 0.993 | 0.0096 | 0.1 | 0.0337 | 0 |

**CAP_HIT endpoints: none.** The largest outer count is 246 of 400. The runner's own summary is `cap_summary.any_cap_hit = false`.

## 2. How the natural stop fired

Source: `evidence/TERMINATION_TAIL_METRICS.json`.

- **It is a first crossing, not a deep convergence.** At every mesh the run stops on the first iteration with ‖Δρ‖₂ < ε, at a ratio of 0.949–0.996. From 240x30 up, the ratio approaches 1 **from above** in the last 5–20 iterations, declining by about 1–3 % per iteration. The preceding minima are 1.004–1.025. At 160x20 the ratio oscillates (1.82, 2.62, 1.13, 1.56, 0.95) before crossing. The stop is therefore a threshold crossing of a slowly decaying design-change norm.
- **The box ceiling is still in use.** The largest per-element move box is **at the ceiling 0.1 at every iteration of every run**, so some elements keep taking monotone steps. The mean box decays to 0.008 (160x20) or 0.022–0.034 (other meshes).
- **The floor criterion caught nothing.** Final max|Δρ| is 7–14 % of the largest box. The preregistered test "stop with every box at the floor 0.002" never fired and cannot have: `n_iter_move_max_at_floor = 0` everywhere.
- **The objective is flat at the stop at eight meshes.** The relative change of ω₁ over the last 20 outer iterations lies between −0.095 % and +0.048 %.
- **800x100 is the exception.** Its ω₁ changed +0.125 % over the last 20 iterations and +0.79 % over the last 50, and its maximum is at the final iterate (246). ω₁ was therefore **still rising slowly** when the design-change criterion fired.
- **Grayness is still falling at every mesh.** The final M_nd equals the minimum of the M_nd history everywhere. The designs were still becoming less gray when the stop fired.
- **Volume constraint.** The final |mean ρ − 0.5| ≤ 1.5e-6. The largest in-run |volume error| grows with mesh, from about 1.0e-4 (160x20 and 240x30) to 5.8e-4 (800x100).
- **Nested MMA.** Every sub-optimization converged by its own test (`innerConv` all true). The inner iterations per outer step were 7–46.

## 3. Assessment

- The **formal** endpoint is clean and uniform. All nine runs are NATIVE_CONVERGED by the implementation's recorded criterion, the log line is present, the ratio is below 1, there is no cap, no guard, and no inner failure.
- **Credibility beyond the formal criterion is limited and mesh-dependent:**
  - (i) the stop is a first crossing just under ε, not a deep settling;
  - (ii) at 800x100, ω₁ had not plateaued;
  - (iii) M_nd was still decreasing at every mesh.

  A somewhat smaller ε would very probably have produced more iterations, less gray designs and, at 800x100, a slightly higher ω₁. That is an **inference from the trends, not tested**, and it was deliberately not tested, because retuning ε is out of scope.
- No termination pathology of the kind the preregistration names occurred: no cap, no box-driven stop, no inner failure.
- The heuristic stop is **not** a KKT certificate (HISTORICAL_KKT_CONTEXT.md).
