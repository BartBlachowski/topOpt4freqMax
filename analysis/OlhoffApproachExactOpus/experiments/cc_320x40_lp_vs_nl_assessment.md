# 320x40 LP reduction vs nonlinear subeigenvalue assessment

Starting point: saved connected 320x40 Case-B local move-control final design.
Both cases use `move_lim = outer_move = 0.05`, 100 restart iterations, and `mult_tol = 0.05` so the N>1 subproblem is exercised. Only `subproblem_formulation` differs.

## Comparison

| case | formulation | first N=2 | max consecutive N=2 | total N=2 iters | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | effCmp8 | final volume | maxDrho final | omega1 slope last20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NL_nonlinear | nonlinear | 84 | 4 | 13 | 411.068 | 431.562 | 0.0499 | 411.068 | 431.562 | 0.0499 | 2 | 1 | 0.5000 | 2.8367e-04 | 0.0092 |
| LP_reduced | lp_reduced | 84 | 2 | 2 | 410.927 | 431.472 | 0.0500 | 410.927 | 431.472 | 0.0500 | 2 | 1 | 0.4999 | 4.2439e-06 | 0.0152 |

## LP off-diagonal diagnostic

For the LP-reduced case, the paired off-diagonal constraints target `f_sk^T Delta rho = 0` for `s != k` inside the inner MMA step.
- Maximum recorded absolute residual: `1.584901e-05`.
- Final recorded absolute residual: `1.584901e-05`.
- The residual is not numerically zero, so the MMA embedding enforces the LP equalities approximately rather than as an exact equality-constrained LP solve.

## Questions

1. LP movement of omega1 toward omega2: no; final gaps are LP `0.0500` and NL `0.0499`.
2. Faster gap decrease: LP has the more negative last-20 gap slope (`-0.00014` vs `-0.00002`).
3. N=2 persistence: LP has `2` N=2 iterations, NL has `13`.
4. Connected topology: preserved in both cases.
5. Optimization path alteration: best omega1 comes from `NL_nonlinear` at `411.068`.
6. Closest to 456.4: final omega1 is LP `410.927` and NL `411.068`; best omega1 is LP `410.927` and NL `411.068`.
7. Benchmark-gap explanation: the best gap/coalescence case is `NL_nonlinear` with final gap `0.0499`.

## Per-case notes

### NL_nonlinear
- Runtime: 53.7 s; iterations: 100.
- Best: iter 100, omega1=411.068, omega2=431.562, omega3=449.280, gap12=0.0499, N=2.
- Final: omega1=411.068, omega2=431.562, omega3=449.280, gap12=0.0499, N=2, volume=0.5000.
- Connectivity: effective components=1, raw components=14, ccSolid=0.221, central density=0.266.
- Last-20 behavior: omega1 slope=0.0092/iter, gap12 slope=-0.00002/iter, omega1 std=0.053.

### LP_reduced
- Runtime: 45.9 s; iterations: 85.
- Best: iter 84, omega1=410.927, omega2=431.472, omega3=449.107, gap12=0.0500, N=2.
- Final: omega1=410.927, omega2=431.472, omega3=449.107, gap12=0.0500, N=2, volume=0.4999.
- Connectivity: effective components=1, raw components=14, ccSolid=0.221, central density=0.266.
- Last-20 behavior: omega1 slope=0.0152/iter, gap12 slope=-0.00014/iter, omega1 std=0.088.

## Final assessment

- Remaining discrepancy classification: B. optimizer path independent of embedding is the primary explanation.
- Connected topology status: preserved.
- Frequency/coalescence status: best final gap is `0.0499` from `NL_nonlinear`; neither case is judged coalesced unless this is near zero.
- The LP-reduced experiment should be interpreted as diagnostic only: it keeps MMA and the rest of the update mechanism unchanged, so it tests the embedding effect without replacing the optimizer by a true standalone LP solver.