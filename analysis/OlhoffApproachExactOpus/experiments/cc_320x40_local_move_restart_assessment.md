# 320x40 local move-control restart assessment

Starting point: saved long-horizon snapshot at iteration 150 (`omega1=388.281`, `omega2=460.834`).
Only `move_lim` and `outer_move` differ across cases; all other settings are inherited from the 320x40 connected run.

## Comparison

| case | move | best iter | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | effCmp8 | maxDrho final | omega1 slope last20 | omega1 std last20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline_move020 | 0.200 | 71 | 388.305 | 455.959 | 0.1742 | 379.460 | 468.925 | 0.2358 | 1 | 1 | 1.0000e-01 | 0.0912 | 6.729 |
| B_move005 | 0.050 | 100 | 409.492 | 435.009 | 0.0623 | 409.492 | 435.009 | 0.0623 | 1 | 1 | 2.9072e-04 | 0.0193 | 0.111 |
| C_move0025 | 0.025 | 100 | 390.629 | 460.695 | 0.1794 | 390.629 | 460.695 | 0.1794 | 1 | 1 | 1.2782e-04 | 0.0209 | 0.120 |

## Per-case notes

### A_baseline_move020
- Runtime: 252.9 s; iterations: 100
- Best: iter 71, omega1=388.305, omega2=455.959, omega3=474.766, gap12=0.1742, N=1
- Final: omega1=379.460, omega2=468.925, omega3=511.288, gap12=0.2358, N=1, volume=0.4999
- Connectivity: effective components=1, raw components=9, ccSolid=0.225, central density=0.295
- Last-20 behavior: omega1 slope=0.0912/iter, gap12 slope=-0.00083/iter, omega1 std=6.729

### B_move005
- Runtime: 66.0 s; iterations: 100
- Best: iter 100, omega1=409.492, omega2=435.009, omega3=448.884, gap12=0.0623, N=1
- Final: omega1=409.492, omega2=435.009, omega3=448.884, gap12=0.0623, N=1, volume=0.4999
- Connectivity: effective components=1, raw components=13, ccSolid=0.222, central density=0.268
- Last-20 behavior: omega1 slope=0.0193/iter, gap12 slope=-0.00015/iter, omega1 std=0.111

### C_move0025
- Runtime: 46.0 s; iterations: 100
- Best: iter 100, omega1=390.629, omega2=460.695, omega3=471.395, gap12=0.1794, N=1
- Final: omega1=390.629, omega2=460.695, omega3=471.395, gap12=0.1794, N=1, volume=0.5000
- Connectivity: effective components=1, raw components=9, ccSolid=0.238, central density=0.285
- Last-20 behavior: omega1 slope=0.0209/iter, gap12 slope=-0.00007/iter, omega1 std=0.120

## Final assessment

- Connected topology status: preserved in all cases.
- Smallest final gap12: B_move005 = 0.0623.
- Highest best omega1: B_move005 = 409.492.
- Multiplicity status: not approaching.
- Obstacle classification: compare the reduction in last-20 omega1 oscillation and final gap across A-C. If smaller move sharply reduces oscillation and gap, obstacle A (step size) dominates; if not, the remaining obstacle is the generalized-gradient/MMA path rather than local move size alone.