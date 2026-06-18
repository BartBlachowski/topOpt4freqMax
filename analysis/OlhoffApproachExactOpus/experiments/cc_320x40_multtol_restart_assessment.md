# 320x40 multiplicity-tolerance restart assessment

Starting point: saved Case B local move-control final design (`omega1=409.492`, `omega2=435.009`).
Only `mult_tol` differs across cases; move limits remain `0.05`.

## Comparison

| case | mult_tol | first N=2 | max consecutive N=2 | total N=2 iters | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | effCmp8 | maxDrho final | omega1 slope last20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0_mult001 | 0.010 | - | 0 | 0 | 411.162 | 430.773 | 0.0477 | 411.162 | 430.773 | 0.0477 | 1 | 1 | 2.8767e-04 | 0.0147 |
| B1_mult005 | 0.050 | 84 | 4 | 13 | 411.068 | 431.562 | 0.0499 | 411.068 | 431.562 | 0.0499 | 2 | 1 | 2.8367e-04 | 0.0092 |
| B2_mult010 | 0.100 | - | 0 | 0 | 410.043 | 437.735 | 0.0675 | 410.043 | 437.735 | 0.0675 | 3 | 1 | 1.8417e-04 | 0.0050 |

## Per-case notes

### B0_mult001
- Runtime: 53.2 s; iterations: 100
- N=2 activation: first=None, total_iters=0, max_consecutive=0
- Best: iter 100, omega1=411.162, omega2=430.773, omega3=449.122, gap12=0.0477, N=1
- Final: omega1=411.162, omega2=430.773, omega3=449.122, gap12=0.0477, N=1, volume=0.4999
- Connectivity: effective components=1, raw components=14, ccSolid=0.220, central density=0.265
- Last-20 behavior: omega1 slope=0.0147/iter, gap12 slope=-0.00014/iter, omega1 std=0.085

### B1_mult005
- Runtime: 55.4 s; iterations: 100
- N=2 activation: first=84, total_iters=13, max_consecutive=4
- Best: iter 100, omega1=411.068, omega2=431.562, omega3=449.280, gap12=0.0499, N=2
- Final: omega1=411.068, omega2=431.562, omega3=449.280, gap12=0.0499, N=2, volume=0.5000
- Connectivity: effective components=1, raw components=14, ccSolid=0.221, central density=0.266
- Last-20 behavior: omega1 slope=0.0092/iter, gap12 slope=-0.00002/iter, omega1 std=0.053

### B2_mult010
- Runtime: 91.2 s; iterations: 100
- N=2 activation: first=None, total_iters=0, max_consecutive=0
- Best: iter 100, omega1=410.043, omega2=437.735, omega3=450.924, gap12=0.0675, N=3
- Final: omega1=410.043, omega2=437.735, omega3=450.924, gap12=0.0675, N=3, volume=0.5000
- Connectivity: effective components=1, raw components=12, ccSolid=0.223, central density=0.267
- Last-20 behavior: omega1 slope=0.0050/iter, gap12 slope=0.00005/iter, omega1 std=0.029

## Final assessment

- Connected topology status: preserved in all cases.
- N=2 activation occurred: yes.
- Smallest final gap12: B0_mult001 = 0.0477.
- Highest best omega1: B0_mult001 = 411.162.
- Bottleneck classification should be based on whether forced/earlier N=2 activation reduces the gap and raises omega1 while preserving connectivity. If activation occurs without improvement, the obstacle is optimizer path/generalized-gradient behavior after activation rather than late detection.