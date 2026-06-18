# 320x40 basin-initialization assessment

All cases use the same stabilized optimizer settings: Heaviside projection continuation `1->2->3` every 25 iterations, `move_lim = outer_move = 0.05`, and `mult_tol = 0.05`. Only the initial design field changes.

## Comparison

| initial state | initial grey | best omega1 | best gap12 | final omega1 | final omega2 | final gap12 | final N | N=2 iters | effCmp8 | final grey | central density | phys vol | class note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| raw | 0.201 | 418.677 | 0.0104 | 418.677 | 423.071 | 0.0105 | 2 | 85 | 1 | 0.150 | 0.254 | 0.5000 | best useful basin |
| thresholded | 0.000 | 7.807 | 0.0046 | 6.794 | 7.038 | 0.0358 | 3 | 16 | 2 | 0.000 | 0.252 | 0.5001 | collapsed low-frequency basin |
| smoothed | 0.311 | 395.396 | 0.0165 | 394.329 | 409.784 | 0.0392 | 2 | 100 | 1 | 0.213 | 0.254 | 0.5000 | lower-frequency basin |
| thresholded_smoothed | 0.236 | 379.165 | 0.0116 | 364.987 | 375.253 | 0.0281 | 2 | 66 | 2 | 0.185 | 0.261 | 0.5006 | lower-frequency basin |

## Questions

1. Different initializations converge to different frequencies: yes. Final omega1 spread is `411.883` rad/s.
2. Movement toward `omega1 ~= omega2 ~= 456.4`: no. The best final omega1 is `418.677` from `raw`, still far below 456.4.
3. Connected topology preservation: effective single-component topology appears in `2/4` cases, but the thresholded case has a collapsed low-frequency response and should not be considered a useful connected optimum.
4. Final state basin-dependence: yes. Raw, smoothed, thresholded, and thresholded+smoothed starts end in distinct frequency/gap regimes.

## Final assessment

- Hypothesis P4 is supported in the weak sense: the optimizer is basin-sensitive.
- It is not supported as the primary explanation for the remaining benchmark gap: none of the alternate initial states finds a higher-frequency coalesced basin near 456.4.
- The useful basin remains the raw connected design under PC1 settings; smoothing/thresholding pushes the run into lower-frequency or collapsed basins.
- Conclusion: basin selection matters, but the missing benchmark mechanism is not simply recovered by these local initial-state perturbations.