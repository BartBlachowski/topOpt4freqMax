# 320x40 sensitivity-filter radius assessment

All cases start from the connected 320x40 Case-B restart design and use the same stabilized optimizer path: Heaviside projection continuation `1->2->3` every 25 iterations, `move_lim = outer_move = 0.05`, and `mult_tol = 0.05`. Only `rmin_elem` changes.

## Comparison

| rmin | best omega1 | best gap12 | final omega1 | final omega2 | final gap12 | final N | N=2 iters | effCmp8 | final grey | central density | ccSolid | diagonal score | phys vol | class note |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2.0 | 420.395 | 0.0109 | 420.395 | 425.220 | 0.0115 | 2 | 86 | 1 | 0.143 | 0.253 | 0.228 | 0.0561 | 0.5000 | highest frequency |
| 2.5 | 418.699 | 0.0105 | 418.699 | 423.108 | 0.0105 | 2 | 86 | 1 | 0.150 | 0.254 | 0.226 | 0.0553 | 0.5000 | smallest gap |
| 3.0 | 416.638 | 0.0105 | 416.638 | 421.378 | 0.0114 | 2 | 86 | 1 | 0.152 | 0.253 | 0.224 | 0.0547 | 0.5000 | wider-filter lower frequency |

## Questions

1. Gap12 dependence on rmin: weak over this range. Final gap spans `0.0105` to `0.0115`.
2. Omega1 toward omega2: all three cases stay near 1.1% gap, but none approaches `omega1 ~= omega2 ~= 456.4`. Best frequency is rmin `2.0` with omega1 `420.395`.
3. Connectivity stability: preserved in all cases.
4. Morphology alteration: no material class change. Central density, ccSolid, and diagonal-score proxies shift modestly, consistent with member-width smoothing rather than a different topology class.

## Final assessment

- Sensitivity-filter radius influences the frequency level more than coalescence in this local range.
- Smaller rmin=2.0 gives the highest final omega1 (`420.395`) but not the smallest gap.
- Reference rmin=2.5 gives the smallest final gap (`0.0105`), only slightly better than rmin=2.0/3.0.
- Larger rmin=3.0 lowers both frequencies (`omega1=416.638`) without improving coalescence.
- Conclusion: member-width regularization affects the branch quantitatively, but it is not the primary control on the remaining benchmark discrepancy.