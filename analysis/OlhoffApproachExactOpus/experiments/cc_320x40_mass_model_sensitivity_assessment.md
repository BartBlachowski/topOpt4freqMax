# 320x40 Du-Olhoff mass interpolation sensitivity assessment

All cases start from the connected 320x40 Case-B restart design and use the same stabilized optimizer path: Heaviside projection continuation `1->2->3` every 25 iterations, `move_lim = outer_move = 0.05`, `rmin_elem = 2.5`, and `mult_tol = 0.05`. Only `mass_mode` changes.

## Comparison

| case | mass mode | best omega1 | best gap12 | final omega1 | final omega2 | final gap12 | final N | N=2 iters | effCmp8 | final grey | central density | phys vol | class note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M1_step | du2007_step | 418.077 | 0.0127 | 418.077 | 423.582 | 0.0132 | 2 | 78 | 1 | 0.151 | 0.253 | 0.5000 | reference-like |
| M2_c0 | du2007_c0 | 419.105 | 0.0107 | 419.105 | 423.622 | 0.0108 | 2 | 83 | 1 | 0.149 | 0.254 | 0.5000 | highest omega1 |
| M3_c1 | du2007_c1 | 418.699 | 0.0105 | 418.699 | 423.108 | 0.0105 | 2 | 86 | 1 | 0.150 | 0.254 | 0.5000 | smallest gap |

## Questions

1. Smallest gap12: `M3_c1` / `du2007_c1` with final gap `0.0105`.
2. Highest omega1 while preserving connectivity: `M2_c0` / `du2007_c0` with omega1 `419.105` and effective components `1`.
3. Stable N=2: yes, all tested models are stable by this criterion (`M2_c0, M3_c1`).
4. Ranking versus coarse-mesh studies: on this connected stabilized branch the ranking is weak and local. It does not reproduce a coarse-mesh-style mass-model dominance; all three remain in the same connected class and close frequency/gap range.

## Final assessment

- Final gap spread is small: `0.0105` to `0.0132`.
- Final omega1 spread is also small among useful connected cases: `418.077` to `419.105`.
- Relative to c1 reference (`omega1=418.699`, gap=0.0105`), c0 raises omega1 slightly but c1 remains marginally better for coalescence.
- Conclusion: the connected branch is mildly sensitive to the Du-Olhoff mass interpolation choice, but mass interpolation is not the primary control on the remaining benchmark discrepancy.