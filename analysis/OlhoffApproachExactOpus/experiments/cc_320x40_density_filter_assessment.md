# 320x40 density-filter stabilized reproduction diagnostic

Classification target: paper-inspired stabilized reproduction, not strict paper reproduction.
S0 is the existing 100-iteration nonlinear reference. S1 adds only density filtering in the design-to-physical density map; no Heaviside projection was run because S1 did not improve the benchmark direction.

Volume convention for S1: the active volume constraint is `mean(rho_phys) <= volfrac`, where `rho_phys = H*x_design/Hs`. The final design-variable volume is reported separately.

## Comparison

| case | filter | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | total N=2 | effCmp8 | grey frac | central density | ccSolid | final phys vol | final design vol | max update final |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0_reference_NL | none | 411.068 | 431.562 | 0.0499 | 411.068 | 431.562 | 0.0499 | 2 | 13 | 1 | 0.199 | 0.266 | 0.221 | 0.5000 | - | 2.8367e-04 |
| S1_density_filter | density filter only | 387.568 | 406.818 | 0.0497 | 387.568 | 406.818 | 0.0497 | 2 | 100 | 1 | 0.311 | 0.268 | 0.200 | 0.5000 | 0.4996 | 1.6981e-04 |

## Required questions

1. Gap12 below current ~5%: S1 final gap is `0.0497` versus S0 `0.0499`. This is not a material reduction.
2. Omega1 toward omega2: no useful movement; S1 lowers the frequency level (`omega1=387.568`) relative to S0 (`omega1=411.068`).
3. N=2 persistence: S1 is more persistent by count (`100` iterations) than S0 (`13`), but without coalescence or frequency gain.
4. Connected Fig. 3c topology: preserved; S1 effective components = `1`.
5. Closer to `omega1 ~= omega2 ~= 456.4`: no. S1 moves both frequencies farther below 456.4.
6. Density filtering changes the optimizer path mainly by smoothing/lowering the stiffness-effective design; it does not create a stronger coalescence path.
7. Projection was not tested because S1 was not promising under the stated criterion.

## Final assessment

- Classification: **C. paper-inspired stabilized reproduction** as an experiment class, but the result is not a successful near-reproduction.
- This is not exact paper reproduction: density filtering in the design-variable-to-physical-density map is not part of the strict paper-derived exact formulation used in the previous studies.
- Remaining obstacle classification: density filtering alone does not explain the benchmark discrepancy; the likely issue remains optimizer path/control or another undocumented numerical choice.
- Recommendation: stop this density-filter-only branch unless the next objective explicitly allows projection/continuation as a separate, non-exact stabilized optimizer study.