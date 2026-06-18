# 320x40 Heaviside projection diagnostic

P0 is the existing 100-iteration nonlinear reference. P1/P2 use elementwise Heaviside projection only in the design-to-physical-density map, with no density filter in that map. The existing sensitivity filter and filter radius are unchanged.

Volume convention for P1/P2: the active volume constraint is `mean(rho_phys) <= volfrac`, where `rho_phys = H_beta(x_design)`. Final design-variable volume is reported separately.

## Comparison

| case | projection | beta | best omega1 | best omega2 | best gap12 | final omega1 | final omega2 | final gap12 | Nfinal | total N=2 | max N=2 run | effCmp8 | grey frac | central density | phys vol | design vol | class |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0_reference | none | - | 411.068 | 431.562 | 0.0499 | 411.068 | 431.562 | 0.0499 | 2 | 13 | 4 | 1 | 0.199 | 0.266 | 0.5000 | - | reference |
| P1_beta1 | Heaviside | 1.0 | 412.166 | 432.787 | 0.0500 | 412.166 | 432.787 | 0.0500 | 1 | 63 | 3 | 1 | 0.191 | 0.264 | 0.4996 | 0.5003 | neutral |
| P2_beta3 | Heaviside | 3.0 | 419.063 | 427.038 | 0.0190 | 419.063 | 427.038 | 0.0190 | 2 | 100 | 100 | 1 | 0.151 | 0.255 | 0.5000 | 0.5026 | helpful |

## Required questions

1. Gap12 reduction: P1 is `0.0500` versus P0 `0.0499`; P2 is `0.0190`. P2 is the only material reduction.
2. Omega1 toward omega2: P2 raises omega1 to `419.063` while lowering the gap; P1 only slightly raises both frequencies without closing the gap.
3. N=2 persistence: P0 `13` iterations, P1 `63`, P2 `100`. P2 is more persistent than P0/P1.
4. Topology connected: effective components are P0 `1`, P1 `1`, P2 `1`.
5. Closer to `omega1 ~= omega2 ~= 456.4`: P2 is closer by coalescence and omega1, but still far below 456.4 (`omega1=419.063`, `omega2=427.038`).

## Final assessment

- P1 beta=1 classification: **neutral**.
- P2 beta=3 classification: **helpful**.
- Hypothesis P1 status: partially supported. Moderate projection materially reduces gap12 and improves N=2 persistence while preserving connected topology, but it does not reach the published 456.4 coalesced benchmark.
- Projection-only is a paper-inspired stabilization, not exact paper reproduction. It should remain diagnostic unless a later production-oriented stabilized solver is explicitly requested.