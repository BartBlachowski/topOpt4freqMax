# 320x40 Heaviside projection continuation assessment

This is a paper-inspired stabilized reproduction experiment, not exact paper reproduction. All cases start from the saved connected 320x40 Case-B design and keep FE model, mass model, nonlinear subeigenvalue formulation, filter radius, move limits, volume constraint, MMA settings, and `mult_tol=0.05` unchanged.

## Comparison

| case | beta schedule | best omega1 | best gap iter | best gap12 | final omega1 | final omega2 | final gap12 | final N | N=2 iters | effCmp8 | final grey | phys vol | final beta | omega1 slope20 | class note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PC0_beta3_fixed | fixed 3.0 | 419.063 | 54 | 0.0169 | 419.063 | 427.038 | 0.0190 | 2 | 100 | 1 | 0.151 | 0.5000 | 3.0 | 0.0237 | fixed-beta reference |
| PC1_beta123_i25 | [1, 2, 3] / 25 | 418.699 | 94 | 0.0105 | 418.699 | 423.108 | 0.0105 | 2 | 86 | 1 | 0.150 | 0.5000 | 3.0 | 0.0646 | best final coalescence |
| PC2_beta1234_i25 | [1, 2, 3, 4] / 25 | 417.010 | 75 | 0.0111 | 408.223 | 424.955 | 0.0410 | 2 | 86 | 1 | 0.128 | 0.5000 | 4.0 | 0.0649 | over-projected final |
| PC3_beta12346_i20 | [1, 2, 3, 4, 6] / 20 | 416.568 | 43 | 0.0121 | 415.710 | 447.413 | 0.0763 | 1 | 47 | 1 | 0.099 | 0.5018 | 6.0 | -0.0286 | over-projected final |

## Required questions

1. Continuation below 1.90% gap: yes. Best final case is `PC1_beta123_i25` with final gap `0.0105`; best transient gap is `0.0105` in `PC1_beta123_i25`.
2. Omega1 upward toward omega2: partly. PC1 keeps omega1 near the beta=3 level (`418.699`), but does not exceed PC0. Stronger beta schedules reduce final omega1.
3. Omega2 stability: PC1 lowers omega2 toward omega1 (`423.108`), which improves coalescence. PC3 drives omega2 away upward at the final beta=6 stage.
4. N=2 persistence: PC0, PC1, and PC2 have persistent N=2; PC3 loses final N=2 despite a long N=2 segment.
5. Connected topology: preserved in all cases.
6. Stronger projection: beta up to 3 helps coalescence; beta 4/6 mostly over-projects or freezes/degrades the branch rather than improving final coalescence.
7. Closest approach to `omega1 ~= omega2 ~= 456.4`: `PC1_beta123_i25` by final coalescence, but `PC0_beta3_fixed` by final omega1. No schedule approaches the published frequency level.

## Final assessment

- Projection role classification: **B. useful stabilizer**.
- It is more than cosmetic: beta continuation 1->2->3 reduces final gap from PC0 `0.0190` to `0.0105` and keeps N=2 active.
- It is not yet a key/dominant reproduction mechanism: the best final frequency remains around `omega1=418.7`, `omega2=423.1`, far below `456.4`, and stronger beta continuation degrades the final state.
- Practical recommendation: keep beta around 3 for this branch; do not continue blindly to beta 4 or 6 without a separate path-control change.