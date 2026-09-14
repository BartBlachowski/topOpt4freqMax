# Discontinuities and alternative explanations

Regimes are evaluated on all points; no smooth-family assumption is made.

| Transition | Observed discontinuity | Most supported reading |
| --- | --- | --- |
| 160→240 | gap 1.45→16.18%; stencil5→9; end holes simplify; IoU 0.797 | Different endpoint morphology/spectrum; filter-support association possible, causality unisolated |
| 240→320 | final stage3→2; Mnd15.60→23.36%; gap remains10.74% | Direct change in where legacy stopping is admitted |
| 320→400 | Mnd23.36→32.33%; density L1=0.0891; omega loses1.85% | Greater premature-stop quality loss; historical E common-mesh comparison supports this |
| 400→560 | gradual objective loss/gray increase; gap falls | Fine legacy evolution, no demonstrated three-rung phenomenon |
| 560→640 | gap enters<1%; grayness keeps increasing | Spectral coalescence trend does not certify density maturity |
| 640→720 | close densities (IoU 0.960); eigensolve/outer jumps; multiple-J warnings7→43 | Apparently stable pair plus separate numerical spectral/cost regime change |
| 720→800 | outer223→170; omega159.086→153.302; Mnd41.24→50.66; IoU 0.753; warnings43→81 | Strong endpoint/stopping regime change; no convincing asymptotic refinement |

## Competing causes ranked by evidence

For **failure to answer the central three-rung question**, the proven cause is unpromoted configuration and discarded histories. FE, filter, MMA or mass interpolation cannot explain away which config was executed.

For **grayness growth and unstable endpoint quality**, rank: (1) stopping/continuation defect—direct logs show every stop immediately after a move halving, and historical same-mesh E paths recover substantially cleaner C320/C400 designs; (2) mesh-dependent topology evolution—direct field differences, but inseparable from stopping; (3) multiplicity/next-mode approximation—direct increasing warning count, unknown impact; (4) filter discretization and FE discretization—real discretization changes, no controlled attribution; (5) mass interpolation and MMA globalization—fixed across meshes, plausible interacting factors but no isolating comparison. The *three-rung* controller defect hypothesis has no fine-mesh test here.

For **C800 discontinuity**, evidence ranks premature admission and altered trajectory ahead of simple FE order arguments. Near-multiple J handling is a stronger specific numerical concern than an alleged radius mismatch. Stable effective cone radius and abrupt loss of iterations weaken a filter-only explanation. A distinct optimization basin is possible, but not established without histories or a controlled same-mesh comparison. Small gap12 at C800 does not exonerate the J=3 approximation.

For **computational expense**, direct cost data rank nested MMA per-step expense first, outer trajectory length second, FE/eigensolve dimension third in total share (although its exponent is higher). No evidence shows increasing mean nested iteration count per outer. Host load/thermal effects could contribute to the late eigensolve kink, but no load telemetry can rank them quantitatively.

For **historical 0.005 pathology**, same-prefix four-rung evidence isolates the extra rung and its persistent low-amplitude/cancellation dynamics as the leading explanation. Physical formulation is fixed. Changing filter, mass or FE is unnecessary to explain 1248 additional C320 iterations with negligible density change.

These rankings concern evidence strength, not asserted causal probabilities. No recommendation to change several ingredients at once follows.
