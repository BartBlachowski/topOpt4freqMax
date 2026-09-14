# Targeted rerun recommendations

No full nine-resolution rerun is justified. The frozen statuses, timing records, and common-evaluator outputs are internally consistent, and censoring protected every fit.

Three narrow diagnostic follow-ups are required before an unqualified paper freeze because the original campaign did not retain enough evidence to close the causal questions. None is intended to improve a benchmark number.

1. **Olhoff 640x80, identical S1 profile and 1600-work horizon, diagnostics only.** Hypothesis: the deterministic late N=1/2 branch switching creates a degenerate LP trajectory that drives `dual-simplex-highs` to exit flag 0. Record the failed-attempt native spectrum, N, gap12, lamref, constraint-row norms/rank, active bounds, and the full `linprog` output/message. Do not change LP options. Confirmation is the same failure at attempted k=1067 with an iteration-limit message and a diagnosed degeneracy/conditioning signature; refutation is a different trajectory or failure mechanism.

2. **Yuksel 800x100, identical 2000-iteration profile, history retention enabled.** Hypothesis: the large terminal max change is localized oscillation/topology turnover rather than a smooth approach to tolerance. Record at least the final 300 Stage-2 iterations (max/RMS dx, objective, volume, grayness, binary turnover, and modal spectrum). Keep the cap and tolerance unchanged. Confirmation is persistent/oscillatory max dx with small RMS/objective change; refutation is a monotone decay truncated near tolerance. Only if the latter occurs should a separately preregistered modest Stage-2 extension be considered.

3. **Proposed 160x20, identical frozen profile, deterministic diagnostic replay.** Hypothesis: the native ~109 triplet consists of low-density/void-model modes made visible by the native `Emin/E0=1e-9` plus linear mass interpolation; common E1's stiffer void floor and E2/E3's low-density mass suppression remove them. Retain the final density, history, and mode shapes/energy localization; do not change settings. Confirmation is bitwise/near-bitwise reproduction of the density and ~109 triplet with modes localized in low-density regions, while frozen common evaluators reproduce ~154 raw and ~163 binary. Refutation is failure to reproduce or global structural modes at ~109.

These follow-ups are diagnostic evidence acquisition, not replacement observations. Existing capped/failed rows remain censored.
