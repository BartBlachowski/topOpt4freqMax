# Persistent MMA After Globalization Report

Generated: `2026-07-08 12:44:39`

## Scope

The two best stabilization variants from the globalization experiment are rerun with identical FE, sensitivity, filter, boundary, generalized-gradient, and globalization settings. Only MMA memory handling is toggled between outer-iteration restart and persistent asymptote/history state.

Paper-like retention: `abs(omega1 - 456.4)/456.4 <= 0.02` and `gap12 <= 0.005` for `20` consecutive outer iterations. Coalescence retention only requires `gap12 <= 0.005` for `20` iterations.

## Summary

| variant | MMA memory | final omega1 | final omega2 | final N | volume | rho min/max | paper streak | coal streak | rejected trials | alpha min/median | move min/median | connected | support-connected |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `D_trust_0p1_restarted` | restarted MMA state each outer iteration | 488.9747 | 489.3673 | 2 | 0.49998 | 0.001 / 1 | 2 | 78 | 0 | 0.5 / 0.5 | 0.02 / 0.02 | no | no |
| `D_trust_0p1_persistent` | persistent MMA asymptotes/history | 13.3289 | 42.6076 | 1 | 0.48872 | 0.001464 / 1 | 0 | 0 | 0 | 0.5 / 0.5 | 0.2 / 0.2 | no | no |
| `E_combined_0p25_restarted` | restarted MMA state each outer iteration | 687.3682 | 687.4501 | 2 | 0.46704 | 0.001 / 1 | 0 | 87 | 641 | 0 / 0 | 0.05 / 0.05 | no | no |
| `E_combined_0p25_persistent` | persistent MMA asymptotes/history | 1181.7773 | 1196.8021 | 1 | 0.34560 | 0.001031 / 0.9996 | 0 | 9 | 562 | 0 / 0 | 0.05 / 0.2 | no | no |

## Pairwise Answer

- `D_trust_0p1`: restarted final `488.9747/489.3673`, persistent final `13.3289/42.6076`; paper streak `2 -> 0`, coalescence streak `78 -> 0`.
- `E_combined_0p25`: restarted final `687.3682/687.4501`, persistent final `1181.7773/1196.8021`; paper streak `0 -> 0`, coalescence streak `87 -> 9`.

## Conclusion

Persistent MMA does not reduce over-optimization back toward the paper basin in this experiment. For `D_trust_0p1`, persistence destroys the retained coalesced state. For `E_combined_0p25`, persistence moves to a higher, non-paper and non-retained-coalesced state. The off-target coalesced optima are retained by the restarted MMA runs, not improved by persistent MMA.

## Evidence Files

- `persistent_mma_after_globalization_results/persistent_mma_summary.csv`
- `persistent_mma_after_globalization_results/<variant>/<variant>_iterations.csv`
- `persistent_mma_after_globalization_results/<variant>/<variant>_rho_final.csv`
- `persistent_mma_after_globalization_results/<variant>/<variant>_topology.png`
- `persistent_mma_after_globalization_results/<variant>/<variant>_result.mat`
