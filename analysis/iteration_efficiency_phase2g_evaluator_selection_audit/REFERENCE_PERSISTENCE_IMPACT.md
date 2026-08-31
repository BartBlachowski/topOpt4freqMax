# Reference and persistence impact

Within the available survey, the selected C frequency is invariant throughout the exact
threshold plateau: density partition `tau=0.02..0.50` with nearby simultaneous-condition cuts (the
intersection contains `0.49..0.54`; at `tau=0.1`, `0.48..0.56`). Thus the available C
quality sequences do not change within that plateau.

Formal `b_ref`, `B_meas`, `k_enter`, and `k_cert` cannot be recomputed. The reference rule
requires a separate density trajectory through `B_ref=3200`; the longest stored production
density history has 1,601 snapshots and the reference-length artifacts retain quality
arrays, not densities. Therefore:

- sensitivity of `b_ref`: **not exercisable**, not zero;
- sensitivity of `k_enter`: **not exercisable**, not zero;
- sensitivity of `k_cert`: **not exercisable**, not zero;
- `B_meas`: cannot be regenerated because it depends on the unavailable C-based `b_ref`.

Once C is implemented, unresolved modal states must make `Q_e(k)` unavailable and cannot
contribute to a valid sustained floor or acceptance window. They must never be imputed.
The formulas, `P`, `B_ref`, `B_meas`, q levels, and topology gates otherwise remain unchanged.
