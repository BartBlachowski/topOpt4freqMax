# Experiment architecture

Entry point: `iteration_efficiency_final(runMode, olhoffVariant)`.

The production path is a two-pass experiment for every selected method/mesh:

1. Capture a lossless double reference trajectory through `B_ref=3200`.
2. Evaluate every post-update state with the single authoritative Candidate-C
   evaluator and repaired exact-count topology gate.
3. Establish causal `b_ref` and `Q_ref`; fail closed if unavailable.
4. Compute `B_meas=min(max(B0,b_ref+P-1),B_ref)` and rerun only that measurement
   horizon.
5. Require an identical reference/measurement trajectory prefix.
6. Extract `k_enter` and `k_cert` independently for every q/P pair, with status
   and achieved E1/E2/E3/robust-Q values.
7. Run observer-free, fixed-horizon native timing replays (one warm-up, three
   retained serial replays) after endpoints are frozen.
8. Apply method-specific accounting, fit `y=C*n^p` with common-support and LOO
   diagnostics, render accepted raw/exact-count topologies through the shared
   renderer, and write machine/paper tables.

State identity is explicit: state 0 is the initial design; state k is the
authoritative post-update design after native update k. Yuksel's state 0 is its
Stage-1/Stage-2 handoff; Stage-1 work remains separately accounted.

Authoritative scientific definitions remain in `+ie2a` and
`study_evaluate_design.m`; the final package orchestrates them and does not copy
their equations. Run outputs are isolated under
`runs/<smoke|production>/<lp|mma|both>/<timestamp>/` with separate reference,
measurement, timing, analysis, tables, figures, topologies, provenance, and
validation directories. Existing evidence is never overwritten.

Production is deliberately locked. No authorization token is embedded or
accepted by this package.
