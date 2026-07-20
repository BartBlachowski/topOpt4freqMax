# A4 Recovery Phase 2 — Implementation Report

**Status:** implementation complete; production five-arm recovery sweep not executed in this implementation pass.

## Implemented protocol

- One constants block defines `W=(20,40,80,160,320)`, `M_max=320`, all four thresholds, the `10^-12` tie tolerance, fixed eigensolver settings, and the 25-point grid `G`.
- Every Phase-2 screening event uses the shared adaptive search, mandatory confirmation expansion, `MAC >= 0.99` stability test, and lower-index tie break.
- Diagnostic screening is read-only, shares one search with a coincident operational refresh, and is applied to `G` plus the final iteration.
- Operational `REFERENCE_UNAVAILABLE` outcomes defer the refresh, retain the previous reference, and never terminate an arm.
- Candidate telemetry contains every §6.1 field for every final-window candidate. Per-iteration reference, objective, feasibility, and design-change histories are retained.
- E-0, E-1, E-2a, E-2b, E-3, E-4, and E-5 and the §8 arm statuses/warnings are implemented. The Phase-2 path cannot emit B3.
- Writers exist for the result/event JSON, long CSVs, MAT result, five topology CSVs, Tables A4-1/A4-2, nine plots, matched manifests, and reports.

## Executed implementation validation

- `test_a4_phase2` protocol fixtures: **15/15 passed**, including diagnostics-on/off bit identity and Phase 1’s 10/10 regression.
- Production-scale V-P2-3 through iteration 30: **PASS**. Iteration 25 selected index 49, MAC 0.9775288450; iteration 30 selected index 37, MAC 0.9663501395. Both are E-1 events.
- `test_a4_pipeline` tiny end-to-end driver/artifact validation: **28/28 passed**.
- Existing `test_a4_refresh`: **22/22 passed** (legacy path remains inert outside Phase 2).
- Existing `test_a4_classifier`: **17/17 passed** (legacy classifier retained outside the Phase-2 execution path).
- `test_a4_phase1`: **10/10 passed**.

## Production work still required

The complete `{inf,50,10,5,1}` sweep, production `N=inf` bit-identity stop gate, finite-`N` full replay, production artifact reconstruction checks, git tracking, and runtime-estimate validation remain pending. Until they pass, the Section 8 run verdict is not available and Phase 2 must not be reported as COMPLETE.

M-1, M-2, M-3, M-7, and M-9 remain out of scope under §7.6. No campaign-level H0/H1 decision or manuscript claim has been emitted.
