# Generation 1 — Historical nested-MMA implementation

**This directory intentionally contains no code.**

The historical nested-MMA implementation was **not moved** during the
2026-08-26 reproduction migration. It remains at its original location:

> ### [`analysis/OlhoffApproachExact/Matlab/legacy/`](../../analysis/OlhoffApproachExact/Matlab/legacy/)

## Why it was not moved

It is referenced from experiment trees and from the frozen protocol ledger
(`examples/Performance/ledger/protocol_ledger.json`). Relocating it during the
paper revision would invalidate recorded experiment paths for no scientific
gain. See the "Why the layout is not literally ..." section of
[`../README.md`](../README.md).

## Status

**Frozen. Provenance and forensic analysis only. Do not develop against it.**

Entry points at the real location:

- `run_simply_simply_exact.m`, `run_clamped_clamped_exact.m`,
  `run_clamped_simply_exact.m`
- `verify_initial_frequencies.m`, `verify_inner_loop.m`,
  `verify_multiplicity.m`, `verify_outer_loop.m`,
  `verify_sensitivity_filter.m`
- `inner_loop_mma.m`, `compute_generalized_gradients.m`

Forensic account of how this generation failed:
[`../reproduction2007/OLHOFFEXACT_FAILURE_POSTMORTEM.md`](../reproduction2007/OLHOFFEXACT_FAILURE_POSTMORTEM.md)
(§2.1 defines the legacy / rebuilt / clean-room vocabulary used throughout).
