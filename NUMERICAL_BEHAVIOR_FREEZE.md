# Archived OlhoffApproachExact numerical-behaviour record

**Date of original freeze:** 2026-06-30

**Archive decision:** 2026-07-13

**Scope:** `OlhoffApproachExact`, CC beam 80x10 diagnostic reconstruction
campaign

**Status:** archived diagnostic; not production; not reviewer evidence

This file formerly froze settings for a planned production
`OlhoffApproachExact` campaign. That production decision is withdrawn. The
completed reconstruction campaign could not establish a paper-faithful
reconstruction of Du and Olhoff (2007), so neither the implementation nor its
stabilized settings can serve as a canonical/reference benchmark.

The Phase 1--4 artifacts are retained only to document diagnostics that were
performed. They may support the negative reconstruction verdict, but not
frequency-gap, convergence, speedup, scaling, optimality, or method-ranking
claims in the revision.

## Final scientific disposition

- FE formulation, interpolation, sensitivity implementation, generalized
  gradients, multiplicity handling, mode tracking, optimizer stabilization,
  persistent MMA, the tested regularization variants, and the tested support
  interpretations did not resolve the discrepancy.
- Benchmark under-specification or unpublished implementation details are the
  most plausible remaining explanation.
- `OlhoffApproachExact` and `OlhoffApproachExactOpus` are closed diagnostic
  archives.
- No additional tuning or production run is planned.
- Active revision comparisons, if retained, use only the local
  `analysis/OlhoffApproach` implementation and identify it as local and
  Olhoff-inspired.

## Historical diagnostic settings

The final Phase 4 diagnostic used the settings below. They are recorded for
traceability only and are not production defaults.

| Setting | Historical value |
|---|---:|
| `inner_max_iter` | 300 |
| `outer_move` | 0.02 |
| `alpha` | 0.5 |
| `persistent_mma_state` | true |
| `mass_mode` | `du2007_c1` |
| `penal` | 3 |
| `rmin_elem` | 2.5 |
| `inner_tol` | 1e-4 |
| `outer_tol` | 1e-3 |
| `mult_tol` | 1e-3 |
| `acceptance_check` | false |
| `move_lim` | Inf |

The corresponding one-parameter studies tested inner-iteration count,
persistent MMA asymptotes, and two outer move limits. Their convergence within
the attempted implementation does not validate the implementation against the
published benchmark.

## Archived artifacts

All paths below are relative to
`examples/Revision_v1/archive/olhoff_exact_reconstruction/output/`, where these
directories were moved when the reconstruction campaign was archived (2026-07-13).
They are no longer under `examples/Revision_v1/output/`.

| Artifact | Diagnostic role |
|---|---|
| `phase1_olhoff_exact_cc_80x10_inner300/` | Inner-iteration diagnostic |
| `phase2_olhoff_exact_cc_80x10_asymptote_persistence/` | Persistent-asymptote diagnostic |
| `phase3_olhoff_exact_cc_80x10_outermove005/` | Outer-move diagnostic (`0.05`) |
| `phase4_olhoff_exact_cc_80x10_outermove002/` | Outer-move diagnostic (`0.02`) |
| `pilot_olhoff_exact_cc_80x10*/` | Earlier reconstruction pilots |

All numeric files, tables, and figures in these directories remain unchanged
historical outputs. They are excluded from production manifests, revised
tables and figures, the response letter, and manuscript claims.

## Closed decision

There is no longer an `OlhoffApproachExact` production freeze. The only valid
use of this record is to explain the archived diagnostic reconstruction and why
it was removed from the reviewer evidence chain.
