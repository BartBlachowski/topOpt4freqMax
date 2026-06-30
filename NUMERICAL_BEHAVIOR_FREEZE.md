# NUMERICAL_BEHAVIOR_FREEZE

**Date:** 2026-06-30
**Scope:** `OlhoffApproachExact`, CC beam 80x10 diagnostic pilot
**Language:** MATLAB only

This note freezes the numerical-stabilization settings for the final
OlhoffApproachExact experimental campaign. It is based only on saved Phase 1-4
artifacts under `examples/Revision_v1/output/`.

---

## Summary

The algorithm-equivalence audit found no deviations from Du and Olhoff (2007)
in the verified FE, eigenproblem, sensitivity, mode-tracking, multiplicity, and
`du2007_c1` interpolation implementation. The remaining failure mode was a
persistent outer two-cycle. Phases 1-4 isolated that behavior to the outer
trust-region size rather than inner MMA convergence or asymptote state.

| Phase | Single change | Result | Decision |
|---|---|---|---|
| Phase 1 | `inner_max_iter: 30 -> 300` | Inner MMA mostly converged, but the outer cycle remained. Cap-hit fraction fell to 1.5%; final design change remained 0.0914. | Inner convergence was not the dominant cause. |
| Phase 2 | Preserve MMA asymptote state | Negligible convergence effect. Final design change 0.0921; mean last-window design change 0.0915. | Asymptote restart was not the dominant cause. |
| Phase 3 | `outer_move: 0.20 -> 0.05` | Strong reduction of oscillation but still capped at 400 iterations. Final design change fell from 0.0921 to 0.0206; beta parity gap fell from 18.50 to 6.03. | Trust-region size was the dominant cause. |
| Phase 4 | `outer_move: 0.05 -> 0.02` | Converged at outer iteration 24 with final design change 1.0179e-4 < 1e-3. S1 found 0/10 localized low-density modes. | Stable enough to freeze. |

---

## Frozen Production Settings

Use these settings for the final OlhoffApproachExact campaign unless a future
explicitly scoped investigation supersedes this freeze:

| Setting | Frozen value |
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

The remaining benchmark settings are unchanged from the Phase 3/4 diagnostic
case: CC beam, 80x10 mesh, volume fraction 0.5, identical initialization,
identical boundary conditions, identical sensitivities, identical update
ordering, and persistent MMA asymptotes enabled.

---

## Justification

The freeze is a numerical stabilization of the paper-ambiguous move-limit
choice, not an algorithmic change.

Du and Olhoff (2007) define the nested first-order update and MMA subproblem,
but the practical trust-region/move-limit scale is a numerical implementation
choice. The verified mathematical components were not changed in Phase 4:

- FE assembly was unchanged.
- Eigenproblem and mode normalization were unchanged.
- Generalized sensitivities were unchanged.
- Mass interpolation remained `du2007_c1`.
- Stiffness interpolation remained SIMP with `p = 3`.
- Objective, constraints, multiplicity handling, and update ordering were unchanged.
- MMA update rules and asymptote logic were unchanged.

Only the outer trust-region parameter was reduced from 0.05 to 0.02. This
reduced first-order linearization error enough for the same algorithmic
formulation to converge.

---

## Artifact Evidence

| Artifact | Role |
|---|---|
| `examples/Revision_v1/output/phase1_olhoff_exact_cc_80x10_inner300/phase1_inner300_summary.md` | Phase 1 inner-iteration test |
| `examples/Revision_v1/output/phase2_olhoff_exact_cc_80x10_asymptote_persistence/phase2_asymptote_persistence_summary.md` | Phase 2 asymptote-persistence test |
| `examples/Revision_v1/output/phase3_olhoff_exact_cc_80x10_outermove005/phase3_outermove005_summary.md` | Phase 3 `outer_move=0.05` test |
| `examples/Revision_v1/output/phase4_olhoff_exact_cc_80x10_outermove002/phase4_outermove002_summary.md` | Phase 4 `outer_move=0.02` convergence result |
| `examples/Revision_v1/output/phase4_olhoff_exact_cc_80x10_outermove002/s1_mode_summary.json` | Final S1 diagnosis: 0/10 localized low-density modes |

---

## Freeze Decision

Freeze the OlhoffApproachExact production settings listed above.

Rationale:

- Phase 1 showed that increasing `inner_max_iter` mostly solved inner MMA cap
  hits but did not remove the outer cycle.
- Phase 2 showed that preserving MMA asymptote state had negligible effect.
- Phase 3 showed that reducing `outer_move` strongly reduced the cycle.
- Phase 4 showed that `outer_move = 0.02` converged rapidly and S1 found
  0/10 localized low-density modes among the first 10 modes.

No further numerical-behaviour tuning is recommended before the final
OlhoffApproachExact experimental campaign.
