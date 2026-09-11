# PERFORMANCE_READINESS — Part L

# `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

**Zero nine-mesh campaign runs were executed.** The campaign was not run here
under any circumstances, as the brief directs.

## The authorization checklist

| Requirement | State |
|---|:--:|
| `THREE_RUNG_PRODUCTION_POLICY_VALIDATED` | ✅ |
| `PRODUCTION_THREE_RUNG_CONTROLLER_PROMOTED` | ❌ not promoted |
| promotion-level provenance PASS | ❌ `PROMOTION_PROVENANCE_BLOCKED` |
| `PROMOTED_PRODUCTION_CONFIG_EQUIVALENCE_PASS` | ❌ not assessable — no promoted config |
| `THREE_RUNG_PROMOTION_EQUIVALENCE_PASS` | ❌ not assessable |
| production freeze PASS | ❌ not performed |
| tests PASS | ⚠️ 5 of 6 repository test files clean; `test_finalization_gate` 4 failures |
| finalization PASS | ❌ G1–G5 FAIL on all three load-bearing studies |

Seven of eight unmet. Authorization requires all eight.

## The fixed campaign, for when it is authorized

```
160x20  240x30  320x40  400x50  480x60  560x70  640x80  720x90  800x100
```

## What this study contributes to campaign readiness

- The controller is **validated at 320×40** and the policy is unambiguous.
- The cost model is now measured rather than projected at that mesh:
  `CONVERGED @352` versus `CAP_HIT @1600`, 6 498 inner MMA iterations versus
  76 532. A four-rung campaign at 320×40 would not have terminated within the
  preregistered cap at all.
- Wall time for the validated run was **1 486.9 s single-threaded** at 320×40 —
  a real anchor for sizing the campaign, though it must not be extrapolated
  across meshes from a single point.

## What must happen first

1. Transfer `C240x30_trajectory.mat` from the machine that produced it and
   verify it against `183d7ce60d512fc2…` (H1 — the binding constraint).
2. Apply the prepared `FINAL_SHA256.txt` repair and the matching
   `three_rung_architecture/EVIDENCE.json` edit (H2).
3. Commit or revert the two `tOuter`-dirty tracked paths (H3).
4. Decide the Class A container question — transfer the originals or re-declare
   the local ones explicitly (`PROVENANCE_REPAIR_STATUS.md` §4).
5. Re-run the finalization gate and `test_finalization_gate` to green (H4, H5).
6. Promote configuration-only, per `PROMOTION.md`, and discharge Part J.

**C320 must not be re-run.** Its validation is complete and is reusable as-is.
