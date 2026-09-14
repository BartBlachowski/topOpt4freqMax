# PERFORMANCE_READINESS — Phase 18

# `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`

**Zero nine-mesh runs were executed.** The campaign is not executed in this task
under any circumstance, and it remains unauthorized.

## The authorization checklist

| Requirement | State |
|---|:--:|
| `THREE_RUNG_PRODUCTION_POLICY_VALIDATED` (reused from retry1) | ✅ |
| `PROMOTION_PROVENANCE_PASS` | ❌ `PROMOTION_PROVENANCE_BLOCKED` |
| `PRODUCTION_THREE_RUNG_CONTROLLER_PROMOTED` | ❌ not promoted |
| `PROMOTED_PRODUCTION_CONFIG_EQUIVALENCE_PASS` | ❌ not assessable — nothing promoted |
| `THREE_RUNG_PROMOTION_EQUIVALENCE_PASS` | ❌ not assessable |
| production freeze PASS | ❌ `PRODUCTION_FREEZE_FAIL` — phase not reached |
| relevant software tests PASS | ⚠️ 4 of 5 run clean; `test_finalization_gate` 4 failures; controller mechanics 29/29 |
| finalization PASS | ❌ 3 of 4 load-bearing studies fail G1–G5 |
| no unresolved scientific/controller blocker | ✅ **none** |

Seven of nine unmet. Authorization requires all nine.

## The one requirement that *is* met, and it matters

**There is no unresolved scientific or controller blocker.** The controller is
validated, its definition is unambiguous, its code is dispatch-verified, and the
promotion delta is three configuration fields with zero unexpected
computational changes. Everything standing between here and authorization is
**provenance bookkeeping and file transfers** — no science, no compute.

## The fixed campaign, for the next task

```
160x20  240x30  320x40  400x50  480x60  560x70  640x80  720x90  800x100
```

Authorization, when issued, is permission for the **next** task only. That task
must preregister the complete nine-mesh scaling/performance campaign before its
first run, and no campaign result may alter A, B, persistence, move levels, stop
semantics, `p`, `q`, the mass model, the filter, `R`, projection, multiplicity
or MMA.

## Sizing anchor already available

From the validated run, at 320×40 single-threaded: `CONVERGED @352` in
**1 486.9 s** with 6 498 inner MMA iterations, against the four-rung arm's
`CAP_HIT @1600` with 76 532. A four-rung campaign at 320×40 would not have
terminated within the preregistered cap at all.

This is **one mesh**. It must not be extrapolated into a scaling law — that is
what the campaign is for.

## What must happen first

1. **Transfer six container artifacts** from the machine(s) that produced them:
   `C240x30_trajectory.mat` (absent), and `C160x20`, `C320x40`, `C400x50`,
   `F400_400x50`, `P400_400x50` (present but regenerated locally, digests
   mismatched). Verify each against its declared digest.
   *Or* make and document an explicit owner decision to re-declare the
   regenerated containers — see `PROMOTION_PROVENANCE.md` §4.
2. Optionally apply the analysed Class C repair to
   `three_rung_architecture/EVIDENCE.json` and cascade to its `FINAL_SHA256.txt`
   (`HISTORICAL_HASH_REPAIR.md` §5).
3. Re-run the finalization gate and `test_finalization_gate` to green.
4. Promote configuration-only, three fields, via the OlhoffCurrent config layer
   (`PROMOTION_DIFF_PLAN.md` §5, route (b)).
5. Discharge config and dispatch equivalence, freeze, then authorize.

**C320 must not be re-run**, and neither must C160, C240 or C400.
