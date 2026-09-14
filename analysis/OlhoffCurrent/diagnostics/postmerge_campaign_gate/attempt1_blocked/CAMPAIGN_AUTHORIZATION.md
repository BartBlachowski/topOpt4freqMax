# CAMPAIGN_AUTHORIZATION

```
OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED
```

> The definitive nine-mesh campaign remains blocked because the finalization-gate `SUPERSEDED_PRODUCTION_SOURCE` rule introduced by migration commit `9b30ec4` (PREREGISTRATION_AMENDMENT_1 §A2) is not strictly fail-closed. A fabricated digest shadowed by a later genuine line for the same path passes (P5). A `git show` failure is masked by the shell pipeline, so the empty-content digest passes at a deletion commit (P9). A local `+impl` edit with a regenerated manifest satisfies the "current source" condition (P10). The amendment review therefore fails, and the merge, post-merge suite, both 160×20 anchors, production freeze and config preview were not executed.

| authorization condition | status |
|---|---|
| amendment review passes | **FAIL** (A2) |
| provenance-gate review passes | **FAIL** (P5, P9, P10) |
| migration commit review passes | **FAIL**: the commit's gate claims contradict the probes; all other criteria pass |
| merge passes | not executed |
| post-merge implementation identity passes | not executed (pre-merge 79/79 upstream-identical) |
| no new test regression | not executed |
| historical 160×20 anchor passes | not executed |
| Pedersen 160×20 anchor passes | not executed |
| nine future configs resolve correctly | not executed |
| telemetry ready | not assessed |
| repository integrity passes | yes |
| no scientific parameter retuned | yes |

## How to lift the block (owner decision)

1. Authorize a small follow-up commit on `migration/olhoffcurrent-upstream-253069` that repairs `local_supersededSource` (remedy 1–4 in FINALIZATION_GATE_REVIEW.md) and adds destructive tests for P5, P9 and P10. The change is outside `+impl`, so byte identity and every trajectory are unaffected.
2. Re-run this gate from Step 1 against the new branch head: review the delta, merge, then Steps 4–11.

The owner may instead accept A2 as-is with an explicit written waiver. This gate may not grant that waiver.
