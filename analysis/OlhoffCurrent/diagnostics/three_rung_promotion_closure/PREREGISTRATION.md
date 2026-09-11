# PREREGISTRATION — three_rung_promotion_closure

## 0. Ordering disclosure, stated first

Phase 0 requires this file's SHA-256 to be frozen **before applying production
promotion**. No production promotion was applied, so that requirement is met by
a wide margin.

But the Phase 4 and Phase 5 provenance repairs **were applied before this file
was written**, and that is disclosed rather than papered over. Both are fully
reversible and their pre-repair state is preserved byte-for-byte in
`evidence/`:

```
evidence/two_branch_FINAL_SHA256.BEFORE.txt                            2fb9fc73…2bf5e0e0
evidence/preserved_working_copies/C320x40_iterations.WORKING_COPY.csv  6ac72fa8…a4325a22
evidence/preserved_working_copies/C320x40_record.WORKING_COPY.json     76560421…5e78c28e
```

Nothing that could bias a *scientific* outcome was touched, because this task
produces no scientific outcome: it executes **zero optimization runs**.

## 1. Mission and the hard lock

No scientific question is open. The three-rung controller was causally
validated in `three_rung_promotion_validation_retry1`. This task exists only to
close provenance, promote, freeze and authorize.

```
SCIENTIFIC OPTIMIZATION RUNS AUTHORIZED IN THIS TASK: ZERO
```

Not C320. Not C160, C240, C400. Not any nine-mesh case. Not a fixed-move arm,
a production baseline, a candidate optimization or a historical reproduction.

Permitted: file transfer, hash verification, provenance repair, static audit,
config audit, dispatch audit, software/unit tests, sub-160 software-mechanics
tests, manifest/finalization checks, offline replay/postprocessing.

Any sub-160 test is **software only** and may never be reported as scientific
validation evidence.

## 2. The frozen scientific result — REUSED, never re-derived

```
move.levels  = [0.04, 0.02, 0.01]
continuation = frozen stageExhaustion
E            = frozen A OR B          (W = 20, P = 20, Wnp = 10)

0.04 -> hold until persistent E -> descend
0.02 -> hold until persistent E -> descend
0.01 -> hold until persistent E -> CONVERGED

no move = 0.005 in canonical production after promotion
beta: optimization variable + diagnostic; ZERO authority over continuation,
      descent or terminal convergence
```

Validated at 320×40 by exactly one run: `CONVERGED @352` versus the four-rung
oracle's `CAP_HIT @1600`; S1 = 274/A, S2 = 313/B, S3 = 352/B; `RHO`/`DRHO`
bitwise identical over all 4 505 600 prefix entries; 6 498 versus 76 532 inner
MMA iterations (−91.51 %), 1248 outer iterations eliminated (−78.00 %).

**This must not be repeated.** Its verdict is reused as
`THREE_RUNG_PRODUCTION_POLICY_VALIDATED [REUSED — NO NEW SCIENTIFIC RUN]`.

## 3. Timing-field interpretation — frozen for this task

```
tOuter   tEig   tGrad   tInner
```

are `toc`-derived nondeterministic timing telemetry and are **EXCLUDED** from
scientific bitwise-equivalence requirements. No optimizer quantity reads any of
them back into computation.

Both facts from the retry are preserved, and neither is rewritten:

- literal enumerated-list reading → mismatch;
- category / scientific-state reading → **PASS**.

These timers may not be used to question the validated C320 result.

## 4. Frozen provenance repair steps

| # | Repair | Scope, fixed in advance |
|---|---|---|
| P4 | `two_branch_controller_validation/FINAL_SHA256.txt` | regenerate **ONLY** that file, and only entries whose target is provably the intended finalized version. No underlying scientific file may be altered to satisfy an old digest. |
| P5 | the two timing-drifted tracked paths | restore to the authoritative version **only after** proving the difference is timing/non-scientific and that no unique scientific state exists in the working copy. Working copies archived first. |

Not in scope: re-declaring container digests to make a gate pass. That is the
move the finalization gate exists to prevent and is defensible only as an
explicit, documented owner decision.

## 5. Frozen promotion criteria

Promotion may proceed **only** if every one of these holds:

1. `VALIDATED_C320_EVIDENCE_REUSE_PASS`;
2. `ORIGINAL_C240_EVIDENCE_TRANSFER_PASS` — the original artifact present at
   the policy location and matching its **full** authoritative digest
   `183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d`
   (131 203 128 bytes). A prefix match is not sufficient;
3. `C240_FINALIZATION_RESTORED_PASS` — `three_rung_resolution_240` G1–G5 PASS;
4. `HISTORICAL_FINAL_SHA256_REPAIR_PASS`;
5. `TIMING_ONLY_DRIFT_SETTLED_PASS`;
6. every load-bearing study G1–G5 PASS —
   `two_branch_controller_validation`, `three_rung_architecture`,
   `three_rung_resolution_240`, `three_rung_promotion_validation_retry1`;
7. `test_finalization_gate` with **0 failures**.

If any fails: `PROMOTION_PROVENANCE_BLOCKED`,
`PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED`, STOP, no scientific run.

## 6. Frozen promotion change set

Configuration/preset only. Exactly three computational fields:

| Field | from | to |
|---|---|---|
| `move.levels` | `[0.04 0.02 0.01 0.005]` | `[0.04 0.02 0.01]` |
| `move.continuation.signal` | `boundVariable` | `stageExhaustion` |
| `stop.rule` | `designChange` | `stageExhaustion` |

No controller code change. No change to A, B, persistence, windows, reset
semantics, thresholds, normalization or tolerance scaling. If **any** further
computational field must change, each one is audited and explained *before*
promotion.

## 7. Frozen scientific config lock

Promoted production must preserve, identical to the validated candidate:
`p = 3`; mass Eq. (4b); `q = 1`; sensitivity filter applied to all; `R = 0.06·b`;
projection OFF; subspace multiplicity of size 2 with diagonal offsets and
off-diagonal terms; published MMA; FE; eigensolver; objective; volume; initial
design; controller-unrelated tolerances; deterministic/thread policy.

## 8. Frozen equivalence criteria

- `PROMOTED_PRODUCTION_CONFIG_EQUIVALENCE_PASS` — promoted production and the
  validated candidate agree on **every** computational field. Permitted
  differences: run name, output path, inert provenance metadata.
- `THREE_RUNG_PROMOTION_EQUIVALENCE_PASS` — canonical production dispatches to
  the exact controller code the validated run executed, proven by dispatch
  audit, qualified-name resolution, source hashes and `+impl` tree identity.

## 9. Frozen freeze criteria

`PRODUCTION_FREEZE_PASS` requires: config equivalence PASS, dispatch
equivalence PASS, relevant software tests PASS, canonical source manifest and
`+impl` hash recorded, production preset/config hash recorded, A/B
implementation hashes recorded, and no unaccounted change under any canonical
production path.

## 10. Frozen campaign-readiness criteria

`NINE_MESH_PERFORMANCE_CAMPAIGN_AUTHORIZED` requires **all** of: policy
validated (reused), promotion provenance PASS, controller promoted, config
equivalence PASS, promotion equivalence PASS, production freeze PASS, software
tests PASS, finalization PASS, and no unresolved scientific or controller
blocker.

Authorization is permission for the **next** task only. The campaign
(160×20 … 800×100) is not executed here under any circumstance, and no campaign
result may later alter A, B, persistence, move levels, stop semantics, `p`, `q`,
the mass model, the filter, `R`, projection, multiplicity or MMA.
