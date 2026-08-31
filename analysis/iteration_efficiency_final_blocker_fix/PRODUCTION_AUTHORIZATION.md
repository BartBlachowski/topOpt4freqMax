# Production authorization

**STATE A — FINAL PRE-PRODUCTION BLOCKERS CLOSED; NINE-MESH PRINCIPAL CAMPAIGN AUTHORIZED**

| blocker | verdict |
|---|---|
| B-1 Yuksel timing replay work identity | **CLOSED** |
| B-2 scaling common support | **CLOSED** |
| B-3 cell-local failure containment | **CLOSED** |

No new production blocker was discovered. All 14 integration tests pass. The nine-mesh production campaign was **not** run.

## What is authorized

| item | value |
|---|---|
| principal comparison | Proposed / Yuksel / Du–Olhoff LP |
| common evaluator | Candidate C — adaptive structural mode |
| Olhoff authoritative trajectories | lossless `double` |
| Olhoff nested MMA | secondary / optional |
| production meshes | 160×20, 240×30, 320×40, 400×50, 480×60, 560×70, 640×80, 720×90, 800×100 |
| paper outputs | iteration efficiency · computational time · scaling C and p · absolute quality · final topologies |

Authorization is technical and scientific clearance of the harness. It is not an instruction to run, and no campaign was run here.

## Production remains code-locked — deliberately

`iefinal.preflight` still fails closed on `run_mode = 'production'` (`iefinal:PreflightFailed: production_authorization`), and `no_self_authorization` still prevents the manifest from authorizing itself. This task did not unlock production, because unlocking is an operator action that should accompany a signed authorization record, not a side effect of a code fix.

## Exact launch procedure

1. Replace the blanket lock in `analysis/iteration_efficiency_final/+iefinal/preflight.m` with an explicit external authorization check — an authorization record outside the manifest naming this correction round, its date, and the HEAD it was verified against. Keep `no_self_authorization` intact.
2. Update `run_final_integration_tests.m::testStaleAndProductionLocks` to assert the new gate: production refused **without** the authorization record, permitted **with** it.
3. Re-run `run_final_integration_tests` (expect 14/14 plus the amended lock test) and `iteration_efficiency_final('smoke','lp')`.
4. Commit the working tree so the campaign has a reproducible HEAD. It is currently dirty with 48 entries, including the instrumented native sources; a campaign launched from an uncommitted tree is not reproducible.
5. Confirm ~110 GB free on the output volume and reserve the machine for roughly five days of uninterrupted serial single-threaded compute (see the prior audit's cost estimate).
6. Launch:

```matlab
cd /Users/piotrek/Programming/topOpt4freqMax
iteration_efficiency_final('production','lp')
```

Run `'both'` only if the MMA scope decision says so. At the measured 68× per-outer cost, a nine-mesh MMA campaign is a months-long run and must not gate the principal comparison.

## Carried-forward non-blocking findings

From the final pre-production audit, unchanged by this task and still worth addressing: F-04 (`ie2a.account_iterations` pinned but not executed), F-05 (contract's Olhoff source hashes stale; `validate_contract` called with `VerifyFiles=false`), F-06 (`tail_truncated` is a P-primary quantity written to P=50/200 rows), F-07 (MMA cost and the LP/MMA filter-radius asymmetry), F-08 (`provenance_hash` empty), F-09, F-10. None blocks production.
