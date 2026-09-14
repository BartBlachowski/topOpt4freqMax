# Production launch path and checklist

**Authorization is NOT issued.** Nothing in the repository was modified to enable production during this audit. This document records what the launch path actually is and the exact minimal action required once the blockers are cleared.

## 1. The production gate, as implemented

`iteration_efficiency_final('production','lp')` → `iefinal.run` → `iefinal.config` → **`iefinal.preflight`**, which fails closed on:

```matlab
checks.no_self_authorization  = ~cfg.manifest.production_authorized;   % manifest must say false
checks.production_authorization = ~strcmp(cfg.run_mode,'production');  % production always fails
```

Two consequences worth stating plainly:

1. Production is **unconditionally locked**. There is no token, environment variable, flag or file that unlocks it — I searched for one. Verified live: `iefinal.preflight(iefinal.config('production','lp'))` throws `iefinal:PreflightFailed: production_authorization`.
2. Setting `production_authorized: true` in `PRODUCTION_MANIFEST.json` does **not** help — it trips `no_self_authorization` instead. The manifest cannot authorize itself, by design.

`analysis/iteration_efficiency_final/run_final_integration_tests.m::testStaleAndProductionLocks` asserts that production preflight throws, so unlocking production requires updating that test in the same change.

## 2. Exact minimal post-audit action (once B-1, B-2, B-3 are cleared)

1. Apply the three corrections in `BLOCKERS.md` and run their stated re-verifications.
2. Replace the blanket lock in `analysis/iteration_efficiency_final/+iefinal/preflight.m` with an explicit external authorization check — an authorization record outside the manifest, naming the audit that granted it, its date and the audited HEAD — keeping `no_self_authorization` intact so the manifest still cannot authorize itself.
3. Update `run_final_integration_tests.m::testStaleAndProductionLocks` to assert the new gate: production still refused **without** the authorization record, permitted **with** it.
4. Re-run `run_final_integration_tests` (expect 11/11 plus the amended lock test) and `iteration_efficiency_final('smoke','lp')`.
5. Commit the working tree so the campaign has a reproducible HEAD. The tree is currently dirty with 47 entries including all the instrumented native sources; a production campaign launched from an uncommitted tree is not reproducible.
6. Launch:

```matlab
cd /Users/piotrek/Programming/topOpt4freqMax
iteration_efficiency_final('production','lp')
```

## 3. Configuration the launch will use

| item | value | source |
|---|---|---|
| run mode | `production` | argument |
| Olhoff selector | `lp` (principal comparison) | argument |
| methods | Proposed, Yuksel, Olhoff-LP | `method_plan` |
| meshes | the nine frozen meshes | `manifest.production_meshes` |
| `B_ref` / reference horizon | 3200 | `config.m` |
| P | 100 primary, 50 / 200 sensitivity | `config.m` |
| q | .98 / .99 / .995 | `config.m` |
| threads | 1 | `maxNumCompThreads(1)` in `run_trajectory` and `run_timing_firewall` |
| timing | 1 warm-up + 3 retained replays | `config.m` |
| output | `runs/production/lp/<timestamp>/` | `new_run_directory` |
| preflight | invoked automatically before any output directory is created | `run.m` |

Run `iteration_efficiency_final('production','both')` only if the MMA scope decision in `PRODUCTION_COST_ESTIMATE.md` says so — at the measured 68× per-outer cost, nine-mesh MMA is a months-long campaign and must not gate the principal one.

## 4. Pre-launch checklist

- [ ] B-1 corrected and re-verified (Yuksel timing replay reproduces recorded stage counts)
- [ ] B-2 corrected and re-verified (`support="available"` and `support="common"` fit families)
- [ ] B-3 corrected and re-verified (cell failure recorded as a censored row, campaign continues, rows checkpointed)
- [ ] `preflight.m` production gate replaced with an external authorization record; `no_self_authorization` retained
- [ ] `run_final_integration_tests` passes with the amended lock test
- [ ] `iteration_efficiency_final('smoke','lp')` passes
- [ ] working tree committed; HEAD recorded in the authorization record
- [ ] ~110 GB free on the output volume (see `PRODUCTION_COST_ESTIMATE.md`)
- [ ] machine reserved for ~5 days of uninterrupted serial single-threaded compute
- [ ] MMA scope decided separately and documented

Recommended non-blocking cleanups to fold into the same change: F-04 (call `ie2a.account_iterations` or unpin it), F-05 (refresh the contract's Olhoff source hashes), F-06 (per-P `tail_truncated`), F-08 (populate `provenance_hash`).
