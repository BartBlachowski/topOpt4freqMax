# MIGRATION_HANDOFF — for the method owner

## Where things are

| | |
|---|---|
| migration branch | `migration/olhoffcurrent-upstream-253069` (created at `013cc48`), in worktree `/Users/piotrek/Programming/topOpt4freqMax-migration-253069` |
| commit | the single commit on that branch after `013cc48` — `git log -1 migration/olhoffcurrent-upstream-253069` (a commit cannot contain its own hash) |
| primary checkout | `/Users/piotrek/Programming/topOpt4freqMax` on `benchmark-methodology-r2`, **untouched**: no switch, no stash, no edit to tracked or untracked files |
| raw evidence (git-ignored, declared in `EVIDENCE.json`) | `analysis/OlhoffCurrent/evidence/upstream_253069_migration/` in the worktree, **and** copied to the same path in the primary checkout, which is git-ignored there too, so its evidence gate can verify after a merge |
| upstream snapshot used | session scratchpad (temporary); reproducible exactly with `git -C /Users/piotrek/Programming/Matlab/Olhoff archive 253069262407885a8b759a9e721c4f0a7d3a397d` (tar SHA-256 `f9112403…`) |
| symlinks in the worktree (git-ignored) | `evidence/{move_activity_400,three_rung_resolution_240,two_branch_controller_validation}` and `campaign_9mesh_r2/benchmark_records.mat` → the primary checkout's ignored copies, so the suites can run in the worktree |

## To adopt the migration

1. Review `REPORT.md`, `PREREGISTRATION_AMENDMENT_1.md` and `HARNESS_UPDATE.md` (the production decision).
2. Merge or cherry-pick the branch into `benchmark-methodology-r2`. It touches only `analysis/OlhoffCurrent/**` and Olhoff-specific `examples/Performance/**` files (DIFF_AUDIT.md). None of the eight untracked diagnostic directories in the primary checkout is in the commit.
3. In the primary checkout, re-run the gate suites. They must run alone, because `test_source_integrity` and `test_currentness` temporarily perturb `+impl` and the manifest:
   ```matlab
   addpath('<repo>/analysis/OlhoffCurrent'); addpath('<repo>/analysis/OlhoffCurrent/tests');
   test_path_isolation(); test_currentness(); test_source_integrity();
   test_preset_identity(); test_pedersen_adaptive_units(); test_cost_reporting();
   test_evidence_retention(); test_finalization_gate();   % 2 pre-existing failures expected (TEST_REPORT)
   test_preset_equivalence(); test_named_preset_reproduction();   % 160x20 solves, ~10 min
   ```
4. Once the untracked `three_rung_canary_preflight` / `nine_mesh_campaign_audit` etc. are present again in the merged checkout, `test_finalization_gate` I will also evaluate them. Their state is theirs, not this migration's.
5. Remove the worktree when done: `git worktree remove /Users/piotrek/Programming/topOpt4freqMax-migration-253069`. The branch and commit remain.

## API changes callers must know

| before | after |
|---|---|
| `olhoffcurrent_preset()` | `olhoffcurrent_preset(name)`; production via `olhoffcurrent_production_preset()` |
| `olhoffcurrent_config(nelx, nely)` | `olhoffcurrent_config(nelx, nely, 'Preset', name)` — refused without a name |
| `olhoffcurrent_run(nelx, nely)` | `olhoffcurrent_run(nelx, nely, 'Preset', name)` |
| `olhoffcurrent_caveat()` | `olhoffcurrent_caveat(name)` |
| `cfg.provenance.productionPreset` | `cfg.provenance.olhoffCurrentPreset` (+ `presetRole`, `upstreamCommit`, `compatibilityAliases`, `presetResolvedVia`) |
| `out.production_preset` = the preset run | `out.preset` = the preset run; `out.production_preset` set only if it is the recorded production preset |
| timing-schema key `DuOlhoffReconstructionM4` | `DuOlhoffReconstruction` (with `preset`) |

The committed historical diagnostic scripts (e.g. `two_branch_controller_validation/scripts/cv_config.m`) call the old unnamed API. If re-run they now **fail closed** rather than silently resolving a different formulation. That is intended. To reproduce them, name `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered`, or use its alias `duOlhoffFixedPenaltySensitivityFiltered`.

## Open items handed on (not done here, by design)

1. Repair of the p-continuation `hist.move` logging defect: upstream-first, separate commit (KNOWN_PREEXISTING_DEFECTS D1).
2. `confbench_selftest` T2 one-line fix (D2).
3. Evidence state of seven older studies (D3): owners' decision.
4. Phase 6 benchmark-policy work: radius re-freeze, Proposed low-density treatment, evaluator ω₁ policy.
5. Nine-mesh campaign: awaiting owner authorization (CAMPAIGN_GATE.md).
