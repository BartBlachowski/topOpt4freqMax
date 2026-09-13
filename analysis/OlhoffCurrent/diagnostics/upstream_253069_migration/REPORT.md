BOTTOM LINE

**OlhoffCurrent now uses the shared upstream implementation derived from Olhoff commit `253069262407885a8b759a9e721c4f0a7d3a397d`.** All 79 `+impl` files are byte-identical to it, with no private solver or controller fork left.

**The historical SIMP + eq. (4b) reconstruction is preserved under its own explicit preset** and reproduces historical 160×20 evidence bit for bit. That covers both the β-stall production formulation and the stage-exhaustion diagnostic controller.

**The successful Pedersen/adaptive-box formulation is a separate, named preset** (`duOlhoffPedersenAdaptiveBoxSensitivityFiltered`). It reproduces the committed upstream S160x20 result bit for bit, and it is now the recorded production preset. **No scientific formulation was silently overwritten**: every configuration call must name its preset, and the old production name resolves only to the historical formulation.

Every migration gate passed. Two points are disclosed because a reviewer may weigh them differently:

- Three test checks fail, identically on a pristine `013cc48`. They are pre-existing and unrelated to the migration.
- One migration-caused failure — a historical study pinning production source files — was resolved by a strict gate refinement instead of editing that study's evidence.

Both are recorded as post-hoc deviations in PREREGISTRATION_AMENDMENT_1.md. No mesh above 160×20 was solved. The nine-mesh campaign is ready for owner authorization, not authorized.

```
UPSTREAM_253069_IDENTITY_PASS
MIGRATION_BYTE_PROMOTION_PASS
HISTORICAL_PRESET_PRESERVED_PASS
HISTORICAL_PRESET_REPRODUCTION_PASS
PEDERSEN_PRESET_DISTINCT_IDENTITY_PASS
PEDERSEN_PRESET_S160_REPRODUCTION_PASS
SHARED_IMPLEMENTATION_EQUIVALENCE_PASS
CONFIG_SCHEMA_MIGRATION_PASS
MANIFEST_PROVENANCE_PASS
BENCHMARK_PREFLIGHT_PASS
RELEVANT_TEST_SUITE_PASS          (pre-existing failures disclosed; Amendment A1)
PHASE6_UNTOUCHED_PASS
UNAUTHORIZED_DIFF_ZERO
OLHOFFCURRENT_MIGRATION_COMPLETE
OLHOFF_NINE_MESH_CAMPAIGN_READY_FOR_AUTHORIZATION
```

## Answers

1. **Was upstream commit 253069… verified exactly?** Yes.
   - Object type is commit; tree `4571029f…`; sole parent `6b08708`; `fsck` clean.
   - `git archive` SHA-256 `f9112403…`; all 1877 extracted files blob-verified.
   - Only committed blobs were used; the upstream working tree (dirty plan §7) was not read.

   SOURCE_IDENTITY.md.
2. **What target branch/HEAD was migrated?** `benchmark-methodology-r2 @ 013cc48451d33bed61c5c4eea174bbd898d548a2`. The migration was done on the new branch `migration/olhoffcurrent-upstream-253069` in a dedicated worktree.
3. **Was target dirty state preserved safely?** Yes.
   - There were no tracked changes; eight untracked diagnostic directories were never switched, stashed, cleaned or edited.
   - The primary checkout's `git status` is unchanged.
   - The only write to the primary checkout is the git-ignored raw evidence copy (MIGRATION_HANDOFF.md).
4. **Which files were promoted byte-for-byte?**
   - Replaced (16): `algo/genGrad.m`, `algo/innerLoopRho.m`, `+olh/+config/{describe,fromLegacy,schema,toLegacy,validate}.m`, `+olh/+move/limit.m`, `+olh/+presets/list.m`, `docs/{CONFIG_REFERENCE,MIGRATION_FROM_LEGACY,PRESETS,SCIENTIFIC_CONFIG_PROVENANCE}.md`, `olhoffSolve.m`, `fem/assemble2D.m`, `fem/eigSolve.m`.
   - Added (4): `+olh/+material/stiffnessInterpolation.m`, `+olh/+presets/{duOlhoffAdaptiveMove,duOlhoffAdaptivePedersen,duOlhoffOuterAsymptotes}.m`.
   - 59 were already identical. PROMOTION_MAP.md.
5. **Which target-specific implementation differences remain?** None. 79/79 BYTE_IDENTICAL_TO_UPSTREAM; 0 INTENTIONALLY_TARGET_SPECIFIC; 0 UNEXPLAINED. `architecture/README.md` stays excluded, as at 695f03b.
6. **Is shared solver/controller code now upstream-identical?** Yes, byte for byte, including `olhoffSolve.m`, `+move/limit.m`, `+move/exhaustion.m` and the config layer. At execution level, target = upstream bitwise for β-stall, stage exhaustion and Pedersen (SHARED_IMPLEMENTATION_EQUIVALENCE.md).
7. **Was the historical Eq.(4b) formulation preserved?** Yes, with its values unchanged.
   - All 81 old configuration leaves are equal for all 14 recorded historical configurations.
   - The 81-row hash recomputed from the migrated configuration reproduces every recorded hash.
   - `duOlhoffFrozenM4.m` is byte-identical.
8. **What is its canonical preset name?** `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` for the production formulation of 2026-09-07 … 2026-09-13; compatibility alias `duOlhoffFixedPenaltySensitivityFiltered`. The stage-exhaustion diagnostic controller is `duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered`. The brief's suggested "Eq4bStageExhaustion" name was not used for the β-stall preset, because its configuration uses the β stall, not stage exhaustion.
9. **Does it still reproduce historical 160×20 evidence?** Yes, bitwise.
   - **β-stall**: equal to the frozen conference record (ρ, ω₁ 169.495227021538, volume, 91/2241) and to the pre-migration run, across the full result.
   - **Three-rung**: equal to the pre-migration run and the upstream-audit target record (180/4283, events 103/142, terminal B at 180, every Δρ).
   - **Four-rung override**: equal to the committed C160x20 target record (ρ SHA `332c00a5…`, 36 `hist` fields, 219 Δρ, CSV, exhaustion record, log).

   HISTORICAL_REPRODUCTION.md.
10. **Was the successful Pedersen formulation added separately?** Yes, as its own registry entry delegating to `duOlhoffAdaptivePedersen` with no overrides. It does not inherit stage exhaustion; the config layer forbids it on the adaptive policy.
11. **What is its canonical preset name?** `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` (provenance aliases `duOlhoffAdaptivePedersen`, S160x20 … S800x100).
12. **Does it reproduce upstream committed S160x20 evidence?** Yes, bitwise.
    - ρ; ω₁ 169.210576386275; 121 outer / 2369 inner; identical log; all 24 non-timing `hist` fields; box trajectory (`hist.move`, `aux.moveMean`); `aux.Mnd`.
    - The configuration differs only in `stop.rule` (absent at 6b08708), `runtime.verbose` and `runtime.name`.

    PEDERSEN_S160_REPRODUCTION.md.
13. **Were scientific identities kept separate?** Yes: three canonical names with the material law and controller in the name, per-preset caveats, per-preset harness assertions (each fails on the other formulation) and distinct configuration hashes. PRESET_IDENTITY.md.
14. **Did any historical alias silently change formulation?** No.
    - The only resolvable alias, `duOlhoffFixedPenaltySensitivityFiltered`, resolves to the β-stall preset with an identical hash.
    - Provenance aliases are refused as names.
    - Unnamed calls are refused, so no call follows the production change.
15. **What schema change occurred?** 81 → 87 rows: `material.stiffness.linearBelow`, `optimizer.inner.asymptoteHistory`, `move.adaptive.grow`, `move.adaptive.shrink`, `stop.guards.settledWindow`, `stop.guards.boxInactiveFraction`. Enum domains also grew (`pedersen`, `adaptive`, `move.initial` up to Inf).
16. **Why do config hashes change?** `olhoffcurrent_config_hash` hashes every schema row, so added rows change every hash even when no scientific value changes. The added rows sit at the values that select the old code path.
17. **Were historical hashes preserved in provenance?** Yes.
    - `PROVENANCE.json` event 1 records the nine campaign hashes; event 2 records the new 87-row hashes of both presets.
    - CONFIG_HASH_TRANSITION.md maps old → new for all 14 configurations.
    - No historical artifact was rewritten.
18. **Is SOURCE_MANIFEST consistent with actual tree?** Yes: 79 files, tree `4ba9a3ae10881344…`. Every file hash equals the live file and the 253069 snapshot; independently computed in Python and by MATLAB.
19. **Is PROVENANCE now accurate?** Yes. Prose, JSON, manifest, tree and README agree (20/20 scripted checks). The stale "74 files / one adaptation" is corrected to 79 files / zero adaptations, with the history explained. The 695f03b record is preserved verbatim in `history[0]`.
20. **Are stage exhaustion and tOuter now supplied by upstream code?** Yes, both in 253069 as default-off options, byte-identical in `+impl`.
21. **Does OlhoffCurrent still need private solver/controller edits?** No.
22. **Was the p-continuation logging defect left untouched and documented?** Yes.
    - A6/A7 on the migrated tree have science digests equal to upstream's.
    - ρ/ω/Δρ equal the committed references; `hist.move` differs only at the p-events (86/97, 61/72).
    - The separate repair recommendation is in KNOWN_PREEXISTING_DEFECTS.md.
23. **Were filter/MMA/multiplicity/material kernels changed beyond approved upstream promotion?** No.
    - Filter, MMA, `innerLoop`, multiplicity and mass-interpolation files were already identical and untouched.
    - `assemble2D`/`genGrad`/`eigSolve` and the new `stiffnessInterpolation` come from 253069 and are inert under SIMP (bitwise historical reproduction).
    - ρ_min, maxCluster, radii and material values are unchanged.
24. **Were other benchmark methods untouched?** Yes. The Proposed/Yuksel code, profiles, radii and evaluator are unchanged. Harness edits are Olhoff-specific (DIFF_AUDIT.md: 0 unauthorized, 0 Phase-6 paths).
25. **Was Phase 6 deferred?** Yes. The R = 0.06 re-freeze for other methods, Pedersen/mass changes in Proposed, and the evaluator ω₁ policy are each classified SEPARATE BENCHMARK POLICY / SCIENTIFIC TASK.
26. **Did all 160×20 gates pass?** Yes: 9 gate solves plus 2 defect anchors, all comparisons pass (`evidence/comparisons.json → pass`, 11/11).
27. **Did all tests pass?** No test failure is attributable to the migration.
    - All OlhoffCurrent suites, all new tests, and the six upstream suites run against the migrated tree have 0 failures; preflight passes 38/38.
    - Pre-existing and identical on pristine `013cc48`: `test_finalization_gate` H/I (other studies' evidence state) and harness self-test T2.
    - One migration-caused failure (a pinned production-source hash in `two_branch_controller_validation`) was resolved by a strict `SUPERSEDED_PRODUCTION_SOURCE` gate rule with new destructive tests, not by editing evidence.

    TEST_REPORT.md, PREREGISTRATION_AMENDMENT_1.md.
28. **What is the selected production preset?** `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, recorded as event 2 of `PROVENANCE.json → production_preset_events` (2026-09-13). The historical β-stall preset is retained and eligible.
29. **Why was it selected?**
    - It is a distinct, defensible reconstruction whose committed nine-mesh sweep terminates without localized-mode spikes. Its success is formulation stability.
    - The historical formulation's cross-mesh termination was judged not credible, and the three-rung successor was never promoted.
    - It is not claimed to be more KKT-stationary, and was not chosen for topology appearance.
30. **What caveat accompanies it?** The per-preset caveat in `olhoffcurrent_caveat(name)`, carried into every result and table. It states:
    - Pedersen low-density stiffness and linear mass;
    - the adaptive per-element box;
    - natural heuristic termination, not a KKT certificate;
    - the fixed physical R = 0.06 (1.2 elements at 160×20, 6 at 800×100);
    - terminal bimodality not expected beyond coarse meshes (native gap 0.7 % at 160×20, 11.8–24.5 % from 240×30);
    - a DISTINCT formulation from the historical eq. (4b) reconstruction, not a bug fix;
    - evaluator model must be named;
    - not "Olhoff 2007".
31. **Are per-outer and total-cost reporting both available?** Yes.
    - Totals are kept; per-outer mean/median outer time, outer-excluding-inner, eig, gradient and total per outer are added.
    - They appear in `rec.times`, the detailed CSV, the notes table, the timing schema and a per-outer power-law fit.
    - Tested by `test_cost_reporting`, self-test T14 and the harness smoke. COST_REPORTING.md.
32. **Is OlhoffCurrent now ready for a new nine-mesh campaign?** Ready for **authorization**, not authorized. CAMPAIGN_GATE.md lists the owner decisions: preset, Phase 6 fairness, evaluator model, cost.
33. **Was any mesh >160×20 run?** No. Configuration *resolution* (no solve) at the recorded historical meshes was used only to reproduce recorded hashes.
34. **What is the exact next action?** The method owner reviews this migration: REPORT, AMENDMENT_1, HARNESS_UPDATE production decision. The owner then merges `migration/olhoffcurrent-upstream-253069` into `benchmark-methodology-r2` and re-runs the OlhoffCurrent suites in the primary checkout (MIGRATION_HANDOFF.md). Only after that, and as a separate explicit decision, authorize or decline the nine-mesh campaign.

## Where to look

| question | document |
|---|---|
| identities, start state, preregistration | SOURCE_IDENTITY.md, TARGET_START_STATE.md, MIGRATION_PREREGISTRATION.md, PREREGISTRATION_AMENDMENT_1.md |
| files | PROMOTION_MAP.md, BYTE_IDENTITY.md, DIFF_AUDIT.md |
| presets | PRESET_IDENTITY.md, HISTORICAL_PRESET.md, PEDERSEN_PRESET.md |
| configuration hashes | CONFIG_HASH_TRANSITION.md |
| provenance | MANIFEST_PROVENANCE.md |
| 160×20 science | HISTORICAL_REPRODUCTION.md, PEDERSEN_S160_REPRODUCTION.md, SHARED_IMPLEMENTATION_EQUIVALENCE.md |
| harness, cost | HARNESS_UPDATE.md, COST_REPORTING.md |
| defects, tests | KNOWN_PREEXISTING_DEFECTS.md, TEST_REPORT.md |
| decision, handoff | CAMPAIGN_GATE.md, MIGRATION_HANDOFF.md |
| machine-readable | METRICS.json, EVIDENCE.json, DATA_MANIFEST.json, `evidence/*.json`, FINAL_SHA256.txt |
