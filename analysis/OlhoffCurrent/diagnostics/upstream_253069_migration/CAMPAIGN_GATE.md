# CAMPAIGN_GATE — Part 25

```
OLHOFF_NINE_MESH_CAMPAIGN_READY_FOR_AUTHORIZATION
```

**This is not an authorization.** No mesh above 160×20 was solved by this task, and none should be until the method owner has reviewed the migration and explicitly authorized a campaign.

## Why "ready for authorization"

Every migration gate passed. Pre-existing test failures are disclosed and proved identical on the baseline (TEST_REPORT.md, PREREGISTRATION_AMENDMENT_1.md). The following statement is now demonstrably true:

> OlhoffCurrent now uses the shared upstream implementation derived from Olhoff commit 253069262407885a8b759a9e721c4f0a7d3a397d. The historical Eq.(4b)-mass reconstruction remains reproducible under its own explicit preset, while the successful Pedersen/adaptive-box formulation is exposed under a separate named preset. No scientific formulation was silently overwritten.

| basis | where |
|---|---|
| 79/79 `+impl` files byte-identical to 253069 | BYTE_IDENTITY.md |
| historical β-stall and stage-exhaustion presets bitwise reproduce historical 160×20 evidence | HISTORICAL_REPRODUCTION.md |
| Pedersen preset bitwise reproduces committed S160x20 | PEDERSEN_S160_REPRODUCTION.md |
| target = upstream on three configurations | SHARED_IMPLEMENTATION_EQUIVALENCE.md |
| benchmark preflight 38/38 at 160×20 with formulation-specific assertions | HARNESS_UPDATE.md |
| total and per-outer cost reporting in place | COST_REPORTING.md |

## What the owner should decide before authorizing

1. **Which preset the campaign runs.** The harness is wired to `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` (`confbench_olhoff_preset.m`), and preflight requires it to equal the recorded production preset. Running the historical preset instead is a one-line change in `confbench_olhoff_preset.m`, plus a new `production_preset_events` entry. The preflight's β-stall assertion set already exists.
2. **Benchmark fairness (Phase 6, deferred).** Proposed and Yuksel still use their frozen element-unit filter radii and their own low-density treatments. Olhoff uses a fixed physical R = 0.06 and the Pedersen law. Whether the conference table needs the separate Phase 6 policy work first is a benchmark-policy decision outside this migration.
3. **Evaluator model.** Olhoff native ω are Pedersen/linear-mass values. The common E1/E2/E3 evaluator still runs outside timing. Every table must name which model it reports.
4. **Expected cost.** Indicative only, from the committed sweep and upstream audit:
   - 121–246 outer and 1913–4650 inner iterations per mesh;
   - about 270 s at 160×20 solo;
   - about 1.1–1.25 s per inner sub-iterate at 800×100, so roughly 1.5 h for that mesh alone.

   Budget several hours for the Olhoff column.
5. **Pre-existing harness defects that do not block a campaign but should be known:** self-test T2 (KNOWN_PREEXISTING_DEFECTS D2), and the finalization-gate state of seven older studies (D3).

## What must not happen

- Running nine meshes directly from this worktree branch without the owner merging or reviewing it.
- Changing Proposed or Yuksel in the same step.
- Re-labelling the Pedersen results as "Du-Olhoff reconstruction (M4)" or as "Olhoff 2007".
