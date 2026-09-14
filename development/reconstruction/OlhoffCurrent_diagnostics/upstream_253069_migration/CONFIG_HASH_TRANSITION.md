# CONFIG_HASH_TRANSITION

```
CONFIG_SCHEMA_MIGRATION_PASS
```

Schema rows: **81 → 87**. Added rows (inserted within their schema blocks; relative order of the old rows preserved = True; removed rows: 0):

- `material.stiffness.linearBelow`
- `optimizer.inner.asymptoteHistory`
- `move.adaptive.grow`
- `move.adaptive.shrink`
- `stop.guards.settledWindow`
- `stop.guards.boxInactiveFraction`

Enum domains also grew (`material.stiffness.model` gains `pedersen`, `move.policy` gains `adaptive`, `move.initial` admits `Inf`); domains are not part of the hash.

## Why every hash changes

`olhoffcurrent_config_hash` is SHA-256 over `path=value` lines for **every schema row** in schema order. Adding six rows changes the hashed text for every configuration, including configurations whose scientific values are untouched. Hash equality old/new is therefore not required and not expected.

## A. Scientific value equivalence (all 14 recorded historical configurations)

For each: resolved on the pre-migration tree (hash must reproduce the recorded one), resolved on the migrated tree, all 81 old leaves compared by value, the 81-row hash recomputed **from the migrated configuration**, and the added rows checked at the values that select the pre-migration code path (`optimizer.inner.asymptoteHistory = inner`, `stop.guards.settledWindow = 1`, `stop.guards.boxInactiveFraction = 0`; `material.stiffness.linearBelow` and `move.adaptive.*` are inert under `simp`/`ladder`). Configuration **resolution only** — no mesh above 160×20 was solved.

| config | mesh | recorded (old) hash | source of record | pre-tree reproduces | 81 leaves equal | old hash from migrated cfg | added rows neutral | new hash (87 rows) | = upstream resolution |
|---|---|---|---|---|---|---|---|---|---|
| β-stall (production until 2026-09-13) | 160×20 | `28756d22aacb…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `c0fe56ce897c…` | True |
| β-stall (production until 2026-09-13) | 240×30 | `0856b13d02e3…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `6578c7846a68…` | True |
| β-stall (production until 2026-09-13) | 320×40 | `2a5b500991ff…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `7bc5b94f4a0b…` | True |
| β-stall (production until 2026-09-13) | 400×50 | `cec3cd6b89ad…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `95d2400f000f…` | True |
| β-stall (production until 2026-09-13) | 480×60 | `a49417d0571d…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `11bcb9a9e4df…` | True |
| β-stall (production until 2026-09-13) | 560×70 | `3478f34841ce…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `1a6c7b95f388…` | True |
| β-stall (production until 2026-09-13) | 640×80 | `e5d868133d52…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `36079c2f075e…` | True |
| β-stall (production until 2026-09-13) | 720×90 | `7efe1ee908be…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `a338c98d5c2a…` | True |
| β-stall (production until 2026-09-13) | 800×100 | `9321858983a3…` | campaign_9mesh_r2/benchmark_results.json | True | True | True | True | `a37164f0895a…` | True |
| three-rung stage exhaustion | 320×40 | `afad9ea4b27d…` | diagnostics/three_rung_promotion_validation_retry1/METRICS.json | True | True | True | True | `a068b77bfb7d…` | True |
| three-rung stage exhaustion | 480×60 | `03097a28b0ad…` | diagnostics/three_rung_canary_preflight/EFFECTIVE_CONFIG.json | True | True | True | True | `7e17eaad4b1b…` | True |
| three-rung stage exhaustion | 800×100 | `7724af5ed267…` | diagnostics/three_rung_canary_preflight/EFFECTIVE_CONFIG.json | True | True | True | True | `a8e21e92117f…` | True |
| four-rung stage exhaustion (override) | 160×20 | `31d2ef382746…` | diagnostics/two_branch_controller_validation/runs/C160x20_record.json | True | True | True | True | `75e5d6527ed5…` | True |
| four-rung stage exhaustion (override) | 320×40 | `2359a1112fce…` | diagnostics/two_branch_controller_validation/runs/C320x40_record.json | True | True | True | True | `910f92554c6e…` | True |

Checks (`evidence/config_transition.json → checks`): allPreHashesReproduceRecorded = True, allOldLeavesEqual = True, allOldSchemaHashesFromPostReproduceRecorded = True, allAddedLeavesNeutral = True, allHistoricalPostEqualsUpstream = True, allPedersenPostEqualsUpstream = True, schemaOnlyGrew = True.

## B. Hash provenance transition

| preset | mesh | old hash (81 rows) | new hash (87 rows) |
|---|---|---|---|
| BETA | 160x20 | `28756d22aacb59726be9f37583deca89fcfcecc93f9b867d46a49223ed1db697` | `c0fe56ce897ce32ec8efc90ca1fb6bdd9c8326ee56700f5ce33adfb26bded17d` |
| BETA | 240x30 | `0856b13d02e3f1065c0ab145213bb57afb1b98570e64c9b023851f696141196c` | `6578c7846a6851ea8eb60a47abb59b4482432f324e3ee908092cabcae0dcf7ee` |
| BETA | 320x40 | `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad` | `7bc5b94f4a0b2aca4f66bbec9cc3664c5181cdaddca6b556eff0b506f8f07b08` |
| BETA | 400x50 | `cec3cd6b89ad67bc5386093a0caaf1b05dc172ddb75f488c52088c69d55d0c20` | `95d2400f000fbd145006524513147b4ace1d1c3f75c339d605a15caac155d84c` |
| BETA | 480x60 | `a49417d0571d3c2406d030cbab1fda58a1dad8d334991c5aaf22fa17f1d7f601` | `11bcb9a9e4df3fdb2f53a82e84df1070cf1a3ce6c97be817a63249ecdb262450` |
| BETA | 560x70 | `3478f34841ce743457aac7d4ba1755f2989453a9f0d79ccfc4f961f89fb7fd64` | `1a6c7b95f388afeb4789413f8ff4bd5cce347f210e2ef6c76b54285a7f891068` |
| BETA | 640x80 | `e5d868133d52a510797e6847b9f113ab3d5424849f8473f23c7827485bff300b` | `36079c2f075ec7f49a31eeaf1fc27ae329ff9b590d52378235b280f0a4b9a235` |
| BETA | 720x90 | `7efe1ee908be54cefd2b9e63f911e337809b92870056c7b1854b46bd41a91078` | `a338c98d5c2aa77c65d754bc03684d66b53ea5b2ccddce7b8ee56d4c9992a35c` |
| BETA | 800x100 | `9321858983a3d7d33ca1a7b5bed9136fb7967e637b7ac5b40dadfad5ffdd4132` | `a37164f0895a0a2abb65f44e12ce05c4517088af638a7ca5a7322a094f39416a` |
| EX3 | 320x40 | `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab` | `a068b77bfb7d05d2898e7100da32a562aa6dab3bac1bcbb068d96b2a2c15cd4b` |
| EX3 | 480x60 | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` | `7e17eaad4b1b83b94142de7ef82a5c8bee1e7c7af4da9343d94fb7e90fb0dbfe` |
| EX3 | 800x100 | `7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe` | `a8e21e92117fc75d8be1f9afc6a89bd38c3ee6a06ddb1eab1b18d0ffcfa62653` |
| EX4 | 160x20 | `31d2ef382746a942a4036d07dd6a1012742432cba0497d2de5ec24b51d2b5904` | `75e5d6527ed5317fabaae6f8a56a71b780c869703bbea888dcb62b69846ff2bb` |
| EX4 | 320x40 | `2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4` | `910f92554c6ef52a7d6558a244689944b3e2e918f9641a3a6b36bcaa49127171` |

### New preset: `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` (no pre-migration hash)

| mesh | new hash | = upstream 253069 resolution | differs from the committed 6b08708 sweep cfg only in |
|---|---|---|---|
| 160x20 | `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 240x30 | `2fac1384239527b1a0108e1d34725021cdc7c5a26f1939b602239415b34847e1` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 320x40 | `a1203d43efb240308cfcb520c92a6d9bc37a0c703c728fc2f3d298f779bc5f6e` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 400x50 | `e40c6ba16c9addbb587d08ecab29669f7db560d09b980ca62c2096856b849280` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 480x60 | `8e55d18251152ec2d26836593489c14f4842390d912d7c66f43f3b18730c83de` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 560x70 | `8b2533a3caf8432d40a54dbcbd8958900e333c410527d5547c53699b9e373ccf` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 640x80 | `cbe8dcf070d3e318a889867b5168fdbb09f1b6babfd46f981447d96e9a073aaf` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 720x90 | `2893ad47136fc9a7775f1f4d9ff1dcba38e7fe14271dd12b5727ff4bd8bb54f3` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |
| 800x100 | `f9138743067afd0edce255ad26a6d7ae5c0f632a537516a4793eafecbacbe4cd` | True | `stop.rule`, `runtime.verbose`, `runtime.name` |

`stop.rule` did not exist at 6b08708 (added by 253069 with default `designChange`, the behaviour 6b08708 hard-wired); `runtime.verbose` and `runtime.name` are not scientific fields.

## Rules

- Old hashes (e.g. campaign 480×60 `a49417d0…`, C480 `03097a28…`) **remain valid identifiers of the historical artifacts that recorded them**. No historical evidence file was rewritten.
- New results carry new hashes. `PROVENANCE.json → production_preset_events` records both, and `tests/fixtures/schema_rows_pre_253069.json` keeps the 81-row list so any old hash can be recomputed from a new configuration (`test_preset_identity` does this for three of them).
