# PROMOTION_MAP — derived from the trees, not from the plan

Inputs: the pre-migration target `+impl` (75 files, tree `edbfe47e…`), the committed upstream tree `253069` (`git ls-tree`, executable subset `algo fem filter mma mma_published architecture/{+olh,olhoffSolve.m,legacy,docs}`), and the audited map `scientific_delta_olhoff_migration/SOURCE_TO_TARGET_FILE_MAP.md` (15 DIFFERENT + 4 SOURCE_ONLY against 6b08708).

## Result

| class | count | notes |
|---|---|---|
| already byte-identical, not copied | 59 | includes `+olh/+move/exhaustion.m` (the target's file, now upstream) |
| differed → replaced by byte copy | 16 | the 15 of the audited map + `docs/MIGRATION_FROM_LEGACY.md` (changed by 253069 itself) |
| upstream-only → added by byte copy | 4 | `stiffnessInterpolation.m` and three presets |
| target-only | 0 | |
| excluded (as at 695f03b) | 1 | `architecture/README.md` |
| **after migration** | **79** | all byte-identical to 253069 |

The plan's "≈15 files" was not trusted: the set was recomputed file by file. It differs from the audited 6b08708 map by exactly one file, `architecture/docs/MIGRATION_FROM_LEGACY.md`, which the capability commit 253069 changed (legacy spellings of the stage-exhaustion option). Of the audited map's seven "changed on both sides" files, all now come from 253069, where the three-way merge was done and verified upstream; no merge was performed in the target.

## The 20 copied files

| path | action | pre-migration target SHA-256 | upstream = post SHA-256 |
|---|---|---|---|
| `algo/genGrad.m` | REPLACED | `13f2ab8541b2f626…` | `6176f68ba89120e2…` |
| `algo/innerLoopRho.m` | REPLACED | `aded1849467e9320…` | `f892f720c29e6ca1…` |
| `architecture/+olh/+config/describe.m` | REPLACED | `e089721ba2b91f5f…` | `ba09e574bde8d46d…` |
| `architecture/+olh/+config/fromLegacy.m` | REPLACED | `7d0bad832c906c53…` | `36fdb3339c39697d…` |
| `architecture/+olh/+config/schema.m` | REPLACED | `d7a808995d9a3481…` | `3fa03435a0fc54f2…` |
| `architecture/+olh/+config/toLegacy.m` | REPLACED | `685caacbd3db2bd1…` | `38fd269c0372c1c1…` |
| `architecture/+olh/+config/validate.m` | REPLACED | `30e495ea8bcc2519…` | `10493fbe6717cfce…` |
| `architecture/+olh/+material/stiffnessInterpolation.m` | ADDED | `ABSENT…` | `6dfedb07f21c06ed…` |
| `architecture/+olh/+move/limit.m` | REPLACED | `61fa923d430121ea…` | `f1d519c813956693…` |
| `architecture/+olh/+presets/duOlhoffAdaptiveMove.m` | ADDED | `ABSENT…` | `3e7962dc0ad0a148…` |
| `architecture/+olh/+presets/duOlhoffAdaptivePedersen.m` | ADDED | `ABSENT…` | `94e924e6ce46ea40…` |
| `architecture/+olh/+presets/duOlhoffOuterAsymptotes.m` | ADDED | `ABSENT…` | `59d154c1198001f5…` |
| `architecture/+olh/+presets/list.m` | REPLACED | `d30b9ab76114e39a…` | `2593695f8801b4eb…` |
| `architecture/docs/CONFIG_REFERENCE.md` | REPLACED | `397939c66043ef01…` | `94a05f13413892ae…` |
| `architecture/docs/MIGRATION_FROM_LEGACY.md` | REPLACED | `1808da8b98cafc87…` | `dd2c9cabf6d82319…` |
| `architecture/docs/PRESETS.md` | REPLACED | `5fc61dde76f71d62…` | `de0baa086173b00d…` |
| `architecture/docs/SCIENTIFIC_CONFIG_PROVENANCE.md` | REPLACED | `22cd312905fe4589…` | `deb3c90cad72b39d…` |
| `architecture/olhoffSolve.m` | REPLACED | `1e5a114cbf91717e…` | `6c891b42a6c3bbdd…` |
| `fem/assemble2D.m` | REPLACED | `7b2f1e10228d5723…` | `ebf543df22d77459…` |
| `fem/eigSolve.m` | REPLACED | `b0784ceeb15fafe1…` | `1f7181e45b57b0db…` |

Every copy was verified immediately after writing (target SHA-256 = snapshot SHA-256); `evidence/promotion_copy_log.csv` holds the full digests.

## Old local divergence eliminated (Part 4)

Pre-migration, seven target files differed from the target's own promoted base 695f03b (commit `1438aa3`: `olhoffSolve.m`, `+move/limit.m`, `+config/{schema,validate,fromLegacy,toLegacy}.m`, new `+move/exhaustion.m`). After migration every one of them hashes to 253069 (`exhaustion.m` was already identical). Remaining local source differences: **none**.

| capability | pre-migration location | now supplied by |
|---|---|---|
| stage-exhaustion controller | target-local edits (1438aa3) | upstream 253069 (`move.continuation.signal`/`stop.rule = stageExhaustion`, default off) |
| `hist.tOuter` | target-local adaptation (695f03b promotion) | upstream 253069 |
