# NINE_MESH_CONFIG_PREVIEW — Step 5C

```
NINE_MESH_CONFIG_PREVIEW_PASS
```

**Nothing was solved.** Each of the nine meshes was resolved through `olhoffcurrent_config(nelx, nely, 'Preset', …)` **and** through the conference harness (`confbench_method_config('olhoff', …)`), under the fail-closed path guard. Machine-readable: `NINE_MESH_CONFIGS.json`; full detail: `evidence/NINE_MESH_PREVIEW.json`; script: `scripts/pmg_preview.m`. The run-1 log (`logs/PREVIEW_nine.run1_script_defect.log.txt`) failed on a field-name defect in the preview script and is kept; the re-run is the result.

| mesh | NE | config hash | ε (l2) | R phys | rminEl |
|---|---|---|---|---|---|
| 160×20 | 3200 | `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4` | 0.05 | 0.06 | 1.2 |
| 240×30 | 7200 | `2fac1384239527b1a0108e1d34725021cdc7c5a26f1939b602239415b34847e1` | 0.075 | 0.06 | 1.8 |
| 320×40 | 12800 | `a1203d43efb240308cfcb520c92a6d9bc37a0c703c728fc2f3d298f779bc5f6e` | 0.1 | 0.06 | 2.4 |
| 400×50 | 20000 | `e40c6ba16c9addbb587d08ecab29669f7db560d09b980ca62c2096856b849280` | 0.125 | 0.06 | 3 |
| 480×60 | 28800 | `8e55d18251152ec2d26836593489c14f4842390d912d7c66f43f3b18730c83de` | 0.15 | 0.06 | 3.6 |
| 560×70 | 39200 | `8b2533a3caf8432d40a54dbcbd8958900e333c410527d5547c53699b9e373ccf` | 0.175 | 0.06 | 4.2 |
| 640×80 | 51200 | `cbe8dcf070d3e318a889867b5168fdbb09f1b6babfd46f981447d96e9a073aaf` | 0.2 | 0.06 | 4.8 |
| 720×90 | 64800 | `2893ad47136fc9a7775f1f4d9ff1dcba38e7fe14271dd12b5727ff4bd8bb54f3` | 0.225 | 0.06 | 5.4 |
| 800×100 | 80000 | `f9138743067afd0edce255ad26a6d7ae5c0f632a537516a4793eafecbacbe4cd` | 0.25 | 0.06 | 6 |

All nine hashes equal `PROVENANCE.json` event 2 `config_hashes` exactly, and the harness route gives the same hash as the API on every mesh.

## Checks (all PASS)

| check | detail |
|---|---|
| same preset | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` everywhere; it is both the recorded production preset and the benchmark's named preset |
| only mesh-bound rows vary | of 87 rows, the ones that vary across meshes are `domain.mesh.nelx`, `domain.mesh.nely`, `stop.tolerance` and `runtime.name` |
| stop tolerance is the fixed rule | `stop.toleranceRule = meshScaled`, ε = 0.05·√(NE/3200) on every mesh; not a hidden override |
| material law | Pedersen stiffness (`linearBelow` 0.1, p = 3) with linear mass eq. (2), q = 1, on all nine |
| no eq. (4b) fallback | none |
| fixed physical radius | `filter.radiusPhysical` = 0.06 on all nine, with b = 1 (R = 0.06·b); `filter.radiusElements` = NaN, derived at run time as R/(b/nely) |
| adaptive box | `move.policy = adaptive`, initial and ceiling 0.10, floor 0.002, ×1.2 / ×0.7 |
| natural stopping | `stop.rule = designChange`; guards settledMove, ladderExhausted and maxDesignChange off |
| absent | stage exhaustion (signal and stop rule), p continuation (stiffness or mass), projection |
| no SOCP | inner optimizer is nested MMA on all nine |
| runtime | `runtime.maxOuter` = 400, single thread, diagnostics off, on all nine |
| distinct | 9 distinct hashes |
