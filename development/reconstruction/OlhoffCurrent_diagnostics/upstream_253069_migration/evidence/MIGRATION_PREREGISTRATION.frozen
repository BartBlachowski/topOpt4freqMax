# MIGRATION_PREREGISTRATION — OlhoffCurrent ← Olhoff@253069

Frozen before any file under `analysis/OlhoffCurrent` (other than this directory) or
`examples/Performance` is modified. SHA-256 recorded in `evidence/preregistration_sha256.txt`.
Written 2026-09-13.

## 0. Identities fixed by this document

| item | value |
|---|---|
| target repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| target branch at start | `benchmark-methodology-r2` |
| target HEAD at start | `013cc48451d33bed61c5c4eea174bbd898d548a2` ("Nine test passed") |
| target tracked changes at start | none |
| target untracked at start | 8 diagnostic directories under `analysis/OlhoffCurrent/diagnostics/` (c480_socp_causal_run, filtered_subproblem_integrability_audit, frozen_inner_solver_study, frozen_problem25_reference, gray_kkt_forensic_audit, nine_mesh_campaign_audit, scientific_delta_olhoff_migration, three_rung_canary_preflight) |
| `+impl` at start | 75 files, tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (live recomputation = SOURCE_MANIFEST.json) |
| upstream repository | `/Users/piotrek/Programming/Matlab/Olhoff` |
| upstream migration commit | `253069262407885a8b759a9e721c4f0a7d3a397d` (tree `4571029f…`) |
| parent successful-science commit | `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` |
| upstream snapshot | `git archive 253069…` → tar SHA-256 `f911240340cab69b9806f98f9413d9038649fde5e14adf3bbe4d57f3ddb0dbeb`, extracted read-only in the session scratchpad, 1877 files blob-verified |

## 1. Isolation

The main checkout carries unrelated untracked user work and an attached MATLAB desktop session.
The migration therefore runs in a **dedicated git worktree**
`/Users/piotrek/Programming/topOpt4freqMax-migration-253069` on new branch
`migration/olhoffcurrent-upstream-253069`, created at `013cc48`. The main checkout is not
cleaned, stashed, switched or edited. Historical evidence that is git-ignored in the main
checkout (`benchmark_records.mat`, `analysis/OlhoffCurrent/evidence/**`) is read by absolute
path or copied into the worktree's ignored locations after SHA-256 verification.

## 2. Promotion map (derived, not assumed)

Upstream executable subset = `algo fem filter mma mma_published architecture/{+olh, olhoffSolve.m, legacy, docs}`
(the layout of the 695f03b promotion). `architecture/README.md` stays excluded, as at 695f03b.

| class | count | files |
|---|---|---|
| already byte-identical to 253069 | 59 | (incl. `+olh/+move/exhaustion.m`) |
| differ → byte-copy | 16 | `algo/genGrad.m`, `algo/innerLoopRho.m`, `+olh/+config/{describe,fromLegacy,schema,toLegacy,validate}.m`, `+olh/+move/limit.m`, `+olh/+presets/list.m`, `docs/{CONFIG_REFERENCE,MIGRATION_FROM_LEGACY,PRESETS,SCIENTIFIC_CONFIG_PROVENANCE}.md`, `olhoffSolve.m`, `fem/assemble2D.m`, `fem/eigSolve.m` |
| upstream-only → byte-copy | 4 | `+olh/+material/stiffnessInterpolation.m`, `+olh/+presets/{duOlhoffAdaptiveMove,duOlhoffAdaptivePedersen,duOlhoffOuterAsymptotes}.m` |
| target-only | 0 | |

Expected result: **79 files, 79 BYTE_IDENTICAL_TO_UPSTREAM, 0 INTENTIONALLY_TARGET_SPECIFIC,
0 UNEXPLAINED_DIFFERENCE.** Copies are taken from the extracted snapshot, never from the upstream
working tree. No file under `+impl` is edited by hand.

## 3. Preset identity (wrapper layer only; nothing added to `+impl`)

| canonical name | resolves as | role |
|---|---|---|
| `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` | `olh.presets.duOlhoffFrozenM4`, unchanged | HISTORICAL production formulation (the preset of the 2026-09-11 nine-mesh campaign) |
| `duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered` | `duOlhoffFrozenM4` + `move.levels=[0.04 0.02 0.01]`, `move.continuation.signal=stageExhaustion`, `stop.rule=stageExhaustion`; runtime cap 1600 | HISTORICAL DIAGNOSTIC: the validated-but-never-promoted C320/C480/C800 controller; not production-eligible |
| `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` | `olh.presets.duOlhoffAdaptivePedersen`, unchanged | the successful source formulation, a DISTINCT scientific method variant |

Rules:
- `duOlhoffFixedPenaltySensitivityFiltered` remains a **compatibility alias of the historical
  β-stall preset only**; it can never resolve to another formulation.
- Configuration resolution requires an explicit preset name. A call that names none fails closed
  (so no unnamed call can silently change formulation when the production default changes).
- Production selection is a recorded provenance event in `PROVENANCE.json`
  (append-only list: old/new name, upstream preset, upstream commits, config hashes, date).
- Intended production decision (Part 13), made from the prior audits and not from this
  migration's runs: `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`. It takes effect only if every
  gate below passes; otherwise nothing is committed.

## 4. Runs (160×20 only, single thread, R2025b, timing excluded from all comparisons)

| id | tree | configuration | reference(s) |
|---|---|---|---|
| PRE_BETA | worktree before promotion (edbfe47) | historical β-stall, cap 400, diag off | — |
| PRE_EX3 | worktree before promotion | historical three-rung stage exhaustion, cap 1600, diag on | — |
| POST_BETA | worktree after migration | canonical β-stall preset | PRE_BETA; frozen conference record `campaign_9mesh_r2/benchmark_records.mat` (rho bitwise, ω₁, counts, status); UP_BETA |
| POST_EX3 | after | canonical three-rung preset, diag on | PRE_EX3; upstream-audit `case_TARGET_EX3_160.mat` (edbfe47); UP_EX3 |
| POST_EX4 | after | β-stall preset + stage exhaustion (4 rungs), cap 1600, diag on | committed target record `two_branch_controller_validation/runs/C160x20_record.json` + `_iterations.csv` + declared trajectory (rho_sha256 `332c00a5…`) |
| POST_PED | after | canonical Pedersen preset through the production wrapper | committed upstream `repro/results/S160x20/res.mat` (blob in 6b08708 and 253069); UP_PED |
| UP_BETA / UP_EX3 / UP_PED | read-only 253069 snapshot | the same resolved configuration via `olh.config.resolve` | POST_* |
| POST_A6 / POST_A7 | after | upstream anchors A6_pdecoupled160 / A7_massp160 (legacy route) | upstream-audit `anchor_CAND_*` science digests; committed anchor references (defect localization) |

No solve at any mesh above 160×20. Configuration **resolution** (no solve) at the recorded
historical meshes is permitted solely to reproduce recorded configuration hashes.

## 5. Comparison standard

- Bitwise (`isequal`/`isequaln`): rho, omega, lambda, volume, every non-timing `hist` field, every
  per-iteration Δρ when recorded, `res.log`, status, nOuter, cumulative inner count,
  `res.exhaustion` (stage-exhaustion runs), `res.aux` (Pedersen), controller/box histories
  (`hist.move`, `hist.stage`, `hist.beta`, `hist.dxOuter`, `hist.dxNorm2`, `aux.moveMean`).
- Timing fields excluded: `tEig`, `tGrad`, `tInner`, `tOuter`, wall-clock scalars.
- PRE vs POST: `hist` field sets may differ ONLY by (a) the 12 `hist.ex*` fields that the old
  target created empty when the controller is off, and (b) fields that upstream adds as pure
  reporting (`res.aux`). Any removed non-empty field, or any other difference, is a FAIL.
- POST vs UP: identical field sets and bitwise equality on all non-timing content.
- POST_PED vs committed S160x20: every field of the committed record bitwise; the migrated result
  may add only `hist.tOuter`.
- Topology metrics are reported, not gated beyond rho equality: M_nd = 100·mean(4ρ(1−ρ));
  gray = fraction with 0.1 < ρ < 0.9; mid = fraction with 0.4 ≤ ρ ≤ 0.6 (checked against
  recorded values before use); gap12 = (ω₂−ω₁)/ω₁; rho hash = SHA-256 of the little-endian
  float64 bytes.

## 6. Configuration schema

Expected 81 → 87 rows. For every historical configuration (β-stall 160×20…800×100 campaign
hashes; three-rung C320 `afad9ea4…`, C480 `03097a28…`, C800 `7724af5e…`; EX4 C160 `31d2ef38…`,
C320 `2359a111…`): resolve under the migrated tree, require (i) every one of the 81 old leaves
equal by value, (ii) the old-schema hash recomputed from the migrated configuration equal to the
recorded hash, (iii) every added row at the value that selects the pre-migration code path.
Old/new hash equality is not required.

## 7. Gates (all required; any FAIL → no commit, `OLHOFFCURRENT_MIGRATION_BLOCKED`)

| gate | pass condition |
|---|---|
| UPSTREAM_253069_IDENTITY_PASS | §0 upstream rows verified from the committed object; snapshot blob-verified |
| MIGRATION_BYTE_PROMOTION_PASS | §2 expected result; per-file SHA-256 source = target |
| HISTORICAL_PRESET_PRESERVED_PASS | β-stall and three-rung presets resolve to their historical values (§6); compatibility alias resolves to β-stall only |
| HISTORICAL_PRESET_REPRODUCTION_PASS | POST_BETA, POST_EX3, POST_EX4 meet §5 against every listed historical reference |
| PEDERSEN_PRESET_DISTINCT_IDENTITY_PASS | distinct canonical name; no alias maps between formulations; resolved material law/controller differ as documented |
| PEDERSEN_PRESET_S160_REPRODUCTION_PASS | POST_PED meets §5 against committed S160x20 |
| SHARED_IMPLEMENTATION_EQUIVALENCE_PASS | POST_* = UP_* for BETA, EX3, PED |
| CONFIG_SCHEMA_MIGRATION_PASS | §6 |
| MANIFEST_PROVENANCE_PASS | tree = manifest = prose = JSON; promoted commit 253069, parent 6b08708; zero adaptations recorded |
| BENCHMARK_PREFLIGHT_PASS | harness preflight (no solve) passes with per-preset assertions of the chosen production formulation; historical assertions retained under the historical preset |
| RELEVANT_TEST_SUITE_PASS | all OlhoffCurrent suites and new migration tests: 0 failures; POST_A6/POST_A7 show no change beyond the documented p-continuation logging defect |
| PHASE6_UNTOUCHED_PASS | no change to Proposed/Yuksel code, profiles, radii, evaluator policy |
| UNAUTHORIZED_DIFF_ZERO | every changed path lies in `analysis/OlhoffCurrent/**` or is an Olhoff-specific harness file in `examples/Performance/**` and is listed in DIFF_AUDIT.md |

## 8. Explicitly out of scope

Fixing the p-continuation `hist.move` logging defect (documented only); SOCP; the C480 causal
run; any change to material law values, ρ_min, maxCluster, filter, MMA, multiplicity; Phase 6
(R = 0.06 re-freeze, Pedersen/mass changes in Proposed, evaluator ω₁ policy); nine-mesh or any
mesh above 160×20; merging upstream history; better-looking topology as a criterion.
