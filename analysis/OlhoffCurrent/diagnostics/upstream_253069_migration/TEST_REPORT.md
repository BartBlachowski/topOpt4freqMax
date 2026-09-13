# TEST_REPORT — Part 21

```
RELEVANT_TEST_SUITE_PASS
```

with the disclosure below.

- **No test failure is attributable to the migration.**
- Three checks fail, and each fails **identically on a pristine `git archive` of `013cc48`** run in the same way: finalization-gate H and I, and harness self-test T2.
- The preregistration said "0 failures". That wording did not anticipate pre-existing failures in suites that also inspect unrelated historical studies. The interpretation is recorded as a post-hoc deviation in PREREGISTRATION_AMENDMENT_1.md.

All runs: MATLAB 25.2.0 (R2025b), single thread, worktree `migration/olhoffcurrent-upstream-253069`. Logs are under `analysis/OlhoffCurrent/evidence/upstream_253069_migration/logs/`; suite results in `T_*.mat`.

## OlhoffCurrent suites (migrated tree)

| suite | covers | checks | failures |
|---|---|---|---|
| `test_path_isolation` | fail-closed dispatch, helper shadowing (A–E) | 6 | **0** |
| `test_currentness` | manifest integrity, provenance, state model (`CURRENT` against the recorded branch), `LOCAL_MODIFIED` reachable, every registered preset delegates to an existing upstream preset, production registered and eligible | 13 | **0** |
| `test_source_integrity` | artifacts ignored, source edits/removals/extra files block, 79-file pin, competing implementation blocks | 10 | **0** |
| `test_preset_identity` (new) | config/schema (87 rows), preset registry, aliases, refusal of unnamed/provenance/unknown names, three resolved formulations field by field, historical hashes 28756d22/afad9ea4/31d2ef38 recomputed from migrated configs, production event log, distinct identity, caveats | 26 | **0** |
| `test_preset_equivalence` (historical, 160×20) | β-stall preset vs frozen conference record: ρ/ω₁/volume bitwise, 91/2241, converged | 6 | **0** |
| `test_named_preset_reproduction('pedersen')` (new, 160×20) | Pedersen preset vs committed S160x20 digests: ρ, ω, λ, 121/2369, status, log, 24 hist fields, aux, additions | 10 | **0** |
| `test_named_preset_reproduction('stageExhaustion')` (new, 160×20) | three-rung preset vs pre-migration OlhoffCurrent digests: ρ, ω, λ, 180/4283, log, 36 hist fields incl. `ex*`, `res.exhaustion`, additions limited to `aux` | 10 | **0** |
| `test_cost_reporting` (new, 160×20 cap 3) | total and per-outer cost fields and identities, CAP_HIT, preset recorded | 8 | **0** |
| `test_pedersen_adaptive_units` (new, no solve) | adaptive box through `olh.move.limit` with the resolved Pedersen preset: initial box, ×1.2 / ×0.7 / zero-step rule, clamps, contraction to the 0.002 floor and regrowth to 0.10, refusal without ρ; Pedersen stiffness through `olh.material.stiffnessInterpolation`: SIMP branch ≥ 0.1, ρ·ρ₀^(p−1) below, C⁰ continuity, SIMP path unchanged, mass/stiffness ratio 100 at ρ_min | 12 | **0** |
| `test_evidence_retention` | evidence gate R1–R10 incl. git-ignored evidence | 15 | **0** |
| `test_finalization_gate` | A–G sandbox, **J1–J3 (new)** superseded production source, H real studies, I legacy ledger | 15 | **2 — pre-existing** (see below) |

## Upstream architecture suites run against the migrated `+impl`

These are the six root-independent suites of the 253069 snapshot, run under OlhoffCurrent's gate with `olhoffSolve` and `olh.move.limit` asserted inside `+impl`. The same suites were also run on the snapshot itself.

| suite | covers | target failures | snapshot failures |
|---|---|---|---|
| `test_config` | schema/validate/resolve rules | 0 | 0 |
| `test_mass` | mass interpolation | 0 | 0 |
| `test_modules` | FE, filter, multiplicity and move-policy modules | 0 | 0 |
| `test_preset_resolution_unchanged` | 13 presets × 3 meshes vs the 6b08708 resolution fixture | 0 | 0 |
| `test_stage_exhaustion` | controller fixtures 1–11, config rules, wiring solves at 160×20 (CAP_HIT, trace, projection refusal, diagnostics inert) | 0 | 0 |
| `test_outer_timing` | `hist.tOuter` on cap / break / stage-exhaustion paths, including two adaptive-Pedersen solves; excluded from the science record | 0 | 0 |

Not re-run: `test_continuation`, `test_legacy_roundtrip`, `test_presets_match_history`, `test_preset_reproduces_anchor`. They hard-code the upstream root. They are covered by byte identity, by the upstream audit's suite run on the candidate (0 failures), and here by the A6/A7 anchor runs.

## Harness

| check | result |
|---|---|
| `confbench_preflight` (160×20, all methods, no solve) | **38/38 PASS** |
| `confbench_selftest` T1, T3–T14 (incl. new T13, T14) | PASS |
| `confbench_selftest` T2 | FAIL — **pre-existing**; identical on `013cc48` (`harness_selftest_BASELINE_013cc48.json`); KNOWN_PREEXISTING_DEFECTS D2 |
| Olhoff smoke + manifest + export | PASS (`harness_check.json`) |

## Controller, timing, adaptive-box and stage-exhaustion coverage (brief's list)

| requested category | covered by |
|---|---|
| config/schema tests | `test_config`, `test_preset_identity` §7, CONFIG_HASH_TRANSITION |
| preset tests | `test_preset_identity`, `test_currentness` §5, `test_preset_resolution_unchanged` |
| historical-equivalence tests | `test_preset_equivalence`, `test_named_preset_reproduction('stageExhaustion')`, HISTORICAL_REPRODUCTION (β-stall, three-rung, four-rung) |
| Pedersen-preset source-equivalence tests | `test_named_preset_reproduction('pedersen')`, PEDERSEN_S160_REPRODUCTION |
| stage-exhaustion tests | `test_stage_exhaustion`, EX3/EX4 runs |
| adaptive-box tests | `test_pedersen_adaptive_units` (rule and clamps, direct); Pedersen box trajectory bitwise over 121 iterations (`hist.move`, `aux.moveMean`); `test_outer_timing` adaptive-Pedersen solves; `test_preset_resolution_unchanged` (adaptive presets); preflight assertions |
| timing tests | `test_outer_timing`, `test_cost_reporting` |
| controller tests | `test_stage_exhaustion` fixtures, A6/A7 anchors |
| manifest/provenance tests | `test_currentness`, `test_source_integrity`, `manifest_provenance_check.json` |
| benchmark preflight tests | preflight 38/38, self-test T13 |
| 160 promotion tests | the 160×20 gate runs (9 solves + 2 anchors) |

## The pre-existing failures, proved identical

Baseline: `git archive 013cc48` extracted to the scratchpad, with the same git-ignored evidence directories linked in.

| check | pristine 013cc48 | migrated |
|---|---|---|
| finalization H `move_activity_400` | FAIL (required trajectory hash mismatch) | FAIL (same) |
| finalization H `two_branch_controller_validation` | PASS | PASS |
| finalization I failing studies | `controller_architecture_offline, move_activity_400, move_ladder_necessity, three_rung_architecture, three_rung_promotion_closure, three_rung_promotion_validation_retry1, two_rung_architecture` | **the same seven** |
| finalization J1–J3 | (not present) | PASS |
| self-test T2 | FAIL | FAIL (same message) |

`test_evidence_retention` on the baseline archive shows R5a/R5/R6 failing only because an extracted archive is not a git repository. In the worktree (a repository) it passes 15/15.

## One migration-caused failure found — and how it was resolved

Before the gate change, `test_finalization_gate` also failed H for `two_branch_controller_validation`. That study's `FINAL_SHA256.txt` pins seven **production source** files by working-tree path: `SOURCE_MANIFEST.json`, `olhoffSolve.m`, `+move/limit.m`, `+config/{schema,validate,toLegacy,fromLegacy}.m`. The promotion legitimately replaced those files.

- **Not done**: editing that historical file, which would rewrite evidence.
- **Done**: `olhoffcurrent_finalization_gate` accepts such a line only as `SUPERSEDED_PRODUCTION_SOURCE`, when the recorded digest equals the file's content in a commit reachable from HEAD and `+impl` verifies against its manifest.
  - All seven lines resolve to commit `1438aa3`.
  - New tests prove the rule is not a pass-through: J1 (historical source digest passes), J2 (fabricated digest fails), J3 (historical digest of a non-source file fails).
  - `evidence/finalization_gate_affected.json` records the verdicts for `two_branch_controller_validation` and this study (both PASS).

## Execution incidents (recorded)

1. The first gate-suite run with the new rule hung. MATLAB's `system()` let `git log` open its pager. The job was killed, and the tree integrity was verified intact immediately afterwards (manifest 79/79, no extra files). The sandbox directory left by the killed test was removed. Fix: `git --no-pager`. The re-run is the reported result.
2. Harness check runs 1–2 failed because of the audit script itself: a missing `cfg.runWarmup`, and records lacking the driver-added `mesh` field. Fixed in the script only; run 3 is reported.
3. The first comparison run mis-encoded the preregistered allowances (HISTORICAL_REPRODUCTION.md). Fixed with a stricter check; data unchanged.
4. `test_pedersen_adaptive_units` run 1 failed P2 because of the new test itself: its 1e-18 tolerance was below the rounding of `0.1^2 = 0.010000000000000002`. The check now asserts the documented branch `g = ρ·ρ₀^(p−1)` exactly, with ρ/100 to 1e-17. The code did not change. Run 1 log: `logs/T_pedersen_adaptive_units.run1.log`.
5. An earlier draft of this report attributed adaptive-box coverage to `test_modules`/`test_config`. Neither contains an adaptive or Pedersen case (checked by search), so the direct unit test above was added instead.
