# VERIFICATION — repository cleanup, 2026-09-14

Branch `benchmark-methodology-r2`, starting HEAD `05919933f3f4af68b36c1981e66130709a263593`.
**Nothing was committed.** All smoke runs are software/path checks, not scientific evidence.

## 0. Sequencing around live work

| time | event |
|---|---|
| 12:33 | baseline recorded (`BASELINE*.tsv`). A MATLAB `-batch` job (bimodality-gap arm `budget400`, 800×100) was running through `analysis/OlhoffCurrent` from `docs/bimodality_gap/scripts`. **No file was moved.** |
| 12:33–13:50 | source-of-truth analysis, classification, dry runs only |
| 13:50 | MATLAB job exited 0 (`CAP_HIT nOuter=400`, results written) |
| 13:51–14:30 | a peer Claude session (`topopt4freqmax-41`) regenerated the study's figures/tables in `docs/bimodality_gap`; migration held until the repository had been write-quiet for 5 min **and** that session reported idle |
| 14:30 | `PREMOVE_INVENTORY.tsv` (6097 files: baseline + files the live study added) → 183 moves executed |
| 14:38 | second pass: 1 missed file (`examples/Performance/conference_benchmark.zip`) moved; rule added to the manifest |

## A. Repository shape

```
topOpt4freqMax/
├── README.md   .gitignore
├── analysis/     Olhoff/  Proposed/  Yuksel/  elastic2D/
├── development/  audits/ benchmark_history/ experiments/ historical_documents/
│                 legacy_examples/ legacy_implementations/ migration_history/
│                 reconstruction/ repository_cleanup/  README.md
├── docs/         topopt_config.schema.json  complexity.tex  complexity/
├── examples/     Building/ ClampedBeam/ ClampedHingedBeam/ HingedBeam/ elastic2D/
│                 Performance/ weightedTopologyResultsHelper.m
├── paper/        reviews/  references/ (literature, incl. library/ from docs/)
├── tests/        test_development_firewall.py  firewall_allowlist.tsv
└── tools/        Matlab/  Python/  compat/
```

`examples/Performance/`: `performance_comparison.m`, `conference_bench/`,
`benchmark_profile/`, `compose_nine_mesh_comparison.m`,
`conference_benchmark/{campaign_9mesh_r2, nine_mesh_pedersen_b21483b, nine_mesh_comparison_pedersen_b21483b}`.

`tools/` stays at top level (deviation from the suggested layout): it is genuinely shared current
code, and `run_topopt_from_json` plus every runner derive the repository root from its depth.

## B. Current examples / entry paths (smoke, budgets cut)

| check | result |
|---|---|
| benchmark path, Proposed 160×20, `max_outer_override=3` via `confbench_method_config` → `confbench_run_case` → `run_topopt_from_json` | ran, `CAP_HIT` (the cap), profile `proposed_practical_move02_tol001` == frozen manifest; `which topopt_freq` → `analysis/Proposed/Matlab` |
| benchmark path, Yuksel 160×20, cap 3 | ran, `CAP_HIT`, profile `yuksel_practical_move01_tol001` == frozen; `which top99neo_inertial_freq` → `analysis/Yuksel/Matlab` |
| benchmark path, Olhoff 160×20, cap 3, preset `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` | ran, `CAP_HIT`, `rec.implementation = analysis/Olhoff` |
| `which study_evaluate_design` / `run_topopt_from_json` | `examples/Performance/benchmark_profile/…` / `tools/Matlab/…` |
| `confbench_frozen_budget` | yuksel 1000, proposed 2000 (read from the moved freeze manifest) |
| retired keys `Olhoff`, `OlhoffExact`, `OlhoffDu2007Repro` | all raise `run_topopt_from_json:RetiredApproach` |
| Python `tools.Python.run_topopt_from_json`: ClampedBeam JSON (`ourApproach`) and elastic2D Cantilever at 160×20, 3 iterations | both ran; solver dirs `analysis/Proposed/Python`, `analysis/elastic2D/Python` |
| Python `import Yuksel.Python.solver` | OK (`analysis/Yuksel/Python/solver.py`) |

## C. Existing test suites

| test | result |
|---|---|
| `test_path_isolation` (A–F; B/D/E contaminate with the **archived** trees) | nFail = 0 |
| `test_currentness` (state `CURRENT`, local integrity PASS after the rename) | nFail = 0 |
| `test_source_integrity` | nFail = 0 |
| `test_preset_identity` | nFail = 0 |
| `test_pedersen_adaptive_units` | nFail = 0 |
| `test_cost_reporting` (160×20, cap 3) | nFail = 0 |
| `test_named_preset_reproduction('pedersen')` (160×20 vs committed S160x20) | nFail = 0 — rho, ω₁, volume BITWISE (231 s) |
| `test_preset_equivalence` (historical preset, 160×20 vs `campaign_9mesh_r2` record) | nFail = 0 (126 s) |
| `confbench_selftest` | 13/14 PASS; **T2 FAIL — pre-existing**: identical failure on a `git archive` of HEAD `0591993` (T2 calls `olhoffcurrent_paths()` without an output argument, which the gate refuses before it checks contamination). Not repaired. |
| `tests/test_development_firewall.py` | PASS |

## D. Dependency firewall

`python3 tests/test_development_firewall.py` → `FIREWALL PASS` (116 allowlisted hits in 30 reviewed
entries; 0 unexplained). Checks F1 `development` (case-insensitive), F2 every historical tree name and the
path forms of the renamed trees, F3 `genpath(`, F4 historical directory names, F5 entry-point uniqueness,
F6 run-time gate forbids `development/`.

**Mutation test:** a planted file with `fullfile(r,'Development','x')`, `fullfile(r,'analysis','OlhoffApproach',…)`,
`genpath(r)`, `fullfile(r,'analysis','ourApproach')` and a directory `analysis/OlhoffM4Reconstruction_probe` →
5 violations reported, commented-out reference ignored; probes removed → PASS.

Allowlist categories (`tests/firewall_allowlist.tsv`): the run-time blacklist itself; the retired-key
`case`; negative isolation tests that put archived trees on the path to prove refusal; the word
"development" meaning the external upstream repository; recorded provenance strings never opened
(PROVENANCE.json, SOURCE_MANIFEST.json, preset citations, frozen profile, compose manifest keys, one fixture
field); the pre-existing dangling optional `mma.py` import in Python Proposed/elastic2D.

Run-time complement: `olhoffcurrent_forbidden_paths` now lists `development` first, so the scrub and the
gate refuse any MATLAB path entry below it.

## E. Historical preservation — files deleted = 0

| comparison | result |
|---|---|
| pre-move inventory (6097 files, SHA-1) → mapped new path, immediately after the moves | **0 missing, 0 hash mismatches**; 8/8 empty dirs present |
| same mapping at the end of the task (size check) | 0 missing; 34 size changes = the reference edits listed in §G, this task's own bookkeeping, 2 regenerated ignored `.pyc` |
| git | 3051 renames (3038 R100 + 13 renamed-and-edited), 11 in-place edits, **0 deletions**; `git ls-files` = 3159, same as baseline |
| baseline (12:33) vs pre-move (14:30) | +19 files, all written by the live bimodality study or this task — none removed |

Moves: 184 operations (118 `git mv`, 66 plain renames), 4899 of 6097 files relocated. Log: `MIGRATION_LOG.tsv`.

## F. Current source uniqueness

Olhoff-family bare-name `.m` definitions in active roots exist only in `analysis/Olhoff/`; the
`+impl/` copies are package-hidden; `tools/Matlab/{mmasub,subsolv}.m` are the declared non-winning
collisions required by Proposed/Yuksel. F5: exactly one definition of `olhoffcurrent_run`,
`topopt_freq` (.m/.py), `top99neo_inertial_freq`, `run_topopt_from_json` (.m/.py),
`study_base_config`, `study_evaluate_design`, `performance_comparison`. One tolerated duplicate:
`BeamTopOpt.m` (per-example scripts in ClampedBeam/HingedBeam).

## G. Current files edited (references only; no numerical content)

`README.md`, `.gitignore`, `analysis/Olhoff/{README.md, olhoffcurrent_{assert_dispatch,caveat,config,currentness,forbidden_paths,provenance,run,source_manifest}.m, tests/test_path_isolation.m, tests/test_source_integrity.m}`,
`analysis/Yuksel/Python/run_{cantilever,fixed_pinned,simply_supported}.py`,
`tools/Matlab/run_topopt_from_json.m` (legacy keys retired; snapshot in `development/legacy_implementations/dispatcher_snapshot/`),
`tools/Python/run_topopt_from_json.py`,
`examples/Performance/{performance_comparison.m, compose_nine_mesh_comparison.m (untracked; addpath + comment), conference_bench/confbench_{caveats,frozen_budget,manifest,method_config,preflight,selftest}.m}`.

`+impl/` untouched (currentness `CURRENT`, tree hash unchanged).
