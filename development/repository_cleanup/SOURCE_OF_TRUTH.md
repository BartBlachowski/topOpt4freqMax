# SOURCE_OF_TRUTH — what is current, and why

Established from repository evidence at HEAD `0591993` (branch
`benchmark-methodology-r2`) **before** anything was moved. Names and
modification times were not used as evidence. Paths below are the
**pre-migration** paths; the post-migration location is given in the last
column of each table.

**Evidence conflict check: none found.** Every source listed below agrees on
the same three implementations. Two documents are *stale* (not conflicting) and
are superseded by later, explicit records; they are listed in §5.

---

## 1. Du–Olhoff — `analysis/OlhoffCurrent` → `analysis/Olhoff`

| item | value |
|---|---|
| public entry point | `olhoffcurrent_run(nelx, nely, 'Preset', name)` in `analysis/OlhoffCurrent/olhoffcurrent_run.m` |
| path gate (installed by the entry point) | `olhoffcurrent_paths` → `olhoffcurrent_assert_dispatch`, `olhoffcurrent_forbidden_paths`, `olhoffcurrent_known_collisions`, `olhoffcurrent_owned_names`, `olhoffcurrent_impl_dirs`, `olhoffcurrent_root`, `olhoffcurrent_scrub_forbidden_paths` |
| configuration | `olhoffcurrent_config`, `olhoffcurrent_preset`, `olhoffcurrent_presets`, `olhoffcurrent_production_preset`, `olhoffcurrent_caveat`, `olhoffcurrent_config_hash`, `olhoffcurrent_legacy_view` |
| provenance / integrity | `olhoffcurrent_currentness`, `olhoffcurrent_provenance`, `olhoffcurrent_source_manifest`, `olhoffcurrent_sha256_file`, `olhoffcurrent_is_artifact`, `PROVENANCE.json`, `PROVENANCE.md`, `SOURCE_MANIFEST.json` |
| solver core | `+impl/` — 79 files (`algo/ fem/ filter/ mma/ mma_published/ architecture/`), byte copy of upstream Olhoff `253069262407885a8b759a9e721c4f0a7d3a397d`; canonical solver `+impl/architecture/olhoffSolve.m`; the sibling layout is load-bearing (`algo/useMMA.m` locates `mma_published/` relative to itself) |
| production preset | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` (latest `PROVENANCE.json → production_preset_events`, 2026-09-13) |
| current tests | `tests/test_path_isolation.m`, `test_currentness.m`, `test_source_integrity.m`, `test_preset_identity.m`, `test_preset_equivalence.m`, `test_named_preset_reproduction.m`, `test_cost_reporting.m`, `test_pedersen_adaptive_units.m`, `olhoffcurrent_test_digest.m`, `fixtures/` |
| **post-migration** | `analysis/Olhoff/` (function names keep the `olhoffcurrent_` prefix — see §6) |

### Evidence that it is current

1. **`analysis/OLHOFF_CURRENT_PROMOTION_REPORT.md`** (2026-09-07, commit
   `cf1b71d` "Establish OlhoffCurrent as sole production implementation"):
   *"`analysis/OlhoffCurrent` — the sole production implementation"*, with
   bitwise equivalence against the frozen M4 reconstruction at 160×20 and
   320×40, and a repointing table of every production script.
2. **`analysis/OLHOFF_IMPLEMENTATION_MAP.md`**: fourteen trees classified,
   *"There is exactly one `PRODUCTION` entry"* — `analysis/OlhoffCurrent`.
   Every other Olhoff tree is `FROZEN_EVIDENCE`, `HISTORICAL`, `AUDIT_ONLY`,
   `EXPERIMENTAL` or `DEVELOPMENT_UPSTREAM`.
3. **Later history re-promotes the same tree, it does not replace it.**
   `9b30ec4` (2026-09-13) "Promote shared Olhoff implementation with named
   formulation presets" re-copies `analysis/OlhoffCurrent/+impl` from upstream
   `253069…`; `b21483b` hardens its provenance gate; `0591993` adds
   diagnostics under it. No commit after `cf1b71d` names another production
   Olhoff tree.
4. **The current runner executes it and nothing else.**
   `examples/Performance/performance_comparison.m` adds
   `analysis/OlhoffCurrent`, scrubs the forbidden trees, and dispatches
   through `confbench_run_case` → `olhoffcurrent_run`.
   `confbench_olhoff_preset()` returns
   `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, identical to the
   recorded production preset.
5. **The newest campaign ran it.**
   `examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b`
   (generated 2026-09-14T05:34) is the Olhoff column of the current
   three-method table `nine_mesh_comparison_pedersen_b21483b`.
6. **Fail-closed gate.** `olhoffcurrent_forbidden_paths` blocks every other
   Olhoff tree; `test_path_isolation` (TEST A–F) proves the refusal.

### Required supporting files outside the tree

* `tools/Matlab/mmasub.m`, `tools/Matlab/subsolv.m` — **declared collisions**
  (`olhoffcurrent_known_collisions`), *not* dependencies: they may exist but
  may never win resolution.
* The external upstream `/Users/piotrek/Programming/Matlab/Olhoff` is read
  (never executed) by `olhoffcurrent_currentness` for information only.

### Not current, although inside the tree

`analysis/OlhoffCurrent/diagnostics/` (30 studies, 1.5 GB) and
`analysis/OlhoffCurrent/evidence/` (1.3 GB, git-ignored raw evidence) are the
record of the experiments, audits, migrations and campaign audits performed
*on* the implementation. The study-governance tooling that exists only to
finalize those studies — `olhoffcurrent_finalization_gate.m`,
`olhoffcurrent_evidence_gate.m`, `olhoffcurrent_evidence_declare.m`,
`EVIDENCE_POLICY.md`, `tests/test_finalization_gate.m`,
`tests/gate_provenance_probes.m`, `tests/test_evidence_retention.m` — is used by
no production script, no benchmark file and no other current test (checked by
caller search; its only callers are each other and scripts under
`diagnostics/`). It is hard-bound to the `diagnostics/` / `evidence/` layout and
to git objects at `analysis/OlhoffCurrent/...` in historical commits. It is
archived **together with** the studies it governs.

---

## 2. Yuksel — `analysis/YukselApproach` → `analysis/Yuksel`

| item | value |
|---|---|
| MATLAB entry point | `top99neo_inertial_freq.m` (two-stage inertial frequency), reached through `tools/Matlab/run_topopt_from_json.m`, approach key `Yuksel` (`addpath(analysis/YukselApproach/Matlab)`) |
| other files | `top99neo_dynamic_freq.m`, `run_cantilever.m`, `run_fixed_pinned.m`, `run_simply_supported.m`, `plot_*_convergence.m`; Python port `Python/solver.py` + runners |
| frozen benchmark profile | `yuksel_practical_move01_tol001` in `analysis/three_method_parametric_study/results/profile_freeze_manifest.json` (`source_implementation: analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`) |
| **post-migration** | `analysis/Yuksel/` |

### Evidence

1. The only Yuksel implementation in the repository (no second tree exists).
2. `confbench_run_case` → `runYuksel` → `run_topopt_from_json` key `yuksel` →
   `analysis/YukselApproach/Matlab`.
3. `confbench_manifest.m` hashes
   `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m` into every
   campaign manifest; `compose_nine_mesh_comparison.m` requires that hash to be
   unchanged between `campaign_9mesh_r2` and `nine_mesh_pedersen_b21483b` for
   the Yuksel rows of the current table.
4. The frozen profile names this file as its source implementation.
5. Last solver change `e361b06` (2026-07-31); the frozen profile
   (2026-08-27) and both recorded campaigns postdate it.

---

## 3. Proposed — `analysis/ourApproach` → `analysis/Proposed`

| item | value |
|---|---|
| MATLAB entry point | `topopt_freq.m`, reached through `run_topopt_from_json` key `ourApproach` |
| Python entry point | `Python/topopt_freq.py`, reached through `tools/Python/run_topopt_from_json.py` |
| frozen benchmark profile | `proposed_practical_move02_tol001` (`source_implementation: analysis/ourApproach/Matlab/topopt_freq.m`) |
| **post-migration** | `analysis/Proposed/` (the JSON approach key stays `ourApproach`) |

### Evidence

1. The only Proposed implementation in the repository.
2. `confbench_run_case` → `runProposed` → `run_topopt_from_json` key
   `ourApproach`; all current user examples
   (`examples/{Building,ClampedBeam,ClampedHingedBeam,HingedBeam}/*.json`) use
   `"approach": "ourApproach"`.
3. Hashed by `confbench_manifest.m`; unchanged-source check in
   `compose_nine_mesh_comparison.m`; named by the frozen profile.
4. Last change `b3524bf` (2026-09-04), before `campaign_9mesh_r2`.

---

## 4. Shared current code and runners

| item | status | evidence |
|---|---|---|
| `tools/Matlab/` (dispatcher `run_topopt_from_json.m`, MMA copy, plotting, history, BC/passive/load-case helpers) | **current, shared** | on the path of the performance runner (`addpath(tools/Matlab)`); `confbench_preflight` asserts `which('run_topopt_from_json')` is this copy; used by Proposed, Yuksel and elastic2D dispatch |
| `tools/Python/` | **current, shared** | Python dispatcher for Proposed and elastic2D; used by every `examples/*/run_*.py` |
| `tools/compat/` | current | added by `run_topopt_from_json` (`compatDir`) |
| `analysis/elastic2D` | **current (auxiliary)** | compliance-minimization solver dispatched by `run_topopt_from_json` keys `elastic2D`/`elastc2D`; current example `examples/elastic2D` |
| `analysis/three_method_parametric_study/{study_base_config.m, study_evaluate_design.m, results/profile_freeze_manifest.json}` | **current benchmark configuration** inside a historical study | `performance_comparison.m` adds the directory; `confbench_method_config`, `confbench_frozen_budget`, `confbench_preflight` (checks 7–8) and `confbench_manifest` read exactly these three files. The rest of the study (stage-A/holdout runners, plots, selection scripts) is used by no current file. → the three files move to `examples/Performance/benchmark_profile/`; the rest is archived. |
| **current performance runner** | `examples/Performance/performance_comparison.m` + `examples/Performance/conference_bench/` | promotion report §8; `confbench_*` is its implementation; produced `nine_mesh_pedersen_b21483b` |
| **current table composition** | `examples/Performance/compose_nine_mesh_comparison.m` (untracked, 2026-09-14) | composes the current three-method table from `campaign_9mesh_r2` (Proposed, Yuksel) and `nine_mesh_pedersen_b21483b` (Olhoff); output `nine_mesh_comparison_pedersen_b21483b` |
| **current recorded results** | `conference_benchmark/{campaign_9mesh_r2, nine_mesh_pedersen_b21483b, nine_mesh_comparison_pedersen_b21483b}` | inputs/outputs of the composition above; `campaign_9mesh_r2` is also read by `test_preset_equivalence.m` |
| current user examples | `examples/{Building, ClampedBeam, ClampedHingedBeam, HingedBeam, elastic2D}` | JSON configs dispatch to `ourApproach` / `elastc2D`; runners `*.m` / `run_*.py` |

---

## 5. Stale (not conflicting) documents

| document | stale statement | superseded by |
|---|---|---|
| `analysis/OLHOFF_IMPLEMENTATION_STATUS.md` | names `analysis/OlhoffM4Reconstruction` "CONFERENCE-ACTIVE" | the promotion report and the implementation map, which say so explicitly ("This map supersedes it") |
| `analysis/OLHOFF_IMPLEMENTATION_MAP.md` (debt §1) | four historical Olhoff trees "untracked in git" | they are tracked at `0591993` (`git ls-files`: 13 / 53 / 35 / 5 files) — a fact change, not a production claim |
| `Matlab/README.md` | "three independent realizations … for the paper-revision phase" | predates the promotion; describes historical generations only |
| root `README.md` | `examples/case1.json`, `examples/BeamTopOptFreq.json` | those files do not exist; replaced by the new navigation README |
| `performance_comparison.m` default `cfg.runLabel = 'campaign_9mesh_r2'` | a re-run would target the existing campaign directory | not a source-of-truth question; the preflight refuses silent overwrite. Reported, not changed (outside the scope of a path migration) |

---

## 6. Deliberate non-changes

* **Function names keep the `olhoffcurrent_` prefix.** They are the API recorded
  in every campaign manifest, error identifier
  (`olhoffcurrent_assert_dispatch:PathContaminated`), preset registry and
  test. Renaming them is an API change, not a path migration. Only the
  *directory* takes the canonical name.
* `PROVENANCE.json`, `PROVENANCE.md`, `SOURCE_MANIFEST.json` are moved
  byte-for-byte. The string `analysis/OlhoffCurrent` inside them is the
  location at promotion time. `olhoffcurrent_source_manifest` verifies files by
  path *relative to* `+impl/` and never compares the recorded `root`, so the
  rename does not change `olhoffcurrent_currentness()`.
* The JSON approach keys (`ourApproach`, `Yuksel`, `elastic2D`) are user-facing
  configuration vocabulary and are unchanged.
