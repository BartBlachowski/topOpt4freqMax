# Production manifest, schema and output isolation audit

## 1. Nine-mesh manifest

`PRODUCTION_MANIFEST.json → production_meshes`:

```
[160,20] [240,30] [320,40] [400,50] [480,60] [560,70] [640,80] [720,90] [800,100]
```

Exactly nine entries, no omission, no duplicate, no hidden development mesh, no stale four-mesh configuration. Identical to `iteration_efficiency_contract.json → production_meshes`. `preflight.m` compares against a hard-coded literal, so a tampered manifest is rejected (verified: substituting a four-mesh list fails closed).

The smoke mesh `[16 2]` lives only in `config.m`'s `smoke` branch and can never reach production, because `runMode='production'` selects `cfg.meshes = manifest.production_meshes`.

All nine meshes are 8:1 with even `nely`, satisfying `topology_metrics`' mid-height-support assertion and the contract's `even_nely_required`.

## 2. Method / profile bindings, verified independently

| id | role | source (hash-verified) | key parameters |
|---|---|---|---|
| Proposed | principal | `analysis/ourApproach/Matlab/topopt_freq.m` | move 0.2, tol 0.01, rmin 2.0 el, B0 900 |
| Yuksel | principal | `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m` | move 0.1, tol 0.01/0.01, rmin 2.5 el, stage1 cap 2000, B0 2000 |
| Olhoff-LP | principal | `analysis/olhoff_stabilization_audit/olhoffOptStabilized.m` | S1, move [0.005, 0.0025], rmin 1.3 el, B0 3200 |
| Olhoff-MMA | secondary | `Matlab/reproduction2007/algo/olhoffOpt.m` | move 0.01, rminPhys 0.06, offDiag, maxInner 300, B0 3200 |

Each binding was checked against the executing code, not just the manifest text (see `METHOD_ROUTING_AUDIT.md`). All parameters match the frozen contract's profile records.

## 3. 800×100 handled honestly

The manifest carries 800×100 as an ordinary production mesh with no special-casing, no inherited value and no lower bound imported from the frozen campaign's `RUN_ERROR`. `config.m`, `run.m` and `run_trajectory.m` contain no mesh-conditional logic whatsoever. Nothing can fabricate or infer an 800×100 result.

**However** — the campaign cannot currently *record* a genuine 800×100 failure either. See finding F-03: a solver failure raises an exception that aborts the whole campaign rather than producing a censored row. Honest attempt: yes. Honest recording of the outcome: not yet.

## 4. Result schema

`RESULT_SCHEMA.json` (`iteration_efficiency_result_v1`, `additionalProperties: false`) declares 50 required fields. `build_rows.localEmpty` constructs every row from one template containing all 50, so route-specific fields are structurally present on every row and default to `NaN` / `''` — they are never silently populated with misleading zeros.

Validated live against representative rows:

| control | result |
|---|---|
| Proposed / Yuksel / Olhoff-LP / Olhoff-MMA rows | valid |
| `trajectory_dtype = 'single'` | **rejected** |
| `evaluator_id` stale | **rejected** |
| LP row carrying `olhoff_mma_total_inner_iterations` | **rejected** |
| MMA row carrying non-zero `olhoff_lp_calls` | **rejected** |
| reference-failure rows (9 per cell, `k_enter = NaN`, status propagated) | valid, N/A preserved |

N/A semantics are `NaN` for numerics and `""` for strings, encoded as `null` in JSON via the schema's `["number","null"]` unions.

Residual (MINOR, F-08): `provenance_hash` is emitted as `''` on every row. The schema permits it, and run identity is recorded in `provenance/run_manifest.json`, but per-row provenance is weaker than §22 intends.

## 5. Output isolation

`new_run_directory` builds `runs/<smoke|production>/<lp|mma|both>/<yyyyMMdd'T'HHmmssSSS>/` and asserts `~isfolder(out)` before `mkdir`, so:

- smoke, qualification and production **cannot** overwrite one another (different `runMode` segment);
- `lp`, `mma` and `both` **cannot** collide (different selector segment);
- reruns are uniquely identified by a millisecond timestamp;
- existing evidence is never overwritten (the assertion refuses rather than clobbers).

Within a run, each cell writes to `reference/<id>` and `measurement/<id>` where `id = <label>_<nelx>x<nely>` — so under `both`, `olhoff_lp_400x50` and `olhoff_mma_400x50` are distinct trees.

Provenance recorded per run: `provenance/run_manifest.json` carries run mode, selector, output directory, start timestamp, the production flag and the full preflight report (every source and component hash). Rows additionally carry method, variant, mesh, `contract_hash`, `evaluator_id` and `source_hashes`. Method, variant, mesh, contract, evaluator, source hashes, run configuration and run identity are all determinable.

The `ie2a` observer enforces its own isolation: `install_observer` rejects any output path outside the allowed Phase-2A / final-harness roots. Verified live — an attempt to write elsewhere failed closed with `ie2a:OutputIsolation`.
