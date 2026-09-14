# HISTORICAL_S160 — Step 5A

```
POSTMERGE_HISTORICAL_S160_PASS
```

**Run.** Exactly one 160×20 solve of `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` (canonical name, not the alias), in the merged normal checkout:
- HEAD `b21483b`, `+impl` clean (tree `4ba9a3ae…`);
- configuration `olhoffcurrent_config(160, 20, 'Preset', name)`;
- dispatch through the fail-closed guard, single thread;
- wall time 124 s (not a science field).

**Evidence.** Full result, cfg and meta in the git-ignored `analysis/OlhoffCurrent/evidence/postmerge_campaign_gate/ANCHOR_HISTORICAL_S160.mat`. Comparison in `evidence/ANCHOR_COMPARISON.json → historical` (`scripts/pmg_compare.m`).

**Standard.** The migration's established `mig_compare` standard (comparator copied verbatim): bitwise class, size and raw bytes, recursive. Excluded: `tEig tGrad tInner tOuter wallclock provenance.resolvedAt`.

## Result

| quantity | value |
|---|---|
| status | CONVERGED |
| outer / inner | **91 / 2241** |
| ω₁ | **169.495227021538** |
| ω₂ | 171.959995918159 (gap 1.454 %) |
| volume | 0.499999008778 |
| M_nd | 13.4025 % |
| gray fraction (0.1 < ρ < 0.9) | 0.149375 |
| ρ SHA-256 (raw bytes) | `22aa4447bc5f2855…` |

## Comparisons

| reference | verdict | detail |
|---|---|---|
| migration post-run `POST_BETA.mat` (same protocol at `9b30ec4`) | **bitwise, strict** | 0 differing leaves, 0 only-in-candidate or only-in-reference, 0 differing schema rows, 0 differing cfg struct leaves. Covers the full ρ, ω, λ, the whole `hist` (move history, stage and controller state, N, nInner, dx…), `aux`, log and status. |
| upstream-snapshot run `UP_BETA.mat` (253069 archive) | **bitwise, strict** | 0 / 0 / 0 / 0 |
| pre-migration run `PRE_BETA.mat` (tree `edbfe47e`) | **pass (preregistered allowances)** | 0 differing values. Differences allowed: `res.aux` added; 12 `hist.ex*` trace fields dropped, all empty; cfg differs only in the 6 rows added to the schema. |
| frozen conference record (`campaign_9mesh_r2/benchmark_records.mat`, olhoff 160×20) | **pass** | ρ bitwise, ω₁ bitwise, volume bitwise, 91 = 91, 2241 = 2241, CONVERGED / NATIVE_CONVERGED |

## Configuration hashes (schema changed 81 → 87 rows)

| hash | value | note |
|---|---|---|
| new 87-row hash | `c0fe56ce897ce32ec8efc90ca1fb6bdd9c8326ee56700f5ce33adfb26bded17d` | equals `PROVENANCE.json` event 2 `old_preset_config_hashes_under_new_schema["160x20"]` |
| old 81-row hash, reconstructed from the migrated cfg over the frozen row list `tests/fixtures/schema_rows_pre_253069.json` | **`28756d22aacb59726be9f37583deca89fcfcecc93f9b867d46a49223ed1db697`** | = the campaign record's `effective_config_hash`; = PROVENANCE.json event 1 |

All 81 old scientific values are identical to the pre-migration configuration.
