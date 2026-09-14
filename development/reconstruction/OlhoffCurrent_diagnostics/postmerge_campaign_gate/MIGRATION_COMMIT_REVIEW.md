# MIGRATION_COMMIT_REVIEW — `9b30ec4` + `b21483b` as the candidate

```
MIGRATION_COMMIT_REVIEW_PASS
```

Read-only git review, done before the merge.

| criterion | result |
|---|---|
| branch tip | `migration/olhoffcurrent-upstream-253069` = `b21483b158f58e05e7b56957f2fbe8e1d2891395` |
| parentage | `b21483b` → `9b30ec45…` → `013cc48…` (= target HEAD = merge-base) |
| commits | exactly 2 on top of the target; target not ahead (0) |
| hardening commit scope | 30 paths, all in the preregistered allowed list; **0** under `+impl`, `SOURCE_MANIFEST.json`, `olhoffcurrent_preset*.m`, `olhoffcurrent_config*.m`, `olhoffcurrent_run.m`, `examples/**`, `tools/**` |
| unchanged by the hardening commit | `olhoffcurrent_presets.m`, `olhoffcurrent_preset.m`, `olhoffcurrent_production_preset.m`, `olhoffcurrent_config.m`, `olhoffcurrent_config_hash.m`, `olhoffcurrent_caveat.m`, `olhoffcurrent_run.m`, `SOURCE_MANIFEST.json` |
| `+impl` at tip = upstream 253069 | 79/79 blob ids equal; 0 differ |
| whole branch scope | every path under `analysis/OlhoffCurrent/**` or `examples/Performance/**`; 0 paths touching Proposed, Yuksel, evaluator, tools, profiles, C480 or SOCP |
| Phase 6 | not entered (same harness diff reviewed in attempt 1; the hardening commit adds no harness change) |
| p-continuation defect D1 | still present: `olhoffSolve.m` byte-identical to upstream |
| historical and new presets | unchanged (above) |
| untracked collisions | 0 of 102 added paths exist on disk in the normal checkout |
| attempt 1's single failure | resolved: the gate claims are now substantiated by committed-code probes (FINALIZATION_GATE_REVIEW.md) |

**Superseded documentation.** `diagnostics/upstream_253069_migration/{REPORT,TEST_REPORT}.md` still describe the original A2 rule as "strict". Those files are that study's pinned historical record, so they were not edited. The current rule is documented in `EVIDENCE_POLICY.md` item 7 and `diagnostics/provenance_gate_hardening/`.
