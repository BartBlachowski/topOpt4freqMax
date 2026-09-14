# POSTMERGE_IDENTITY — normal checkout at `b21483b`

```
POSTMERGE_IMPLEMENTATION_IDENTITY_PASS
```

| check | result |
|---|---|
| `+impl` file count | **79** |
| every `+impl` blob at HEAD = upstream `253069:<path>` | 79/79 equal, 0 differ |
| working tree = HEAD, by raw bytes (`git hash-object --no-filters` per file) | 79/79 equal |
| untracked non-artifact files or symlinks in `+impl` | none |
| private solver/controller copies outside `+impl` | none |
| stage exhaustion and `hist.tOuter` | present in the upstream-identical `olhoffSolve.m` |
| implementation tree hash (recomputed from working-tree bytes) | `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` |
| `SOURCE_MANIFEST.json` | `tree_sha256` equals the recomputed value; `n_files` 79; rows equal; file SHA-256 `aee44aa9172a8105314bbb9475e28bed8a53d4d154824a8748abf64a208b1836` |
| `PROVENANCE.md` source-tree row | the same tree (one row) |
| `PROVENANCE.json` | `source.commit` = `253069…`; production event = `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` |
| provenance self-contained | every committed acceptance-evidence reference resolves: `Olhoff@253069:repro/audits/upstream_olhoffcurrent_capabilities` EXISTS, `Olhoff@6b08708:repro/results/S160x20` EXISTS, `diagnostics/upstream_253069_migration` IN HEAD, `diagnostics/provenance_gate_hardening` IN HEAD; three local folders explicitly supplementary |
| historical/current distinction in this checkout | `two_branch_controller_validation`: 7 × `HISTORICAL_VERIFIED` at `bba45e7`/`1438aa3` and `CURRENT_SOURCE_HASH_VERIFIED` (H2 in `logs/POST_gates.log.txt`) |
| dirty production-source detection | the 39 J probes of `test_finalization_gate` ran against this HEAD: 39/39 (`logs/POST_gates.log.txt`) |
| G6 over every study in this checkout | `CURRENT_SOURCE_HASH_VERIFIED` for all 30 studies (`evidence/POST_studies.json`) |
| currentness | `test_currentness` 0 failures |
