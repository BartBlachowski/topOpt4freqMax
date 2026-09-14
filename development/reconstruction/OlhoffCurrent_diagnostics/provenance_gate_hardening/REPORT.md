BOTTOM LINE

Migration commit `9b30ec4` added a provenance exception to `olhoffcurrent_finalization_gate` (Amendment A2) that was not fail-closed. It has been replaced by a rule that keeps two separate facts apart:
- **`HISTORICAL_SOURCE_HASH_VERIFIED`**: per line, against the study's single freeze commit.
- **`CURRENT_SOURCE_HASH_VERIFIED`**: `+impl` against committed HEAD objects. The manifest is only a consistency check. Git reads cannot be redirected or replaced.

What the evidence shows:
- **Adversarial probes:** 39/39 behave as preregistered. On the same probes the original `9b30ec4` rule is fail-open on 24, and my own first repair attempt (`adf86a3`, never merged) on 6. That attempt was found wanting by self-review before any merge, and is superseded by this commit.
- **Legitimate history:** the seven superseded lines of `two_branch_controller_validation` still verify: same paths, freeze commit `bba45e7`, last changed in `1438aa3`.
- **Worktree suites:** only the two pre-existing `test_finalization_gate` failures remain.
- **Scope:** no `+impl` file, preset, configuration or benchmark method was touched.
- **Provenance:** committed provenance no longer depends on untracked local studies.

```
PROVENANCE_GATE_HARDENING_PASS
COMMITTED_PROVENANCE_SELF_CONTAINED_PASS
GATE_REPAIR_SCOPE_PASS
```

Preregistration, frozen before any gate code changed and unchanged since:

| document | SHA-256 |
|---|---|
| PREREGISTRATION.md | `de82b1a02e6f1907d11a438b52512ae849ffc97210164cdb3a7376bdee939833` |
| PREREGISTRATION_ADDENDUM_1.md | `6f52d940480f2c3d32a5c56a7815bc8c9dc7ceb134918d8a998108fa1e0a64e4` |

The addendum was frozen after self-review of `adf86a3` and before the changes it describes.

## 1. Defects and repair

| defect | mechanism | repair |
|---|---|---|
| **P5** fabricated line rescued by a later genuine line (`9b30ec4`) | digests in a path-keyed `containers.Map`; the last line wins | every digested line gets its own verdict, compared against **its own** digest; duplicates reported, each validated |
| **P9** empty digest passes where `git show` fails (`9b30ec4`) | `git show c:p \| shasum`, whose exit status is `shasum`'s | `git cat-file -e`, then `-t` = `blob`, then `cat-file blob > tmp`, each status checked; no pipeline. A genuinely empty committed blob still validates (P19). |
| **P10** edited `+impl` plus regenerated manifest passes (`9b30ec4`) | "current" = `+impl` versus the mutable manifest | gate **G6** for every study: HEAD `+impl` blobs equal the working tree by raw SHA-256 (the index is never read); no extra source and no symlinks; manifest equals its HEAD blob, rows and tree; PROVENANCE.md tree row equals HEAD's tree |
| **W1** study-local shadow copy satisfies a source line (`9b30ec4`) | study-local-first resolution | source lines resolve only at `<repo>/<path>` and must be canonical (P21) |
| **W2** any ancestor commit accepted (`9b30ec4`) | `git log HEAD -- path`, first match | exactly one admissible commit per study (conditions below; P12, P22–P24) |
| **A1** `OlhoffCurrent//+impl`, `/./+impl` plus shadow (`adf86a3`) | substring detection missed normalization variants | a line **names** production source by normalized text or logical path (case, `\`, `//`, `./`, `..`); anything not exactly canonical is malformed (P30–P32, P34) |
| **A2** `+impl//architecture` treated as canonical (`adf86a3`) | `strsplit` collapsed empty segments | split without collapsing (P34) |
| **A3** `git replace` of HEAD blobs makes an edited `+impl` pass G6 (`adf86a3`) | git honors replace refs in `ls-tree` and `cat-file` | `--no-replace-objects` on every git call (P35, P37) |
| **A4** `GIT_DIR` redirects `git -C` to another repository (`adf86a3`) | the environment overrides discovery | every git call runs with `env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_OBJECT_DIRECTORY -u GIT_ALTERNATE_OBJECT_DIRECTORIES -u GIT_COMMON_DIR -u GIT_NAMESPACE -u GIT_REPLACE_REF_BASE -u GIT_CONFIG_PARAMETERS -u GIT_CONFIG_COUNT` (P36) |
| links and relative paths into production (`adf86a3`) | non-source lines were resolved anywhere | a non-source line whose file resolves (links followed; case-insensitive on macOS) into `+impl` or `SOURCE_MANIFEST.json` is malformed (P33, P38) |

The single admissible historical commit, per study (PREREGISTRATION §3 S3). All of these must hold:

| # | condition |
|---|---|
| 1 | `FINAL_SHA256.txt` and `EVIDENCE.json` are committed and unmodified |
| 2 | freeze commit = `git log -1 HEAD -- <study>/FINAL_SHA256.txt`, and the hash file there equals the working tree |
| 3 | the declared `sourceTree`/`impl_tree_sha256` equals both the `+impl` tree computed from the committed blobs at that commit and the manifest's tree there |

## 2. Adversarial probes

Sources: PROBE_TABLE.md, `evidence/probes.json`, `evidence/discrimination.json`.

| gate version | probes behaving as the repaired rule requires | fail-open probes |
|---|---|---|
| original rule, `9b30ec4` | 15/39 | **24** (P5, P8, P9, P10, P12, P14, P15, P21–P33, P35–P38) |
| first repair, `adf86a3` (superseded) | 33/39 | **6** (P31, P32, P33, P36, P37, P38) |
| **this repair** | **39/39** | **0** |

- **Positive controls, all PASS:** P0; P1 (genuine superseded line); P11 (identical duplicates, reported); P19 (genuinely empty committed file).
- **Negative controls, all FAIL:** 35 probes.
- **Status separation:** P8, P10, P36 and P37 report historical `VERIFIED` but current `NOT_VERIFIED`, and the study fails.

## 3. Real studies and worktree suites

Sources: `evidence/real.json`, `evidence/tests.json`.

**R1 — `two_branch_controller_validation`: PASS.**
- 7 lines are `HISTORICAL_VERIFIED`: `SOURCE_MANIFEST.json`, `olhoffSolve.m`, `+move/limit.m` and `+config/{schema,validate,toLegacy,fromLegacy}.m`.
- Freeze commit is `bba45e72ea18…`; every line's source commit is `1438aa3f4bd9…`.
- The declared tree `edbfe47e…` equals the tree at the freeze commit.
- `+move/exhaustion.m` is `CURRENT_MATCH`.
- Current source is `CURRENT_SOURCE_HASH_VERIFIED` against HEAD tree `4ba9a3ae…`. The two statuses are reported separately, and the old digests are never presented as the new source.

**R2 — the gate over every study.** The failing set is exactly:
- the 8 legacy ledger studies: `admission_rule, dynamical_regime, fixedmove_400_dynamics, move_activity_offline, move_stop, move_transition, topology_maturity_transition, two_branch_maturity_240`;
- the pre-existing seven: `controller_architecture_offline, move_activity_400, move_ladder_necessity, three_rung_architecture, three_rung_promotion_closure, three_rung_promotion_validation_retry1, two_rung_architecture`.

That is the same set as at `9b30ec4` and on pristine `013cc48` (migration TEST_REPORT). `move_ladder_necessity` also pins `olhoffcurrent_finalization_gate.m` and `test_finalization_gate.m` by their `013cc48` digests; that non-source line already mismatched at `9b30ec4`.

**Worktree suites:**

| suite | failures |
|---|---|
| `test_path_isolation` | 0 |
| `test_currentness` | 0 |
| `test_source_integrity` | 0 |
| `test_evidence_retention` | 0 |
| `test_preset_identity` | 0 |
| `test_finalization_gate` | 2: H `move_activity_400` and I (the seven above), both VERIFIED PRE-EXISTING |

Within `test_finalization_gate`, everything else passes: A–G and D2, all 39 J probe lines plus the 2 J harness lines, H `beta_transition_mechanism` and `two_branch_controller_validation`, and both H2 lines.

**Existing hash files:** all 8 source-naming paths in every hash file (tracked and untracked) are canonical, so the stricter naming rule changes no existing verdict.

## 4. Acceptance (PREREGISTRATION §5; ADDENDUM_1)

| item | verdict |
|---|---|
| P5_FAILS_AS_EXPECTED | PASS |
| P9_FAILS_AS_EXPECTED | PASS |
| P10_FAILS_AS_EXPECTED | PASS |
| ALL_NEGATIVE_PROBES_FAIL_AS_EXPECTED (35) | PASS |
| ALL_LEGITIMATE_HISTORICAL_LINES_PASS (P1, P11, P19, R1's seven) | PASS |
| CURRENT_SOURCE_TIED_TO_HEAD_PASS (P10, P28, P29, P36, P37 fail; R1 current-verified) | PASS |
| DIRTY_IMPL_REJECTED_PASS (P8, P10, P26, P27, P28, P36, P37) | PASS |
| MANIFEST_TAMPER_DOES_NOT_OVERRIDE_HEAD_PASS (P10, P25, P37) | PASS |
| HISTORICAL_CURRENT_STATUS_SEPARATION_PASS | PASS |
| COMMITTED_PROVENANCE_SELF_CONTAINED_PASS | PASS (§5) |
| GATE_REPAIR_SCOPE_PASS | PASS (§6) |
| ALL_RELEVANT_TESTS_PASS_OR_ONLY_VERIFIED_PREEXISTING_FAIL | PASS |

## 5. Committed provenance (Part E, option B)

**Question:** does committed provenance depend on evidence that a clean clone lacks?

**Before:** yes, in three places.
- `PROVENANCE.md` "Acceptance evidence" and `PROVENANCE.json` `source.acceptance_evidence` cited the untracked `diagnostics/scientific_delta_olhoff_migration`.
- `production_preset_events[2].rationale` cited it and `nine_mesh_campaign_audit` with "Per …".

**After:**
- **Acceptance evidence cites only committed material and immutable identities:**
  - `Olhoff@253069…:repro/audits/upstream_olhoffcurrent_capabilities` (verified present at that commit);
  - `Olhoff@6b08708…:repro/results/S*x*` and `SWEEP_R06.md` (verified present);
  - the committed `diagnostics/upstream_253069_migration`, including the production decision in HARNESS_UPDATE.md;
  - this study.
- **Supplementary block.** A new `supplementary_local_evidence_not_required` block, plus a matching PROVENANCE.md row, names `scientific_delta_olhoff_migration`, `nine_mesh_campaign_audit` and `three_rung_canary_preflight` as not committed and not required.
- **Rationale.** It keeps every scientific sentence, but now cites the committed record. No statement, preset event, hash or commit changed.
- **Not edited: `olhoffcurrent_presets.m`.** It is a preset file. Its two `evidence` strings (historical three-rung → canary; Pedersen → scientific delta) already say "untracked at 013cc48" and sit next to committed evidence. They are documentary pointers, not validation inputs.
- **Not edited: `upstream_253069_migration`.** References inside that committed study are its pinned historical record, each disclosed as untracked.
- **Option A was rejected.** Committing into a path the primary checkout holds as untracked would make the merge refuse to overwrite untracked files.

## 6. Scope (Part F)

| class | files |
|---|---|
| gate implementation | `analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m` |
| gate tests | `analysis/OlhoffCurrent/tests/test_finalization_gate.m` (J = probe suite; H2 added; H/I ledger unchanged), `analysis/OlhoffCurrent/tests/gate_provenance_probes.m` (new) |
| provenance docs | `analysis/OlhoffCurrent/EVIDENCE_POLICY.md` (item 7), `analysis/OlhoffCurrent/PROVENANCE.md`, `analysis/OlhoffCurrent/PROVENANCE.json` |
| repair audit artifacts | `analysis/OlhoffCurrent/diagnostics/provenance_gate_hardening/**` |

Unauthorized: 0. There is no diff under `+impl`, `SOURCE_MANIFEST.json`, `olhoffcurrent_preset*.m`, `olhoffcurrent_config*.m`, `olhoffcurrent_run.m`, `examples/**` or `tools/**`.

## 7. Incidents and history (all recorded; no expectation edited)

1. **First repair commit superseded.** `adf86a3f8126e2099436b068ed3bd8a1921a2405` passed the 31 preregistered probes. Self-review then found A1–A4 (the scripts and the `adf86a3` gate are in `selfreview/`). It was never merged or pushed. ADDENDUM_1 was frozen, the gate fixed, P31–P38 added, and the commit **amended** into this single repair commit. `9b30ec4` is untouched. The 31-probe, test and real runs of `adf86a3` were replaced by the runs recorded here; their outcomes (31/31; the same two pre-existing test failures; R1 PASS) are summarised in this list.
2. **First probe run hung.** `git log` opened `less`, because MATLAB's `system()` runs on a TTY. The harness now passes `--no-pager` and sets `GIT_PAGER=cat`.
3. **P28 harness error (30/31).** In a sparse checkout git re-derives skip-worktree bits, so the edit could not be hidden. The harness now uses a full clone. The log is `logs/incident_P28_sparse_checkout_harness_defect.log.txt`.
4. **zsh glob.** An unquoted `--format=%(refname)` in the harness reset was treated as a glob by zsh, the shell MATLAB spawns. It is now quoted.
5. **Empty environment variable.** Restoring an unset `GIT_DIR` with `setenv(name,'')` left an empty value that git rejects. The harness now uses `unsetenv`.
6. **Stale self-declaration.** While a job ran, test I and R2 briefly listed this study as failing: the job overwrote its own listed log file. Logs are now moved into place only when a job ends, the study is re-declared before each block, and the recorded runs show it compliant.
7. **Development-run names.** Early evidence files named `*_dev` were renamed or re-run under final names. `_done.txt` keeps a comment marking the superseded `adf86a3` runs.
8. **Normal-checkout baseline runner.** Its first run lost the MATLAB path after `test_path_isolation` (`postmerge_campaign_gate/logs/BASE_gates.run1_harness_path_defect.*`).

## 8. Next

Restart the merge and promotion gate from Step 1, recorded in the normal checkout's `diagnostics/postmerge_campaign_gate/`.
