# PREREGISTRATION — provenance / finalization-gate hardening

This file was frozen **before** any gate code changed. Its SHA-256 is recorded in `PREREGISTRATION.sha256` (written right after this file) and in the repair REPORT.

Starting states:
- **Migration worktree:** `migration/olhoffcurrent-upstream-253069` @ `9b30ec45b038fb36e7cf20d57679b71cfd099fb3`, clean.
- **Normal checkout:** `benchmark-methodology-r2` @ `013cc48451d33bed61c5c4eea174bbd898d548a2`; 1 commit ahead of `origin`; 9 untracked diagnostics directories, including the first gate attempt `postmerge_campaign_gate`.

## 1. Frozen defect

The defect is in `local_supersededSource` of `analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m` at `9b30ec4` (Amendment A2). It was reproduced in `postmerge_campaign_gate/evidence/gate_probe.log`.

| id | input | observed | mechanism |
|---|---|---|---|
| P5 | fabricated digest, then genuine digest, same path | PASS | digests held in a path-keyed `containers.Map`; the last line wins, and each mismatching line is re-checked against `rec(path)`, not its own digest |
| P9 | empty-file digest; path absent at a commit in `git log -- path` | PASS | `git show c:p 2>/dev/null \| shasum` — the status is `shasum`'s; a failed `git show` hashes empty stdin |
| P10 | `+impl` edited and `SOURCE_MANIFEST.json` regenerated | PASS | "current source" is `+impl` versus the mutable working-tree manifest only |

Two related weaknesses found while designing this repair are in scope, because they belong to the same production-source identity rule:

- **W1.** A production-source line is resolved study-local first, so a copy at `<study>/analysis/OlhoffCurrent/+impl/...` can satisfy a "current match" without touching the real `+impl`.
- **W2.** Candidate commits are *any* commit reachable from HEAD that touched the path, which is broad ancestry search.

## 2. Allowed files (exhaustive)

| file | role |
|---|---|
| `analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m` | gate implementation |
| `analysis/OlhoffCurrent/tests/test_finalization_gate.m` | gate test (J section rewritten; H/I ledger unchanged) |
| `analysis/OlhoffCurrent/tests/gate_provenance_probes.m` (new) | adversarial probe harness shared by the test and the audit |
| `analysis/OlhoffCurrent/EVIDENCE_POLICY.md` | description of the rule |
| `analysis/OlhoffCurrent/PROVENANCE.md`, `analysis/OlhoffCurrent/PROVENANCE.json` | Part E citations only; no scientific statement, preset event, hash or commit changes |
| `analysis/OlhoffCurrent/diagnostics/provenance_gate_hardening/**` (new) | repair audit artifacts |

**Forbidden:**
- `analysis/OlhoffCurrent/+impl/**`, `SOURCE_MANIFEST.json`, `olhoffcurrent_preset*.m`, `olhoffcurrent_config*.m`, `olhoffcurrent_run.m`
- `examples/**`, every other study directory (including `upstream_253069_migration`, whose `FINAL_SHA256.txt` pins its documents), `tools/**`
- anything related to Proposed, Yuksel or the evaluator

A change outside the allowed list is `GATE_REPAIR_SCOPE_FAIL`.

## 3. Required semantics

### Definitions

- **Production-source path:** a digested line's path token that starts with `analysis/OlhoffCurrent/+impl/` or equals `analysis/OlhoffCurrent/SOURCE_MANIFEST.json`.
- **Canonical path:** no backslash, no empty segment, and no `.` or `..` segment.

### S1 (C1) — per-line validation

Every digested line (`^[0-9a-f]{64}\s+(\S+)`) gets its own verdict:

| verdict | condition |
|---|---|
| `CURRENT_MATCH` | the working-tree file has the line's digest |
| `HISTORICAL_VERIFIED` | see S2/S3 |
| `MISSING` | the file is not found |
| `MISMATCH` | anything else |

The line passes only if its verdict is `CURRENT_MATCH` or `HISTORICAL_VERIFIED`. G4 passes only if every line passes; there is no aggregation over lines or paths. Duplicate paths are validated one by one and reported in `st.duplicatePaths`.

A line that names a production-source path but is not a valid digest line, or whose path is not canonical, is `MALFORMED_SOURCE_LINE` and fails G4. Parsing of non-source lines is unchanged.

**W1 fix:** a production-source line is resolved only at `<repo>/<path>`, never study-local and never relative to the current directory.

### S2 (C2) — existence before hashing

Historical content is read only after all of these succeed, each exit status checked on its own:
1. `git cat-file -e <C>:<path>`;
2. `git cat-file -t <C>:<path>` returns `blob`;
3. `git cat-file blob <C>:<path> > tmp`.

No pipeline is used. A missing object fails with reason `HISTORICAL_PATH_ABSENT`, never "empty". A genuinely empty committed blob hashes to the empty digest and can validate.

### S3 (C3) — one admissible historical commit per study

A `MISMATCH` production-source line may become `HISTORICAL_VERIFIED` only if all of these hold:

1. **The study's hash file is committed and unmodified.** The working-tree `FINAL_SHA256.txt` bytes equal its HEAD blob.
2. **The freeze commit is fixed.** `C_freeze = git log -1 --format=%H HEAD -- <study>/FINAL_SHA256.txt`, and the hash-file blob at `C_freeze` equals the working-tree bytes. `C_freeze` is the **only** admissible commit; no other ancestor is searched.
3. **The study declares its source identity.** The study's `EVIDENCE.json` is committed and unmodified (working tree = HEAD blob), and declares `sourceTree` and/or `impl_tree_sha256` (equal if both are present).
4. **The declared identity matches the freeze commit.** The declared tree equals the `+impl` tree hash computed from the committed blobs at `C_freeze`, using the `olhoffcurrent_source_manifest` algorithm. It also equals `SOURCE_MANIFEST.json@C_freeze → tree_sha256`.
5. **The line's own digest matches at the freeze commit.** S2 succeeds for `C_freeze:<path>`, and the SHA-256 of that blob equals **this line's** digest.

The "source commit" `git log -1 --format=%H C_freeze -- <path>` is recorded per line for reporting; it is not an acceptance criterion.

### S4 (C4, C5, C6) — current source tied to HEAD (new gate G6)

This check applies to **every** study the gate evaluates, so a dirty production state never finalizes anything. With `repo = RepoRoot`:

1. `git rev-parse --verify HEAD^{commit}` succeeds.
2. `git ls-tree -r HEAD -- analysis/OlhoffCurrent/+impl` has only regular blobs (mode 100644 or 100755).
3. Every HEAD blob exists in the working tree with **identical raw-byte SHA-256**; the blob bytes come from `git cat-file`.
4. No non-artifact working-tree file exists under `+impl` that HEAD lacks (`olhoffcurrent_is_artifact` decides), and there are no symlinks under `+impl`.
5. The working-tree `SOURCE_MANIFEST.json` bytes equal its HEAD blob, **and** its rows (path, sha256), `n_files` and `tree_sha256` equal the values derived from HEAD.
6. `PROVENANCE.md` contains exactly one `**Source tree SHA-256**` row, and its value equals the HEAD-derived tree.

The git index is not trusted: no `git diff` or `git status`, so skip-worktree and assume-unchanged cannot hide an edit. The status is `CURRENT_SOURCE_HASH_VERIFIED` or `CURRENT_SOURCE_HASH_NOT_VERIFIED`, with reasons.

### S5 (C7) — two separate statuses

`st.historicalSource.status` is one of `HISTORICAL_SOURCE_HASH_VERIFIED`, `HISTORICAL_SOURCE_HASH_NOT_VERIFIED` or `NO_HISTORICAL_SOURCE_LINES`. `st.currentSource.status` is as in S4. Both are printed and never merged into one statement.

Per-line records (`st.sourceLines`) carry line number, path, digest, verdict, freeze commit, source commit and reason. Superseded lines keep being listed in `st.supersededSource`.

### S6 — outcome

`st.ok = G1 && G2 && G3 && G4 && G5 && G6`. G1, G2, G3 and G5 are unchanged.

## 4. Adversarial probe suite (expectations frozen)

**Harness.** Built in a throwaway `git clone --shared` (sparse: `analysis/OlhoffCurrent`) of the repository HEAD. Each probe creates committed history as described.

**"Promotion scenario".** A sandbox study is committed first; that commit is `C_freeze`, with the declared tree equal to that commit's tree. A later commit then modifies `olhoffSolve.m`, `+move/limit.m` and `README.md`, regenerates `SOURCE_MANIFEST.json` and updates the PROVENANCE.md tree row. The gate runs at that HEAD.

| id | probe | expected `ok` |
|---|---|---|
| P0 | compliant study, no source lines | PASS |
| P1 | genuine `olhoffSolve.m@C_freeze` digest (promotion scenario) | PASS |
| P2 | single fabricated digest on a source path | FAIL |
| P3 | `limit.m@C_freeze` digest on the `olhoffSolve.m` line (wrong path, valid digest) | FAIL |
| P4 | non-source path (`README.md`) carrying its `C_freeze` digest after it changed | FAIL |
| P5 | fabricated, then genuine, same path | FAIL |
| P6 | genuine, then fabricated, same path | FAIL |
| P7 | empty-file digest on `olhoffSolve.m` (exists, non-empty at `C_freeze`) | FAIL |
| P8 | genuine superseded line; `+impl` locally edited, manifest unchanged | FAIL |
| P9 | empty digest; probe path existed, was deleted before `C_freeze`, re-added after | FAIL |
| P10 | genuine superseded line; `+impl` edited and manifest regenerated (uncommitted) | FAIL |
| P11 | two identical genuine lines, same path | PASS (duplicate reported) |
| P12 | `olhoffSolve.m` digest from an older commit (`cf1b71d`), not `C_freeze` (wrong commit) | FAIL |
| P13 | digest of a non-source file placed on a source-path line | FAIL |
| P14 | malformed: uppercase 64-hex digest on a source line | FAIL |
| P15 | malformed: 63-hex digest on a source line | FAIL |
| P16 | non-canonical path (`.../architecture/../architecture/olhoffSolve.m`), genuine digest | FAIL |
| P17 | missing: a source path absent from working tree and history | FAIL |
| P18 | empty digest; path never existed at `C_freeze`, added later | FAIL |
| P19 | empty digest; path is a genuinely empty committed file at `C_freeze`, non-empty now | PASS |
| P20 | empty digest; path non-empty at `C_freeze` and changed later | FAIL |
| P21 | study-local shadow copy of a source path with a matching fabricated digest (W1) | FAIL |
| P22 | genuine superseded line; EVIDENCE.json declares no source tree | FAIL |
| P23 | genuine superseded line; declared tree ≠ tree at `C_freeze` | FAIL |
| P24 | genuine superseded line; hash file has an uncommitted extra line | FAIL |
| P25 | `SOURCE_MANIFEST.json` tampered (`+impl` clean) | FAIL |
| P26 | untracked extra source file in `+impl` and manifest regenerated | FAIL |
| P27 | `+impl` file deleted (uncommitted) | FAIL |
| P28 | `+impl` edit hidden with `git update-index --skip-worktree` | FAIL |
| P29 | committed `+impl` change and committed manifest, PROVENANCE.md tree not updated | FAIL |

Each probe records: `probe, expected, actual, historical status, current status, reason`.

**Real-repository controls:**
- **R1.** `two_branch_controller_validation` passes. Exactly 7 lines are `HISTORICAL_VERIFIED` and 1 is `CURRENT_MATCH`. For each of the 7: same logical path, recorded digest = blob at `C_freeze = bba45e72ea18…`, source commit `1438aa3f4bd9…`, declared tree `edbfe47e…` = tree at `C_freeze`. Current status is `CURRENT_SOURCE_HASH_VERIFIED`.
- **R2.** Across all studies, the set that fails the gate equals the pre-existing set, study for study.

## 5. Repair acceptance (Part G)

Every item is required. Any failure: **no commit, no merge, `OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED`**.

| acceptance item | how it is met |
|---|---|
| `P5_FAILS_AS_EXPECTED`, `P9_FAILS_AS_EXPECTED`, `P10_FAILS_AS_EXPECTED` | P5, P9, P10 |
| `ALL_NEGATIVE_PROBES_FAIL_AS_EXPECTED` | P2–P10, P12–P18, P20–P29 |
| `ALL_LEGITIMATE_HISTORICAL_LINES_PASS` | P1, P11, P19 and R1 |
| `CURRENT_SOURCE_TIED_TO_HEAD_PASS` | P10, P28, P29 fail, and R1 is current-verified |
| `DIRTY_IMPL_REJECTED_PASS` | P8, P10, P26, P27, P28 |
| `MANIFEST_TAMPER_DOES_NOT_OVERRIDE_HEAD_PASS` | P10, P25 |
| `HISTORICAL_CURRENT_STATUS_SEPARATION_PASS` | in P8 and P10: historical = VERIFIED, current = NOT_VERIFIED; in P1: both VERIFIED, reported separately |
| `COMMITTED_PROVENANCE_SELF_CONTAINED_PASS` | Part E (§6) |
| `GATE_REPAIR_SCOPE_PASS` | §2 |
| `ALL_RELEVANT_TESTS_PASS_OR_ONLY_VERIFIED_PREEXISTING_FAIL` | see below |

**Tests in the worktree:**
- `test_finalization_gate`: the J section (the full probe suite) all pass, and H/I failures are exactly `H move_activity_400` plus the seven-study I-set;
- `test_path_isolation`, `test_currentness`, `test_source_integrity`, `test_evidence_retention`, `test_preset_identity`: 0 failures.

**100 % of probes must behave as expected.** Expectations are not edited after results are seen. A defect in the probe harness itself may be fixed only if it is recorded as such, and the probe re-run with the same expectation.

## 6. Part E decision (fixed in advance)

**Option B** is chosen. Committed provenance will cite only committed evidence and immutable commit identities:
- `diagnostics/upstream_253069_migration`;
- this study;
- upstream `Olhoff@253069…:repro/audits/upstream_olhoffcurrent_capabilities`;
- upstream `Olhoff@6b08708…:repro/results/S*x*` and `SWEEP_R06`.

Untracked local folders are listed as *supplementary, not required*. The scientific wording and every decision stay unchanged.

Option A is rejected. Committing into a path the normal checkout holds as untracked would make the merge refuse to overwrite untracked files.

The evidence strings inside `olhoffcurrent_presets.m` are forbidden to edit (preset file). They already say "untracked at 013cc48" and each stands next to committed evidence. They are classified, not changed.

## 7. Commit, merge and gate rules

- **One** new descendant commit on the migration branch ("Harden historical source provenance verification"). `9b30ec4` is not amended. The worktree must be clean afterwards.
- **Re-run from Step 1:**
  1. amendment review (A1, repaired A2, A3);
  2. branch review (`9b30ec4` plus the repair commit);
  3. re-verify the target HEAD and merge into `benchmark-methodology-r2` — fast-forward if possible, no ad-hoc conflict resolution, `origin` untouched;
  4. post-merge identity and the full suite (MIGRATION_HANDOFF list, the six upstream root-independent suites against `+impl`, `confbench_preflight`, `confbench_selftest`).
- **Failure classification** (NEW REGRESSION / VERIFIED PRE-EXISTING / ENVIRONMENTAL / TEST DEFECT) is made against a baseline run **in the normal checkout before merge**, with the same untracked content present.
- **Anchors (Steps 5A/5B):** one 160×20 run per preset. The comparison is the migration's established standard (`mig_compare`): bitwise class, size and raw bytes, recursive; excluded are `tEig tGrad tInner tOuter wallclock provenance.resolvedAt`.
  - **Historical:** against `POST_BETA`/`PRE_BETA` and the frozen campaign record: 91/2241, ω₁ 169.495227021538, and the 81-row hash `28756d22…` reconstructed exactly.
  - **Pedersen:** against upstream `S160x20/res.mat` and `POST_PED`/`UP_PED`: 121/2369, ω₁ 169.210576386275.
- **Freeze and preview (5C/5D/5E)** as in the brief. Nothing above 160×20 is solved.

## 8. Stop conditions

Any of these means stop and block:
- an acceptance item fails;
- a scope violation;
- `+impl` differs from upstream 253069;
- a merge conflict or an unexpected target HEAD;
- a new test regression;
- either anchor not reproducing;
- any pressure to edit expectations, presets or scientific code.
