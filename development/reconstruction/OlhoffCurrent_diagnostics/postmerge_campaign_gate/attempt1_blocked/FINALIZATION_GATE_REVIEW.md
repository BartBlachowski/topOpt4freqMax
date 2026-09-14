# FINALIZATION_GATE_REVIEW — Amendment A2 (`SUPERSEDED_PRODUCTION_SOURCE`)

```
FINALIZATION_GATE_PROVENANCE_LOGIC_FAIL
```

**Reviewed:** `analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m` at `9b30ec45b038fb36e7cf20d57679b71cfd099fb3`, specifically `local_supersededSource`.

**Result:** the rule is **not strictly fail-closed**. Three constructed inputs make the gate return `ok = true` when it should fail. Each was run end to end through `olhoffcurrent_finalization_gate`.

## What the rule does

A mismatching `FINAL_SHA256.txt` line becomes `SUPERSEDED_PRODUCTION_SOURCE` (not a failure) when all of these hold:

1. its path starts with `analysis/OlhoffCurrent/+impl/` or equals `analysis/OlhoffCurrent/SOURCE_MANIFEST.json`;
2. `olhoffcurrent_source_manifest('Verify', true).ok`, meaning `+impl` matches the working-tree `SOURCE_MANIFEST.json`;
3. for some commit in `git log HEAD -- <path>`, `git show <commit>:<path> | shasum -a 256` equals `rec(<path>)`.

`rec` is a `containers.Map` keyed by path, built from the whole hash file.

## Requirement checklist

| requirement (brief) | met? | evidence |
|---|---|---|
| exact historical digest | **NO** | The digest compared is `rec(path)`, the **last** line for that path, not the line under test. See P5. |
| exact logical file path | yes | The same `path` is used in `git log -- path` and `git show c:path`, without `--follow`. P3 fails correctly. |
| identifiable historical commit | yes | The commit is recorded in `st.supersededSource.commit`. |
| digest actually present at that commit | **NO** | The exit status is the pipeline's, i.e. `shasum`'s, not `git show`'s. When `git show` fails (e.g. at a deletion commit), the empty string's SHA-256 matches. See P9. |
| current source independently validated | **NO** | `+impl` is checked only against the **mutable working-tree** manifest, not HEAD, PROVENANCE or upstream. A local edit plus a regenerated manifest passes. See P10. The header's claim "never a local edit" is false. |
| fabricated digest fails | **only sometimes** | P2 (single line) fails correctly; P5 and P9 pass. |
| unrelated ancestor file cannot satisfy | yes | P3 (another source file's historical digest) and P4 (non-source file) both fail. |
| historical vs current distinguished | yes (equivalent) | Superseded lines are listed separately and printed as `SUPERSEDED_PRODUCTION_SOURCE`; working-tree lines are counted in `nHashed`. The gate does not claim historical lines match current source. |

A further way the rule is broader than necessary: any version of the path from **any** commit reachable from HEAD is accepted. That includes versions written after the study froze. It could be limited to ancestors of the commit that added the study's `FINAL_SHA256.txt`.

## Probe results

Run in a throwaway `git clone --shared` at `9b30ec4` in the session scratchpad. Scripts: `scripts/gate_probe.m` and `scripts/gate_probe_p10.m`. Log: `evidence/gate_probe.log`. Machine-readable: `evidence/gate_probe_results.json` and `evidence/gate_probe_p10_result.json`. MATLAB 25.2.0.2998904 (R2025b). The clone was restored to `9b30ec4` with a clean status after the runs.

| id | input | expected | gate.ok | verdict |
|---|---|---|---|---|
| P0 | compliant sandbox study | PASS | PASS | ok |
| P1 | real historical digest of `olhoffSolve.m` (@013cc48), same path (J1 replica) | PASS | PASS | ok |
| P2 | fabricated digest `c…c` (J2 replica) | FAIL | FAIL | ok |
| P3 | `limit.m`'s historical digest placed on the `olhoffSolve.m` line | FAIL | FAIL | ok |
| P4 | historical digest of `README.md`, a non-source file | FAIL | FAIL | ok |
| **P5** | **fabricated line, then a real historical line, same path** | FAIL | **PASS** | **FAIL-OPEN**: both lines are reported as superseded with the *real* digest, so the fabricated digest is never shown |
| P6 | real historical line, then fabricated line, same path | FAIL | FAIL | ok (order-dependent) |
| P7 | empty-content digest, path never deleted | FAIL | FAIL | ok |
| P8 | real historical digest, `+impl` locally edited, manifest not regenerated | FAIL | FAIL | ok |
| **P9** | **empty-content digest; path deleted and re-added in history** | FAIL | **PASS** | **FAIL-OPEN**: recorded as superseded at the deletion commit, where the file does not exist |
| **P10** | **real historical digest; `+impl` locally edited and `SOURCE_MANIFEST.json` regenerated** | FAIL | **PASS** | **FAIL-OPEN**: `man.ok = 1` |

## Reachability in the real repository today

- **P9 is latent.** No `+impl` path or `SOURCE_MANIFEST.json` has ever been deleted in history reachable from `9b30ec4` (`git log --diff-filter=D`: none).
- **P5 is latent.** No study hash file has duplicate source-path lines.
- **P10 needs a manifest rewrite.** `olhoffcurrent_source_manifest` warns against doing that.
- **The one real use is legitimate.** `two_branch_controller_validation/FINAL_SHA256.txt` has eight source lines. Seven differ from the current tree, and all seven equal the blob at the **same path** in `1438aa3` ("A & B tests", 2026-09-09, an ancestor), checked independently with `git cat-file`. The eighth (`+move/exhaustion.m`) matches the current tree. So the historical evidence itself is sound.

This is a defect in how general the rule is, not a wrong verdict on existing evidence. The brief makes strict fail-closedness a blocking condition ("If this rule is broader than necessary: BLOCK") and does not allow silent waivers, so this review is **FAIL**.

## Minimal remedy (not applied — outside this task's authorization)

1. Evaluate each digested line on its own, comparing the line's own digest; or refuse duplicate source-path lines.
2. Require `git cat-file -e <commit>:<path>` to succeed, and hash `git cat-file blob` output without a pipeline that hides the exit status. Treat any git failure as a mismatch.
3. Tie "current source" to committed identity: `+impl` and `SOURCE_MANIFEST.json` unmodified against HEAD with no untracked source, and a tree hash equal to the recorded `4ba9a3ae…`. For example, require `olhoffcurrent_currentness` state `CURRENT`.
4. Optional: limit candidate commits to ancestors of the commit that introduced the study's `FINAL_SHA256.txt`.
5. Add destructive tests mirroring P5, P9 and P10 (e.g. J4–J6).

The change lives outside `+impl`. It cannot affect byte identity or any trajectory.
