# PREREGISTRATION_ADDENDUM_1 — self-review findings on the first repair commit

**Timing:** written **before** any code change it describes. PREREGISTRATION.md (SHA-256 `de82b1a0…`) is unchanged; every original expectation stands. This addendum only **adds** semantics and probes, all of which make the gate stricter.

## Context

The first repair commit `adf86a3f8126e2099436b068ed3bd8a1921a2405` passed all 31 preregistered probes. Before the Step 1 re-review, I attacked it further, using the committed gate in throwaway clones. Scripts: `scratchpad/run/selfreview_probe.m`, `selfreview2_probe.m`; their outputs are reproduced in REPORT.md.

| finding | observed on `adf86a3` | why it matters |
|---|---|---|
| **A1** `analysis/OlhoffCurrent//+impl/...` and `analysis/OlhoffCurrent/./+impl/...` plus a study-local shadow copy | `ok = 1`: the line is not recognised as production source and resolves to the shadow | W1 is not closed for normalization variants |
| **A2** `+impl//architecture/...` | classified as a canonical source line (`strsplit` collapses empty segments) | "canonical" is broader than intended; not fail-open, since git then fails |
| **A3** `git replace` of the HEAD blobs of an edited `olhoffSolve.m` and a regenerated `SOURCE_MANIFEST.json`, with the PROVENANCE row updated | **G6 = 1** | current source is not tied to the committed HEAD objects (C4) |
| **A4** `GIT_DIR` in the environment pointing at a repository whose HEAD commits the same edit | **G6 = 1** | `git -C <repo>` obeys `GIT_DIR`; the gate is not reading the repository it names |

A3 and A4 also apply to the historical check: a replace ref can forge `<freeze>:<path>`.

**Decision:** `adf86a3` is **not** accepted. It was never merged or pushed.

## Added semantics

### S7 — git invocation hygiene (every git call in the gate)
- Run as `env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_OBJECT_DIRECTORY -u GIT_ALTERNATE_OBJECT_DIRECTORIES -u GIT_COMMON_DIR -u GIT_NAMESPACE -u GIT_REPLACE_REF_BASE -u GIT_CONFIG_PARAMETERS -u GIT_CONFIG_COUNT git --no-pager --no-replace-objects -C "<repo>" …`.
- Objects, trees and history are read as committed in the repository named by `RepoRoot`, never as redirected or replaced.

### S8 — detecting production-source naming
1. **Canonical check.** The canonical-path check splits on `/` without collapsing, so an empty segment is non-canonical.
2. **Normalized name.** A digested line **names production source** if its path token, after lower-casing, converting `\` to `/`, dropping empty and `.` segments and resolving `..`, contains `olhoffcurrent/+impl` or `olhoffcurrent/source_manifest.json`. A non-digest line names production source if its text, normalized the same way (repeated `/` collapsed, `./` segments dropped), contains either string. Such a line that is not exactly canonical is `MALFORMED_SOURCE_LINE`.
3. **Resolution check.** A non-source digested line whose resolved file lies, by canonical real path (symlinks resolved; case-insensitive on macOS and Windows), inside `<repo>/analysis/OlhoffCurrent/+impl` or equals `<repo>/analysis/OlhoffCurrent/SOURCE_MANIFEST.json` is also `MALFORMED_SOURCE_LINE`. Production source may be referenced **only** by its canonical logical path.

## Added probes (expectations frozen here)

| id | probe | expected `ok` |
|---|---|---|
| P31 | `analysis/OlhoffCurrent//+impl/architecture/olhoffSolve.m` + study-local shadow + matching fabricated digest | FAIL |
| P32 | `analysis/OlhoffCurrent/./+impl/architecture/olhoffSolve.m` + study-local shadow + matching fabricated digest | FAIL |
| P33 | study-relative `../../+impl/architecture/olhoffSolve.m` (resolves to the real production file) with its current digest | FAIL |
| P34 | `analysis/OlhoffCurrent/+impl//architecture/olhoffSolve.m` with the genuine freeze digest (promotion scenario) | FAIL |
| P35 | promotion scenario; the freeze-commit blob of `olhoffSolve.m` forged via `git replace` to fabricated content, whose digest is on the line | FAIL |
| P36 | `+impl` edited, manifest and PROVENANCE row regenerated (uncommitted); `GIT_DIR` points at a clone whose HEAD commits that edit | FAIL |
| P37 | `+impl` edited, manifest and PROVENANCE row regenerated (uncommitted); HEAD blobs of both files replaced via `git replace` | FAIL |
| P38 | study-local **symlink** to the real `olhoffSolve.m`, line with its current digest | FAIL |

All 31 original probes must still behave as preregistered. The suite becomes 39 probes, and **39/39** is required. R1/R2 and the test expectations are unchanged.

## Commit handling

The repair must remain **one** focused commit descending from `9b30ec4`. `adf86a3` will be amended into a single replacement commit; `9b30ec4` is untouched. REPORT.md records the superseded hash `adf86a3…` and the reason.
