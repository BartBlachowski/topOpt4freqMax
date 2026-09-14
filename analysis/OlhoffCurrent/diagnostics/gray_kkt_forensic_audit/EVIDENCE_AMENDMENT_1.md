# EVIDENCE_AMENDMENT_1 — historical source-manifest pin

**Date:** 2026-09-14. **Authority:** the study owner's decision (option 2 of `postmerge_campaign_gate/CAMPAIGN_AUTHORIZATION.md`).

## Why

This audit analysed the pre-migration OlhoffCurrent implementation, tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`. Its `EVIDENCE.json` pinned the production manifest `analysis/OlhoffCurrent/SOURCE_MANIFEST.json` as a **required** artifact, at SHA-256 `431e7b3084507cd8b53862847b828cfd9f8948172e349bbae0c72749e8134aeb`.

The upstream 253069 promotion (`9b30ec4`, merged into `benchmark-methodology-r2` at `b21483b`) legitimately replaced that file (now `aee44aa9…`, tree `4ba9a3ae…`). The evidence gate therefore reported `REQUIRED_HASH_MISMATCH`, and the study failed the finalization gate. That would have let a legitimate promotion masquerade as lost evidence.

## What changed

1. **`EVIDENCE.json`, one artifact only (`analysis/OlhoffCurrent/SOURCE_MANIFEST.json`).**
   - `class` changed from `required` to `optional`.
   - Annotations added: `historical: true`, `original_class: required`, `historical_note`, `historical_identity`, `superseded_by`.
   - The recorded digest is **unchanged**. The evidence gate now reports it as `OPTIONAL_HASH_MISMATCH` (warned, visible), not as a pass.
   - A top-level `evidence_amendments` record was appended.
2. **`FINAL_SHA256.txt`.**
   - The `EVIDENCE.json` line now carries the amended digest.
   - Lines were appended for this file and the two preserved originals.
   - Every other line is unchanged.

## What did not change

- The other 57 required and 3 scratch artifacts.
- Every report, verdict, metric, figure, script and evaluation of this study.
- `DATA_MANIFEST.json`. It keeps recording the study as frozen, including the original `EVIDENCE.json` digest `6a89e7ca…`, which is the preserved copy below.
- The historical digest itself.

## Originals, preserved byte-identical

| file | SHA-256 |
|---|---|
| `evidence_record_history/EVIDENCE.pre_253069_promotion.json` | `6a89e7caf63f22c88e3508a81fe0976ebbabfdc94a799d0b409459cb0182c551` |
| `evidence_record_history/FINAL_SHA256.pre_253069_promotion.txt` | `5cc4d29db3c865ab0aa017ad7e5555eaad96b4553c172515e75cb6e79448c860` |

## The historical claim, verified against git (2026-09-14)

`git cat-file blob <commit>:analysis/OlhoffCurrent/SOURCE_MANIFEST.json` has SHA-256 `431e7b30…` and `tree_sha256` `edbfe47e…` at:
- `1438aa3f4bd9…`, where the manifest last changed before the promotion;
- `bba45e72ea18…`;
- `013cc48451d3…`.

At `9b30ec45b038…` and `b21483b158f5…` it is `aee44aa9…` with tree `4ba9a3ae…`. The study's declared `sourceTree` and `implTree` are `edbfe47e…`, consistent with this.

## Limitation

The evidence gate does not verify historical annotations; it only warns on the optional mismatch. This study is untracked, so the finalization gate's commit-based historical verification cannot apply to it. The verification above is recorded here instead, and is reproducible with the command shown.
