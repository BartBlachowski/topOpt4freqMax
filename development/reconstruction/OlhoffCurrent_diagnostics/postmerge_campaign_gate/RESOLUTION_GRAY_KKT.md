# RESOLUTION_GRAY_KKT — lifting the attempt-2 blocker (2026-09-14)

## Blocker (attempt 2, 2026-09-13)

The untracked study `gray_kkt_forensic_audit` declared the pre-migration `analysis/OlhoffCurrent/SOURCE_MANIFEST.json` (`431e7b30…`) as REQUIRED evidence. After the merge it failed G2, making it a new entry in `test_finalization_gate` I. The blocked records are preserved byte-identical in `attempt2_blocked_gray_kkt/`.

## Owner decision

**Option 2:** update the study's own evidence record to mark the pin as historical, then re-run `test_finalization_gate` and confirm the I-set equals the verified pre-existing set.

## What was done — only in `diagnostics/gray_kkt_forensic_audit/`; see its `EVIDENCE_AMENDMENT_1.md`

- **Historical claim verified against git.** `431e7b30…` is `SOURCE_MANIFEST.json` at `1438aa3` (last change before the promotion), `bba45e7` and `013cc48`, with `tree_sha256` `edbfe47e…`, which is the study's declared `sourceTree`. It was superseded at `9b30ec4`.
- **Originals preserved byte-identical** in `evidence_record_history/`: EVIDENCE.json `6a89e7ca…`, FINAL_SHA256.txt `5cc4d29d…`.
- **`EVIDENCE.json`: one artifact changed.**
  - `class` `required` → `optional`, with `historical: true`, `original_class`, `historical_identity` and `superseded_by` added.
  - Recorded digest unchanged.
  - `evidence_amendments` record appended.
  - New file SHA-256 `9ff544e4…`.
- **`FINAL_SHA256.txt`.** Only the EVIDENCE.json line changed; three lines were appended (amendment record, two originals). It self-verifies with 98 lines.
- **Left unchanged:** the other 57 required artifacts, `DATA_MANIFEST.json` (which still names the preserved original), and every report, verdict and metric.

## Re-verification in the merged normal checkout (HEAD `b21483b`, unchanged)

| check | result |
|---|---|
| `gray_kkt_forensic_audit` evidence gate | PASS: 57/57 required match, 1 optional **OPTIONAL_HASH_MISMATCH (warned, visible)**, 3 scratch |
| `gray_kkt_forensic_audit` finalization gate | PASS (G1–G6) |
| `test_finalization_gate` (`logs/RESOLVE_gates.log.txt`) | 2 failures: H `move_activity_400`, and I = **exactly** the verified pre-existing set (the seven tracked plus `frozen_inner_solver_study`, `frozen_problem25_reference`, `scientific_delta_olhoff_migration`); all 39 J probes, H `beta_transition_mechanism` and `two_branch_controller_validation`, and both H2 pass |
| `test_path_isolation`, `test_currentness`, `test_source_integrity`, `test_evidence_retention` | 0 each |
| every study vs pre-merge baseline (`evidence/RESOLVE_studies.json` vs `logs/BASE_studies.json`) | identical verdicts and G1–G5 flags for all studies; only this gate study changed (now compliant); G6 `CURRENT_SOURCE_HASH_VERIFIED` for all 30 |

## Unaffected by the resolution

Untracked content does not change HEAD, `+impl`, any configuration or any solve. The other suites, both anchors, the nine-config preview and `CAMPAIGN_IDENTITY.json` all stand as recorded at HEAD `b21483b`.
