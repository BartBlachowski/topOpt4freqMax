# GATE_PREREGISTRATION — merge and promotion gate, attempt 2

## Attempt 1 is preserved, not overwritten

- **Location.** Attempt 1 (2026-09-13, `OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED`) is in `attempt1_blocked/`.
- **How it was preserved.** The whole folder was moved there before attempt 2 began. All 24 files were verified byte-identical after the move against a full SHA-256 list taken before it, and its own `FINAL_SHA256.txt` still self-verifies.
- **What remains stale.** Its `EVIDENCE.json` still names the old evidence root. Those evidence files now sit under `attempt1_blocked/evidence/`, and they are hashed by attempt 2's `FINAL_SHA256.txt`.
- **What changed since.** The owner authorised a repair of the gate rule, and this attempt restarts from Step 1.

## Disclosure

The pass criteria, stop rules and verdicts are fixed by the owner's second brief. This file records them after the fact and changes none of them.

**Repair-phase preregistration** (frozen before any code change), in `provenance_gate_hardening/` of commit `b21483b`:

| document | SHA-256 |
|---|---|
| PREREGISTRATION.md | `de82b1a0…` |
| PREREGISTRATION_ADDENDUM_1.md | `6f52d940…` |

**Pre-merge baseline.** It was run in this checkout before the merge, with the same untracked content present, so failures could be classified against it: `logs/BASE_gates.*`, `logs/BASE_studies.*`, `logs/BASE_selftest.*`.

## Fixed identities

| | |
|---|---|
| migration commits | `9b30ec45b038fb36e7cf20d57679b71cfd099fb3` + repair `b21483b158f58e05e7b56957f2fbe8e1d2891395` |
| upstream implementation | `253069262407885a8b759a9e721c4f0a7d3a397d` (archive SHA-256 `f9112403…`) |
| target | `benchmark-methodology-r2`, expected start `013cc48…` |
| historical anchor | `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered`, 160×20: 91 / 2241 / ω₁ 169.495227021538; 81-row hash `28756d22…` |
| production anchor | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`, 160×20: 121 / 2369 / ω₁ 169.210576386275 |

## Stop and block rules

| step | rule |
|---|---|
| 1–3 | a FAIL stops the gate |
| 4 | any NEW REGRESSION blocks; only failures demonstrated before the merge may remain |
| 5A/5B | anchor FAIL blocks; comparison uses the migration's `mig_compare` standard |
| 5C | the production identity is frozen only if both anchors pass |
| all | nothing above 160×20 is solved |
