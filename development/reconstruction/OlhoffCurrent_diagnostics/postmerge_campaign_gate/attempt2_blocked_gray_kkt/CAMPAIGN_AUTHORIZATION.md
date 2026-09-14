# CAMPAIGN_AUTHORIZATION — attempt 2

```
OLHOFF_NINE_MESH_CAMPAIGN_BLOCKED
```

> The definitive nine-mesh campaign remains blocked because the post-merge test suite contains one new, merge-induced failure. The untracked local study `analysis/OlhoffCurrent/diagnostics/gray_kkt_forensic_audit` declares the pre-migration `analysis/OlhoffCurrent/SOURCE_MANIFEST.json` (SHA-256 `431e7b30…`) as a REQUIRED evidence artifact. After the merge it therefore fails the finalization gate (G2, `REQUIRED_HASH_MISMATCH`), and `test_finalization_gate` I lists it as a new failing study. The owner's rule allows only previously demonstrated failures to remain, and this one passed before the merge.

## Authorization conditions

| condition | status |
|---|---|
| provenance repair accepted | **PASS** (`PROVENANCE_GATE_HARDENING_PASS`) |
| P5 / P9 / P10 negative controls | **PASS**: all FAIL as required, from committed code |
| full adversarial gate suite | **PASS**: 39/39 (original rule fail-open on 24; first repair on 6) |
| committed provenance self-contained | **PASS** |
| amendment review | **PASS** |
| migration review | **PASS** |
| merge | **PASS** (fast-forward to `b21483b`) |
| post-merge identity | **PASS** |
| **post-merge tests** | **FAIL**: one NEW failure (`gray_kkt_forensic_audit`, untracked, merge-induced); every other failure verified pre-existing |
| historical S160 | **PASS** (bitwise) |
| Pedersen S160 | **PASS** (bitwise at established level) |
| nine-config preview | **PASS** |
| telemetry readiness | **PASS** |
| repository integrity | **PASS** |

## What the failure is, and is not

- **It is** a genuine consequence of the merge in this checkout. The study's pin was valid before the merge and is not now.
- **It is not** a code, science or committed-repository regression:
  - `+impl` is byte-identical to 253069;
  - G6 passes for every study;
  - both anchors reproduce;
  - the study is absent from a clean clone of `b21483b`;
  - the evidence gate code is unchanged by both commits, so `9b30ec4` alone would have caused it too.

## How to lift the block (owner decision; not taken here)

Any one of these:

1. **Rule it out of scope.** Declare in writing that untracked local studies are outside the post-merge regression scope, i.e. classify this failure ENVIRONMENTAL. Every other condition already passes, and `CAMPAIGN_IDENTITY.json` stays valid, because untracked content does not change HEAD or the configurations. No re-run is needed.
2. **Update the study's own evidence record.** As `gray_kkt_forensic_audit`'s owner, record that its `SOURCE_MANIFEST.json` pin is historical (it studied tree `edbfe47e…`). Then re-run `test_finalization_gate` in this checkout and confirm the I-set equals the verified pre-existing set. This task was not authorized to edit that study.
3. **Commit it under a separately authorized gate change.** Commit the study, and extend the historical-source verification to committed `EVIDENCE.json` production-source artifacts. That is a new gate change requiring its own authorization and review.

## Frozen identity awaiting that decision

See `CAMPAIGN_IDENTITY.json`.

| | |
|---|---|
| HEAD | `b21483b158f58e05e7b56957f2fbe8e1d2891395` |
| production preset | `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` |
| production config hash (160×20) | `b1a5744df798ad624fcd0b8b4306888eb99816ee5ce60f51040e39e03e90d4f4` |
| `+impl` tree | `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` |
| nine config hashes | `NINE_MESH_CONFIGS.json` |

No mesh above 160×20 was solved, and the campaign was not run.
