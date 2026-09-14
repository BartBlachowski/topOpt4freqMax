# MANIFEST_PROVENANCE — Part 9

```
MANIFEST_PROVENANCE_PASS
```

Scripted agreement check: `evidence/manifest_provenance_check.json`, 20/20 true.

| source of truth | files | tree SHA-256 | promoted commit | parent | adaptations |
|---|---|---|---|---|---|
| actual tree (recomputed in Python, independent of MATLAB) | 79 | `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` | — | — | — |
| `SOURCE_MANIFEST.json` (written by `olhoffcurrent_source_manifest('Write',true)`; every file hash = live) | 79 | same | — | — | — |
| `BYTE_IDENTITY.md` / `evidence/byte_identity.json` | 79, all = 253069 | same | 253069 | — | 0 target-specific |
| `PROVENANCE.json` | 79 | (live values added by `olhoffcurrent_provenance`) | `253069262407885a8b759a9e721c4f0a7d3a397d`, the head of the recorded branch | `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` | `[]` |
| `PROVENANCE.md` | 79 | same | same | same | "Adaptations: none" |
| `README.md` §8 | 79, byte-identical to `253069` | — | — | — | no adaptations |

The manifest was written in one deliberate step (`scripts/mig_write_manifest.m`). That step refused to keep the file unless:

- the MATLAB tree hash equals the independently computed one;
- the file count is 79;
- every manifest file hash equals the 253069 snapshot's file.

## What changed in provenance

| item | before | after |
|---|---|---|
| schema | `olhoff_current_provenance/1` | `/2` (the `/1` record preserved verbatim in `history[0]`) |
| source commit / branch | `695f03b` / `architecture/canonical-config` | `253069` / `migration/upstream-olhoffcurrent-capabilities`, with `parent_successful_science_commit = 6b08708` (`repro/natural-convergence`) |
| prose file count / adaptations | "74 files", "the one adaptation" (**stale since `1438aa3`**) | 79 files, none; §7 explains the stale history |
| production preset | one static `production_preset` block | append-only `production_preset_events` (event 1 reconstructed from the /1 record; event 2 this change) |
| config hashes | not recorded | per event: old 81-row campaign hashes, new 87-row hashes for both presets |
| known defects | — | `PCONT_MOVE_HISTORY_LOGGED_BEFORE_RESET` |
| lineage | — | 695f03b promotion → 1438aa3 local edits → 253069 promotion |

## Upstream lineage (explicit)

```
upstream source commit:              253069262407885a8b759a9e721c4f0a7d3a397d
parent successful-science commit:    6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7
previous promotion (history[0]):     695f03bdac20c423a4e1d389cf9db9187597bcc3
```

## Currentness model correction (required for this promotion)

`olhoffcurrent_currentness` used to test `merge-base --is-ancestor <promoted> HEAD` against the development checkout's HEAD. The promoted commit now lives on a branch that is not checked out upstream (the checkout is `repro/natural-convergence @ 6b08708`). The old test would therefore have reported `PROVENANCE_MISMATCH` for a correct promotion.

The function now tests against the recorded branch `refs/heads/<source.branch>`:

- a missing branch is `PROVENANCE_MISMATCH`, with its own message;
- the checked-out branch and HEAD are still reported, as information only.

Result after migration: `CURRENT`, 0 commits ahead. `test_currentness` and `test_source_integrity` pass; both require `CURRENT`, and the latter still requires `LOCAL_MODIFIED` on any source edit.

## Evidence-gate rule added for promotions

`olhoffcurrent_finalization_gate` now accepts a study's pinned production-source line only as `SUPERSEDED_PRODUCTION_SOURCE`. Two conditions must both hold: the recorded digest is that path's content in a commit reachable from HEAD, and `+impl` verifies against its manifest. See `EVIDENCE_POLICY.md` rule 7 and TEST_REPORT.md.

This prevents the promotion from falsely invalidating `two_branch_controller_validation` without editing that study's hash file. The seven superseded lines all resolve to commit `1438aa3`.
