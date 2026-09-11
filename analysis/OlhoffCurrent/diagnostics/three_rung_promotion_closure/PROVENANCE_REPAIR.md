# PROVENANCE_REPAIR — consolidated record of every repair

Two repairs applied, one analysed and deliberately deferred, one class not
repairable on this host. Detail lives in `HISTORICAL_HASH_REPAIR.md` and
`TIMING_TELEMETRY_AUDIT.md`; this is the ledger.

## Repair 1 — `two_branch_controller_validation/FINAL_SHA256.txt` — **APPLIED**

```
old sha256  2fb9fc731d816e5c47c9835d5f777f8bb3be2fe61773ca00119049ed2bf5e0e0
new sha256  a7ef91f9cc279b46288f25f6f453495c5d358d96e1ff5ad33a8d87e61c73bd3e
archived    evidence/two_branch_FINAL_SHA256.BEFORE.txt
changed     3 of 143 lines; 71 entries before and after; identical key sets
```

| Entry | old | new | underlying file |
|---|---|---|---|
| `PROVENANCE.md` | `42037f81…3ca90790` | `0e987a9b…35459a33` | **unchanged; disk == git HEAD** |
| `BASELINES.md` | `74629a16…d2299b21` | `2212a66a…baa4f0df` | **unchanged; disk == git HEAD** |
| `evidence/baselines.json` | `08b48346…10b4c571` | `a4b55671…8738f420e` | **unchanged; disk == git HEAD** |

**Proof scientific content is unchanged:** each target is byte-identical to
`git show HEAD:<path>`. The files were never wrong; the digests had gone stale
because the documents were edited after the hash file was written and both
states were committed together. Machine-independent — it reproduces anywhere.

**Reason for repair:** a self-verifying manifest that disagrees with its own
committed files is a bookkeeping defect that blocks G4 on every host, and it
masks real staleness.

`git status` at the time of repair: the three targets **clean**; only the two
timing paths modified (settled in Repair 2).

## Repair 2 — the two timing-drifted tracked paths — **APPLIED**

Archived first, then restored from HEAD. Nothing destroyed.

| Path | working copy (archived) | restored to |
|---|---|---|
| `runs/C320x40_iterations.csv` | `6ac72fa8…a4325a22` | `ff570d6e…563b356c` |
| `runs/C320x40_record.json` | `76560421…5e78c28e` | `3fc68c05…4dc236be` |

Archives: `evidence/preserved_working_copies/`.

**Verification before restoring** — exhaustive, not sampled:

- CSV: 55 columns × 1600 rows compared cell by cell. **Only `tOuter` differs**;
  the 54-column scientific projection is byte-identical
  (`5f9f1896…8f6bb53d`).
- JSON: flattened leaf comparison. **Exactly 3 leaves differ** — `wall_s`
  (timing), `trajectoryBytes` (container size), `matlab` (toolchain version).
  `status`, `nOuter`, `innerTotal`, `rho_sha256`, `cfgHash`, `implTree`,
  `omega`, and the full event log are identical.

**No unique scientific state existed in either working copy.** Both digests
after restore equal what `FINAL_SHA256.txt` and `EVIDENCE.json` already
recorded — the hash file was correct and the working copy had drifted.

## Repair 3 — `three_rung_architecture/EVIDENCE.json` — **ANALYSED, NOT APPLIED**

Records the same stale `baselines.json` digest `08b48346…`. Investigated via
`git log` and a semantic diff: the change (commit `4dd5760`) is **purely
additive** — 12 new keys, `old=<absent>` for every one, **no existing value
modified**. The file's own note: *"Recovered after the freeze; frozen scalars
re-derived from this file and matched to <=1e-9. No frozen value was edited."*

Not applied because (a) the brief scopes Phase 4 to *"ONLY FINAL_SHA256.txt"*,
and (b) it would change no gate outcome — that study's G2 fails on container
digests regardless — while cascading into its own `FINAL_SHA256.txt`
(`EVIDENCE.json` is hashed there as `47560482…947183f6`).

The same stale digest also appears in `two_rung_architecture/EVIDENCE.json` and
`move_ladder_necessity/FINAL_SHA256.txt`, and in several
`provenance_start.json` / `provenance_final.json` snapshots. **The snapshots
must never be edited** — they record state at run time.

## Class 4 — git-ignored containers — **NOT REPAIRABLE HERE**

Six artifacts. One absent, five present with mismatched container digests:

```
MISSING   evidence/three_rung_resolution_240/C240x30_trajectory.mat
MISMATCH  evidence/two_branch_controller_validation/C160x20_trajectory.mat
MISMATCH  evidence/two_branch_controller_validation/C320x40_trajectory.mat
MISMATCH  evidence/two_branch_controller_validation/C400x50_trajectory.mat
MISMATCH  evidence/move_activity_400/F400_400x50_trajectory.mat
MISMATCH  evidence/move_activity_400/P400_400x50_trajectory.mat
```

Requires a file transfer or an explicit, documented owner decision to
re-declare. **Not taken unilaterally:** re-declaring a digest to make a gate
pass is the move the finalization gate exists to prevent.
`PROMOTION_PROVENANCE.md` §4.

## Net effect

`two_branch_controller_validation`'s G4 mismatch list went from **10 entries to
5**; the five remaining are all git-ignored `.mat` containers. Its G2 list is
unchanged (containers only). No study's gate flipped to PASS, because in every
case a container blocker remains.
