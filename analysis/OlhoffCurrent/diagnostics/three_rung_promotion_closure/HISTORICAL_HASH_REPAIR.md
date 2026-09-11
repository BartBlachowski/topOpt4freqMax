# HISTORICAL_HASH_REPAIR — Phase 4

# `HISTORICAL_FINAL_SHA256_REPAIR_PASS`

`two_branch_controller_validation/FINAL_SHA256.txt` carried three digests that
disagreed with the study's own **committed, clean** files. The manifest was
regenerated; **no underlying file was touched**.

## 1. Before and after

| | SHA-256 |
|---|---|
| before (archived at `evidence/two_branch_FINAL_SHA256.BEFORE.txt`) | `2fb9fc731d816e5c47c9835d5f777f8bb3be2fe61773ca00119049ed2bf5e0e0` |
| after | `a7ef91f9cc279b46288f25f6f453495c5d358d96e1ff5ad33a8d87e61c73bd3e` |

71 digested entries before and after, **identical key sets**, 68 of 71 entries
byte-identical. Exactly 3 changed.

## 2. The exact entries repaired

| Entry | old digest | new digest |
|---|---|---|
| `PROVENANCE.md` | `42037f8196cf9768477fb0473ec8eed45445c5da23dec2649774cb093ca90790` | `0e987a9bc7b39425661ffcee0cd81ebc5fa88010d35e094daff8d4df35459a33` |
| `BASELINES.md` | `74629a1634bb5452ea1e91161e9e652acf51a5d431ee156767ee339dd2299b21` | `2212a66a50c62f118c0a17047c8e2c8fa55bf88b9edfe0a2c21bcf50baa4f0df` |
| `evidence/baselines.json` | `08b48346da8685398e293a320f1124b5d069344a265209279571570710b4c571` | `a4b55671c26187bb2aaf7c466eb990bf1164a7d9fe14dddc1b5f4bd8738f420e` |

## 3. Verification performed before applying — all six checks

| # | Check | Result |
|---|---|:--:|
| 1 | hash the existing manifest | ✅ `2fb9fc73…` |
| 2 | compare old versus proposed | ✅ 3 of 71 entries differ |
| 3 | identify exactly which entries differ | ✅ table above |
| 4 | every target is the intended finalized version | ✅ each file on disk is **byte-identical to `git show HEAD:<path>`** |
| 5 | affected files clean versus authoritative git state | ✅ same check — disk == HEAD for all three |
| 6 | does any scientific file need changing | ✅ **no** |

The decisive observation: all three targets are byte-identical to their
committed state. The **files** were never wrong; the **digests** were stale.
They were edited after the hash file was written and both states were committed
together. This reproduces on any machine and is machine-independent bookkeeping,
not a host artifact and not scientific damage.

The proposed manifest came from the retry
(`three_rung_promotion_validation_retry1/evidence/PROPOSED_two_branch_FINAL_SHA256.txt`)
but was **not copied blindly**: every entry was re-derived and re-checked here
before it was applied.

## 4. Why this is not "re-declaring a digest to pass a gate"

The distinction matters, because the finalization gate exists to prevent exactly
that move. A digest may be rewritten only when the target is provably the
intended finalized artifact and the mismatch is a bookkeeping lag. Here:

- the artifacts are tracked in git and **clean**;
- their content is reachable and verifiable from `HEAD` by anyone;
- nothing about the science changed.

Contrast with the five `.mat` **container** digests that remain mismatched
(`C160`, `C320`, `C400`, `F400`, `P400`). Those files are git-ignored, were
regenerated locally, and their originals are on another machine. Rewriting
*those* digests would assert that the local containers are the declared ones,
which is unverifiable from the repository. **They were deliberately left alone.**
See `PROMOTION_PROVENANCE.md` §3.

## 5. A second occurrence of the same defect — found, documented, NOT applied

`diagnostics/three_rung_architecture/EVIDENCE.json` records the **same stale**
`baselines.json` digest `08b48346…`, and that mismatch still fails its G2.

It was investigated rather than assumed:

```
git log -- .../evidence/baselines.json   ->   changed in 4dd5760 ("C320")
```

A semantic diff of the two versions shows the change is **purely additive** —
12 new keys, `old=<absent>` for every one, **no existing value modified**:

```
m160/m320 . rho_recovered, rho_recovered_file, rho_recovered_note,
            rho_recovered_sha256, rho_recovered_trajectorySha,
            rho_sha256_frozen_state
```

The file states its own provenance: *"Recovered after the freeze; frozen scalars
re-derived from this file and matched to <=1e-9. No frozen value was edited."*
So every scalar `three_rung_architecture` consumed is byte-identical in the
current file, and repairing that digest would assert nothing false.

**It was nevertheless not applied**, for two reasons:

1. The brief scopes Phase 4 explicitly — *"replace/regenerate **ONLY**
   FINAL_SHA256.txt"*. `EVIDENCE.json` is a different file in a different study.
2. It would change nothing: `three_rung_architecture`'s G2 still fails on the
   C160/C320/C400 container digests regardless, so the edit would be churn on a
   committed historical study for zero gate benefit, and it would cascade into
   that study's own `FINAL_SHA256.txt` (which hashes `EVIDENCE.json` as
   `47560482e8351d776fa5fcabe5c51722a8e72d96dca7f16448a7b1a9947183f6`).

Recorded here so a future task finds it already analysed. The same stale digest
also appears in `two_rung_architecture/EVIDENCE.json` and
`move_ladder_necessity/FINAL_SHA256.txt`, and in several
`evidence/provenance_start.json` / `provenance_final.json` snapshots — the
latter are **historical records of state at run time and must never be edited**.

## 6. Effect

Before the repair, `two_branch_controller_validation` G4 listed 10 mismatches.
After this repair **and** the Phase 5 settlement, it lists 5 — all of them
git-ignored `.mat` containers. `PROVENANCE.md`, `BASELINES.md`,
`evidence/baselines.json`, `runs/C320x40_iterations.csv` and
`runs/C320x40_record.json` have all cleared.
