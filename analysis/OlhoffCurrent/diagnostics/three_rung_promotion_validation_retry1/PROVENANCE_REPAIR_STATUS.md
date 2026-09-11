# PROVENANCE_REPAIR_STATUS — what is broken, what was repaired, what was not

Three distinct defects, kept apart (`PROVENANCE.md` §6). Each is stated with its
class, its status, and — where this retry did not apply the repair — the exact
reason and the exact steps to close it.

## Summary

| # | Defect | Class | Status after this retry |
|---|---|---|---|
| 1 | `C240x30_trajectory.mat` absent on this host | B | **OPEN — cannot be closed here.** Requires a file copy from the producing machine. |
| 2 | `two_branch_controller_validation/FINAL_SHA256.txt` stale, 3 entries | C | **REPAIR PREPARED AND VERIFIED, NOT APPLIED** — see §2 |
| 3 | `tOuter`-only drift in 2 tracked working-tree files | — | **DIAGNOSED AND DOCUMENTED, NOT APPLIED** — see §3 |
| 4 | `C160/C320/C400` `.mat` container digests stale | A | **OPEN — owner decision required**, see §4. Scientifically immaterial for C320, which is verified to the bit. |

## 1. Defect 1 — C240 raw evidence (Class B)

```
analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat
  status  REQUIRED_MISSING (parent directory does not exist on this host)
  bytes   131 203 128
  sha256  183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d
```

Classification: **`REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL`**, not loss. The artifact
was produced by the prior `three_rung_resolution_240` task **on a different
machine**, and `analysis/OlhoffCurrent/evidence/` is git-ignored wholesale, so
it never travelled. `evidence/.gitignore` states the expected consequence
outright: *"On a fresh clone this directory is empty and the gate reports
REQUIRED_MISSING. That is the correct, honest answer."*

**Not re-run**, by direction and on merit — re-running would produce a different
container and would not restore the declared digest anyway.

**To close:** copy the file from the producing machine into
`analysis/OlhoffCurrent/evidence/three_rung_resolution_240/` and verify it
against the digest above, which is already declared in that study's
`EVIDENCE.json`. No regeneration.

This alone blocks promotion (Part H, H1), independently of everything else in
this retry.

## 2. Defect 2 — stale `FINAL_SHA256.txt` (Class C)

Three entries in `two_branch_controller_validation/FINAL_SHA256.txt` disagree
with the study's own **committed and clean** files. Both the files and the hash
file are committed, so this reproduces on **every** machine — it is genuine
bookkeeping staleness, not a host artifact:

| Entry | recorded | on disk **= git HEAD** |
|---|---|---|
| `PROVENANCE.md` | `42037f8196cf9768477fb0473ec8eed45445c5da23dec2649774cb093ca90790` | `0e987a9bc7b39425661ffcee0cd81ebc5fa88010d35e094daff8d4df35459a33` |
| `BASELINES.md` | `74629a1634bb5452ea1e91161e9e652acf51a5d431ee156767ee339dd2299b21` | `2212a66a50c62f118c0a17047c8e2c8fa55bf88b9edfe0a2c21bcf50baa4f0df` |
| `evidence/baselines.json` | `08b48346da8685398e293a320f1124b5d069344a265209279571570710b4c571` | `a4b55671c26187bb2aaf7c466eb990bf1164a7d9fe14dddc1b5f4bd8738f420e` |

**Proof the underlying scientific files were not modified by anyone, including
this task:** each of the three hashes to the same value on disk as `git show
HEAD:<path>` produces. They are byte-identical to what was committed. The
documents were edited *after* the hash file was written and both states were
committed together; nothing has been touched since.

### The prepared repair

```
evidence/PROPOSED_two_branch_FINAL_SHA256.txt
  sha256  a7ef91f9cc279b46288f25f6f453495c5d358d96e1ff5ad33a8d87e61c73bd3e
current file
  sha256  2fb9fc731d816e5c47c9835d5f777f8bb3be2fe61773ca00119049ed2bf5e0e0
```

Exactly **3 of 143 lines** change — only the three digests above. Line count,
ordering, path spellings, byte-size annotations, `MISSING` markers and every
other entry are preserved verbatim.

### Why it was prepared but NOT applied

Promotion is blocked by Defect 1 regardless of this repair, and applying it
would write into a **committed historical study** that this task was told not to
rewrite. Doing so would leave the working tree carrying an unrequested diff
against a study whose gate still cannot pass. The repair is therefore delivered
as a verified artifact, ready to drop in, together with the audit trail above.

**To apply** (one command, after Defect 1 is closed):

```
cp analysis/OlhoffCurrent/diagnostics/three_rung_promotion_validation_retry1/evidence/PROPOSED_two_branch_FINAL_SHA256.txt \
   analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/FINAL_SHA256.txt
```

Note that `three_rung_architecture/EVIDENCE.json:` also declares the **old**
`baselines.json` digest `08b48346…`, so closing this fully requires
re-declaring it there too. That is a second, equally mechanical edit, and it is
called out here so it is not discovered later as a surprise.

## 3. Defect 3 — `tOuter`-only working-tree drift

Two tracked files are dirty, and were dirty **before this task started**:

```
 M diagnostics/two_branch_controller_validation/runs/C320x40_iterations.csv
 M diagnostics/two_branch_controller_validation/runs/C320x40_record.json
```

| File | working tree | git HEAD | what `FINAL_SHA256.txt` records |
|---|---|---|---|
| `C320x40_iterations.csv` | `6ac72fa880f2e9fb…` | `ff570d6e4d024f36…` | `ff570d6e4d024f36…` — **HEAD** |
| `C320x40_record.json` | `765604214a4074ab…` | `3fc68c050a9d81b5…` | `3fc68c050a9d81b5…` — **HEAD** |

So the hash file is **correct**; the working copy is what drifted. The prior
attempt established the drift is `tOuter`-only (the 54-column scientific
projection is byte-identical, `5f9f1896…8f6bb53d`) and that the JSON deltas are
`wall_s`, `trajectoryBytes` and the MATLAB version string — consistent with the
C320 run having been re-executed locally and **reproducing its science
bit-for-bit**.

**Settled without changing any scientific state, in two ways:**

1. **Declared out of scope for acceptance.** `PREREGISTRATION.md` §6 excludes
   `tOuter` from bitwise scientific equivalence *before* the run, on the stated
   ground that it is nondeterministic wall-clock telemetry that
   `olhoffSolve.m` itself says nothing reads back.
2. **Routed around entirely.** The oracle CSV used for Part E was extracted from
   **git HEAD**, not the working tree
   (`evidence/oracle_C320x40_iterations_HEAD.csv`, `ff570d6e…`), so the drift
   cannot contaminate the comparison even in principle.

**Not applied:** `git checkout` of the two paths would discard the user's
uncommitted local re-run timings. That is the user's call, not this task's.
Either action — committing them or restoring them from HEAD — closes the
G2/G4 entries they currently break.

## 4. Defect 4 — regenerated containers (Class A)

```
C160x20_trajectory.mat   recorded 4d11a2fd…   on disk 81244cf5…
C320x40_trajectory.mat   recorded c9d4d766…   on disk 4892c10e…
C400x50_trajectory.mat   recorded fa0e714c…   on disk 673be8c7…
```

These are **container** digests. The science inside them is verified: all three
final-`rho` digests match the values committed in git-tracked
`runs/*_record.json`, and for C320 this retry re-verified the prefix anchors
`RHO[:,1:352]` and `omega(1:2,1:352)` exactly (`PROVENANCE.md` §A1). A `.mat`
container is not byte-reproducible across writes, and `evidence/` is git-ignored,
so a locally regenerated copy will never match a digest recorded elsewhere.

**Owner decision required**, and it is genuinely a choice, not a repair:

- **(a)** transfer the original containers from the machine that produced them,
  so the recorded digests match again; or
- **(b)** re-declare the local containers, recording explicitly that the science
  was verified unchanged against the committed `rho_sha256` values.

This retry takes neither option. Option (b) in particular must not be taken
silently: re-declaring a digest to make a gate pass is exactly the move the
finalization gate exists to prevent, and it is defensible only as a deliberate,
documented decision by the repository owner.

## Bottom line

Defect 1 is the binding constraint and cannot be closed on this host. Defects 2
and 3 are fully diagnosed with repairs prepared and verified; defect 4 needs a
decision, not a fix. None of them touches the C320 scientific oracle, which is
verified to the bit and was the only prior evidence this retry's scientific run
actually consumed.
