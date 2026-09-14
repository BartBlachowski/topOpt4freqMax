# PROMOTION_PROVENANCE — Part H, the stricter gate

This gate is **stricter than the run-permission gate** of `PROVENANCE.md`.
Passing Part G does not open it. Its five requirements were fixed in
`PREREGISTRATION.md` §1 **before** the scientific run, so its outcome is not
contingent on what the run showed.

## H1 — original C240 evidence transfer status

```
analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat
  required   131 203 128 bytes
  sha256     183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d
  on this host   ABSENT  (parent directory does not exist)
```

Classification: **`REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL`** — the artifact was
produced on a different machine and `evidence/` is git-ignored wholesale, so it
never travelled. It is **not** lost and is **not** re-run.

The brief's own rule applies without interpretation:

> Preferred: original `C240x30_trajectory.mat` copied from the producing machine
> and verified against its original authoritative digest.
> **If not local: promotion remains blocked.**

It is not local, and it cannot be made local from this host — no copy exists in
`~/.Trash`, in a full-disk search, or in a Time Machine backup here. A file
transfer from the producing machine is required, and that is outside what this
task can perform.

# **H1 — FAIL (blocking)**

## H2 — stale `two_branch_controller_validation/FINAL_SHA256.txt`

Three entries genuinely stale against the study's own committed, clean files.
Fully diagnosed, with a repaired file generated and verified, and with proof
that the underlying scientific files were not modified by anyone:
`PROVENANCE_REPAIR_STATUS.md` §2.

```
prepared   evidence/PROPOSED_two_branch_FINAL_SHA256.txt
           sha256 a7ef91f9cc279b46288f25f6f453495c5d358d96e1ff5ad33a8d87e61c73bd3e
current    diagnostics/two_branch_controller_validation/FINAL_SHA256.txt
           sha256 2fb9fc731d816e5c47c9835d5f777f8bb3be2fe61773ca00119049ed2bf5e0e0
changed    3 of 143 lines — PROVENANCE.md, BASELINES.md, evidence/baselines.json
```

**Prepared, not applied.** Since H1 blocks promotion regardless, writing into a
committed historical study would leave an unrequested diff against a study whose
gate still cannot pass. A second, equally mechanical edit is also required —
`three_rung_architecture/EVIDENCE.json` declares the same stale
`baselines.json` digest — and is called out so it is not discovered later.

# **H2 — NOT CLOSED (repair prepared and verified)**

## H3 — `tOuter`-only tracked drift

Settled without changing any scientific state, and in a way that makes it
incapable of affecting this study's result:

1. `PREREGISTRATION.md` §6 excluded `tOuter` from bitwise scientific
   equivalence **before** the run, on the stated ground that it is
   nondeterministic wall-clock telemetry which `olhoffSolve.m` says nothing
   reads back.
2. The oracle CSV for Part E was taken from **git HEAD**
   (`evidence/oracle_C320x40_iterations_HEAD.csv`, `ff570d6e4d024f36…` — the
   digest `C320_ORACLE.md` and `three_rung_architecture/EVIDENCE.json` both
   declare), never from the dirty working tree.

Note that `FINAL_SHA256.txt` records the **HEAD** digests for both files, so the
hash file is correct and the working copy is what drifted. The two dirty paths
were left exactly as found: restoring them from HEAD would discard the user's
uncommitted local timings, which is the user's call, not this task's.

# **H3 — DOCUMENTED AND SETTLED for this study; the working-tree drift itself remains for the owner to commit or revert**

## H4 — repository finalization gate G1–G5, all load-bearing studies

Run with the repository's own `olhoffcurrent_finalization_gate`, which resolves
the mixed study-relative / repo-relative path convention. Full output:
`evidence/gates.json`.

| Study | G1 | G2 | G3 | G4 | G5 | result |
|---|:--:|:--:|:--:|:--:|:--:|---|
| `two_branch_controller_validation` | ✅ | ❌ | ✅ | ❌ | ✅ | **FAIL** |
| `three_rung_architecture` | ✅ | ❌ | ✅ | ✅ | ✅ | **FAIL** |
| `three_rung_resolution_240` | ✅ | ❌ | ✅ | ✅ | ✅ | **FAIL** |

The failures are exactly the classes already enumerated, and nothing else:

- **G2** everywhere — `REQUIRED_HASH_MISMATCH` on the `C160/C320/C400`
  containers (Class A) and, for `three_rung_resolution_240`,
  `REQUIRED_MISSING` on the C240 trajectory (Class B);
- **G4** on `two_branch_controller_validation` only — the three Class C stale
  entries, plus the two `tOuter`-dirty working-tree paths.

# **H4 — FAIL**

## H5 — `test_finalization_gate`

```
test_currentness           0 failures
test_source_integrity      0 failures
test_path_isolation        0 failures
test_preset_equivalence    0 failures
test_evidence_retention    0 failures
test_finalization_gate     4 failures
```

The gate's own self-tests **A–G all pass** — it is operational and fails closed.
Cases **H** and **I** fail, for the same three provenance classes. This is the
same regression the stopped attempt reported, unchanged.

# **H5 — FAIL**

## Verdict

| | |
|---|---|
| H1 C240 transfer | **FAIL — blocking, and not closable from this host** |
| H2 stale `FINAL_SHA256` | NOT CLOSED (repair prepared, verified, not applied) |
| H3 `tOuter` drift | settled for this study; working-tree state left to the owner |
| H4 G1–G5 | **FAIL** |
| H5 `test_finalization_gate` | **FAIL** |

# `PROMOTION_PROVENANCE_BLOCKED`

Therefore, and regardless of the scientific outcome:

```
THREE_RUNG_POLICY_VALIDATED_BUT_PROMOTION_PROVENANCE_BLOCKED
PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED
NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED
```

**The C320 scientific validation is NOT discarded.** It stands on its own
evidence, it is complete, and it remains valid for reuse once provenance is
closed. C320 must not be re-run for it. What is required before promotion is a
**file transfer and two hash-file edits** — no compute, no optimization, no
re-derivation.
