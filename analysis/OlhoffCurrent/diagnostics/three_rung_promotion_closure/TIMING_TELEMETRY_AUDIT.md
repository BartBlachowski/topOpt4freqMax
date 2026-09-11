# TIMING_TELEMETRY_AUDIT — Phase 5 and the frozen timer interpretation

# `TIMING_ONLY_DRIFT_SETTLED_PASS`

## 1. The frozen interpretation

Four fields of `res.hist` are `toc`-derived nondeterministic timing telemetry:

| Field | produced at | read back by the optimizer? |
|---|---|---|
| `tEig` | `olhoffSolve.m:215 → 405` | **no** |
| `tGrad` | `olhoffSolve.m:287 → 406` | **no** |
| `tInner` | `olhoffSolve.m:351 → 407` | **no** |
| `tOuter` | `olhoffSolve.m:558` | **no** — the source comment states *"Nothing reads it back."* |

All four are **EXCLUDED** from scientific bitwise-equivalence requirements.

`tInner` is the **wall time** of the inner solve, not inner **work**; inner work
is `nInner`/`cumInner`, which are deterministic and were compared exactly in the
validated run (`cumInner = 6498` at S3, identical to the oracle).

### Both facts from the retry are preserved

The retry disclosed an enumeration defect in its own preregistration: it named
`tOuter` but not the other three, because it enumerated against the 55-column
telemetry CSV, where `tOuter` is the only timer column. Mechanical checking
established those three are the **only** additional differences. Both readings
stand, and neither is rewritten:

```
literal enumerated-list reading   : mismatch
category / scientific-state read  : PASS
```

`three_rung_promotion_validation_retry1/PREREGISTRATION.md` is **unchanged**
(`6b0638d987b244f6be609d2eb7cb619248bfab1b25ace15d6600b6107fa5345d`), with its
incomplete list intact. These timers may not be used to question the validated
C320 result.

## 2. The two timing-drifted tracked paths

```
analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_iterations.csv
analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_record.json
```

Both were dirty **before** this task and before the retry. Neither was created
or modified by either.

## 3. Exhaustive verification, cell by cell

### `C320x40_iterations.csv`

Compared against `git show HEAD:…` over **every cell** — 55 columns × 1600 rows:

| Check | Result |
|---|---|
| header identical | ✅ |
| row count | 1600 = 1600 |
| **columns containing any differing cell** | **`tOuter` only** (1600 of 1600 rows) |
| all other 54 columns | **0 differing cells** |
| 54-column scientific projection | **byte-identical** |
| projection SHA-256 | `5f9f18960dfc4a990c15d1b101b256e006336a8d6ca41f560ec6d6b58f6bb53d` |

This is `tOuter`-only in the strict sense.

### `C320x40_record.json`

Compared leaf by leaf after flattening. **Exactly 3 leaves differ**, and none is
scientific state:

| Leaf | HEAD | working copy | class |
|---|---|---|---|
| `matlab` | `25.2.0.2998904 (R2025b)` | `25.2.0.3042426 (R2025b) Update 1` | toolchain version string |
| `trajectoryBytes` | `283657283` | `283658112` | `.mat` **container** size |
| `wall_s` | `34373.83334995833` | `37410.57288875` | wall-clock runtime |

Identical, therefore: `status`, `nOuter`, `innerTotal`, `innerMax`,
`innerNonConv`, `stage_final`, `move_final`, `cfgHash`, `implTree`,
`rho_sha256`, `omega`, `gap12`, `volume_final`, `Mnd_final`, `gray_final`,
`mid_final`, `tol`, `cap`, and the complete exhaustion/event log.

**Precision matters here and the brief's phrasing is slightly narrowed:** the
JSON drift is *not* strictly "`tOuter`-only". It is timing (`wall_s`) **plus**
container metadata (`trajectoryBytes`) **plus** a toolchain version string.
None of the three is scientific state, so the settlement is justified — but the
label "tOuter-only" would be inaccurate for this file and is not used.

**No unique scientific state exists in either working copy.** The two together
record that the C320 run was re-executed locally under R2025b Update 1 and
reproduced `rho_sha256` and every event exactly — a fact of provenance interest,
now preserved in §4 and in the retry's `PROVENANCE.md`, not of scientific
content.

## 4. How they were settled

**Archived first, then restored** — nothing was destroyed:

```
evidence/preserved_working_copies/C320x40_iterations.WORKING_COPY.csv
    6ac72fa880f2e9fb545afcabbd092fe6be32bc132845e55d61b6d5baa4325a22
evidence/preserved_working_copies/C320x40_record.WORKING_COPY.json
    765604214a4074ab133d131befec243d93fd502de404570be76d4a5e78c28e55
```

then

```
git checkout -- <the two paths>
```

restoring the authoritative committed versions:

| Path | digest after restore | recorded in `FINAL_SHA256.txt` / `EVIDENCE.json` |
|---|---|---|
| `runs/C320x40_iterations.csv` | `ff570d6e4d024f360974ab065c3042127404b7073a66ed972bfd8718563b356c` | **same** |
| `runs/C320x40_record.json` | `3fc68c050a9d81b5a6ad6ab482abda9679d5cd5f2455967dc4a3a8b64dc236be` | **same** |

Note the direction of the defect: the **hash file was already correct** and
recorded the HEAD digests; the *working copy* was what had drifted. Restoration
therefore cleared these entries from both the G2 and G4 mismatch lists.

The tracked working tree is now clean. No historical telemetry migration was
attempted and none is implied: the policy remains that **timing telemetry is
nondeterministic and scientific state is deterministic**.
