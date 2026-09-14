# C320_ORACLE_REUSE — reusing the frozen oracle, and what was re-checked

The C320 oracle was **not regenerated** and the old four-rung C320 run was
**not re-executed**. This document records exactly what was reused, what was
re-verified, and why the reuse is sound.

## 1. What is reused

`diagnostics/three_rung_promotion_validation/C320_ORACLE.md`
— SHA-256 `d09579b80c0d187b3800e71e22538a1c482e60d79505d44457ffc9d57e8b82d9`,
**unchanged** from the stopped attempt (it matches that study's own
`FINAL_SHA256.txt` entry byte for byte).

Its source is arm `C` of `two_branch_controller_validation` at 320×40:
frozen two-branch stage exhaustion `E = A OR B`, ladder
`[0.04 0.02 0.01 0.005]`, cap 1600, result **`CAP_HIT @1600`**,
`stage_final = 4`, `implTree edbfe47e…`, `rho_sha256 0348b288…`.

## 2. What this retry re-checked — and what it deliberately did not

The stopped attempt already established, cryptographically:

1. the final `RHO` hash matches the git-tracked `rho_sha256`;
2. across all 1600 iterations `maxAbs`, `omega1`, `omega2`, `move`, `stage`,
   `exE` and `cumInner` match the tracked scientific CSV exactly / bitwise;
3. `l2` and `volume` recomputations differ only at ~1e−15 from summation order;
4. the dirty-CSV drift is `tOuter`-only;
5. stripping `tOuter` gives byte-identical scientific CSV content.

`C320_ORACLE_SCIENTIFICALLY_VERIFIED` is therefore an **already-established
premise**, and no substantial compute was spent re-proving it. Per the brief,
only currentness and hashes were re-checked.

### Re-checked (cheap, decisive)

| Check | Result |
|---|---|
| `C320_ORACLE.md` digest unchanged | ✅ `d09579b8…7e8b82d9` |
| raw trajectory present on this host | ✅ `evidence/two_branch_controller_validation/C320x40_trajectory.mat`, `RHO` 12800×1600 |
| `local_vecHash(RHO(:,end))` | ✅ `0348b288…1795bfb3` |
| **`local_vecHash(RHO(:,1:352))`** | ✅ `b8c0f18d887c7f5a81ece452f0a4097a0cc1a67710a137d66e677fcfd78b5ba3` |
| **`local_vecHash(omega(1:2,1:352))`** | ✅ `fd63608399edaede2c4f928a25d5478eacc3711f6d626ae770ad107e7f4c488d` |
| trajectory's own `meta.implTree` | ✅ `edbfe47e…` = the tree on disk now |
| `+impl` manifest | ✅ verified, 75 files, `CURRENT` |
| oracle arm config reconstructs | ✅ `cfgHash 2359a111…` = the committed `cfgHash` |

### Not repeated

The full 1600-iteration cross-validation of the tracked CSV against the
container. It was performed once, recorded, and its conclusion is inherited.

## 3. Frozen event structure — re-read from the raw container

Read from `exh` inside the `.mat` (not only from the tracked JSON):

```
W = 20, P = 20, Wnp = 10, tol = 0.1
stageStarts : [1, 275, 314, 353]
descents    : [275 1 274 255 ; 314 2 313 294 ; 353 3 352 333]
              ( = [iterApplied, stageFrom, declIter, declBegin] )
eventBranch : A  B  B
```

| Event | declBegin | **declIter (S)** | move at decl | branch | four-rung descent applied |
|---|---|---|---|---|---|
| S1 | 255 | **274** | 0.04 | **A** | 275 |
| S2 | 294 | **313** | 0.02 | **B** | 314 |
| S3 | 333 | **352** | 0.01 | **B** | 353 |

**The oracle is `declIter`** — 274 / 313 / 352 — not the descent iteration and
not the window start.

## 4. Frozen expected S3 terminal state (iteration 352)

Recomputed in this retry from the raw trajectory; identical to `C320_ORACLE.md`.

| Quantity | Value |
|---|---|
| `omega1` | `166.42726927757769` |
| `omega2` | `203.58101247948915` |
| `gap12` | `0.22324312213489572` |
| `volume` | `0.49999874883108736` |
| `Mnd` | `12.940093529981478` |
| `gray` | `0.15296874999999999` |
| `mid` | `0.030156249999999999` |
| `move` | `0.01` |
| `stage` | `3` |
| `multN` | `2` |
| `nInner` | `19`, `innerConv = 1` |
| `cumInner` | `6498` |

## 5. Why container-digest failure does not invalidate this oracle

The repository's evidence gate reports `REQUIRED_HASH_MISMATCH` for the C320
`.mat` **container** (recorded `c9d4d766…`, on disk `4892c10e…`). That is a
statement about the file wrapper, not about the numbers inside it. `evidence/`
is git-ignored wholesale, so these containers never travelled with git and the
copy on this host was regenerated locally; a `.mat` container is not
byte-reproducible across writes.

What *is* committed to git, and what this retry checks against, is the
**scientific digest**: `rho_sha256` in `runs/C320x40_record.json`, plus the
per-iteration scientific CSV. Every anchor derived from the container reproduces
the committed values exactly. The science is intact; the wrapper is stale.

Classification: `CONTAINER_DIGEST_STALE / HOST-SPECIFIC`, **not**
`SCIENCE_INVALID`. See `PROVENANCE.md` §6 Class A.

## 6. Prediction carried forward — and its status

`three_rung_promotion_validation/CONTROLLER_RECOVERY.md` §8 predicted that at
iteration 352 the three-rung run would report `CONVERGED` instead of descending
to `move = 0.005`, eliminating 1248 outer iterations (78.00 %) and 70 034 inner
MMA iterations (91.51 %).

That was a **prediction from existing evidence**, explicitly *not* a validated
result — the stopped attempt never ran it. This retry executes exactly that one
run. Until `C320_TERMINATION_VALIDATION.md` reports, the prediction must not be
cited as an outcome.
