# C320_ORACLE — the four-rung C320 trajectory, verified

The oracle this task would have validated against. **Verified intact and
hash-valid**, independently of the finalization-gate failures described in
`PROVENANCE.md`. Frozen here so a re-attempt need not repeat the verification.

## 1. Identity

| Field | Value |
|---|---|
| source study | `diagnostics/two_branch_controller_validation` |
| arm | `C` — candidate, frozen two-branch stage exhaustion, `E = A OR B` |
| config | `duOlhoffFrozenM4` + `{move.continuation.signal, stop.rule} = 'stageExhaustion'` |
| move ladder | `[0.04 0.02 0.01 0.005]` (four rungs) |
| mesh | 320×40, NE = 12800, `tol = 0.1` |
| cap | 1600 |
| status | **`CAP_HIT` @1600**, `stage_final = 4` |
| inner MMA total | 76 532 (`innerMax` 139, `innerNonConv` 0) |
| implTree | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| final rho | `rho_sha256 = 0348b288da711f2bdcd89263f7feaae33ccaedcb73a133a1a5e941421795bfb3` |

Scalar/telemetry evidence (git-tracked, HEAD `60f5b72`):
```
ff570d6e4d024f360974ab065c3042127404b7073a66ed972bfd8718563b356c  runs/C320x40_iterations.csv
```
Raw trajectory (git-ignored, on disk):
`analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C320x40_trajectory.mat`
— `RHO`, `DRHO` both `[12800 × 1600]` double, plus `cfg`, `exh`, `hist`, `log`, `meta`, `move`.

## 2. Verification performed in this task

The on-disk `.mat` differs from the container digest recorded by downstream
studies. It was therefore verified **against git-tracked scientific digests**:

1. **Recorded-digest check.** `local_vecHash(RHO(:,end))` =
   `0348b288…1795bfb3` = the `rho_sha256` committed in `runs/C320x40_record.json`
   at HEAD. **MATCH.**
2. **Cross-validation against the tracked CSV**, all 1600 iterations:

| Quantity | Comparison | Result |
|---|---|---|
| `max|drho|` | `max(abs(DRHO))` vs CSV `maxAbs` | **bitwise identical**, 0 differing elements |
| `omega1` | `hist.omega(1,:)` vs CSV `omega1` | **bitwise identical** |
| `omega2` | `hist.omega(2,:)` vs CSV `omega2` | **bitwise identical** |
| `move` | `hist.move` vs CSV `move` | **bitwise identical** |
| `stage` | `hist.stage` vs CSV `stage` | identical (max abs dev 0) |
| `exE` | `hist.exE` vs CSV `exE` | identical (max abs dev 0) |
| `cumInner` | `hist.cumInner` vs CSV `cumInner` | identical (max abs dev 0) |
| `‖drho‖₂` | recomputed vs CSV `l2`/`exAmp` | max rel dev 1.83e−15 (summation order) |
| `volume` | `mean(RHO)` vs CSV `volume` | max rel dev 3.22e−15 (summation order) |

`maxAbs` being bitwise identical is decisive: `max` is an exact selection, so
`DRHO` itself is bit-identical. The 1e−15 residuals on `l2`/`volume` are
recomputation summation-order effects, not data differences.

3. **Working-tree drift.** The dirty `runs/C320x40_iterations.csv` differs from
   HEAD **only** in the `tOuter` column; the 54-column scientific projection is
   byte-identical (`5f9f1896…8f6bb53d` for both).

**Conclusion: the C320 oracle is sound.** Its failure to satisfy the
finalization gate is a container-digest staleness issue, not evidence damage.

## 3. Frozen event structure

From `exhaustion` in the tracked `runs/C320x40_record.json`:
```
W = 20, P = 20, Wnp = 10, tol = 0.1
stageStarts : [1, 275, 314, 353]
descents    : [[275,1,274,255], [314,2,313,294], [353,3,352,333]]
              (= [iterApplied, stageFrom, declIter, declBegin])
```

| Event | declBegin | **declIter (S)** | move at decl | branch | descent applied | new move |
|---|---|---|---|---|---|---|
| S1 | 255 | **274** | 0.04 | **A** | 275 | 0.02 |
| S2 | 294 | **313** | 0.02 | **B** | 314 | 0.01 |
| S3 | 333 | **352** | 0.01 | **B** | 353 | 0.005 |

Detector state at each declaration (from the tracked CSV):

```
iter 274: stage=1 move=0.04  exA=1 exB=0 exE=1 exNA=20 exNB=0 exDecl=1 stageStart=1
iter 313: stage=2 move=0.02  exA=0 exB=1 exE=1 exNA=0 exNB=20 exDecl=1 stageStart=275
iter 352: stage=3 move=0.01  exA=0 exB=1 exE=1 exNA=0 exNB=20 exDecl=1 stageStart=314
```

These match the task brief's stated endpoints (S1=274 A, S2=313 B, S3=352 B)
exactly. **The oracle is `declIter`** — 274 / 313 / 352 — not the descent
iteration and not the window start.

## 4. Frozen expected S3 terminal state (iteration 352)

The three-rung candidate's terminal state must equal this:

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

Exactness anchors computed from the verified trajectory:
```
RHO[:, 1:352]        sha256 = b8c0f18d887c7f5a81ece452f0a4097a0cc1a67710a137d66e677fcfd78b5ba3
omega(1:2, 1:352)    sha256 = fd63608399edaede2c4f928a25d5478eacc3711f6d626ae770ad107e7f4c488d
```
(`local_vecHash` convention: SHA-256 over `typecast(double(v(:)),'uint8')`,
column-major.)

## 5. Predicted three-rung outcome — from existing evidence only

Per `CONTROLLER_RECOVERY.md` §8, ladder length is inert until `stage == 3`.
At iteration 352, `atLastLevel = (3 >= 3)` becomes **true**, so
`convOuter = declared && atLastLevel = true` and the run terminates
**`CONVERGED @352`** instead of descending to `move = 0.005`.

Cost that the fourth rung consumes at this mesh (derived from the tracked CSV,
no new run required):

| Metric | Four-rung | Three-rung (predicted) | Eliminated |
|---|---|---|---|
| outer iterations | 1600 (`CAP_HIT`) | 352 (`CONVERGED`) | **1248 — 78.00%** |
| cumulative inner MMA | 76 532 | 6 498 | **70 034 — 91.51%** |

This is a **prediction from existing evidence**, not a validated result. It was
NOT confirmed: the authorized C320 three-rung run was never executed, because
the task stopped at Phase 1. It must not be cited as a validated outcome.
