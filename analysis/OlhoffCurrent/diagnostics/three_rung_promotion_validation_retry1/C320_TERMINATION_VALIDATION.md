# C320_TERMINATION_VALIDATION — Part F

## 1. The run stopped where the oracle descended

```
status      CONVERGED
nOuter      352
stage_final 3
move_final  0.01
innerTotal  6498   (innerMax 40, innerNonConv 0)
wall        1486.9 s
cfgHash     afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab
implTree    edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb
rho_sha256  0c928dc01f56ee25f19b688e63b3844141f71e3130d3a03b3f9e4255659bb925
```

At iteration 352 the frozen detector declared `E` on **Branch B** with full
persistence (`declBegin 333`, `declIter 352`, `P = 20`). With
`move.levels = [0.04 0.02 0.01]`, `atLastLevel = (3 >= 3)` is **true**, so
`convOuter = ex.declared && atLastLevel` is true and the run terminated.

| Requirement | Result |
|---|:--:|
| reports `CONVERGED` | ✅ |
| terminates at exactly 352 | ✅ |
| does **not** execute a scientific iteration 353 | ✅ — `nOuter = 352` |
| never enters `move = 0.005` | ✅ — moves visited are exactly `{0.04, 0.02, 0.01}` |
| final stage ≤ 3 | ✅ — `max(stage) = 3` |
| no inner solve failed | ✅ — `innerNonConv = 0` |
| did not hit the cap | ✅ — `CONVERGED`, not `CAP_HIT`; cap was 1600 |

`rho_sha256` differs from the oracle's `0348b288…` and must: the oracle's final
design is at **iteration 1600, stage 4**, after 1248 further iterations on the
removed rung. The correct comparison at 352 is the prefix anchor, and it matches
exactly (`C320_PREFIX_EQUIVALENCE.md` §1).

## 2. Terminal scientific state — 13 of 13 exact

Compared against the values frozen in `C320_ORACLE.md` before the run:

| Quantity | Oracle | Obtained | |
|---|---|---|:--:|
| `omega1` | `166.42726927757769` | `166.42726927757769` | ✅ |
| `omega2` | `203.58101247948915` | `203.58101247948915` | ✅ |
| `gap12` | `0.22324312213489572` | `0.22324312213489572` | ✅ |
| `volume` | `0.49999874883108736` | `0.49999874883108736` | ✅ |
| `Mnd` | `12.940093529981478` | `12.940093529981478` | ✅ |
| `gray` | `0.15296874999999999` | `0.15296874999999999` | ✅ |
| `mid` | `0.030156249999999999` | `0.030156249999999999` | ✅ |
| `move` | `0.01` | `0.01` | ✅ |
| `stage` | `3` | `3` | ✅ |
| `multN` | `2` | `2` | ✅ |
| `nInner` | `19` | `19` | ✅ |
| `innerConv` | `1` | `1` | ✅ |
| `cumInner` | `6498` | `6498` | ✅ |

Equality is tested with `==`, not a tolerance. All 13 hold.

# `C320_THREE_RUNG_TERMINATION_PASS`

## 3. What the removed rung cost

| Metric | Four-rung oracle | Three-rung candidate | Eliminated |
|---|---|---|---|
| outer iterations | 1600 (`CAP_HIT`) | **352 (`CONVERGED`)** | **1248 — 78.00 %** |
| cumulative inner MMA | 76 532 | **6 498** | **70 034 — 91.51 %** |

`three_rung_promotion_validation/CONTROLLER_RECOVERY.md` §8 predicted `CONVERGED
@352`, −1248 outer (78.00 %) and −70 034 inner (91.51 %). **The prediction is
confirmed to the digit**, and it is now a validated result at this mesh rather
than a derivation.

The asymmetry is worth stating carefully. The fourth rung did not merely cost
1248 iterations — it **never terminated at all** within the preregistered cap of
1600, while producing no change whatsoever to the design that the third rung had
already reached at 352. That is the operational pathology the removal addresses.

## 4. Scope of the claim

This is **one mesh**. 320×40 is the only mesh at which the three-rung policy has
been validated end to end, and nothing here establishes a mesh law. The
architecture evidence supporting `[0.04 0.02 0.01]` across meshes is the
pre-existing `three_rung_architecture` and `three_rung_resolution_240` work,
which this study consumed as an input and did not re-derive.

No outcome-driven repair was performed. Nothing was tuned, relaxed or retried
after the run began; the criteria above are those `PREREGISTRATION.md` §5 fixed
before the result existed.
