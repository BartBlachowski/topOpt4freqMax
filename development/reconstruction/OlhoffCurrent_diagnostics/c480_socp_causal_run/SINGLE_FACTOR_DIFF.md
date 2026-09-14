# SINGLE_FACTOR_DIFF — Part 2

```
C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS
```

## Configuration: zero differences

`scripts/cs_config_diff.m` → `evaluations/single_factor_diff.json`.

| | control | treatment |
|---|---|---|
| cfg object | struct stored in the control trajectory | **the same struct**, passed unmodified |
| `isequal(cfg_control, cfg_treatment)` | — | **true** |
| config hash | `03097a28…782e` | `03097a28…782e` |
| fresh `cp_config(480,60)` hash | `03097a28…782e` | — |
| schema rows compared | 81 | 81 |
| rows that differ | — | **0** |

## The one scientific difference

| factor | control | treatment |
|---|---|---|
| `treat.innerSolver` | `repeatedMMA`: `+impl/algo/innerLoop.m` → `mma_published/mmasub.m` | `exactSOCP`: `scripts/cs_socp_inner.m` (`fp_problem` SOC; `coneprog` schur, then augmented; `cs_socp_certify`) |

It is carried outside `cfg` because the canonical schema has no inner-SOCP field and
production may not be edited. Under `exactSOCP`, the fields
`optimizer.inner.{type, variant, tolerance, minIterations, maxIterations}` stay in
`cfg` unchanged but are **inert**. `hist.nInner` then records interior-point
iterations, not MMA sub-iterations. It is telemetry, and no controller reads it.

## Driver source: every change tagged, production reconstructed exactly

`scripts/cs_diff_audit.py` → `evaluations/diff_audit.json`; the full unified diff
is `evaluations/olhoffSolve_vs_study_driver.diff` (10 hunks).

| | |
|---|---|
| production `+impl/architecture/olhoffSolve.m` | SHA-256 `1e5a114c…ba3`, 647 lines, **unmodified** |
| study `scripts/cs_olhoffSolveSOCP.m` | SHA-256 `48f5fa77…084`, 720 lines |
| rebuilt production = copy − tagged lines + `%CS-ORIG%` lines | **exact** |
| replaced production lines | 3: the function signature, `for outer = 1:maxOuter`, `if innerLP` |

| tag | lines | what |
|---|---|---|
| signature | 7 | name, `treat` argument, header comment |
| dispatch | 9 | `if exactSOCP → cs_socp_inner … elseif innerLP` (production branches verbatim below it) |
| failclosed | 24 | rejection → no update + `break`; NaN/Inf checks after FE, gradients, update; status override |
| checkpoint | 18 | resume load, `startOuter`, checkpoint save every 25 iterations |
| identity | 6 | iteration-1 FE spectrum must equal the control's `hist.omega(:,1)` bitwise |
| telemetry | 5 | SOCP record capture, `res.socp`, `res.treat` |
| preflight | 4 | `stopAfter` hook (0 in the treatment) |

No numeric expression of the outer loop is touched. FE, eigensolve, multiplicity,
generalized gradients, offsets, filter, move controller, clamp update, exhaustion
detector, stop logic and final analysis are all untouched. The reused formulation
and certificate files `fp_problem.m`, `fp_dualbound.m` and `fp_kkt.m` are
**byte-identical** copies of the certified reference study's scripts (SHA-256
`5ae495fe…`, `f643384c…`, `7a632985…`).

## Preflight parts

| part | result | evidence |
|---|---|---|
| P1 config identity | PASS | hashes above |
| P2 source-diff audit | PASS | exact reconstruction; only permitted categories |
| P3 control replay | PASS | study driver in `repeatedMMA` mode, 3 outer iterations: DRHO, RHO, hist.omega, hist.beta, hist.nInner, exAmp, exCos **bitwise** equal to the control |
| P4 oracle known-answer | PASS (after Amendment 1) | schur primary reproduces the certified oracle bitwise: \|Δbs\| = 0, d2 = 0, dinf = 0, gap 6.45e-12. The original augmented primary FAILED the design-space bars (d2 0.0107, dinf 0.745). See `PREREGISTRATION_AMENDMENT_1.md` |
| P5 fail-closed tests | PASS | N=3 → E1; inconsistent offsets → E3,E4; asymmetric F → E2; offDiag=false → E6; volFun → E7 (all rejected **before** solving); infeasible x (row 1e-6) and suboptimal feasible x (gap 1.79e-3) rejected by the certificate; the oracle itself certifies (gap 1.46e-12) |
| P6 toy smoke test 96×12 | PASS | 6 certified SOCP iterations; resume 3+3 = straight 6 **bitwise** (ρ, hist, Δρ); evidence/CSV/record pipeline without error. Outputs excluded from all analyses |
| P7 `+impl` tree | PASS | `edbfe47e…52cb` |

The treatment code (19 MATLAB files) was hashed into `evaluations/preflight.json`
after preflight. The runner re-verified all 19 at launch.
