# PREREGISTRATION — three_rung_promotion_validation_retry1

**Frozen before the C320 three-rung run was executed.** No trajectory, CSV or
record for the candidate existed on disk at the moment this file was written;
`FROZEN_BEFORE.txt` records the SHA-256 of this file and of every script that
will produce the result, together with proof that `runs/` and
`evidence/three_rung_promotion_validation_retry1/` were empty.

## 1. What this retry is

Attempt 1 (`diagnostics/three_rung_promotion_validation/`) stopped at Phase 1
with **`THREE_RUNG_PROMOTION_PROVENANCE_FAIL`**,
`PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED`,
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED`, and **scientific runs = 0**. That
result is preserved unchanged and is not rewritten.

It stopped because the repository's own finalization gate fails G2 on this host
for all three load-bearing studies, and `test_finalization_gate` reports the same
regression. This retry does not dispute that. It separates two questions the
stopped attempt had to answer together:

| | question | this retry |
|---|---|---|
| **scientific validation dependency** | is the specific evidence *this run consumes* verified? | **yes** — `DEPENDENCY_SPECIFIC_SCIENTIFIC_PROVENANCE_PASS` |
| **historical package finalization** | can every load-bearing study be re-verified end to end on this host? | **no** — and Part H keeps promotion blocked on it |

Why the first now permits the run: the C320 oracle is verified
cryptographically, on this host, against digests committed to git — final `rho`,
`RHO[:,1:352]` and `omega(1:2,1:352)` all reproduce exactly, and the oracle arm's
configuration reconstructs to the committed `cfgHash`. The failing gates are
container staleness (Class A), a file that lives on another machine (Class B)
and a stale hash file (Class C) — none of which is the evidence this run reads.
Details: `PROVENANCE.md` §6.

### Unresolved promotion-level provenance items, declared in advance

1. `C240x30_trajectory.mat` — `REMOTE_REQUIRED_EVIDENCE_NOT_LOCAL`. Cannot be
   satisfied on this host. **Promotion will be blocked by this regardless of
   how the run turns out.** It will not be re-run.
2. `two_branch_controller_validation/FINAL_SHA256.txt` — Class C stale entries.
3. `tOuter`-only drift in two tracked C320 files.
4. `C160/C320/C400` container digests — Class A.

Declaring these *before* the run means the promotion outcome is not contingent
on what the run shows.

## 2. The one authorized scientific run

```
tag    C320x40_three_rung
mesh   320 x 40        NE = 12800        tol = 0.05*sqrt(NE/3200) = 0.1
cap    1600
```

**Exactly one.** No other mesh, no repeat, no variant. `tr_run.m` hard-codes the
mesh rather than accepting it as an argument.

Software mechanics may use meshes below the 160×20 scientific floor (48×6 is
used); those are never interpreted and never cited as scientific results.

## 3. The frozen candidate

```matlab
% cv_config('C',320,40) -- the exact arm that produced the oracle -- replayed
% through its own recorded override list, plus ONE field.
cfg = olh.config.resolve('duOlhoffFrozenM4', ov{:}, ...
        'move.levels',  [0.04 0.02 0.01], ...
        'runtime.name', 'TR3_C_320x40');
```

```
move.levels  = [0.04, 0.02, 0.01]
transition   = stageExhaustion
E            = frozen A OR B          (W = 20, P = 20, Wnp = 10)
semantics    : move 0.04  hold until persistent E  -> descend
               move 0.02  hold until persistent E  -> descend
               move 0.01  hold until persistent E  -> CONVERGED
there is no production move = 0.005
beta         : optimization variable and diagnostic; NO continuation authority,
               NO terminal authority
```

Config hashes, frozen now: candidate `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab`;
four-rung baseline `2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4`
(= the `cfgHash` committed in `two_branch_controller_validation/runs/C320x40_record.json`).

## 4. The frozen oracle — predictions stated before the result

From `three_rung_promotion_validation/C320_ORACLE.md`
(SHA-256 `d09579b80c0d187b3800e71e22538a1c482e60d79505d44457ffc9d57e8b82d9`).

**Events** (the oracle is `declIter`, not the descent iteration):

```
S1  declIter 274  Branch A   (window 255-274, move 0.04)
S2  declIter 313  Branch B   (window 294-313, move 0.02)
S3  declIter 352  Branch B   (window 333-352, move 0.01)
```

**Exact anchors:**

```
RHO[:,1:352]       b8c0f18d887c7f5a81ece452f0a4097a0cc1a67710a137d66e677fcfd78b5ba3
omega(1:2,1:352)   fd63608399edaede2c4f928a25d5478eacc3711f6d626ae770ad107e7f4c488d
```

**Terminal state at 352:** `omega1 = 166.42726927757769`,
`omega2 = 203.58101247948915`, `gap12 = 0.22324312213489572`,
`volume = 0.49999874883108736`, `Mnd = 12.940093529981478`,
`gray = 0.15296874999999999`, `mid = 0.030156249999999999`, `move = 0.01`,
`stage = 3`, `multN = 2`, `nInner = 19`, `innerConv = 1`, `cumInner = 6498`.

**Predicted outcome:** `CONVERGED @352`; the run must **not** execute a
scientific iteration 353 at `move = 0.005`.

## 5. Pass criteria, fixed now

| Verdict | Requires |
|---|---|
| `C320_THREE_RUNG_PREFIX_EQUIVALENCE_PASS` | `RHO[:,1:352]` and `omega(1:2,1:352)` hashes match the frozen anchors **exactly**; every compared telemetry column bitwise identical over 1…352; S1 = 274/A, S2 = 313/B, S3 = 352/B reproduced exactly |
| `C320_THREE_RUNG_TERMINATION_PASS` | status `CONVERGED`, `nOuter = 352`, `stage = 3`, `move = 0.01`, no iteration 353, terminal state equals the oracle table above to the printed precision |
| `THREE_RUNG_PRODUCTION_POLICY_VALIDATED` | all of the above **and** dependency provenance PASS, single-factor PASS, software tests PASS, exactly one scientific run, no inner failure, no cap, no scientific retuning |

Anything less is reported as `PARTIALLY_VALIDATED`, `REJECTED` or
`INCONCLUSIVE` — never quietly repaired.

## 6. What is compared, and what is excluded — declared before the run

**Compared bitwise** over iterations 1…352: `move`, `stage`, `exA`, `exB`,
`exE`, `exNA`, `exNB`, `exDecl`, `exAmp`, `exCos`, `exNet`, `exMedcos`,
`exMednet`, `exTol`, `exStageStart`, `beta`, `betaStallRel`, `betaStallFires`,
`maxAbs`, `l2`, `rms`, `ratio`, `stepNorm`, `path_W`, `net_W`, `Mnd`, `gray`,
`mid`, `volume`, `volErr`, `omega1`, `omega2`, `gap12`, `cosT`, `net_ratio`,
`cosT_unsat`, `net_ratio_unsat`, `boundFrac`, `revFrac`, `nInner`, `cumInner`,
`innerConv`, `multN`, `multJ`, `degen`, `moveChanged`, `descent`, `prodTol`,
`prodStopRaw`, `prodSettled`, `prodStopAdmit`, plus the raw `RHO` and `omega`
anchors.

**Excluded, and why:**

| Column | Reason |
|---|---|
| `tOuter` | nondeterministic wall-clock telemetry; `olhoffSolve.m` states nothing reads it back. **Wall-clock timing is not a bitwise reproducibility requirement.** |
| `prodStageShadow`, `prodMoveShadow` | post-hoc **counterfactual** diagnostics about the discarded production beta-stall ladder; `cv_telemetry.m` computes them by replaying that ladder, so they read `move.levels` and are parameterized by the arm under test. Whether they in fact differed is reported, not hidden. |

No other exclusion is permitted. In particular, a mismatch in any scientific,
controller, optimization or inner-work quantity is a **FAIL**, not a candidate
for reclassification after the fact.

## 7. Scope lock — what may not change

A, B, persistence, history/reset semantics, windows, tolerance scaling, `p`,
mass interpolation, `q`, filter, `R`, projection, multiplicity, MMA, FE,
eigensolver, objective, volume, initialization, deterministic/thread policy.
No Branch C. No mesh dependence. No restoration of beta continuation authority.
No architecture re-mining — `THREE_RUNG_ARCHITECTURE_SUPPORTED` and
`THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED` are inputs, not questions.

## 8. Stop conditions

Stop before the run if dependency provenance fails, an oracle hash differs,
frozen A/B is ambiguous, the static prefix proof fails, single-factor fails or
the software gate fails. **After the run starts, no outcome-driven repair of any
kind is permitted.** If the run validates but promotion provenance stays
blocked, the validation is kept, C320 is not re-run, nothing is promoted.

## 9. Not authorized here

The nine-mesh performance campaign (160×20 … 800×100). Zero campaign runs. It
may be authorized only after every gate in Parts G–K passes, and this task stops
before running it in any case.
