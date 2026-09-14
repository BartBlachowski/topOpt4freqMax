# C320_PREFIX_EQUIVALENCE — Part E

The three-rung candidate compared against the frozen four-rung C320 oracle over
iterations 1…S3 = 352. Machine-readable: `evidence/prefix_equivalence.json`.

Both runs executed against the **same** implementation tree,
`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`.

## 1. The raw anchors — exact

| Anchor | Expected (frozen before the run) | Obtained | |
|---|---|---|:--:|
| `RHO[:,1:352]` | `b8c0f18d887c7f5a81ece452f0a4097a0cc1a67710a137d66e677fcfd78b5ba3` | identical | ✅ |
| `omega(1:2,1:352)` | `fd63608399edaede2c4f928a25d5478eacc3711f6d626ae770ad107e7f4c488d` | identical | ✅ |

And, independently of the hashes, element by element against the oracle
container:

| Array | over 1…352 | |
|---|---|:--:|
| `RHO` (12800 × 352) | **bitwise identical** | ✅ |
| `DRHO` (12800 × 352) | **bitwise identical** | ✅ |

`DRHO` matching bitwise is the stronger statement: the increments the inner
sub-problem returned are identical at every one of the 4 505 600 entries, so the
agreement is not a coincidence of clamping.

## 2. Telemetry — 52 of 52 compared columns bitwise identical

Every compared column of the 55-column per-iteration CSV is bitwise identical
over 1…352 — **zero differing elements**, not "agreeing to tolerance":

`omega1`, `omega2`, `gap12`, `volume`, `volErr`, `Mnd`, `gray`, `mid`,
`move`, `stage`, `moveChanged`, `descent`, `beta`, `betaStallRel`,
`betaStallFires`, `prodTol`, `prodStopRaw`, `prodSettled`, `prodStopAdmit`,
`exA`, `exB`, `exE`, `exNA`, `exNB`, `exDecl`, `exAmp`, `exCos`, `exNet`,
`exMedcos`, `exMednet`, `exTol`, `exStageStart`, `cosT`, `net_ratio`,
`cosT_unsat`, `net_ratio_unsat`, `boundFrac`, `revFrac`, `maxAbs`, `ratio`,
`l2`, `rms`, `stepNorm`, `path_W`, `net_W`, `nInner`, `cumInner`, `innerConv`,
`multN`, `multJ`, `degen`, `outer`.

The oracle CSV was taken from **git HEAD**
(`evidence/oracle_C320x40_iterations_HEAD.csv`, `ff570d6e4d024f36…`), never from
the working tree, so the pre-existing `tOuter` drift could not enter the
comparison.

**36 of 36** compared `res.hist` fields are likewise bitwise identical.

## 3. Events — exact

| Event | expected | obtained | |
|---|---|---|:--:|
| S1 | `declIter 274`, Branch **A** | `274`, **A** | ✅ |
| S2 | `declIter 313`, Branch **B** | `313`, **B** | ✅ |
| S3 | `declIter 352`, Branch **B** | `352`, **B**, `declBegin 333` | ✅ |

`stageStarts = [1 275 314]` and exactly **two** descents,
`[275 1 274 255; 314 2 313 294]`, with `eventBranch = {A, B}`.

The asymmetry at S3 is the candidate's whole point and is asserted rather than
worked around: the three-rung run **consumes** the S3 declaration as
convergence, so it never records it as a descent. It appears instead in the
solver's terminal fields — `terminalDeclared = 1`, `terminalDeclIter = 352`,
`terminalDeclBegin = 333`, `terminalBranch = 'B'`.

**beta caused no transition.** `move.continuation.signal = 'stageExhaustion'`
throughout; both descents carry a recorded branch (`A`, `B`) and a
`declIter`/`declBegin` window. `betaStallRel` and `betaStallFires` are recorded
and are bitwise identical to the oracle's, confirming that the beta stall
detector was computed and simply had no authority.

## 4. The exclusions — full disclosure, including one I got wrong in advance

### 4.1 What was excluded as preregistered

`PREREGISTRATION.md` §6, frozen before the run, named three exclusions:

| Column | Differed? | Reason declared in advance |
|---|---|---|
| `tOuter` | yes, all 352 | nondeterministic wall-clock telemetry |
| `prodStageShadow` | yes, 201 of 352 | post-hoc counterfactual replay of the **discarded** production beta-stall ladder; it reads `move.levels` and is therefore parameterized by the arm under test |
| `prodMoveShadow` | yes, 201 of 352 | as above |

The two shadow columns differing is exactly what `SINGLE_FACTOR_AUDIT.md` §3
predicted **before the run**: production's beta-stall ladder would have reached
stage 4 in the four-rung arm and caps at stage 3 in the three-rung arm
(max deviation 1 stage, 0.005 move). They are diagnostics *about* the policy
that was discarded, not solver state.

### 4.2 A discrepancy in my own preregistration, stated plainly

`PREREGISTRATION.md` §6 also said *"No other exclusion is permitted."* It named
its exclusions by enumerating the **55-column telemetry CSV**, in which `tOuter`
is the only wall-clock column. But `res.hist` — which the CSV does not carry in
full — holds **three further timers** that the enumeration therefore missed:

```
hist.tEig   = toc(te)     olhoffSolve.m:215 -> 405
hist.tGrad  = toc(tg)     olhoffSolve.m:287 -> 407
hist.tInner = toc(ti)     olhoffSolve.m:351 -> 407
```

All three differ between the runs, by 0.18 s / 0.09 s / 4.97 s at most. They are
`toc` values and **nothing in the solver reads any of them back** — the only
other mention of any of them is the `hist` initialization at line 62.

**This is a defect in my enumeration, not a finding about the candidate**, and
it must not be resolved quietly. So both readings are computed and both are
reported:

| Reading | Result | Failing fields |
|---|---|---|
| **(a)** literal — only the three names `PREREGISTRATION.md` §6 listed | **FAIL** | `tEig`, `tGrad`, `tInner` — and nothing else |
| **(b)** category — wall-clock timing telemetry excluded as a class | **PASS** | *none* |

`prefix_equivalence.json` records `prefixPassLiteral = 0`,
`prefixPassCategory = 1`, and a mechanical check
`timingOnlyDivergence = 1` asserting that **every** field separating the two
readings is a member of the timing set.

### 4.3 Which reading governs, and why

The task brief's own language is a **category**, not a list of names:

> Exclude only: `tOuter` / **nondeterministic timing telemetry** from bitwise
> scientific equivalence.
>
> Use: scientific state, controller state, optimization state, **inner work**
> for exact validation. Do not use: **wall-clock timing as a bitwise
> reproducibility requirement.**

`tEig`, `tGrad` and `tInner` are wall-clock timing telemetry beyond argument.
Note especially that `tInner` is the **wall time** of the inner solve, not inner
**work** — inner work is `nInner` and `cumInner`, which are compared and match
exactly (`cumInner = 6498` at iteration 352, identical to the oracle).

Reading (b) therefore governs, and three things are worth stating so this cannot
be mistaken for a result-driven rescue:

1. **It changes no scientific, controller, optimization or inner-work
   quantity.** Every one of those is bitwise identical; the divergence between
   the two readings is *entirely* four timer fields.
2. **The decision does not depend on the outcome.** The category rule was in the
   brief before the run, and the preregistration's own stated *rationale* for
   excluding `tOuter` — "nondeterministic wall-clock telemetry that
   `olhoffSolve.m` says nothing reads back" — applies verbatim to all three. Had
   I enumerated `res.hist` rather than the CSV, these three would have been named
   in the frozen list and the verdict would be identical.
3. **Nothing was rewritten.** `PREREGISTRATION.md` stands as frozen, with its
   incomplete list intact, and `FROZEN_BEFORE.txt` still records its original
   digest `6b0638d987b244f6be609d2eb7cb619248bfab1b25ace15d6600b6107fa5345d`.

## 5. Verdict

| Check | Result |
|---|---|
| `RHO[:,1:352]` hash | ✅ exact |
| `omega(1:2,1:352)` hash | ✅ exact |
| `RHO`, `DRHO` vs oracle | ✅ bitwise |
| 52 telemetry columns | ✅ bitwise, 0 differing elements |
| 36 `hist` fields | ✅ bitwise |
| S1 = 274/A, S2 = 313/B, S3 = 352/B | ✅ exact |
| beta caused a transition | ✅ no |
| divergence outside wall-clock timing | ✅ none |

# `C320_THREE_RUNG_PREFIX_EQUIVALENCE_PASS`

The static proof of `SINGLE_FACTOR_AUDIT.md` §2 — that ladder length is
computationally inert until the detector declares at `stage == 3` — is now
confirmed empirically at full scientific resolution, to the bit.
