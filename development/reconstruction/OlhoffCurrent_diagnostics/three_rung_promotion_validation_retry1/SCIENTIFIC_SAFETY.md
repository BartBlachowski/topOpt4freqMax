# SCIENTIFIC_SAFETY — the run lock, and what was and was not executed

## 1. The count

| | |
|---|---|
| **Candidate scientific runs executed in this retry** | **1** |
| mesh | **320×40** and no other |
| nine-mesh campaign runs | **0** |
| C240 re-runs | **0** |
| old four-rung C320 re-runs | **0** |
| runs discarded, repeated or re-tuned | **0** |

`scripts/tr_run.m` **hard-codes** `NELX = 320; NELY = 40` rather than accepting a
mesh argument. There is no path through it that solves another mesh, so the lock
is enforced by the code and not by discipline.

## 2. Solves that are NOT candidate scientific runs

Declared explicitly, so the count above is unambiguous:

| Solve | Mesh | Why it is not a scientific run |
|---|---|---|
| `tr_tests.m` end-to-end mechanics | **48×6** (NE = 288) | far below the 160×20 scientific floor; three solves (three-rung, four-rung, `cap = 5`) exercising *mechanics only*. No number from them is interpreted or cited. Permitted by the brief: *"Sub-160 meshes may be used for software mechanics only."* |
| `test_preset_equivalence` | 160×20 | the **repository's own** pre-existing frozen equivalence fixture — production preset, four-rung ladder, **not this candidate**. It is part of the standard test suite and was run unmodified. |

The 160×20 mesh-resolution floor applies to *scientific* claims. No claim in
this study rests on a sub-160 solve, and the one 160×20 solve that occurred was
a repository test of the production preset, not of the candidate.

## 3. What was never touched

Not modified by this retry, at any point:

- **the controller** — `exhaustion.m`, `limit.m`, `olhoffSolve.m`; `+impl` tree
  hash is `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`
  at task start and unchanged at task end, manifest-verified, 75 files;
- **A, B, persistence, windows, tolerance scaling, history/reset** — the
  detector takes no ladder argument and is provably ladder-blind
  (`SOFTWARE_VALIDATION.md` §1);
- **the physics and the formulation** — `p`, mass interpolation, `q`, filter,
  `R`, projection, multiplicity, MMA, FE, eigensolver, objective, volume,
  initialization;
- **the deterministic/thread policy** — `runtime.singleThread`, threads forced
  to 1;
- **production** — no canonical or production path was changed. `move.levels`
  in `duOlhoffFrozenM4` still resolves to `[0.04 0.02 0.01 0.005]`;
- **the prior stopped study** — all seven files of
  `three_rung_promotion_validation/` hash to its own `FINAL_SHA256.txt`;
- **the two dirty tracked files** — left exactly as found.

## 4. No outcome-driven repair

`PREREGISTRATION.md` was frozen, with its hashes recorded in `FROZEN_BEFORE.txt`,
**before** the candidate trajectory, CSV or record existed on disk — the freeze
file records that `runs/` held 0 entries and the candidate evidence directory
did not exist. Pass criteria, the comparison set and the two exclusions
(`tOuter`; `prodStageShadow` / `prodMoveShadow`) were fixed in that document in
advance.

After the run started, **nothing was tuned, relaxed, re-classified or repaired**
in response to what it showed. No threshold was moved, no exclusion was added,
no criterion was reinterpreted. The verdicts in `POLICY_VALIDATION.md` are the
ones `PREREGISTRATION.md` §5 defined before the result existed.

## 5. Architecture not reopened

`THREE_RUNG_ARCHITECTURE_SUPPORTED`, `THREE_RUNG_POLICY_PREREGISTRATION_JUSTIFIED`,
`C240_THREE_RUNG_COUNTERFACTUAL_EXACT` and
`THRESHOLD_SPLITTING_CONCERN_RESOLVED` were treated as **inputs**. No
architecture mining was performed, no Branch C was added, no window or
persistence was re-derived, no mesh dependence was introduced, and beta's
continuation authority was not restored.
