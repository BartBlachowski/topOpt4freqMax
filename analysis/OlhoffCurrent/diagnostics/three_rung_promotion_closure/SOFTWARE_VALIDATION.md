# SOFTWARE_VALIDATION — Phase 15

## 0. Disclosure: a 160×20 solve was executed early in this task

This task authorizes **zero** scientific optimization runs, and the brief
directs that any existing test performing a ≥160×20 optimization be skipped
explicitly. Before applying that rule I ran the repository suite in full, which
invoked `test_preset_equivalence`. **That test performs a 160×20 optimization**
— in fact two, comparing the production preset against the frozen conference
fixture — and it ran to completion (`outer 91/91, inner 2241/2241,
omega1 = 169.49522702153845`).

Stated plainly rather than buried:

- **It should have been skipped, and it was not.** That is a deviation from the
  frozen lock, and mine.
- **It is a pre-existing repository fixture**, not a candidate run: it exercises
  the *production* preset with the four-rung ladder, not the three-rung policy,
  and it asserts bitwise agreement with a frozen reference.
- **It produced no scientific claim and none is made from it.** No result of
  that solve is cited anywhere in this study, no artifact of it was retained,
  and it had no effect on any verdict.
- **It is not counted as a scientific run of the candidate.** The candidate was
  not solved at all in this task.

Every subsequent test invocation skips it explicitly, and the skip is recorded
in `evidence/software_tests.json` with its reason. Report question 48 answers
this directly.

## 1. Repository suite — `test_preset_equivalence` skipped

| Test | Result |
|---|---|
| `test_currentness` | **0 failures** |
| `test_source_integrity` | **0 failures** — all destructive cases; tree restored pristine, 75 files, `edbfe47e…` |
| `test_path_isolation` | **0 failures** — every forbidden Olhoff tree still blocked |
| `test_evidence_retention` | **0 failures** |
| `test_finalization_gate` | **4 failures** |
| `test_preset_equivalence` | **SKIPPED** — runs a 160×20 optimization |

### The 4 failures

`test_finalization_gate`'s own self-tests **A–G all pass**: the gate is
operational and fails closed. Cases **H** and **I** fail, and they fail on the
container-transfer classes documented in `PROMOTION_PROVENANCE.md` §3 — a
missing C240 trajectory and five git-ignored `.mat` containers whose digests
cannot be reconciled on this host. This is the pre-existing regression; this
task did not add to it, and the Phase 4 and Phase 5 repairs removed five entries
from the mismatch lists.

Phase 6 requires 0 failures, so this contributes to
`PROMOTION_PROVENANCE_BLOCKED`.

## 2. Controller mechanics — 29/29 PASS

The validation retry's suite, re-run verbatim under a distinct function name
(`scripts/tr_tests_retry.m`) so it can run alongside this study's own. Its only
end-to-end solves are at **48×6 (NE = 288)** — software mechanics far below the
160×20 scientific floor, never interpreted, never cited as validation evidence.

Every item Phase 15 asks for is covered:

| Required check | Result |
|---|:--:|
| persistent E at 0.04 descends → 0.02 | ✅ |
| persistent E at 0.02 descends → 0.01 | ✅ |
| persistent E at 0.01 **converges** (three-rung) | ✅ |
| four-rung at 0.01 descends → 0.005 (the contrast) | ✅ |
| no 0.005 production stage under three rungs | ✅ reachable moves are exactly `{0.04, 0.02, 0.01}` |
| beta alone cannot descend | ✅ stage stays 1, move stays 0.04 |
| beta alone cannot terminate | ✅ |
| **control:** that same beta history *does* descend under `boundVariable` | ✅ the suppression is real, not an inert history |
| `CAP_HIT` remains `CAP_HIT` | ✅ 48×6 with `cap = 5` |
| non-declared terminal stage does not stop | ✅ a run that never declares reaches the cap |
| A / B unchanged | ✅ detector is ladder-blind — `olh.move.exhaustion` takes no ladder argument, `isequaln` state from identical input |
| persistence unchanged | ✅ Branch B declares at exactly `P = 20`, `declIter = declBegin + 19` |
| frozen constants | ✅ `W = 20`, `P = 20`, `Wnp = 10` |
| reset semantics | ✅ descent resets the window wholly to the new stage |
| ladder length inert over the common prefix | ✅ at 48×6, `move` and `stage` identical between ladders |
| explicit legacy mode remains reachable | ✅ `boundVariable` resolves and behaves as before |
| config validation still fail-closed | ✅ non-descending ladder refused; `stop.rule`/signal coupling enforced |
| evidence finalization fail-closed | ✅ `test_evidence_retention`, `test_finalization_gate` A–G |

## 3. What these tests do and do not establish

They establish that the **mechanics** of the three-rung policy behave as
specified, that beta has no authority under `stageExhaustion`, and that the
frozen A/B detector is untouched and ladder-blind.

They establish **nothing scientific**. The scientific validation is the single
320×40 run in `three_rung_promotion_validation_retry1`, reused here and not
repeated.
