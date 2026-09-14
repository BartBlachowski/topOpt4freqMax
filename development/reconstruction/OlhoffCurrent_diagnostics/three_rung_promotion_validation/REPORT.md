# REPORT — three_rung_promotion_validation

## Verdict

**`THREE_RUNG_PROMOTION_PROVENANCE_FAIL` — stopped at Phase 1.**

Zero scientific optimizations executed. No configuration, source or production
path changed. `HEAD` unchanged, no commit created.

## Why

Phase 0/1 require `two_branch_controller_validation`, `three_rung_architecture`
and `three_rung_resolution_240` to pass finalization G1–G5. **On this machine all
three fail G2**, and the repository's own `test_finalization_gate` independently
reports the same regression (4 failures).

The cause is **evidence locality plus one stale hash file — not scientific
damage.** `analysis/OlhoffCurrent/evidence/` is git-ignored wholesale, and its
`.gitignore` states the expected consequence outright: *"On a fresh clone this
directory is empty and the gate reports REQUIRED_MISSING. That is the correct,
honest answer."* Three classes, detailed in `PROVENANCE.md` §4:

- **(a)** `C160/C320/C400` trajectories present but locally regenerated —
  container digests stale, **science verified bit-identical** against the
  `rho_sha256` committed in git.
- **(b)** `C240x30_trajectory.mat` absent **because it was produced on another
  machine** and never transferred. Recoverable by file copy.
- **(c)** `two_branch_controller_validation/FINAL_SHA256.txt` is stale against
  its own *committed* `PROVENANCE.md`, `BASELINES.md` and
  `evidence/baselines.json`. Machine-independent; repairable by regenerating
  one hash file.

All gate results here come from the repository's own
`olhoffcurrent_finalization_gate`, which resolves the mixed study-relative /
repo-relative path convention. The failures are **content** mismatches, not
unresolved paths.

The stop is still correct — this machine cannot verify the prior evidence it is
required to verify — but the remedy is a file transfer and a hash-file
regeneration, not a re-run. **The C320 oracle this task would actually have
consumed is present and cryptographically proven sound** (`C320_ORACLE.md`).

## Answers to the brief's required questions (as far as reached)

1. **Branch / starting HEAD** — `benchmark-methodology-r2` / `60f5b72519aeba942b650d5408339f7ffe6b978b`.
2. **Starting dirty state** — 2 modified paths, both timing-only.
3. **Which dirty paths pre-existed** — both: `two_branch_controller_validation/runs/C320x40_iterations.csv` and `…/C320x40_record.json`. Neither was touched by this task.
4. **Starting `+impl` hash** — `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`.
5. **Did required prior finalization gates pass?** — **No, on this machine.** All three FAIL G2; `two_branch_controller_validation` also fails G4. Cause is evidence locality (`evidence/` is git-ignored; the 240 trajectory was produced on another machine) plus one stale committed hash file — not scientific damage. This is the blocker.
6. **Exact A definition** — `med20 cos < 0 AND med20 net < 0.5 AND amp >= tol`. See `CONTROLLER_RECOVERY.md` §4.
7. **Exact B definition** — `amp < tol AND med20 cos > 0`.
8. **Windows / persistence / reset** — `W = 20`, `P = 20`, `Wnp = 10`, `tol = 0.05*sqrt(NE/3200)` (= 0.1 at 320×40); declaration at `P` consecutive true, counters reset to 0 on any false; on descent the detector window resets wholly to the new stage (`stageStart = outer`, counters cleared, `declared = false`), with `net`'s anchor `rho_{s−1}` the sole pre-stage quantity. §5–6.
9. **Preregistration frozen before scientific output?** — Not applicable; no preregistration was frozen and no scientific output was generated. The task stopped before Phase 2 by direction.
10. **Exact three-rung policy frozen** — none frozen. The policy that *would* have been frozen is recorded in `CONTROLLER_RECOVERY.md` §8 as a configuration-only delta.
11–15, 21–45. **Not reached** — no single-factor audit, no software validation, no scientific run, no prefix/termination/cost result. Nothing may be cited as validated.
16. **Old C320 trajectory serving as oracle** — arm `C` of `two_branch_controller_validation`, `CAP_HIT @1600`, `rho_sha256 = 0348b288…`, verified intact.
17. **S1** — `declIter = 274`, window 255–274, move 0.04, Branch **A**, descent applied at 275.
18. **S2** — `declIter = 313`, window 294–313, move 0.02, Branch **B**, descent applied at 314.
19. **S3** — `declIter = 352`, window 333–352, move 0.01, Branch **B**, descent applied at 353 → stage 4.
20. **Branches expected at S1/S2/S3** — A, B, B. Confirmed from tracked evidence, matching the brief.
46. **Did every mandatory gate pass?** — No. V1 partially (source/currentness/dispatch PASS; tests FAIL), V2 FAIL, V22 FAIL. V3 PASS (A/B recovered exactly, unambiguous). V4–V21 not reached.
47. **Production-policy validation verdict** — none issued; the task stopped before any scientific gate could be evaluated.
48. **Was production promoted?** — No.
49. **What exact files/config changed?** — None under any canonical or production path. Only this new diagnostics directory was created.
50. **Was A changed?** — No.
51. **Was B changed?** — No.
52. **Was persistence changed?** — No.
53. **Was 0.005 the only move level removed?** — No level was removed; production still resolves to `[0.04 0.02 0.01 0.005]`.
54–57. **Not reached** — no candidate, no promotion, no equivalence audits.
58. **Final `+impl` hash** — `edbfe47eb32109a2…` (unchanged).
59. **Final production-config hash** — unchanged; production preset untouched.
60. **Is production frozen/manifested?** — Unchanged from task start; no new freeze performed.
61. **Was a commit created?** — No.
62. — n/a.
63. **Final HEAD** — `60f5b72519aeba942b650d5408339f7ffe6b978b`.
64. **Final dirty state** — the same 2 pre-existing timing-only paths, plus this new untracked diagnostics directory.
65. **Are all task-created artifacts accounted for?** — Yes; every file is listed in `FINAL_SHA256.txt` and `DATA_MANIFEST.json`.
66. **Did fail-closed finalization pass?** — For the prior studies, no, on this machine — that is the finding. The gate itself is operational and failed closed correctly (self-tests A–G pass), behaving exactly as `evidence/.gitignore` documents for a host that does not hold the raw evidence.
67. **Is any controller blocker unresolved?** — No *controller* blocker exists. The recovered A/B rule is unambiguous and the three-rung change is provably configuration-only. The unresolved blocker is **evidence locality on this host**, plus one stale committed hash file.
68. **Is the nine-mesh campaign authorized?** — **No.**
69. **Were zero nine-mesh campaign runs executed?** — Yes, zero.
70. **Was any outcome-driven repair performed?** — No. Nothing was tuned, repaired or relaxed; the stop preceded any scientific result.
71. **Does anything justify projection?** — No. Nothing observed bears on it, and projection is refused outright under stage exhaustion.
72. **Does anything justify changing R?** — No.
73. **Does anything justify changing p / mass / q?** — No.

## Work preserved for a re-attempt

- `CONTROLLER_RECOVERY.md` — the exact frozen A/B rule, its authoritative
  sources, its reset/indexing semantics, and the proof that the three-rung
  change is configuration-only (a static audit of every `move.levels` read).
- `C320_ORACLE.md` — the oracle, cryptographically verified, with the frozen
  S1/S2/S3 event structure and the exact expected S3 terminal state.

Both remain valid while `+impl` tree `edbfe47e…` and the tracked C320 evidence
are unchanged. `PROVENANCE.md` §6 lists what must be repaired first.
