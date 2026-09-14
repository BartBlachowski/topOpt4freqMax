# MIGRATION_COMMIT_REVIEW — `9b30ec45b038fb36e7cf20d57679b71cfd099fb3`

```
MIGRATION_COMMIT_REVIEW_FAIL
```

**Why FAIL:** there is **one** failing criterion. The commit does not match what its own migration report says. The gate header says "A digest that never existed in history still fails" and "(the change is a recorded promotion, never a local edit)". TEST_REPORT says the J tests prove the rule "is not a pass-through" and calls the rule "strict". Probes P5, P9 and P10 disprove all of these (FINALIZATION_GATE_REVIEW.md).

Every other criterion passes (table below). This was a read-only static review; nothing was run from the commit except the gate probes.

| criterion | result | how verified |
|---|---|---|
| branch points to the commit | PASS | `git rev-parse migration/olhoffcurrent-upstream-253069` = `9b30ec45…` |
| expected parentage | PASS | sole parent `013cc48451d33bed61c5c4eea174bbd898d548a2` = `benchmark-methodology-r2` HEAD = merge-base |
| no later commits included | PASS | `013cc48..branch` = 1 commit; `branch..013cc48` = 0 |
| upstream identity | PASS | `253069…` is a commit whose parent is `6b0870850d74…` |
| `+impl` = upstream 253069 | PASS | all 79 blob ids at `9b30ec4:analysis/OlhoffCurrent/+impl/<p>` equal `253069:<p>` in `/Users/piotrek/Programming/Matlab/Olhoff`; 0 differ, 0 missing; all mode 100644 |
| no private solver/controller copy outside `+impl` | PASS | no `olhoffSolve/limit/exhaustion/innerLoop/mmasub/subsolv/assemble2D/eigSolve/genGrad` among tracked OlhoffCurrent files outside `+impl`/diagnostics |
| stage exhaustion and `hist.tOuter` from upstream | PASS | both present in the upstream-identical `olhoffSolve.m` |
| manifest ↔ tree | PASS | tree hash recomputed from git blobs = `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` = `SOURCE_MANIFEST.json` tree = PROVENANCE.md; rows equal (79) |
| changed paths confined | PASS | 121 paths, all under `analysis/OlhoffCurrent/**` or `examples/Performance/**` |
| Proposed / Yuksel / evaluator / tools / profiles / C480 / SOCP untouched | PASS | no path matches; harness diffs read line by line: every functional change is in an `olhoff` branch, an Olhoff-only record/CSV column (`N/A` otherwise), or the per-outer fit, which needs `counts.outer_iterations` that only Olhoff records carry |
| Phase 6 not entered | PASS | no radius, low-density or evaluator change for other methods (DIFF_AUDIT Phase-6 table agrees) |
| p-continuation defect documented and unfixed | PASS | `olhoffSolve.m` byte-identical to upstream, which carries the defect; KNOWN_PREEXISTING_DEFECTS D1 |
| consistency with BYTE_IDENTITY / DIFF_AUDIT / MANIFEST_PROVENANCE | PASS | the figures above match (79/79; 20 promoted; 0 unauthorized) |
| consistency with PRESET_IDENTITY | NOT RE-RUN | resolving presets needs MATLAB on the merged tree (Steps 4–5); not reached |
| **consistency with REPORT / TEST_REPORT on the gate rule** | **FAIL** | see top |

## Observations (non-blocking, for the owner)

- **O1.** `PROVENANCE.md` at `9b30ec4` cites `diagnostics/scientific_delta_olhoff_migration` as acceptance evidence. That directory is **untracked** in the primary checkout and not in the commit, so committed provenance points at uncommitted material.
- **O2.** Local `benchmark-methodology-r2` (`013cc48`) is one commit ahead of `origin/benchmark-methodology-r2` (`bba45e7`).
