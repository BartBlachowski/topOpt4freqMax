# PROVENANCE — move-ladder necessity audit

| | at task start | at task end |
|---|---|---|
| branch | `benchmark-methodology-r2` | `benchmark-methodology-r2` |
| HEAD | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` (unchanged) |
| `git status` | dirty, 17 paths | dirty — the previous study's 17 plus this study's deliverables |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) | **unchanged** |

**The `+impl/` tree is byte-identical at task end.** This audit executed no
optimization and touched no file under `+impl/`. The only production-adjacent
addition is `olhoffcurrent_finalization_gate.m`, which lives *outside* `+impl/`
by design so that it cannot alter the canonical tree hash.

## Environment

MATLAB 25.2.0.2998904 (R2025b) was used only to run the software tests
(`test_finalization_gate`, and the repository suite). All scientific analysis is
Python/offline over existing trajectories: NumPy 2.4.2, h5py 3.16.0,
Matplotlib 3.10.8.

## Preregistration

`PREREGISTRATION.md`, SHA-256
`a08b879b9d4893bc55dd9e5dfc2c26f897cb9d49355f0557ccc8644c7fc6ca3e`, frozen
2026-09-09T15:17:49Z — before the lower-rung decomposition was computed. Its §8
discloses exactly which downstream quantities were already visible from the
previous study, because this audit is not blind to its own headline and saying
otherwise would be dishonest. The materiality thresholds in §6 are anchored to
bars the project had already committed to (20 % relative `M_nd`, 1 % relative
`ω₁`), each taken an order of magnitude tighter so the test is generous to the
ladder.

## Runs

**Zero.** No mesh, no fixed-move arm, no production arm, no controller arm, no
software solve of any size. The only MATLAB executions were the retention-gate
self-tests, which build a sandbox study out of `rand(32,12)` and never call the
optimizer.

## Changes made to the repository

| path | change | why |
|---|---|---|
| `analysis/OlhoffCurrent/olhoffcurrent_finalization_gate.m` | **new** | fail-closed finalization gate (Phase 20) |
| `analysis/OlhoffCurrent/tests/test_finalization_gate.m` | **new** | 12 checks incl. 7 destructive fail-closed cases |
| `two_branch_controller_validation/FINAL_SHA256.txt` | one digest refreshed, correction recorded in the file | its `DATA_MANIFEST.json` line had gone stale (`RETENTION_AUDIT.md` R-F4) |
| `diagnostics/move_ladder_necessity/**` | **new** | this study |

Nothing else was modified. No prior evidence was deleted or rewritten.

## Gates

| gate | result |
|---|---|
| previous study `FINAL_SHA256.txt` re-verified | **71/71** after the R-F4 correction |
| previous study `DATA_MANIFEST.json` | **53/53** |
| previous study `EVIDENCE.json` | **3/3 required present and matching** |
| previous study preregistration vs frozen copy | byte-identical |
| exhaustion-event independent recomputation | matches element-wise on all three meshes |
| mid-task commit audit | `MIDTASK_COMMIT_NONINTERFERENCE_VERIFIED` |
| repository test suite | 6/6 suites, 0 failures |
