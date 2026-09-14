# MID-TASK COMMIT AUDIT — commit `1438aa3`

> ## `MIDTASK_COMMIT_NONINTERFERENCE_VERIFIED`

---

## 1. What happened

| | |
|---|---|
| commit | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| parent | `b6014ba8bca41f85671d79ab4c8bdee7419880bb` (the previous task's starting HEAD) |
| author / committer | `piotrek <nabucco33@tlen.pl>` — the repository owner, **not** the agent session |
| date | 2026-09-09 07:28:32 +0200 |
| subject | `A & B tests` |
| scope | 36 files, +7148 / −22 |

C400 was running at the time. The commit was made externally while a scientific
optimization was in progress, so it must be audited rather than assumed benign.

## 2. Which changed files could touch scientific execution?

Seven of the 36 are under `+impl/`, i.e. inside the executable production tree:

```
+impl/architecture/olhoffSolve.m
+impl/architecture/+olh/+move/limit.m
+impl/architecture/+olh/+move/exhaustion.m
+impl/architecture/+olh/+config/schema.m
+impl/architecture/+olh/+config/validate.m
+impl/architecture/+olh/+config/toLegacy.m
+impl/architecture/+olh/+config/fromLegacy.m
```

plus `SOURCE_MANIFEST.json`. These are exactly the files C400 loaded and
dispatched. The remaining 28 are study documents, telemetry CSVs, records and
scripts, none of which the solver reads.

So the question is not "were scientific files in the commit" — they were — but
**"did the commit change them?"**

## 3. Evidence of non-interference

### 3.1 A commit records content; it does not write to tracked files

`git commit` snapshots the index into a new object. It does not modify the
working-tree files it records. So a commit can only interfere if the *files on
disk* changed at that moment — which is a filesystem question, answered below.

### 3.2 Committed blob == on-disk file, for all seven

```
IDENTICAL   architecture/+olh/+config/fromLegacy.m
IDENTICAL   architecture/+olh/+config/schema.m
IDENTICAL   architecture/+olh/+config/toLegacy.m
IDENTICAL   architecture/+olh/+config/validate.m
IDENTICAL   architecture/+olh/+move/exhaustion.m
IDENTICAL   architecture/+olh/+move/limit.m
IDENTICAL   architecture/olhoffSolve.m
IDENTICAL   SOURCE_MANIFEST.json
```

### 3.3 Every `+impl` mtime predates the first run

```
2026-09-08 20:54:10   +olh/+move/exhaustion.m
2026-09-08 20:54:43   +olh/+move/limit.m
2026-09-08 20:54:58   +olh/+config/{schema,toLegacy,fromLegacy}.m
2026-09-08 20:56:15   olhoffSolve.m, +olh/+config/validate.m
2026-09-08 21:23:27   SOURCE_MANIFEST.json
```

C160 started ≈ 21:30 on 2026-09-08. **No scientific source file was written on
2026-09-09 at all**, and none at 07:28. The commit wrote nothing.

### 3.4 Each run stamped the tree hash it executed under

Recorded *by the run itself*, at run time, into its own record:

```
C160x20  implTree = edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb
C320x40  implTree = edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb
C400x50  implTree = edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb
current  SOURCE_MANIFEST tree = edbfe47eb...152cb   (75 files)
```

All three ran under the tree that exists now, which is the tree the commit
recorded. Had the commit altered anything, the current hash would differ from
the stamped one.

### 3.5 MATLAB had already loaded the functions

C400's MATLAB process started at 07:15:52 and resolved `olhoffSolve`,
`olh.move.limit`, `olh.move.exhaustion` and the config package during its first
iterations, ≈ 12 minutes before the commit. Even a hypothetical file rewrite at
07:28 would not have been re-read mid-run. This is a secondary argument; §3.2–3.4
are the primary ones.

### 3.6 The bitwise prefix result spans the commit instant — the decisive check

Reconstructing C400's per-iteration timeline from `hist.tOuter`:

```
MATLAB launch                    07:15:52
commit                           07:28:32   =  760 s after launch
cumulative solver time at iter 190   758.4 s
cumulative solver time at iter 191   763.3 s
-> the commit landed during outer iteration ~191 of 505
```

C400 iterations **1–369** are bitwise identical to `F400_400x50_trajectory.mat`
— an arm produced on 2026-09-07, two days earlier, under the *pre-controller*
source tree — across ρ (20000 × 369), all five ω, β, `‖Δρ‖₂`, inner iteration
counts, volume and relative gap.

Iteration 191 lies **inside** that verified prefix. So the bitwise identity is
not merely adjacent to the commit; it **straddles** it. Any perturbation at
07:28 would have shown up as divergence from F400 somewhere in iterations
191–369. There is none.

## 4. Config and manifest changes

`SOURCE_MANIFEST.json` was included in the commit, but its on-disk content is
identical to the committed blob and its mtime is 2026-09-08 21:23 — it was
rewritten by `olhoffcurrent_source_manifest('Write',true)` *before* C160, as the
previous study's `IMPLEMENTATION.md` §2 records. No configuration file changed
at commit time; the three per-run `cfgHash` values recorded at run time still
match what `cv_config` produces today (verified in the previous study's
single-factor gate).

## 5. Verdict

> ## `MIDTASK_COMMIT_NONINTERFERENCE_VERIFIED`

**C400 remains scientifically valid**, and so do C160 and C320 (both had
finished before the commit: C160 at 21:37 on 2026-09-08, C320 at 07:10 on
2026-09-09).

The commit is a bookkeeping event: the repository owner recorded work that was
already on disk. The one lasting consequence is administrative — the previous
study's HEAD moved mid-task, which is why its `PROVENANCE.md` records a starting
HEAD of `b6014ba` and a final HEAD of `1438aa3`.

**Residual note, not interference:** the commit captured the study mid-flight, so
`C400x50`'s outputs, the final documents and `EVIDENCE.json` are still
uncommitted. That is a retention exposure, not a validity problem, and it is
carried into `RETENTION_AUDIT.md` as finding R-F5.
