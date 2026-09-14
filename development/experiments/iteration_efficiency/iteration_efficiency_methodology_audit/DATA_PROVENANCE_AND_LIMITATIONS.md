# Data provenance and limitations of this audit's recomputation

Recorded during the audit, on discovery. Read this before relying on any number in
`TOPOLOGY_GATE_AUDIT.md` §4 or Finding C1.

## 1. The Olhoff artifacts were being regenerated during this audit

All nine files under `examples/Performance/final_campaign/raw/olhoff/` carry mtimes from the
**same day as this audit**, written in strict mesh order:

| file | mtime | gap from previous |
|---|---|---|
| `s1_160x20.mat` | 12:20:06 | — |
| `s1_240x30.mat` | 12:22:58 | 0:02:52 |
| `s1_320x40.mat` | 12:28:20 | 0:05:22 |
| `s1_400x50.mat` | 12:37:40 | 0:09:20 |
| `s1_480x60.mat` | 12:42:36 | 0:04:56 |
| `s1_560x70.mat` | 12:50:50 | 0:08:14 |
| `s1_640x80.mat` | 13:14:53 | 0:24:03 |
| `s1_720x90.mat` | 14:04:13 | 0:49:20 |
| `s1_800x100.mat` | 15:14:15 | 1:10:02 |

The gaps grow monotonically with mesh cost, and the two short gaps (480x60, 560x70)
correspond to the runs that terminate early on LP failure at outer iterations 358 and 400.
This is the signature of a sequential regeneration job, not of static frozen artifacts.

A MATLAB process was confirmed running at ~108 % CPU while this audit was in progress.

**Implication for the audit's framing.** The main audit document describes these as "frozen
campaign artifacts opened read-only". The read-only part is accurate — every access used
`h5py.File(..., 'r')`. The "frozen" part is not: these files were produced hours before they
were read, by a job that was still running.

**Content consistency was checked and holds.** For the meshes read, the artifact contents
match `examples/Performance/final_campaign/table1_performance.csv` on every field
cross-checked: `nOuter = 1600`, `status = CAP_HIT`, `res.trigger_iterations = 245` at
160x20, `res.cfg.rminEl = 1.3`, `move = 0.005`, `rhomin = 0.001`, `maxOuter = 1600`,
`unique(hist.nInner) = [1]`. So the audit's numbers describe the same campaign the protocol
describes. But the artifacts are not immutable, and any re-verification must re-check
content identity first.

## 2. The 800x100 cell could not be independently re-verified

`s1_800x100.mat` was readable when this audit's trajectory scans ran (both completed
successfully and reported 1601 snapshots). It was subsequently truncated to 0 bytes at
15:14:15 by the regeneration job, which was mid-write at the time of this note.

A third, **independently implemented** scan — using a pure-Python BFS component labeller
rather than `scipy.ndimage.label` — was running concurrently. It completed eight of nine
meshes and then failed on 800x100 with
`OSError: Can't synchronously read data (filter returned failure during read)`, i.e. the
file was truncated underneath it.

### What that independent scan did confirm

Exact agreement with the reported values on all eight meshes it completed, from a different
component-labelling implementation:

| mesh | T1 % (scipy) | T1 % (BFS) | longest T1 run (scipy) | longest T1 run (BFS) | T0 % (scipy) | T0 % (BFS) |
|---|---:|---:|---:|---:|---:|---:|
| 160x20 | 66.31 | 66.31 | 967 | 967 | 63.56 | 63.56 |
| 240x30 | 77.75 | 77.75 | 1132 | 1132 | 75.69 | 75.69 |
| 320x40 | 56.50 | 56.50 | 861 | 861 | 35.12 | 35.12 |
| 400x50 | 37.44 | 37.44 | 347 | 347 | 2.75 | 2.75 |
| 480x60 | 4.48 | 4.48 | 11 | 11 | 2.80 | 2.80 |
| 560x70 | 1.25 | 1.25 | 5 | 5 | 1.25 | 1.25 |
| 640x80 | 0.56 | 0.56 | 5 | 5 | 0.19 | 0.19 |
| 720x90 | 16.50 | 16.50 | 261 | 261 | 12.88 | 12.88 |

A separate cross-implementation check had already passed independently: the nine
**final-state** results in `TOPOLOGY_GATE_AUDIT.md` §4 were computed with the pure-Python
BFS on the `float64` `res.rho` field and agreed exactly with the `float32`
`scipy.ndimage`-based trajectory results at those same states, including 800x100
(29 components, largest detached 4, total detached 32).

### What remains unverified

The **800x100 trajectory** figures — T1 pass rate 0.00 %, longest T1 run 0, longest T1a run
298 — rest on two runs of the *same* `scipy.ndimage` implementation. They are consistent
with the independently verified 800x100 final state (detached total 32, far above the
aggregate cap of 4), but they have not been reproduced by a second implementation.

**Re-verification is straightforward** once the regeneration job completes: re-run the
trajectory scan with both labellers on the restored file, after first confirming content
identity against `table1_performance.csv`.

## 3. Does Finding C1 depend on the unverified cell?

**No.** C1 survives even if 800x100 is set aside entirely:

- T1 admits no `P = 100` window at **480x60 (11), 560x70 (5) and 640x80 (5)** — all
  double-verified by two implementations.
- The diagnosis that the **aggregate clause** is the culprit is anchored independently at
  **640x80**, where the largest detached component is 4 (the per-component clause passes)
  while the detached total is 20, and where deleting the aggregate clause restores a run of
  226. That mesh is double-verified.
- The 800x100 final state — independently verified with the `float64` field — has largest
  detached 4 and total detached 32, so it fails T1 on the aggregate clause alone at the one
  state where two implementations agree.

800x100 makes C1 more vivid ("zero states pass in an entire trajectory"). It is not what
makes C1 true. If the 800x100 trajectory figure could not be reproduced, the finding would
lose one sentence and keep its substance and its severity.

## 4. Actions taken and not taken

- No file under `examples/Performance/` was written, moved, deleted or opened for writing by
  this audit. Every access used read-only mode.
- On discovering the truncation, all further reads of that directory were stopped rather
  than retried, to avoid interfering with the running job.
- No attempt was made to recover, recreate or substitute the file. The files are untracked
  by git (blanket `*.mat` ignore, with `final_campaign/raw/olhoff/` not among the
  `.gitignore` exceptions), so no version-control copy exists — recovery is the running
  job's business, not the auditor's.
