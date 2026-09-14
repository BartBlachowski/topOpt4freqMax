# PROVENANCE — offline move-activity characterisation

**Task type: OFFLINE ANALYSIS ONLY.** No optimiser was run. No production solver,
preset or configuration was modified. No previous diagnostic directory was
written to. Everything below is derived from already-committed artifacts.

## Repository state

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| HEAD at start of this task | `7154d8201e9defb06d0d758da866c3769c07179a` ("move transitions") |
| Working tree at start | **clean** |
| Working tree change made by this task | one new untracked directory, `analysis/OlhoffCurrent/diagnostics/move_activity_offline/` |
| Date | 2026-09-07 |
| Tooling | Python 3 (`csv`, `json`, `hashlib`, `math`) + matplotlib 3.10.7. NumPy present but unused; pandas absent. MATLAB **not invoked**. |

### Discrepancy against the stated accepted HEAD — read this

The task statement gives the current accepted HEAD as `a1f2c6c`. The actual HEAD
is `7154d82`, one commit later. This is not a detached or dirty state; it
matters for a specific reason:

```
a1f2c6c  Diagnose a move-invariant convergence admission rule
7154d82  move transitions                                     <-- HEAD
```

`git ls-tree a1f2c6c -- analysis/OlhoffCurrent/diagnostics/` contains only
`move_stop/` and `admission_rule/`. **The `move_transition/` study — the "latest
experiment" whose conclusions this task builds on — is not present at `a1f2c6c`
at all.** It was added by `7154d82` (`git log --diff-filter=A` confirms
`move_transition/REPORT.md` first appears there).

So the evidence base for this task exists only at `7154d82`. Working from
`a1f2c6c` would have made the premise unreadable. Two further notes:

- `7154d82` is a **mixed commit**. Besides the `move_transition/` diagnostic
  (37 files) it also adds ~unrelated `analysis/OlhoffApproach/` MATLAB/Python
  files including a binary `.pkl`. Bundling a frozen diagnostic with unrelated
  work weakens the freeze; a diagnostic that is cited as evidence should land in
  a commit of its own.
- The stated accepted HEAD should be advanced to `7154d82` (or the
  `move_transition` study re-landed alone) so that "accepted HEAD" and "the
  study we are reasoning from" refer to the same tree.

## Production source integrity

`analysis/OlhoffCurrent/SOURCE_MANIFEST.json` declares 74 files under `+impl/`.
Each was re-hashed with SHA-256 and compared:

```
+impl source manifest: 74 OK, 0 mismatched, 0 missing  (of 74)
manifest tree_sha256 : c1455374d5f8e256c2678a679a5fc7955d9a8a77c410a41cd30f1c189910208c
```

That `tree_sha256` is byte-identical to `source_tree_sha256` recorded in
`move_transition/METRICS.json`, so the production implementation is unchanged
since the study this task builds on.

Scope note on how that was checked: verification here is **per-file**, which is
the stronger check. The manifest's own `tree_sha256` is a digest whose exact
serialisation recipe lives in `olhoffcurrent_source_manifest.m`; reproducing it
would require MATLAB, which this task does not invoke. It is therefore carried
across as a recorded value, not independently recomputed. `currentness` was
likewise not re-evaluated (it is a MATLAB predicate); the last recorded value,
from `move_transition/METRICS.json`, is `CURRENT`.

## Integrity of the prior diagnostic directories

Every prior study was re-verified against its own `FINAL_SHA256.txt`. **Nothing
was modified.**

| study | entries OK | mismatched | missing |
|---|---|---|---|
| `move_stop` | 20 | **0** | **4** |
| `admission_rule` | 21 | **0** | 0 |
| `move_transition` | 36 | **0** | 0 |

**Zero mismatches anywhere: every surviving artifact is bit-identical to what
was recorded.** The four missing entries are all in `move_stop`:

```
runs/baseline_160x20.mat      runs/fixedmove_160x20.mat
runs/baseline_320x40.mat      runs/fixedmove_320x40.mat
```

These held the per-element density history. They are absent from disk and are
untracked by design — `analysis/OlhoffCurrent/diagnostics/.gitignore` re-includes
`*.png`, `*.csv` and `*.txt` inside this tree but keeps `*.mat` ignored, with the
stated rationale that the raw state is "fully reproducible from `code/`". The
consequence for this task is severe and is the subject of
[`DATA_INVENTORY.md`](DATA_INVENTORY.md).

`admission_rule` and `move_transition` also wrote `.mat` files carrying `RHO`
(see `ar_run.m` line 81, `mt_run.m`'s `save(f,'out','cfg','tcfg','RHO','-v7.3')`),
but those studies never listed them in their `FINAL_SHA256.txt` at all — their
manifests cover only CSVs, code and figures. Those files are likewise gone.

No surviving copy exists anywhere in the repository: a repo-wide search for
`*.mat`/`*.npy`/`*.npz`/`*.h5` found no OlhoffCurrent diagnostic state, and
`analysis/OlhoffArchive.zip` contains an unrelated older `OlhoffApproachExact`
tree.

## Exact input files used

All ten per-iteration CSVs were read; none was written.

| file (relative to `diagnostics/`) | bytes | SHA-256 |
|---|---:|---|
| `move_stop/runs/baseline_160x20_iterations.csv` | 22658 | `f438612943aee5789b3f289eb0563159efd1d5121de147381cf94fa164308a20` |
| `move_stop/runs/baseline_320x40_iterations.csv` | 33214 | `73baa20070133e9e30e073438927f401b557a962a8e05bf31700101e41642804` |
| `move_stop/runs/fixedmove_160x20_iterations.csv` | 99468 | `102309499785bcace0bce20bdcb72afa1f2421d5abecfd3db7b95e0358fb3a68` |
| `move_stop/runs/fixedmove_320x40_iterations.csv` | 54614 | `675e5ae0143e74704bc66f011388e47b9ab752e2db9e14f106d3423a6e8c7917` |
| `admission_rule/runs/unstopped_160x20_iterations.csv` | 156663 | `8114d92c0dc31a1fc417f58069caa57c3b6797314ca8fbb7cb3fb5652e821240` |
| `admission_rule/runs/unstopped_320x40_iterations.csv` | 158019 | `5e7008fe6d2e0236ab5ba8926824a2c1f8e069496cfa5279a47123ac597261f3` |
| `move_transition/runs/armP_160x20_iterations.csv` | 165868 | `9fb22bff8cb65f4df2733bdcad7ae9641611b99864ec3bd504abd9faf102736b` |
| `move_transition/runs/armP_320x40_iterations.csv` | 167146 | `1df7b1aa304be32a4465ca43644f9977db49dd287094378e435e14172bd8eba7` |
| `move_transition/runs/armU_160x20_iterations.csv` | 162121 | `a5f48446f0977cd443b0bfc8b0694930bab012870b1415e1aa4b188f71f8baca` |
| `move_transition/runs/armU_320x40_iterations.csv` | 167087 | `67bbd33c73c268f647253cfe1a919aaa1d8e81077038d209c54f73d5ad6035aa` |
| `move_transition/METRICS.json` | 26282 | `4bf41a575429589033ce1de533c47ff092e30195ef1c19f9320c1d6e561cbe38` |

Also read (not modified): `SOURCE_MANIFEST.json`, the three prior
`FINAL_SHA256.txt` manifests, `+impl/architecture/olhoffSolve.m`,
`+impl/architecture/+olh/+move/limit.m`, and the prior studies' `code/` for
semantics (`ms_run.m`, `ar_run.m`, `mt_run.m`, `mt_telemetry.m`, `mt_spatial.m`,
`mt_utilCount.m`, `mt_export.m`).
