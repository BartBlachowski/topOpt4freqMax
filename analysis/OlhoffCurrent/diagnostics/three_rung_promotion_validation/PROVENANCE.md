# PROVENANCE — three_rung_promotion_validation

**Task outcome: `THREE_RUNG_PROMOTION_PROVENANCE_FAIL` — STOPPED at Phase 1.**

Zero scientific optimizations were executed. No candidate configuration was
created. No production path was modified.

## 1. Repository state at task start (and at task end — unchanged)

| Field | Value |
|---|---|
| branch | `benchmark-methodology-r2` |
| starting HEAD | `60f5b72519aeba942b650d5408339f7ffe6b978b` |
| final HEAD | `60f5b72519aeba942b650d5408339f7ffe6b978b` (no commit created) |
| MATLAB | `25.2.0.3042426 (R2025b) Update 1` |
| `maxNumCompThreads` | 10 (production runs force 1 via `runtime.singleThread`) |
| `+impl` tree hash | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` |
| upstream promoted commit | `695f03bdac20c423a4e1d389cf9db9187597bcc3`, 0 commits ahead |
| currentness | `CURRENT` |

### Pre-existing dirty paths (2, both present before this task)

```
 M analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_iterations.csv
 M analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation/runs/C320x40_record.json
```

Both are **proven non-scientific**. Verified in this task:

- `C320x40_iterations.csv` — projecting away the final column (`tOuter`,
  per-iteration wall time) makes the working tree **byte-identical** to HEAD
  across all 1600 rows, all 54 remaining columns.
  Projection SHA-256 (both): `5f9f18960dfc4a990c15d1b101b256e006336a8d6ca41f560ec6d6b58f6bb53d`
- `C320x40_record.json` — the only deltas are `wall_s`, `trajectoryBytes` and the
  MATLAB version string. `status`, `nOuter`, `innerTotal`, `innerMax`,
  `innerNonConv`, `stage_final`, `rho_sha256`, `implTree` and the full event log
  are unchanged.

These are consistent with the C320 run having been re-executed under R2025b
Update 1 and **reproducing its science bit-for-bit**. No changes were introduced
by this task.

## 2. Phase 0 requirements — results

| Requirement | Result |
|---|---|
| currentness `CURRENT` | **PASS** |
| canonical source integrity | **PASS** — 75 files, tree `edbfe47e…` |
| published MMA wins | **PASS** (`olhoffcurrent_assert_dispatch`) |
| sensitivity filter wins | **PASS** (`olhoffcurrent_assert_dispatch`) |
| forbidden Olhoff trees absent | **PASS** — 13 blacklisted trees, none resolve |
| C320 causal trajectory present + hash-valid | **PASS** — see `C320_ORACLE.md` |
| finalization gate operational | **PASS** — self-tests A–G all pass, fails closed |
| `three_rung_architecture` finalization | **FAIL** (G2) |
| `three_rung_resolution_240` finalization | **FAIL** (G2) |
| tests PASS | **FAIL** — `test_finalization_gate` 4 failures |

## 3. Phase 1 — required prior-evidence verification

Run via the repository's own `olhoffcurrent_finalization_gate`, which resolves
mixed study-relative / repo-relative manifest paths (`local_resolvePath` tries
study, study/runs, repo, absolute). The failures below are **not** a
path-convention artifact.

| Study | G1 | G2 | G3 | G4 | G5 | Result |
|---|:--:|:--:|:--:|:--:|:--:|---|
| `two_branch_controller_validation` | ✅ | ❌ | ✅ | ❌ | ✅ | **FAIL** |
| `three_rung_architecture` | ✅ | ❌ | ✅ | ✅ | ✅ | **FAIL** |
| `three_rung_resolution_240` | ✅ | ❌ | ✅ | ✅ | ✅ | **FAIL** |

`tests/test_finalization_gate` independently reports the same regression
(4 failures): studies that were compliant when finalized no longer are —
`move_activity_400`, `beta_transition_mechanism`,
`two_branch_controller_validation`, `three_rung_architecture`,
`three_rung_resolution_240`.

## 4. The three failure classes

**Corrected after the repository owner disclosed that the prior
`three_rung_resolution_240` task executed on a DIFFERENT MACHINE.** That fact
reclassifies the largest finding below: it is a machine-locality gap, not damage.

`analysis/OlhoffCurrent/evidence/` is git-ignored **wholesale** (`*`, with only
`.gitignore` and `README.md` re-admitted). Its own header states the expected
consequence verbatim:

> "On a fresh clone this directory is empty and the gate reports
> REQUIRED_MISSING. That is the correct, honest answer: the clone does not have
> the raw evidence, and EVIDENCE.json says exactly which files and hashes are
> needed."

### Class (a) — locally regenerated trajectories; science VERIFIED identical

`C160x20`, `C320x40`, `C400x50` trajectory `.mat` files are present on this
machine but their container digests differ from those recorded by the consuming
studies. Because `evidence/` is untracked, these files did not travel with git;
the copies here were produced locally. The working-tree `C320x40_record.json`
records `wall_s` and MATLAB `25.2.0.3042426 … Update 1`, i.e. a local re-run.

Scientific content verified unchanged against the digests committed in
git-tracked `runs/*_record.json`, recomputed with the study's own `local_vecHash`:

| Artifact | final-RHO SHA-256 | recorded | match |
|---|---|---|:--:|
| `C160x20_trajectory.mat` | `332c00a5…dcd2624a` | `332c00a5…dcd2624a` | ✅ |
| `C320x40_trajectory.mat` | `0348b288…1795bfb3` | `0348b288…1795bfb3` | ✅ |
| `C400x50_trajectory.mat` | `1f648be8…81d459d7` | `1f648be8…81d459d7` | ✅ |

**Not damage.** The container digest is stale; the science is bit-identical.

### Class (b) — evidence never transferred to this machine (NOT loss)

```
analysis/OlhoffCurrent/evidence/three_rung_resolution_240/C240x30_trajectory.mat
  class : required        bytes : 131 203 128
  sha256: 183d7ce60d512fc2c045c3cb575404b00c8f0e223adaf97db86c9eef1fd50b0d
  status on THIS machine : ABSENT (parent directory does not exist)
```

Not in `~/.Trash`; absent from a full-disk `find`; no Time Machine backup on this
host. **This is expected**: the artifact was produced by the prior task on
another machine, and `evidence/` is git-ignored, so it was never transferred.
It is not lost — it exists where it was produced.

**An earlier draft of this document called this "genuine, irrecoverable evidence
loss". That was wrong** and is corrected here. The remedy is a file copy, not a
re-run.

### Class (c) — stale `FINAL_SHA256.txt` in a committed study (machine-independent)

`two_branch_controller_validation/FINAL_SHA256.txt` is stale relative to its own
**committed** files. These three are clean (identical to git HEAD) yet their
recorded digests do not match:

| File | FINAL_SHA256 records | on disk = git HEAD |
|---|---|---|
| `PROVENANCE.md` | `42037f81…3ca90790` | `0e987a9b…35459a33` |
| `BASELINES.md` | `74629a16…d2299b21` | `2212a66a…baa4f0df` |
| `evidence/baselines.json` | `08b48346…710b4c571` | `a4b55671…8738f420e` |

The study's documents were edited after its hash file was written, and both
states were committed. This will reproduce on **any** machine, including the one
that produced the prior report. It is repairable by regenerating that one hash
file; it involves no scientific content.

Note on path conventions: `two_branch_controller_validation/FINAL_SHA256.txt`
does mix bases (20 repo-relative among 71 digested entries), and a single-base
`shasum -c` cannot resolve it. All gate results in this document were produced
by the repository's own `olhoffcurrent_finalization_gate`, which tries study,
`study/runs`, repo and absolute bases. The failures above are **content**
mismatches, not unresolved paths. `three_rung_architecture` (43 entries) and
`three_rung_resolution_240` (50 entries) are purely study-relative and their G4
passes here.

## 5. Decision

The task brief makes `V2` (required prior studies / finalization PASS) and `V22`
(finalization G1–G5) **mandatory**, and forbids promotion if any mandatory gate
fails. On this machine they fail.

Because the path forward cost ~10 h of compute and ended in production
promotion, the judgment was put to the repository owner, who directed:
**STOP now — `THREE_RUNG_PROMOTION_PROVENANCE_FAIL`.**

No compute was spent on the C320 candidate run. The stop remains correct: this
machine cannot verify the prior evidence it is required to verify. But the cause
is **evidence locality plus one stale hash file**, not scientific damage — and
the C320 oracle this task would actually have consumed is present and proven
sound (`C320_ORACLE.md`).

## 6. What must happen before this task can be re-attempted

1. **Transfer the raw evidence.** Copy `analysis/OlhoffCurrent/evidence/` —
   at minimum `three_rung_resolution_240/C240x30_trajectory.mat` — from the
   machine that produced it, and verify against the digests already declared in
   each study's `EVIDENCE.json`. No re-run, no regeneration.
2. **Decide class (a).** Either transfer the original `C160/C320/C400`
   trajectories so the recorded container digests match, or re-declare the
   locally regenerated ones, recording that the science was verified unchanged
   against the committed `rho_sha256`.
3. **Repair class (c).** Regenerate
   `two_branch_controller_validation/FINAL_SHA256.txt` against its committed
   files. Consider normalizing it to a single path base while doing so.
4. **Resolve the `tOuter` drift.** Either commit the two dirty timing-only paths
   or restore them from HEAD, so the tree is unambiguous.
5. Re-run `test_finalization_gate` to green, then re-attempt this task.

Alternatively, re-attempt on the machine that already holds the complete
evidence set, where the prior task reported finalization PASS.

Nothing in Phases 2–27 was executed. The recovered controller definitions
(`CONTROLLER_RECOVERY.md`) and the verified C320 oracle (`C320_ORACLE.md`)
remain valid and can be reused directly, so the re-attempt is cheap.
