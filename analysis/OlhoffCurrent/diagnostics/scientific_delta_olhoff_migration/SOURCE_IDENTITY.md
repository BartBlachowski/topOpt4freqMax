# SOURCE_IDENTITY — Parts 1 and 3

```
OLHOFF_SOURCE_COMMIT_VERIFIED
SOURCE_SWEEP_EVIDENCE_PASS
```

## 1. Commit

| | |
|---|---|
| repository | `/Users/piotrek/Programming/Matlab/Olhoff` (read-only throughout) |
| commit | `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` |
| subject | `Nine resolution test - ultimate results` |
| author date | 2026-09-13 16:17:23 +0200 |
| branch containing it | `repro/natural-convergence` (the only branch; it is HEAD) |
| parent | `695f03bdac20c423a4e1d389cf9db9187597bcc3` — the commit OlhoffCurrent was promoted from; the source is exactly one commit ahead |
| tree | `809c671e2d1b15232f8b75ca0f07237c0e8e37ac` |
| files at commit | 1 838 (455 921 894 bytes) |
| checkout state at audit start (16:29) | clean, HEAD = commit |
| checkout state observed at 16:56 | **dirty**: `repro/PLAN_OLHOFFCURRENT_UPDATE.md` modified at 16:32:45 by an external edit (+55 lines, "§7 Fair-comparison settings for all three methods"). HEAD unchanged. Not made by this audit (which ran only `git archive`, `ls-tree`, `show`, `status`, `diff`). Preserved as `evaluations/source_worktree_uncommitted_diff.patch` (SHA-256 `212952322dd9…`). Not audited evidence; flagged in MIGRATION_CLASSIFICATION.md. |

## 2. Snapshot

| | |
|---|---|
| full-tree `git archive --format=tar` SHA-256 (streamed, not stored; git 2.50.1) | `527b56d9f708da8466bc0422b721d697a3a38f3b5c4ce1f60d864c5fad6d29c0` (457 441 280 bytes) |
| subset archive SHA-256 (extracted) | `0dd479ff5d2ea8580497f447fa49d86639b8b18ac6510df2a9ee20423746424a` (90 081 280 bytes) |
| extracted to | `source_snapshot/+olhoff_6b08708/` (405 files, 86 MB; `+` folder so `genpath` never reaches it) |
| blob verification | 405 / 405 files re-hashed as git blobs equal the committed blob ids |
| snapshot tree SHA-256 (sorted `path  sha256` lines) | `b2efb8a006ae6fb177ec70a6d61849dbca7e3f0b6c508eba94ffcc59969d6a57` |
| manifest | `evaluations/source_snapshot_manifest.json` |

Subset: `.gitattributes .gitignore CLAUDE.md NOTES.md EVIDENCE_MANIFEST.sha256
OLHOFFEXACT_FAILURE_POSTMORTEM.md setpaths.m top88.m algo fem filter mma mma_published
architecture/{+olh, olhoffSolve.m, README.md, docs, legacy, tests, anchors/code} repro`. Excluded
(not evidence for this question): legacy `audit_*`, `results/`, `runs/`, PDFs, anchor binaries.

Source files used for semantics: every file of `algo/ fem/ filter/ mma/ mma_published/
architecture/+olh/ architecture/olhoffSolve.m`, `repro/{run_repro,evaluate_runs,sweep_table}.m`,
`repro/PLAN_OLHOFFCURRENT_UPDATE.md` (committed version), `NOTES.md` §32, `CLAUDE.md`.

## 3. Claimed evidence is committed at the hash

All present in the commit tree (not merely on disk): `repro/results/SWEEP_R06.md` + `.csv`,
`SWEEP_R13EL.md` + `.csv`, and for each of the 18 runs `S<mesh>/` and `Rel<mesh>/`:
`describe.txt`, `res.mat` (cfg + res + out), `summary.json`, `hist_vs_paper.png`, `topo.png`,
`topo_vs_paper.png`, plus `<run>.log`. `NOTES.md` §32 (line 1521 onward) and
`repro/PLAN_OLHOFFCURRENT_UPDATE.md` are committed. Largest sweep container `S800x100/res.mat`
is 4.9 MB, below the upstream 5 MB exclusion rule, so no evidence lives outside git. Per-file
SHA-256 digests: `evaluations/source_snapshot_manifest.json` (e.g. `S480x60/res.mat`
is listed there). No discrepancy → no stop.

## 4. Sweep evidence recomputed (Part 3)

`scripts/sd_verify_sweeps.m` loads each committed `res.mat` and recomputes every column of
the committed CSVs with the arithmetic of `repro/sweep_table.m` (one SIMP + eq.(4) FE assembly
and `eigs` per run, no optimization). Output `evaluations/sweep_verification.json`.

| check | result (18 runs) |
|---|---|
| every numeric column vs committed CSV | max relative difference 4.1e−15 (text rounding); status column equal |
| natural stop consistent (`converged` log line, final ‖Δρ‖₂ < ε, nOuter < 400) | 17 CONVERGED; `Rel800x100` CAP_HIT at 400 as reported |
| `summary.json` vs `res` (status, nOuter, inner total, ω₁ native, M_nd) | 18/18 equal |
| `aux.Mnd(end)` vs recomputed M_nd | 18/18 bitwise |
| stiffness / mass model in cfg | `pedersen` (linearBelow 0.1) / `eq2` in all 18 |
| radius | S: 0.06 physical = 1.2 … 6.0 elements; Rel: 1.3 elements |
| move policy | adaptive, initial 0.10, floor 0.002, ×1.2 / ×0.7, in all 18 |
| inner loop | tol 0.05, min 5, max 500; 0 non-converged inner loops in every S run; max inner 29–46 |
| localized-mode spike events ω₁(k) < 0.7 ω₁(k−1) | 0 in all 18 histories |
| single thread | true in all 18 |

The brief's table matches the committed files: e.g. 480×60 → outer 112, ω₁/ω₂ 166.3/203.6,
gap 22.4 %, M_nd 0.131; 800×100 → 246, 165.8/195.9, 18.2 %, 0.165.

Claims checked against data:

| claim | verified value |
|---|---|
| natural termination at every mesh (R = 0.06) | yes, 9/9 |
| 18–21 inner sub-iterations per outer | 18.2 – 20.6 |
| M_nd ≈ 0.115–0.165 | 0.1146 – 0.1655 |
| ω₁ drift ≈ 2 % | 169.74 → 165.7 (min at 720×90): 2.4 % |
| only 160×20 terminally bimodal | gap 0.65 % at 160×20; ≥ 11.7 % elsewhere |
| smooth monotone per-iteration costs | eig/outer 0.053 → 1.359 s and per inner sub-iterate 0.168 → 1.251 s, monotone in NE |
| comparable topology family | images present and committed; not numerically verified by this audit |

**Reporting caveat (class E, material for cross-implementation tables).** The sweep tables' ω
columns are *re-evaluations* of the final designs under SIMP p = 3 + eq.(4) with cut-off 0.1 —
neither the native Pedersen/eq.(2) values nor the target's eq.(4b). At 480×60: native 166.009 /
203.441, table 166.3 / 203.6. Target tables report native SIMP/eq.(4b) values.
