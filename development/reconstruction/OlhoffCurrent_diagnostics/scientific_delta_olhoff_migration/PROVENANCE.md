# PROVENANCE — scientific_delta_olhoff_migration

## Identities

| | |
|---|---|
| audit date | 2026-09-13, 16:29 – 17:25 (+0200); closing integrity check 17:22:28 (`evaluations/final_integrity.json`) |
| host | PMS.local, Apple M1 Max, 10 cores, macOS 26.6; MATLAB R2025b (25.2.0.2998904); Python 3.13 (numpy 2.3.4, scipy 1.16.3, matplotlib 3.10.7, h5py); git 2.50.1 |
| target | `/Users/piotrek/Programming/topOpt4freqMax`, `benchmark-methodology-r2` @ `013cc48451d33bed61c5c4eea174bbd898d548a2` (unchanged at end) |
| target implementation | `analysis/OlhoffCurrent/+impl`, 75 files, tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (verified; no tracked change at end) |
| source | `/Users/piotrek/Programming/Matlab/Olhoff` @ `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` (HEAD unchanged at end) |
| source snapshot | `source_snapshot/+olhoff_6b08708/`, subset archive `0dd479ff…424a`, full archive `527b56d9…29c0`, 405 files blob-verified at start and end |
| preregistration | `AUDIT_PREREGISTRATION.md` SHA-256 `ae998b0ff10bcbd5bde552e284626ae43929cc459885577ca2dbfc7f053dc28f`, frozen 16:41:50 before any analysis or run |
| amendment 1 | `PREREGISTRATION_AMENDMENT_1.md` SHA-256 `9b90de7be7cd54e3c19c982eda3df759e7851c01909de7a8ae121a1c45bbcb67`, frozen before the M1 launch (resolver metadata leaves in the preflight) |

## What was written, and where

Only inside `analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration/`. Nothing was
written to `+impl/`, other OlhoffCurrent files, other diagnostics, the evidence store, or the source
repository. One stray temporary file list was briefly written to `/tmp` early in the session and
deleted immediately; the tar archives used for hashing were created in the session scratchpad and
deleted after hashing.

## Computation performed

| item | count | details |
|---|---|---|
| source-side 480×60 trajectory | 1 | M1: snapshot code, `duOlhoffAdaptiveMove` + `move.initial = 0.10`, `runtime.diagnostics = true`; 16:44:04 → 16:50:45, 64 outer, 395.7 s, CONVERGED (in a spike state) |
| target-side trajectories | 0 | retained C480 canary reused |
| other meshes, nine-mesh, SOCP | 0 | |
| offline same-state evaluations | 27 | 9 frozen states × {T, S1, S0}; each: FE, eigs, gradients, filter, rows, 1–2 full inner solves; zero ρ updates |
| sweep verification | 18 | one SIMP + eq.(4) FE + eigs per committed run |
| retained single-factor config checks | 3 | A2 vs S240 (MATLAB leaves), P1 vs P2 (leaves), C3 vs S800 (describe text) |

## Deviations and events

1. **Snapshot relocation.** Initially extracted to `source_snapshot/`; moved to
   `source_snapshot/+olhoff_6b08708/` (before preregistration) because bare Olhoff function names under
   `analysis/` would be picked up by the six `addpath(genpath(<repo>/analysis))` scripts and make the
   production path gate fail closed. Re-verified 405/405 afterwards.
2. **Amendment 1** (resolver metadata in the preflight), see file.
3. **External edit of the source working tree** at 16:32:45 (`repro/PLAN_OLHOFFCURRENT_UPDATE.md`
   +55 lines, uncommitted), not by this audit; recorded in `evaluations/source_worktree_*`; the audit
   binds to the committed snapshot. The diff was byte-identical at start of recording and at the end.
4. **Pre-existing executed C480 SOCP treatment** in the target tree, contrary to the brief's premise;
   not executed or modified here (TARGET_IDENTITY.md).
5. Tooling fixes during analysis, none affecting a verdict: class-method candidates excluded from the
   path assertion (same rule as the production gate); the target session's path guard kept alive; a
   numpy-vs-BLAS norm comparison replaced by a vector comparison; matrix-vs-vector `mean` replaced by the
   solver's per-vector form in the M1 replay check; a CSV boolean parsing fix.
6. `node` is not installed, so the chart palette validator could not run; figures use the reference
   palette's documented passing categorical order unchanged.
7. Exploratory, not preregistered and not used for verdicts: island-detachment measure; low-density
   band occupancy; kinetic energy by density band.

## Reproduce

```
cd analysis/OlhoffCurrent/diagnostics/scientific_delta_olhoff_migration
python3 scripts/sd_snapshot_manifest.py                                  # snapshot identity
matlab -batch "addpath('scripts'); sd_target_identity()"                # Part 2
matlab -batch "addpath('scripts'); sd_verify_sweeps()"                  # Part 3
matlab -batch "addpath('scripts'); sd_m1_run()"                         # the one run (≈7 min)
matlab -batch "addpath('scripts'); sd_m1_post()"                        # replay + P-prefix
matlab -batch "addpath('scripts'); sd_same_state_run('T','retained')"   # and ('S','retained'), ('T','m1'), ('S','m1')
matlab -batch "addpath('scripts'); sd_config_dump()"
cd scripts && python3 sd_trajectories.py && python3 sd_analysis.py && python3 sd_same_state_compare.py \
  && python3 sd_lowdensity_kkt.py && python3 sd_band.py && python3 sd_islands.py && python3 sd_tables.py \
  && python3 sd_delta.py && python3 sd_docs_generated.py && python3 sd_figures.py && python3 sd_diagrams.py \
  && python3 sd_finalize.py
```
