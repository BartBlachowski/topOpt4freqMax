# PROVENANCE

## Identity

| | |
|---|---|
| repo / branch / HEAD | topOpt4freqMax / `benchmark-methodology-r2` / `013cc48451d33bed61c5c4eea174bbd898d548a2` |
| implementation | `analysis/OlhoffCurrent`, `+impl` tree `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (verified before preflight, at launch, after the run, and at finalization) |
| production `olhoffSolve.m` | SHA-256 `1e5a114c…ba3`, unmodified |
| MATLAB | 25.2.0.2998904 (R2025b), `/Applications/MATLAB_R2025b.app`, `-batch`, one numerical thread |
| Python | 3.13, numpy 2.3.4, scipy 1.16.3, matplotlib 3.10.7, h5py 3.16.0; nothing installed |
| preregistration | `AUDIT_PREREGISTRATION.md` SHA-256 `5b6186f2f2b438f73ce4cfeae8e4565326dc91d99b19e207aa47604c32286ddd`, frozen 11:22 |
| amendment 1 | `PREREGISTRATION_AMENDMENT_1.md` SHA-256 `298efd23763f678f7519d37dc7c40f6e693bad2bec8a1933e4d853e0f09fa340`, frozen 11:37 (pre-launch) |
| treatment code freeze | 19 MATLAB files hashed into `evaluations/preflight.json`; all verified by the runner at launch and at finalization |

## Timeline, 2026-09-13 (+0200)

| time | event |
|---|---|
| ≤ 11:20 | read-only inspection of production code and prior evidence |
| 11:22 | preregistration frozen |
| 11:23–11:25 | Part 1 control identity: `C480_CONTROL_EVIDENCE_PASS` |
| 11:26–11:34 | driver copy, adapter, certificate, preflight scripts written; P1, P4 run |
| 11:34 | **P4 FAILED** with the preregistered `augmented` primary (d2 0.0107, dinf 0.745) |
| 11:35–11:36 | pre-launch degeneracy diagnostic (`cs_preflight_degeneracy.m`) |
| 11:37 | Amendment 1 frozen: `schur` first, cross-solver telemetry every iteration |
| 11:38–11:42 | P4, P5 re-run and passed; P3, P6, P7 passed; P2 diff audit passed after one software fix (below); verdict `C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS`; code freeze; dry run |
| **11:43:07** | **launch** (pid 1109), lock written |
| 11:45–11:46 | control-side spectral evaluation (separate single-thread MATLAB process, ~30 s) and Python analyses ran concurrently with the treatment |
| **11:54:15** | **termination** `SOCP_CERTIFICATE_FAILURE` at outer 15; evidence written; process exit 0 |
| 11:55–12:08 | termination record extracted; post-hoc apex diagnostic (sequential, after the treatment process ended); post-run analyses; figures; documents |

## Deviations, fixes and disclosures

1. **Amendment 1 (pre-launch).** The preregistered primary backend `augmented` failed P4's
   design-space bars on the frozen oracle problem because problem (25) has a numerically
   flat optimal face there. The attempt order was swapped to `schur` first, and the
   cross-solver diagnostic runs every iteration. No threshold changed and no bar was
   waived. The failing P4 record is retained as `evaluations/preflight_P4_original_augmented_FAIL.json`.
2. **Software fixes before launch.** No C480 treatment state existed when these were made.
   - P2 flagged one untagged blank line in the driver copy; it was removed and P2, P3 and
     P6 were re-run.
   - Checkpoints were switched to `-nocompression` (P6 re-run).
   - An unused helper was removed from `cs_socp_table.m`.
   - A `mu = max(mu, ‖p‖)` roundoff projection was added to the certificate before any
     test used it. It is mathematically a dual-feasible projection and keeps every bound valid.
   - A missing `bestGap` default was added for early-return certificates.
   - A `dryrun` mode and a strict file-count check were added to the runner.
   - The code freeze was re-issued after each change, before launch.
3. **No change after launch** to any treatment code, threshold, formulation, controller,
   move or filter. No repair, no resume and no rerun. The checkpoint (every 25) was never
   written, because the run ended at 15.
4. **Concurrency.** A short control-side MATLAB spectral evaluation and Python analyses ran
   on other cores during the treatment. Timings are indicative only (preregistration §15).
5. **Post-run analysis additions.** These were not preregistered and are descriptive only.
   - A matched-iteration control state (ρ₁₄: `cs_endpoint_spectral('control14')`,
     stationarity, geometry, topology, MASTER_METRICS column) was added because the
     treatment has no endpoint.
   - Figures were adapted to the terminated run (zoom panels, matched-control panels,
     summary figure).
   None of these enters a verdict.
6. **Post-hoc apex diagnostic** (`cs_posthoc_apex.m`). This is a frozen re-posing of the
   rejected outer-15 problem: no design update, no continuation. It is labelled POST-HOC
   and excluded from all verdicts. It first failed on a `sparse` indexing typo in the
   diagnostic itself, which was fixed and re-run. Dual solve: 467 s.
7. **Canary manifest header.** The control study's `FINAL_SHA256.txt` header text
   ("scientific runs 0 … NOT_REACHED") is stale; all hashes in its body verify
   (`CONTROL_IDENTITY.md`). Recorded, not repaired.
8. **Toy smoke outputs** (`evaluations/toy_smoke/`, 96×12). These are software-verification
   artifacts, not scientific evidence, and are excluded from all analyses.
9. **`hist.nInner` semantics** under the treatment are interior-point iterations. This is
   telemetry, read by no controller.
10. **Launch wrapper** `nohup caffeinate -i` prevents idle sleep. It has no arithmetic effect.

## Raw evidence (repository evidence policy)

Stored in `analysis/OlhoffCurrent/evidence/c480_socp_causal_run/` (git-ignored) and
declared with SHA-256 in `EVIDENCE.json`:

- `C480x60_socp_trajectory.mat`: RHO/DRHO 28 800×14, move, hist, cfg, meta, exh, log, all 15 SOCP records including the rejected one, final ω/λ, status, treat
- `C480x60_socp_state.mat`: ρ₁₄, cfg, final spectrum
- `C480x60_socp_diag.mat`: `res.diag`
- `LAUNCHED.lock`: the one-run lock

The control trajectory was reused read-only (SHA-256 `a87546bc…ab9b`, re-verified at finalization).
