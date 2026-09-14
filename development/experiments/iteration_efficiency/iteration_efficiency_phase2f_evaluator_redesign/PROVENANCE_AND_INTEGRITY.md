# WP0 / WP26 — Provenance, integrity, and forensic history
PHASE 2F — EVIDENCE GENERATION ONLY — NO METHODOLOGY AMENDMENT — NO REFREEZE — NO PRODUCTION

## WP0 — state at the start of this phase

    branch  benchmark-methodology-r2
    HEAD    632e9b01811845709de33f93051fd853373ed5e1
    git status  19 entries (3 modified tracked files, 16 untracked directories/files),
                unchanged from the Phase-2D and Phase-2E baselines

Full hash records: `WP0_INTEGRITY_pre.json` and `WP0_INTEGRITY_post.json`
(script `scripts/wp0_hashes.py`).

| group | entries | mismatches (pre) | mismatches (post) |
|---|---|---|---|
| `protected_numerical_sources` (Phase-2A record) | 6 | **0** | **0** |
| `profile_sources` | 3 | **0** | **0** |
| `audit_records` | 6 | **0** | **0** |
| Phase-2D declared-unchanged file list | 9 | **0** | **0** |
| Phase-2D `SHA256SUMS.txt` self-check | 28 files | **0** | **0** |
| Phase-2E `SHA256SUMS.txt` self-check | 54 files | **0** | **0** |

The frozen common evaluator
(`analysis/three_method_parametric_study/study_evaluate_design.m`,
`22a1b974c251dbe7baa6499a5aca11e6bde68469b6d2e80fd4274c9447f31343`) and the Phase-2A
contract (`46318e6c…`, matching the hard-coded `ie2a.frozen_contract_sha256()`) are
byte-identical to their pinned digests. All eight stored Olhoff density trajectories were
hashed and are unchanged.

**Nothing outside `analysis/iteration_efficiency_phase2f_evaluator_redesign/` was modified
by this phase.** No optimizer was run, no trajectory regenerated, no methodology or contract
touched, no authorization token set.

The two stale contract-pinned `normative_documents` digests recorded as Phase-2E finding
**N1** remain stale. This phase does not correct them — correcting them is a refreeze
action, and no refreeze is authorised.

## Independence of this phase's computation

The modal engine (`scripts/modal_engine.py`) is written from the mathematical specification
of the frozen evaluator, in Python/NumPy/SciPy, and **extended** to return eigenvectors and
per-mode energy diagnostics that the frozen evaluator does not compute. It was validated
three ways:

| check | result |
|---|---|
| frozen E1/E2/E3 `omega_raw(1)` at k = 252 vs the Phase-2D MATLAB pipeline | agree to **4.6e-12 … 6.5e-12** relative |
| ARPACK sparse shift-invert vs dense LAPACK `eigh` (non-iterative, no shift), 14 modes | agree to **2.5e-09** |
| assembled operator symmetry `‖K − Kᵀ‖∞` | 9.3e-10 absolute on a matrix of scale 1e7, i.e. round-off; the frozen `(K+Kᵀ)/2` symmetrisation is retained for fidelity |

All production computation is **sparse** (`scipy.sparse` assembly, ARPACK shift-invert at
σ = 0 over a sparse LU). Profiling at 720×90 (131 222 DOF): eigensolve 2.69 s, assembly
0.13 s, symmetrise+slice 0.04 s — the eigensolve dominates at 94%, and neither CHOLMOD
(`sksparse`) nor PARDISO is installed, so no reusable symbolic factorisation is available.
The single dense solve in this phase is the deliberate non-iterative cross-check above.

## WP26 — forensic classification of the preceding phases

Each description below was checked against the evidence, not accepted from its own report.

### Phase 2B — VALID NEGATIVE PRECISION RESULT UNDER THE PRE-AMENDMENT Eq. (4)

Confirmed. Phase 2E re-implemented the frozen reference/persistence engines independently
and reproduced Phase-2B's stored outputs **exactly**: `b_ref` 2200 (double) vs 2100
(single); `k_enter` 233/315/609 vs 232/309/524; `k_cert` 332/414/708 vs 331/408/623;
`Q_ref` to 11 digits; `B_meas` 3200 both. The experiment was correct and the mechanism it
identified was correct. It tested the instrument frozen at the time, and it does not
transfer to any continuous replacement law.

### Phase 2C — VALID DIAGNOSIS OF THE Eq. (4) DISCONTINUITY

Confirmed. Phase 2E independently reproduced its central measurements: the binding-evaluator
instability 751/3200 = 23.47% on the 96×12 horizon-3200 trajectory under the frozen `Q_ref`
normalisation, with binding shares (2695, 20, 485) matching Phase-2C's
`EVALUATOR_BINDING_SHARE.csv` exactly; and the 4.0021e-03 and 2.6736e-02 sensitivity
figures to five significant figures. Its reading of the Du & Olhoff source was verified
against the primary PDF.

### Phase 2D — VALID NUMERICAL DIAGNOSIS; REJECTED METHODOLOGY PROPOSAL

Both halves confirmed, and they must not be conflated.

**What Phase 2D got right.** Every stability claim it made is reproducible. Eq. (4a) does
remove the discontinuity: branch-side sensitivity on 1600 production states falls from
2.6496e-02 to 2.6560e-10, matching the branch-free E1 control at 2.6252e-10; float32
sensitivity falls from 2.6736e-02 to 5.5960e-08 against E1's 5.5949e-08. Its scope
discipline was exemplary — two functional lines, E1 untouched, every protected hash
preserved, the frozen evaluator and contract deliberately left alone, and the loss of
native identity disclosed rather than concealed. Its source provenance for Eq. (4)/(4a)/(4b)
verified against the primary PDF without correction.

**Why the proposal was nevertheless rejected.** Every Phase-2D experiment measured the
evaluator's response to a *perturbation* of a state; none measured the *value* at a state.
The spurious-mode reintroduction was present in Phase-2D's own published
`AMENDED_OLHOFF_TRAJECTORY_EVALUATION.csv` and was not analysed.

**Phase 2D is not globally wrong.** It is a sound numerical diagnosis attached to a
methodology proposal that later physical evidence refuted.

### Phase 2E — INDEPENDENT AUDIT; ARTIFICIAL-MODE DISCOVERY; REFREEZE CORRECTLY REFUSED

Confirmed. Its decisive k = 252 finding is reproduced in this phase to 5e-12 against the
Phase-2D MATLAB values and cross-checked against a dense non-iterative solver
(`K252_MODAL_REPRODUCTION.csv`). Its reference-length ruling (gap non-blocking, with a
measured 922× decision margin) is not reopened here, per this phase's instructions.

## What this phase adds

Phase 2E discovered the artificial-mode reintroduction at a handful of states and bounded
its extent by a screening ratio. This phase characterises the phenomenon systematically:
its population structure, its mode-count requirements, its threshold behaviour, its spatial
form, and whether any of candidates A–D yields a method-independent, physically meaningful,
numerically stable structural eigenfrequency for gray intermediate states.
