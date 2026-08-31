# Phase 2 implementation and execution requirements

This is a design contract, not authorization to implement or run it.

Phase-1C delta: Phase 2 now runs **two trajectories per (method, mesh)** — a reference
trajectory that freezes `Q_ref` under `REFERENCE_QUALITY_SPEC.md`, and a separate
measurement trajectory scanned for the q-level endpoints. Endpoints exist at
q = 98%, 99%, 99.5% under the all-evaluator gate, and the topology gate is the repaired
rule in `TOPOLOGY_SANITY_SPEC.md`. Reference work is reported but never charged to
`k_enter`/`k_cert`.

## 1. Isolation and immutability

- Create a new runner/config/result namespace for the iteration-efficiency study.
- Do not edit, overwrite, import results into, or relabel anything under
  `examples/Performance/final_campaign/` or either completed audit directory.
- Bind the exact selected practical algorithm profiles by ID and source hash. Budget and
  observer instrumentation changes must be explicit new-study fields, never edits to the
  old manifests.
- Disable plots and common evaluator calls inside optimization loops.
- Refuse an output directory outside the new study result root.

## 2. Frozen algorithm profiles

Recommended new-study bases:

- Proposed: `proposed_practical_move02_tol001`;
- Yuksel: `yuksel_practical_move01_tol001`;
- Olhoff: `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1`, with its
  algorithmic S1 policy retained but its old 1600 value treated only as prior budget
  evidence.

No move, tolerance, filter, interpolation, lower bound, optimizer, stage rule, or S1
trigger may be tuned in this study. The profile choice is fixed by the completed
pre-study freeze and is an explicit independent-auditor challenge, not a result-contingent
Phase-1A decision.

## 3. Reference and measurement budgets

### 3.1 Reference phase

The reference trajectory uses the single common cap

`B_ref = 3200` acceptance-eligible method-level updates,

fixed in `REFERENCE_QUALITY_SPEC.md` before Phase 2. It is a censoring/resource boundary
only. **There is no terminal-cap fallback**: if the stabilization rule does not fire by
`B_ref` the cell returns `REFERENCE_NOT_ESTABLISHED`, and if a required solver terminates
first it returns `REFERENCE_SOLVER_TERMINATION`. This is what makes `Q_ref` independent of
the resource horizon (finding C2), so the engine must never substitute the best floor at
the cap.

### 3.2 Measurement phase

Use only completed pre-study evidence. For each native stage/outer loop let `K_prior` be
the largest prior budget or successful count in the frozen nine-mesh campaign and set

`B0 = ceil_to_100(max(2*K_prior, K_prior + 5*P))`.

With `P=100`, the current evidence yields:

- Proposed: `B0_OC = 900` (largest prior count 330);
- Yuksel: `B0_stage1 = 2000`, `B0_stage2 = 2000` (prior per-stage horizon 1000);
- Olhoff: `B0_outer = 3200` (prior horizon 1600).

These are lower-bound safety horizons, not convergence thresholds. After the reference pass
has frozen `b_ref`, set the exact per-cell measurement horizon to

\[
B_{meas}=\min\{\max(B_0,b_{ref}+P-1),B_{ref}\}.
\]

Here `B0` is the lower bound for the acceptance-eligible measurement stream: Proposed OC,
Yuksel Stage 2, or Olhoff outer updates. Yuksel Stage 1 retains its separately frozen
`B0_stage1=2000` handoff budget.

This automatic rule is the complete measurement-budget contract. It is deterministic,
method-blind, parameter-free beyond the already-frozen `B0`, `b_ref`, `P`, and `B_ref`, and
is fixed before the measurement scan. It provides the persistence tail through
`b_ref+P-1` whenever that tail lies within the common demonstrated reference cap; the outer
`B_ref` bound forbids extrapolation beyond the reference trajectory. The prior
progress-triggered tranche is deleted. There is no masked progress review and no
result-contingent measurement extension. Reaching `B_meas`
without a certified window produces the applicable final `NOT_REACHED` subclass.

The measurement trajectory must be a clean run from the same initialization to `B_meas`
unless checkpoint resume has separately proved bit-identical. Its entire retained prefix
must match the reference trajectory at shared counts; otherwise classify an implementation
failure rather than joining two different trajectories.

### 3.3 Disclosed non-neutral budget change (finding M6)

Yuksel Stage 1 hit its own 1000-update cap at **640x80, 720x90 and 800x100** in the frozen
campaign. `B0_stage1 = 2000` therefore lets Stage 1 run longer at those three meshes,
changing its endpoint and hence Stage 2's initial design. Raising a previously binding cap
is not a neutral safety horizon: it changes the realized algorithm at three of nine meshes.

Every report must state that at those three meshes `N_stage1`, the Stage-2 trajectory, and
the chronological total are **not comparable** to frozen campaign values, which remain
budget evidence only. Retaining the binding 1000 cap was considered and rejected: it would
make Stage-1 reference stabilization unreachable by construction.

## 4. Return-equivalent trajectory recorder

Record every method-level state needed by the exact scan:

- global count, native stage and local count;
- return-equivalent physical density field and raw design field where different;
- state-before/state-after mapping and a content fingerprint;
- raw volume, native objective, change metrics, move activity;
- cumulative kernel timing excluding recorder I/O;
- native stop flag/iteration without breaking the observer run;
- stage/policy transitions and continuation counters;
- required solver flags and finite-state checks;
- Olhoff native spectrum/multiplicity and LP output including `output.iterations`;
- Yuksel mode-estimate diagnostics naturally available without extra solves.

Store densities losslessly in double precision in chunked `-v7.3` artifacts. A compressed
single-precision path for **new** trajectories is permitted only after a no-solve/short-run
validation proves that every gate decision, `k_enter`, `k_cert`, and E1/E2/E3 frequency
within a documented bound is identical to the double path. A checksum is not a substitute
for a recoverable field.

**Historical single-precision path (finding Mo9).** The frozen Olhoff `res.rho_snapshots`
are `float32` while `res.rho` is `float64`, so the double-path validation above cannot be
run retrospectively. The historical snapshots are nonetheless **explicitly permitted** for
offline rescans at the eight available meshes, on the recorded verified equivalence: at
those final states the `float32` snapshot and the `float64` field give identical component
counts and detached areas under the exact-count projection, and about 7-digit precision is
comfortable against a 1% spectral band. Olhoff 800x100 is unavailable (`RUN_ERROR`, E1
`N/A`) and is not part of that equivalence claim. The permission is scoped: any *new* offline
quantity derived from the historical snapshots must repeat the equivalence check and record
it. This resolves the contradiction between the evidence matrix and this section rather
than leaving a rule the historical data cannot satisfy.

Checkpoint identity test: stop a matched run at several counts, execute its ordinary
finalization, and prove its returned field equals the recorder's `X(k)`. This specifically
guards Proposed/Yuksel/Olhoff update-order differences.

## 5. Observational invariance

- Reuse extension modes that suppress only native termination; do not suppress Yuksel's
  Stage-1 handoff or Olhoff's S1 trigger.
- At one coarse and one fine mesh per method, compare observer-on versus observer-off
  prefixes through the native stop: all fields and scalar histories must match exactly
  except declared timing/I/O fields.
- Capture `linprog` output without changing matrices, bounds, objective, algorithm,
  ordering, or update decision; repeat the existing Olhoff mirror-identity style test.
- **Do not edit the frozen reproduction tree (finding Mo8).** Solver-internal LP iterations
  require the fourth output of the `linprog` call in
  `Matlab/reproduction2007/algo/innerLoopLP.m`, which is guarded by
  `repro2007_tree_hash.m`. Editing that file would break the tree hash and cross the
  harness/algorithm boundary this protocol itself draws, even though the call is
  behaviourally identical. Instrument instead via a **diagnostic mirror outside the frozen
  tree**, following the existing `innerLoopLPDiagnostic.m` and
  `olhoffOptStabilizedDiagnostic.m` pattern, with a mirror-identity test against the frozen
  routine. If the mirror is not built or its identity test fails, the quantity is `NA` and
  the solver-internal curve stays supplementary; it never licenses an edit to the frozen
  tree, and it is never inferred from `st.nInner=1`.
- Do not add common eigensolves to Proposed or Yuksel optimization loops.

## 6. Offline acceptance engine

The engine must:

1. validate profile/source/result hashes;
2. build the exact binary projection with stable index tie-break;
3. compute support-footprint components and all topology metrics, applying the repaired
   gate `C_required=1 AND max detached physical area < A_sig` with mesh-specific
   `a_sig(j)=ceil(A_sig/A_e(j))`, and recording aggregate detached area, component count
   and LCC as diagnostics only — **no aggregate-area veto**;
4. compute E1, E2 and E3 for every return-equivalent state required for an exact
   earliest-window scan, caching by density fingerprint; no evaluator is privileged;
5. compute raw and binary evaluator trajectories without changing the optimizer;
6. recompute native Olhoff `N` and gap on the same return-equivalent field, avoiding the
   existing pre-update history offset;
7. **run the reference pass first**: on the reference trajectory only, build `F_e(b)`,
   evaluate `g_e(b)` at block endpoints, stop the logical scan at the first `b_ref` where
   all three evaluators satisfy `g_e <= epsilon_ref`, and freeze
   `(Q_ref_E1,Q_ref_E2,Q_ref_E3)` with provenance hashes. The scan must be causal: it may
   not inspect later quality to select a different `b_ref`, and it must emit
   `REFERENCE_NOT_ESTABLISHED` rather than any cap-based value;
8. **run the measurement pass second**, receiving only the frozen triplet and its
   provenance, computing `r_e=Q_e/Q_ref_e` and `r_all=min_e r_e`, and scanning for
   `k_enter(q)`/`k_cert(q)` at each of q = 0.980, 0.990, 0.995 plus the per-evaluator
   decompositions. The measurement pass must be structurally incapable of recomputing a
   reference from its own horizon;
9. apply status precedence and produce an auditable Boolean reason per state;
10. emit the paired `k_enter`/`k_cert` family, `k_gate`, `k_native`, budgets, failure
    attempts, `N_reference`, and reference definitions;
11. verify reference/measurement trajectory fingerprints at shared counts and at every
    reported endpoint; a mismatch is an implementation failure, never a new reference
    opportunity;
12. regenerate every table solely from machine-readable trajectory/gate outputs.

No manual image judgment or interactive threshold change may enter the engine.

## 7. Timing replays

After the blinded offline acceptance freeze, run lightweight fixed-horizon timing replays
at both endpoints as specified in `TIMING_SPEC.md`. Verify trajectory fingerprints at
`k_enter` and `k_cert`. Do not charge observer continuation, gate evaluation, or image
generation to minimum-work time.

## 8. Automatic scaling and figure production

Generate the mandatory observations, fits, status-aware figures, topology grids, and
plot-data exports in `SCALING_AND_FIGURE_SPEC.md` from the canonical result tables. The
pipeline must enforce the three-valid-mesh fit minimum; exclude every censored/nonpositive
cell with a recorded reason; report `C,p,R2_log,n_valid,Ne_min,Ne_max`; limit fit lines to
the observed valid range; and verify `k_cert-k_enter=P-1` for every finite pair.

Capture Olhoff `linprog` solver iterations only from genuine solver output. Never infer
them from the selected code's `st.nInner=1`. If they are unavailable, write `NA` and keep
the solver-internal curve supplementary. Export vector figures, 300-dpi PNGs, exact source
tables, and hashes. Labels must describe empirical tested-range scaling, not asymptotic
complexity.

## 9. Required result artifacts

At minimum:

- immutable study manifest and protocol hash;
- per-run configuration/source hashes and platform record;
- full return-equivalent density trajectory or lossless chunk references;
- scalar trajectory CSV/Parquet with data dictionary;
- per-state gate-reason and topology tables;
- common evaluator cache with solver status;
- frozen reference record per cell: `b_ref`, `Q_ref_E1/E2/E3`, all `g_e(b)` values,
  `F_e(b)` trajectories, `N_reference`, `T_reference`, `B0`, `B_meas`, the four inputs to the
  frozen budget equation, and the superseded-horizon diagnostic at 900/1600/2000/3200;
- paired R `k_enter`/`k_cert` summary at every q level with `k_gate` and `k_native`; an A
  summary only if independent `Omega_req` is present, validated, and provenance-locked;
- timing replay raw/summary tables;
- scaling observation/fit tables and F1--F7 source-data slices;
- status/budget/failure ledger;
- standardized accepted-state images;
- independent audit report and checksums.

## 10. Development/production sequence

### Phase A — implementation and no-solve preflight

Schema, hash gates, exact projection/component unit tests, status precedence, synthetic
trajectory tests, budget-equation boundary tests, state-index contracts, the
reference/measurement pass separation test,
the causal first-passage `b_ref` test (including a synthetic trajectory that improves again
after stabilizing, which must **not** revise a frozen reference), the LP diagnostic mirror
identity test, and disk/memory estimates.

**Resource estimate (finding Mi4).** The audit estimated the offline acceptance engine from
frozen Olhoff eigensolve scaling (`t = 1.316e-6 * Ne^1.214` s for a 5-mode solve) at about
**6.4 h** single-threaded for E1-raw across nine meshes and three methods at full budgets,
about **38 h** for the complete E1/E2/E3 x raw/binary set, and about **20 GB** uncompressed
trajectory storage. Phase 1C roughly **doubles** both figures because each cell now carries
a reference trajectory in addition to the measurement trajectory: budget on the order of
**75 h** offline evaluation and **40 GB** storage, with the reference-phase share listed
separately so it remains visible as calibration cost. The q-family adds scan cost but not
eigensolve cost, since the three levels reuse one cached quality trajectory. These are
indicative extrapolations; actual cost depends on how many states the earliest-window scan
must evaluate and on caching effectiveness.

### Phase B — tiny engineering smoke tests

Nonproduction even-`nely` meshes; exercise both stages, S1 trigger simulation, solver
failure injection, chunk recovery, and table regeneration. Results are not scientific.

### Phase C — representative frozen pilot

Required at 240x30 after every instantiated scientific threshold is frozen. R proceeds
without A; if no independently sourced `Omega_req` exists, A is absent rather than an
implementation blocker. The pilot must exercise the **full two-phase sequence**: reference
run, reference freeze, then measurement run and scan. The pilot's purpose is checkpoint
identity, observer invariance, reference/measurement separation, evaluator feasibility, and
evidence completeness—not threshold selection or profile tuning. Mask method identities in the
gate report. Any correction must be an engineering fix with a repeated identity test.

### Phase D — full nine-resolution production

Run all reference trajectories and freeze every `Q_ref` first; then run the measurement
trajectories; then freeze the offline scan; then timing replays. Accept all outcomes,
including `NOT_REACHED`, `REFERENCE_NOT_ESTABLISHED`, and a ranking that changes with `q`.

### Phase E — independent post-run audit

Recompute a sample of density fingerprints, projections, evaluator values, component
labels, k windows, timing boundaries, frozen horizons, and table rows from raw artifacts.

A representative pilot is necessary because return-equivalent indexing and full
trajectory instrumentation are new for Proposed/Yuksel. Existing evidence removes the
need for a result-seeking parameter pilot; it does not remove this engineering/scientific
integrity gate.

## 11. Result firewall

All acceptance constants (`A_sig`, `P`, `L_ref`, `epsilon_ref`, `B_ref`, the q levels,
volume tolerance, Olhoff gap), the repaired topology gate and the demoted T0 diagnostic
role, primary-R hierarchy, endpoint definitions,
censoring, the deterministic `B_meas` equation, fit model, mesh range, and figure/table inclusion rules are protocol-hashed
before the first production identity is unblinded. Completed-campaign data remain
read-only contextual evidence. Production results cannot trigger a threshold, profile,
budget-formula, per-cell horizon, fit-range, or display change; any necessary engineering correction is
documented, identity-tested, and rerun uniformly without inspecting comparative ranks.

## Phase 2H implementation gate

The harness shall call the Candidate C evaluator once per stored state, archive complete
per-evaluator modal evidence, and feed only its actual-gray selected frequencies into Q.
Adaptive search is 3, 6, 12, 24, ... with no scientific ceiling. Every numerical or
resource exhaustion path must return `STRUCTURAL_MODE_NOT_FOUND` and NaN Q. Preflight
must bind the evaluator, contract, and refreeze hashes and reject missing, stale,
wrong-classifier, wrong-scope, or wrong-route qualification artifacts. Production remains
locked until precision, cross-method, and reference-length qualifications all pass.
