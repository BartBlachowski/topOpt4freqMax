# Timing specification

Iteration count is primary. Timing is platform-dependent supporting information.

Phase 1C preserves the Phase-1A timing architecture. The reference trajectory is a
separate calibration activity; its time is disclosed as `T_reference` and is never folded
into time-to-measurement endpoints.

## 1. Time boundaries

For a certified result, define:

- `T_init`: method initialization required before its first method-level iteration,
  including the Proposed frozen-solid reference eigensolve;
- `T_loop_to_enter`: cumulative native optimization-loop time through `k_enter`;
- `T_loop_to_cert`: cumulative native optimization-loop time through `k_cert`;
- `T_native_finalize`: method's ordinary work required to return its native terminal
  spectrum/state when stopped at `k_cert`;
- `T_result_to_cert = T_init + T_loop_to_cert + T_native_finalize`;
- `T_gate_offline`: common E1/E2/E3, topology, and persistence analysis time;
- `T_observer_after_cert`: measurement-trajectory continuation after `k_cert` used only
  for audit diagnostics; it cannot establish or alter the separate reference.
- `T_reference`: complete optimization plus offline E1/E2/E3 reference-generation cost;
  separately reported calibration work.

Define `T_result_to_enter` analogously, using loop work through `k_enter` and the ordinary
finalization of a replay stopped there. The main supporting times are the paired
`T_result_to_enter` and `T_result_to_cert`. `T_gate_offline` and
`T_observer_after_cert` are disclosed but are not charged as minimum optimizer work.
Common evaluator time is excluded equally for all methods because it is an experiment
measurement, not part of any native algorithm.

Time to enter is a retrospective maturation quantity; time to certify is the conservative
prospective quantity. Both are main, and persistence cannot yet be known at entry.

## 2. Measurement strategy

The trajectory-discovery run records densities and diagnostics and may be I/O-heavy.
Do not publish its wall time as algorithm time.

After offline determination of all q-level endpoints, run deterministic, lightweight
timing replays of the same frozen profile with fixed horizons at every distinct robust
`k_enter(q)` and `k_cert(q)`, deduplicating equal horizons, without
common evaluators, images, density-history disk writes, or post-certification continuation.
The replay may suppress native stopping only to reach the already-determined fixed count;
the prefix must pass the established extension-invariance check.

Use one discarded warm-up per method outside the production mesh set, then three serial
single-thread timing replays per `(method,mesh,endpoint)`. Report median and range (and MAD
in supplementary machine-readable output). Randomize run order with a frozen balanced
schedule. A failed or trajectory-mismatched replay is not discarded as a timing outlier;
it is investigated and visibly excluded with reason.

If the computational cost of three large replays is rejected by human review, report only
the instrumented cumulative kernel timers and label the timing evidence `single-run
descriptive`. Do not mix the two policies across methods, meshes, q levels, or endpoints.

## 3. Method-specific quantities

### Proposed

- `T_init`;
- `T_OC_loop_to_cert`;
- `T_OC_loop_to_enter`;
- `N_OC_to_cert`;
- mean loop seconds per OC update to enter and to certify;
- `T_native_finalize` (terminal native eigensolve/post-analysis);
- `T_result_to_enter` and `T_result_to_cert`.

### Yuksel

- `T_init`;
- complete `T_stage1` and `N_stage1`;
- `T_stage2_to_enter`, `T_stage2_to_cert`, and their counts;
- separate mean seconds per Stage-1 and Stage-2 iteration;
- `T_result_to_enter` and `T_result_to_cert`;
- combined mean only as `T_loop_to_cert/N_total_to_cert`, never as a replacement for
  stage means.

### Olhoff

- `T_init` if separable, otherwise `NA` plus unattributed time;
- cumulative eigensolve, gradient, and LP-call time through each endpoint;
- mean seconds per outer iteration to each endpoint;
- successful/attempted LP calls;
- MATLAB-reported LP solver iteration total/mean/median/max if Phase 2 captures it;
- `T_native_finalize` for evaluation of the return-equivalent terminal state;
- `T_result_to_enter` and `T_result_to_cert`.

The frozen audit runner currently records `tEig`, `tGrad`, and `tInner`, and wall time,
but does not separate initialization/finalization. Missing values remain `NA`, not zero.

## 4. Main versus supplementary timing

Main timing evidence:

- median `T_result_to_enter` and `T_result_to_cert`;
- mean seconds per method-level iteration to both endpoints (stage-specific for Yuksel);
- a compact platform key.

Supplement:

- `T_init`, loop, finalization, and gate-evaluation decomposition;
- all three timing replay values, median/range/MAD;
- Yuksel stage times;
- Olhoff eigen/gradient/LP time and LP solver-iteration statistics;
- observer continuation and diagnostic I/O time.
- reference-run optimization, offline evaluator, and stabilization-scan time.

## 5. Platform record

Archive for every timing batch:

- CPU model and core topology;
- RAM size;
- OS name/build and architecture;
- MATLAB version/update and platform architecture;
- requested and active MATLAB computation thread count;
- BLAS/library information available from MATLAB;
- Optimization Toolbox and `linprog`/HiGHS version and exact options;
- eigensolver settings, tolerance, start-vector policy, and requested mode count;
- repository commit, dirty-tree manifest, profile/config hashes;
- numerical precision;
- process isolation, run order, warm-up identifier, power/thermal policy if controlled;
- timestamp and timezone.

Use the same reference platform, one thread, serial execution, and disabled unrelated
plotting/I/O for every method. State explicitly that iteration counts are intended to be
more portable; no hardware-independent timing claim is permitted.

## 6. Mesh-scaling outputs

For every method, plot mean native-loop cost per method-level iteration against `Ne` for
both endpoints. Proposed uses OC updates, Yuksel shows Stage 1 and Stage 2 separately plus
the explicitly labelled chronological combined mean, and Olhoff uses outer updates.
Separately plot `T_result_to_enter` and `T_result_to_cert` against `Ne`. Fit and censor
these series only under `SCALING_AND_FIGURE_SPEC.md`; timing trends are descriptive for
the recorded reference platform, not portable asymptotic complexity claims.

Primary timing curves use q=98/99/99.5 enter endpoints as defined in
`QUALITY_EFFORT_SPEC.md`. Certification times remain visible companions. Reference time is
shown in a separate calibration-cost panel and never added to `T_enter/T_cert`.

## Phase 2H timing amendment

Candidate C eigensolves, modal diagnostics, binary endpoint diagnostics, topology gates,
and persistence scans are common offline assessment and remain excluded from native
endpoint timing. Report evaluator cost separately. Olhoff-LP and Olhoff-MMA receive
separate one-thread serial timing rows and warmups; timing from one route cannot proxy the
other.
