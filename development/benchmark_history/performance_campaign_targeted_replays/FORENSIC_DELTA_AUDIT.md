# Forensic-Audit Delta

This is a narrow delta against `PERFORMANCE_CAMPAIGN_FORENSIC_AUDIT.md`; the original report and its tables remain unchanged.

| Prior finding | New evidence | Delta | Publication consequence |
|---|---|---|---|
| Campaign is trustworthy for terminal/status, timing and common-evaluator observations | All three targets reproduce original status/endpoints; original hashes unchanged | **CONFIRMED** | Core campaign observations may be frozen |
| Exactly five observations are censored and excluded from fits | No original row or mask changed; replays remain diagnostic | **CONFIRMED** | Publish censoring explicitly |
| Proposed 109-to-159 transition is primarily model/interpolation dependence | Direct density and modes show weak-material local modes; same field rises to 153.68 raw E1 and 162.76 binary E1 | **REFINED** | Mechanism is now publication-ready with mesh-specific qualification |
| Proposed native coarse triplet is genuine under its frozen native model | Two identical runs exactly reproduce 109.05/109.49/112.92 and the original fingerprint | **CONFIRMED** | Report as deterministic native-model behavior |
| Proposed low-density interpolation sensitivity was inferred from scalar evidence | Mode energy/localization directly places essentially all three modes in rho<=0.1 material | **REFINED** | Can call them weak-material/local modes at 160x20 |
| Proposed coarse endpoints are fully stationary/KKT solutions | History now exists, but no KKT residual or basin/restart experiment was authorized | **UNCHANGED_EVIDENCE_GAP** | Do not make KKT claims |
| Proposed/Yuksel topologies were unavailable | Proposed 160 density and binary topology are retained and connected; Yuksel cross-method topology equivalence remains unproved | **REFINED** | Use Proposed topology figure; avoid topology-equivalence claim |
| Olhoff 640 first failed LP attempt is k=1067 | Bit-identical replay again fails after 1066 successful updates at attempted k=1067 | **CONFIRMED** | Failure location is publication-ready |
| Olhoff failure is dual-simplex-highs exitflag 0 / MATLAB iteration-limit class | Direct output: 38 iterations, message 'Solver stopped prematurely', empty point; local linprog source maps flag 0 to maximum iterations | **REFINED** | Report exact output and avoid implying a user-set 38-iteration cap |
| Olhoff failure may involve modal branching/LP degeneracy | Direct failed state is N=2 with gap12=0.00256, but normalized constraint rows are full rank with Gram rcond 0.0206 and no point exists for residuals | **REFINED** | Only generic LP iteration-limit causation is supported |
| Olhoff failure island is nonmonotonic and trajectory-dependent | Exact trajectory/failure reproduction plus larger healthy meshes from prior evidence | **CONFIRMED** | Not a monotonic resource-size claim |
| Olhoff best-prior states are diagnostic and not campaign successes | Replay preserves SOLVER_FAILURE and does not promote any prior state | **CONFIRMED** | Keep row censored |
| Yuksel 800 late mechanism was indeterminate | Final 300 show max 0.011-0.1, median max/RMS 97, 61 full-move hits, 3.2-3.5% active variables and small objective increments | **REFINED** | Classify localized irregular oscillation / persistent nonconvergence |
| Yuksel 800 might be a simple cap limitation | No late sample meets tolerance and the trajectory is not monotone decay | **REFUTED** | Do not recommend a modest extension as convergence completion |
| Yuksel 640 mechanism remains unresolved | 640x80 was not one of the authorized closure replays | **UNCHANGED_EVIDENCE_GAP** | Does not block freeze; retain as censored with prior qualification |
| Per-iteration exponents are Olhoff 1.194, Yuksel 0.975, Proposed 1.189 | No original timing or fit was replaced; replays reproduce numerical endpoints | **CONFIRMED** | Publish with one-sample and stage qualifications |
| Yuksel Stage-1/Stage-2 per-iteration fits must remain separate | Replay retains exact 1000+1000 semantics; timing stays diagnostic only | **CONFIRMED** | Keep stage-specific reporting |
| Total-time exponents are practical endpoint fits, not intrinsic complexity | Nothing in the replays changes cost-count decomposition | **CONFIRMED** | Preserve endpoint semantics |
| RAM measurements are unreliable | No memory repair or new memory benchmark was performed | **CONFIRMED** | Exclude quantitative RAM claims |
| Olhoff common-raw advantage is qualified, not universal topology superiority | No new evidence contradicts prior raw/binary evaluator findings | **CONFIRMED** | Retain the qualified wording |
| No full nine-resolution rerun is required | All three exact targets reproduced; no implementation corruption or changed campaign observation found | **CONFIRMED** | Freeze without broad rerun |
| Three narrow diagnostic follow-ups were required before freeze | All authorized replays and the permitted Proposed repeat are complete | **REFINED** | No further runs required |

## Focus findings

- **Proposed:** CONFIRMED and refined. MODEL / INTERPOLATION DEPENDENCE remains primary; the low triplet is now directly shown to be weak-material/local rather than evidence of a disconnected load-carrying skeleton.
- **Olhoff:** REFINED. The exact deterministic LP failure is confirmed, but the simple matrix diagnostics do not demonstrate degeneracy, scaling failure, or modal causation. The supported class is `GENERIC_LP_ITERATION_LIMIT_ONLY`.
- **Yuksel:** REFINED from indeterminate to `PERSISTENT_NONCONVERGENCE`, expressed as localized irregular oscillation with a practically stable but slowly drifting objective.
- **Campaign:** CONFIRMED. No replay invalidates an observation, timing record, or admissible-row scaling fit, and no broad rerun is required.

## Remaining gaps that do not block freeze

The precise internal reason HiGHS stops the Olhoff LP after 38 reported iterations is not exposed by the returned point (none exists) or output structure. Yuksel 640x80 was not replayed. Proposed KKT stationarity and cross-method topology equivalence were not tested. None is needed to state the benchmark results with the qualifications above.
