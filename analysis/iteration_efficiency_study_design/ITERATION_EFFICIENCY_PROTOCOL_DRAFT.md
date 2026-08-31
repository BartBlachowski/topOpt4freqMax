# Iteration-efficiency study — Phase 1C protocol (post-audit repair)

Supersedes the Phase-1A revision. The independent methodology audit returned **NOT READY —
MAJOR METHODOLOGICAL REVISION REQUIRED** (2 CRITICAL, 8 MAJOR, 9 MODERATE, 5 MINOR). All 24
findings are accepted; the one-to-one response is in `PHASE1C_AUDIT_RESPONSE.md` and
`PHASE1C_FINDING_CLOSURE.csv`. Superseded Phase-1A reasoning is retained inline and marked,
not deleted.

## Scope and outcome

This is a design-only protocol for measuring the method-level optimization work Proposed,
Yuksel, and Du–Olhoff/OlhoffApproach need to reach a declared fraction of their attainable
quality. Iteration count is primary; reference-platform time is secondary.

The completed performance/scaling campaign is separate frozen evidence. This design did
not edit it, launch MATLAB, run an optimizer, tune a profile, or create production results.

The primary estimand is **R**, renamed **self-referenced maturation work**: the work to
enter and certify a state attaining a declared fraction `q` of the quality the same method
sustains, where that reference is frozen on a **separate reference trajectory** rather than
inside the measurement horizon (finding C2). R measures maturation and does **not** equalize
achieved quality, so absolute achieved and reference quality accompany every count.

**The primary result is a quality–effort family, not a single threshold** (finding M1). The
preregistered levels are `q ∈ {0.980, 0.990, 0.995}`, co-primary. A single landmark may
anchor prose but may never stand alone.

**A is conditional secondary**, instantiated only if a mesh-specific engineering target
`Omega_req` exists independently of all three new trajectories. No such target currently
exists, so A is omitted unless one is supplied and provenance-locked before production. The
symmetric **best-observed benchmark is now mandatory** (finding M3), but it is not A and
never an absolute requirement.

The paired main iteration outcomes are `k_enter`, the retrospective location where the
accepted persistent regime begins, and `k_cert`, the conservative point where that regime
has been demonstrated. `k_enter` leads the maturation claim and the scaling analysis;
`k_cert` appears beside it with equal prominence as certification cost.

### Phase-1C deltas at a glance

| Area | Phase 1A (superseded) | Phase 1C | Finding |
|---|---|---|---|
| Topology gate | per-component **and aggregate** detached area `< a_res = 5`, derived from Olhoff's `rmin=1.3` and named `r_common` | aggregate veto **deleted**; `a_res`/`r_common` retired; significance is the FE-geometric physical area `A_sig = 4·A_e(160x20) = 0.01` | C1, Mo6, Mo7 |
| Reference quality | best sustained floor **inside the method's own observer horizon** (900 / 2000 / 3200) | frozen on a **separate reference trajectory** by a causal first-passage stabilization rule with **no cap fallback** | C2 |
| Quality threshold | `δ_R = 1%` baseline plus rescans | `q ∈ {98%, 99%, 99.5%}` **co-primary** | M1 |
| Evaluator | E1 primary, E2/E3 sensitivity | no evaluator primary; acceptance requires `min_e [Q_e/Q_ref_e] >= q`; E1/E2/E3 co-equal | M4 |
| Absolute quality | optional descriptive benchmark | mandatory best-observed benchmark; quality in Main Table 1; figure F4 | M3 |
| Yuksel Stage 1 | "design variable is not returned into Stage 2" (**false**) | `x = xPhys` carries the filtered field forward; eligibility rests on objective mismatch alone | M7 |
| Scaling | `k_enter` and `k_cert` fitted as co-equal layers | `k_enter` primary; `k_cert` fit descriptive; common-support companions; LOO ranges; weak-identification labels | Mo1, Mo2, Mo4, M5 |

## WP0 — authoritative evidence inspected

### Profiles and implementation paths

- Proposed selected practical profile: `proposed_practical_move02_tol001`, OC,
  single optimization loop, frozen solid reference, native raw-design max-change stop.
  Evidence: `PROPOSED_NATIVE_PROFILE_AUDIT.md`,
  `analysis/three_method_parametric_study/results/profile_freeze_manifest.json`,
  `analysis/ourApproach/Matlab/topopt_freq.m`.
- Yuksel selected practical profile: `yuksel_practical_move01_tol001`, two sequential OC
  stages with separate stops/times. Evidence: the same freeze manifest and
  `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`.
- Olhoff selected profile:
  `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1`, reproduced
  generalized-gradient method, one LP call per outer update, causal S1 move reduction.
  Evidence: `analysis/olhoff_stabilization_audit/selected_profile.json`,
  `olhoffOptStabilized.m`, `Matlab/reproduction2007/algo/innerLoopLP.m`, and
  `OLHOFF_STABILIZATION_AUDIT.md`.

The new study should reuse these algorithm profiles but must create new budget and
instrumentation manifests. It must not relabel the old fixed 1600 Olhoff endpoint as
convergence.

### Completed campaign and audits

Inspected:

- `examples/Performance/FINAL_CAMPAIGN_PREFLIGHT.md` and the frozen final-campaign
  configuration/results;
- `analysis/performance_campaign_forensic_audit/PERFORMANCE_CAMPAIGN_FORENSIC_AUDIT.md`
  and its evidence map;
- `analysis/performance_campaign_targeted_replays/TARGETED_REPLAY_REPORT.md`, forensic
  delta, and final freeze gate;
- Olhoff native/fixed-budget/stabilization audits;
- the independent Yuksel iteration-count audit;
- shared history recorders and stop-rule/extension-invariance evidence.

Repository facts that directly constrain this design:

1. The 27 performance rows have incompatible terminal semantics: six healthy fixed-work
   Olhoff endpoints, three Olhoff solver failures, seven native-converged plus two capped
   Yuksel endpoints, and nine native-converged Proposed endpoints.
2. Proposed coarse native spectra contain deterministic weak-material local modes. Common
   evaluation materially changes their interpretation, so native omega cannot be the
   universal quality gate.
3. The unchanged common evaluator provides E1/E2/E3 on raw and exact-count binary fields.
   E1 raw was preregistered as primary in the prior campaign only; Phase 1C privileges no
   evaluator, and no common model is truth.
4. Existing topology utilities use stable exact-count binarization and four-connectivity,
   but their `left_right_connected` field is less precise than the support-footprint rule
   required here.
5. The selected Olhoff LP path sets `nInner=1` for one `linprog` call. It does not retain
   HiGHS internal work on successful calls. The targeted failed call reported 38 solver
   iterations, proving these levels differ.
6. Olhoff final-campaign MAT files retain every successful-update density snapshot and
   scalar history. Proposed/Yuksel final-campaign artifacts retain terminal checksums and
   scalars, not recoverable trajectories.
7. Proposed 160x20 and Yuksel 800x100 targeted histories are useful diagnostics but do
   not fill the nine-mesh trajectory gap. Most new `k*` cells cannot be recovered offline.

The detailed availability classification is in `EVIDENCE_AVAILABILITY_MATRIX.csv`.

## Research questions and claim boundaries

For every method/mesh:

- `R(q)`: how many native method-level iterations were required to obtain and certify a
  structurally valid state attaining fraction `q` of that method's **independently frozen**
  sustained reference quality, for each of q = 98%, 99%, 99.5%?
- **absolute quality**: what quality does the method's reference actually attain, in common
  units, relative to the best observed across methods at that mesh?
- conditional `A`: if an independent target exists, how many iterations were required to
  obtain and certify a structurally valid state meeting it?
- supporting: what reference-platform computation accompanied those counts, and what did
  the separate reference phase cost?

These are two distinct dimensions and are never collapsed into one scalar efficiency score.

The study does not claim that an OC update, a Yuksel stage iteration, an Olhoff outer
update, an LP solver iteration, or one second are equivalent work units. It reports native
structure rather than concealing it.

Any ordering is acceptable. The protocol remains valid if Olhoff has the fewest outer
updates, Yuksel has the fewest total updates or the lowest time, Proposed has more but
cheaper updates, Olhoff has the best absolute quality, **the ranking changes with the
requested quality level**, **the quality–effort curves cross**, or any method is
`NOT_REACHED` or `REFERENCE_NOT_ESTABLISHED`. The method-blind firewall re-audit is in
`PHASE1C_AUDIT_RESPONSE.md` WP14.

## What exactly is a proper result?

For a return-equivalent state `X_mj(k)`, proper means all of the following:

1. every required solve through `k` is successful and finite;
2. raw physical volume has relative residual at most `1e-3`;
3. its exact-count binary projection contains a four-connected support-spanning component
   and **no individually significant detached component**, significance being the
   method-neutral physical area `A_sig = 0.01` (superseded: per-component *and aggregate*
   `< a_res = 5`, deleted under C1);
4. it attains fraction `q` of the frozen reference quality **under all three evaluators**,
   i.e. `min_e [Q_e(k)/Q_ref_e] >= q`;
5. it passes only scientifically justified method-specific conditions: Stage 2 for
   Yuksel; S1 policy stage 2 plus native `N=2` and gap12<=1% for Olhoff; none for Proposed —
   with the gate satisfaction index `k_gate` reported beside every endpoint (M8);
6. all five conditions remain true for `P=100` consecutive method-level iterations.

"Proper" is therefore q-indexed: there is one endpoint per quality level, not one canonical
endpoint.

The full mathematical definition, state-index convention, reference construction,
status precedence, and sensitivity set are in `ACCEPTANCE_GATE_SPEC.md`.

Persistent acceptability is deliberately not the same as density stationarity. Density
change, spectral range, component events, and binary turnover are reported throughout the
window, but no cross-method stationarity tolerance is invented.

## Spectral design and threshold firewall

### Common evaluators — no privileged model

**Superseded:** "use unchanged common raw E1 omega1 as the baseline; E2/E3 are a one-factor
sensitivity." The audit verified to 9–11 significant digits that **E1 is Proposed's own
interpolation** up to floor values, E2 was Yuksel's piecewise `x^6` mass law, and E3 was
Olhoff's `rho_min=1e-3` model. Calling E1 neutral is withdrawn (M4).

**Amended (Phase 2D):** E2 and E3 are **no longer** the literal Yuksel and Olhoff native mass
interpolations. Their low-density mass branch now uses the source-defined continuous
Du & Olhoff (2007) Eq. (4a), `g(x) = 1e5*x^6` for `x <= 0.1`, instead of the native
discontinuous Eq. (4). Stiffness laws and floors are unchanged and remain method-specific.
The native optimizers are untouched and still run Eq. (4). The reason is recorded in
`analysis/iteration_efficiency_evaluator_discontinuity_audit/`: Eq. (4) is source-faithful
and intentionally discontinuous by factor 1e5 at `x = 0.1`, the Olhoff update law parks
densities deterministically at that value, and one-double-ULP and float32 branch crossings
moved E2/E3 by up to 4.0e-3 and 2.7e-2 respectively — material against the 0.5–2% quality
bands this protocol uses. Any claim of exact native-interpolation identity for E2/E3 is
retired.

No evaluator is primary. All three are recomputed offline from stored return-equivalent
fields; common eigensolves never enter a method's timed loop. Acceptance requires the
relative threshold under **all three**, expressed as a minimum over dimensionless
attainment ratios. Per-evaluator endpoints are co-equal mandatory decompositions, and
disagreement in status or ordering is labelled `MODEL_DEPENDENT`.

The mitigating empirical fact is stated wherever the objection is raised: the three
evaluators agree within **0.429%** and preserve the ordering wherever all evaluator values
are available. That figure predates the Phase-2D amendment and must be recomputed before it
is cited again. E2 and E3 share the same piecewise low-density mass law — after the
amendment, Eq. (4a) rather than Eq. (4) — and differ only in stiffness floor, so the minimum
is closer to two-way in evidential terms. This bounds the objection at
endpoints; trajectory-level robustness is still required, which is why the all-evaluator
gate applies at every state.

### Relative definition R — horizon-independent reference

**Superseded:** "the reference is the best sustained floor over any valid window *inside the
method's observer horizon*, and the accepted level is 99% of it." Because the horizons were
900 (Proposed), 2000 per stage (Yuksel) and 3200 (Olhoff), a shorter horizon could only
lower the reference, lower the bar, and move `k_enter` earlier — so the shortest-budget
method had the easiest self-reference (C2).

Phase 1C separates the two phases. A dedicated **reference trajectory** freezes
`Q_ref_e = F_e(b_ref)` under a causal method-independent first-passage stabilization rule
(`P=100`, `L_ref=500`, `epsilon_ref=0.001`), applied to all three evaluators simultaneously.
`B_ref = 3200` is a censoring boundary only: if the rule does not fire, the cell returns
`REFERENCE_NOT_ESTABLISHED` and **no cap-based value is substituted**. A separate
**measurement trajectory** from the identical initialization is then scanned against the
frozen, provenance-hashed triplet, which it cannot recompute. Reference work is published as
`N_reference`/`T_reference` and never charged to `k_enter`/`k_cert`. Full rule in
`REFERENCE_QUALITY_SPEC.md`.

The accepted levels are `q ∈ {0.980, 0.990, 0.995}`, **co-primary**. The audit showed
crossings move 3–6x and the fitted exponent moves from +0.145 to +0.479 between 1% and
0.5%, so no single tolerance is canonical (M1). The three values are exactly the Phase-1A
baseline and its two declared sensitivities, elevated together before any ranking was
consulted. Details in `QUALITY_EFFORT_SPEC.md`.

### Absolute definition A

Use `omega1_E1_raw >= Omega_req(mesh)`, where `Omega_req` is independent of all three new
trajectories. No such requirement currently exists. Its absence does not block primary R;
A is simply not instantiated. Existing endpoints cannot be used to hand-pick a level.

The **best-observed benchmark** `Q_BO_e,j = max_m Q_ref_e,mj` is now **mandatory**, not
optional (M3): it is the only symmetric cross-method quality comparison the study can offer,
and the frozen campaign already shows Olhoff leading common raw-E1 endpoint `omega1` by
**6.2–8.5% over Proposed** and **5.9–7.7% over Yuksel** across the eight meshes with a
complete method triple. The Olhoff 800x100 endpoint is `RUN_ERROR` with E1 `N/A`, so no
nine-mesh inference is made. This comparison is symmetric and descriptive, and it is never
A, absolute adequacy, or an engineering requirement.

## Iteration accounting

The binding definitions are in `ITERATION_ACCOUNTING_SPEC.md`.

- Proposed headline pair: `N_OC` / single-loop updates to `k_enter` and `k_cert`; terminal
  eigensolve excluded from iterations and timed separately. A Proposed OC update contains
  **no eigensolve** under the frozen solid-reference profile, against a frozen Olhoff
  eigensolve that was ~75% of outer-update cost at 800x100 (Mi5).
- Yuksel: retain `N_stage1`, `N_stage2_to_cert`, and their sum. **Correction (M7):**
  `top99neo_inertial_freq.m:237` is `x = xPhys;` — Stage 1's filtered physical field *does*
  become Stage 2's design variable and its displacement the initial mode estimate, so the
  design state is continuous across the handoff. The Phase-1A claim that it "is not returned
  into Stage 2" was false and is withdrawn. Stage 1 remains separately reported for its
  distinct objective, update regime, and timing, and Stage-2-only eligibility now rests
  solely on objective mismatch. **Disclosure (M6):** Stage 1 hit its 1000-update cap at
  640x80, 720x90 and 800x100 in the frozen campaign, so a 2000 budget changes the handoff
  state and breaks count comparability at those three meshes.
- Olhoff: retain successful and attempted outer counts, LP calls and flags. The selected
  path's code `nInner=1` is one LP call, so it is redundant with healthy outer updates.
  MATLAB/HiGHS-reported LP iterations belong in supplementary work diagnostics.

## Topology sanity and images

The binding rules are in `TOPOLOGY_SANITY_SPEC.md`.

The gate is quantitative, binary, volume-preserving, stable-tie, and support-footprint
based, and it answers **only** whether a topology is grossly or pathologically invalid.

**Superseded:** "T1, derived from the smallest frozen filter footprint (`a_res=5`), is
baseline; strict zero-island T0 is sensitivity." T1 required each detached component *and
their aggregate* to be under five elements and nearly excluded available fine-mesh evidence;
at 640x80 it admitted 0.6% of states and no `P=100` window (C1). The inherited 800x100
figures are not reused because the artifact is unavailable and the endpoint is
`RUN_ERROR`/E1 `N/A`.

Phase 1C deletes the aggregate veto and retires `a_res`/`r_common`, which were derived from
Olhoff's `rmin=1.3` and so were never method-neutral (Mo7). The hard gate is
**support-to-support connectivity AND no individually significant detached component**, with
significance set by the FE-geometric physical area `A_sig = 4·A_e(160x20) = 0.01`, i.e.
`a_sig = 4…100` elements from 160x20 to 800x100. A constant *element* count was rejected
because its physical area shrinks 25-fold across the family; a constant *physical area* is
the mesh-invariant statement. Aggregate detached area, component count and LCC remain
mandatory diagnostics. T0 is demoted to a known strict diagnostic with its outcome stated up
front, and the permissiveness sensitivity is method-neutral 1x1/3x3 FE patch scales (Mo6).
Fixed-fractional-LCC T2 remains rejected. Support connectivity is traced to the problem
definition: the eigenproblem has no common loaded region, so support-to-support is the only
common hard path; even `nely` is an explicit precondition (Mi1).

Read-only recomputation on the eight available frozen Olhoff trajectories confirms the
repaired rule is satisfiable and still rejects unformed states — longest accepted runs
957 / 925 / 1517 at 160x20 / 640x80 / 720x90, against 5 for T1 at 640x80. The zero-byte
800x100 artifact is `UNVERIFIABLE_AT_PRESENT`; its frozen endpoint is `RUN_ERROR`/E1 `N/A`.

Images support interpretation only. Show paired raw/binary accepted fields with common
orientation, scale, supports, and no method-specific pseudo-load. Main grid: all methods
at 160x20, 320x40, 560x70, 800x100; complete 27-cell grid in the supplement.

## Persistence, k_enter, and k_cert

`k_enter` is the first state of the earliest 100-state all-pass window and leads the main
maturation result. `k_cert=k_enter+99` is the iteration at which the window has been
observed and is the paired conservative certification result. Both counts and both times
are main outputs.

The same absolute `P` applies to every method-level count. It is described accurately as a
**convention inherited from Olhoff stabilization evidence and applied uniformly to impose
the same proof length**, not as a value derived from all three methods; the Phase-1A
implication otherwise is withdrawn. Method-specific P is prohibited. `P=50` and `P=200` are
the predefined sensitivity at every q level.

The proportional burden is unequal and must be disclosed: `P-1` is roughly 30–93% of
Proposed's native run versus about 6% of Olhoff's fixed horizon, and Proposed certification
may extend past its native stop, so `k_native` is printed beside `k_cert` (Mo3).

Quality–effort curves use the **persistent `k_enter`**, with certified `k_cert` as a
companion panel. Instantaneous crossings are diagnostics only: a single-state crossing is
exactly the transient the persistence rule exists to exclude, and it would be worst at the
tightest q level.

For successful rows the `+99` leaves absolute count differences unchanged, but it
compresses ratios and flattens the `k_cert` power-law exponent relative to `k_enter`,
especially at small counts — on frozen Olhoff data the exponent fell from +0.1451 to +0.0991,
a 32% reduction from bookkeeping alone, and the effect is worst for Proposed at 107–330
native iterations (Mo1). **`k_enter` scaling is therefore primary and the `k_cert` power fit
is secondary/descriptive**, with a mandatory caption caveat and a pipeline check that fitting
`k_cert-(P-1)` reproduces the `k_enter` fit exactly. `P=50/200` sensitivity is reported; P is
not enlarged to make counts look more substantial.

## Safety budgets and NOT_REACHED policy

The **reference phase** uses the single common cap `B_ref = 3200` acceptance-eligible
method-level updates. It is a censoring boundary only; it never supplies a reference value,
which is what makes `Q_ref` independent of the resource horizon (C2).

Initial measurement lower bounds are derived from completed prior work, not from future rankings:

`B0=ceil_to_100(max(2*K_prior, K_prior+5P))`.

At `P=100`, this gives Proposed 900 OC updates, Yuksel 2000 per stage, and Olhoff 3200
outer updates. After `b_ref` is frozen, the exact per-cell horizon is
`B_meas=min(max(B0,b_ref+P-1),B_ref)`. This deterministic, method-blind rule uses only frozen
reference quantities, provides the available persistence tail relative to `b_ref`, and
never extrapolates past `B_ref`. It applies to Proposed OC, Yuksel Stage 2, and Olhoff outer
updates; Yuksel Stage 1 retains its separate frozen 2000-update handoff budget. Old
convergence tolerances, move limits, filters,
interpolation, and stage/S1 rules remain unchanged. Because `Q_ref` is frozen elsewhere, a
measurement budget can only affect whether an endpoint is **observed**; it cannot move any
quality bar.

**Disclosed non-neutral change (M6):** Yuksel Stage 1 hit its own 1000-update cap at 640x80,
720x90 and 800x100 in the frozen campaign, so `B0_stage1 = 2000` changes the realized
algorithm at three of nine meshes. `N_stage1`, the Stage-2 trajectory and the chronological
total are **not comparable** to frozen campaign values there.

The prior progress-triggered extension is deleted. There is no discretionary or
result-contingent extension beyond `B_meas`; reaching
that frozen horizon without certification produces the applicable `NOT_REACHED` subclass.

Final classes distinguish:

- `SOLVER_TERMINATION`: required solve failure before certification, **always with the
  backend subclass**. The known Olhoff event carries
  `GENERIC_LP_ITERATION_LIMIT_ONLY: dual-simplex-highs returned exit flag 0 in the recorded
  MATLAB version` and is never generalized to failure of the Du–Olhoff formulation (Mo5).
  The Phase-1A name `GENUINE_SOLVER_FAILURE` is withdrawn as a method indictment;
- `PERSISTENT_NONACCEPTANCE`: repeated/irregular gate exits without a certified window;
- `INVALID_TOPOLOGY`: topology hard gate remains limiting;
- `QUALITY_NOT_REACHED`: healthy structurally valid trajectory misses the level `q`;
- `REFERENCE_NOT_ESTABLISHED`: the stabilization rule did not fire by `B_ref`;
- `REFERENCE_SOLVER_TERMINATION`: a required solver terminated before `b_ref`;
- `OTHER`: reason mandatory.

The two reference classes are new in Phase 1C and are publishable outcomes, not defects:
they are the honest price of refusing a cap-based fallback.

An already certified state remains a minimum-work success if an observer run later fails;
the row becomes `PASS_WITH_LATER_FAILURE`. This measures what was obtained by `k_cert`
without hiding the later failure.

## Timing

Use lightweight deterministic replays stopped at already-frozen `k_enter` and `k_cert`,
rather than publishing I/O-contaminated observer wall time. Report median of three serial,
single-thread runs after method warm-up, plus range. The main supporting quantity is

`T_result_to_cert = T_init + T_native_loop_to_cert + T_native_finalize`.

Offline common evaluation, topology analysis, post-certification continuation, image
generation, and trajectory disk I/O are separately disclosed and not charged. Yuksel
stage timings remain separate. Olhoff eigensolve/gradient/LP times and solver-iteration
statistics are supplementary. Missing Olhoff init/post boundaries remain `NA`, not zero,
unless observational instrumentation separates them.

Record CPU, MATLAB/update, thread count, OS, RAM, solver versions/settings, eigensolver
settings/start vector, precision, code/profile hashes, run order, warm-up, timestamps, and
process isolation. No hardware-independent timing claim is permitted. See `TIMING_SPEC.md`.

## Mandatory efficiency and scaling story

The paper must keep four layers distinct and adjacent:

1. `k_enter(Ne|q)`: primary empirical maturation count and primary count-fit family;
2. `k_cert(Ne|q)`: prominent conservative certification companion, including the explicit
   `P-1` burden; its power fit is secondary/descriptive;
3. mean reference-platform cost per native iteration type versus `Ne`;
4. `T_result_to_enter(Ne)` and `T_result_to_cert(Ne)`: total practical consequences.

The mandatory figures also decompose Yuksel Stage 1/Stage 2/total and Olhoff outer
updates/LP calls, with solver-internal LP iterations supplementary only where genuinely
captured. Eligible `k_enter` series are q-conditioned and use the descriptive free fit
`y=C Ne^p`; every full-range fit has a common-support companion and reports
`C,p,R2_log,n_valid`, fitted range, and leave-one-valid-mesh-out `p` range. Raw/censored
points are shown and at least three valid mesh values are required. No fit extrapolates
beyond its valid data, no censored result is treated as success, and no exponent is called
asymptotic complexity. The completed campaign's old fixed-`p=1.5` plot is retired. Binding
details and every old-plot mapping are in `SCALING_AND_FIGURE_SPEC.md`.

## Paper evidence package

The compact main paper uses:

1. paired R `k_enter/k_cert` iteration outcomes with Yuksel stage decomposition and status;
2. paired reference-platform times and mean native-iteration costs;
3. co-equal E1/E2/E3 endpoint decompositions plus robust all-evaluator acceptance;
4. q-conditioned scaling curves for primary `k_enter`, descriptive `k_cert`, iteration
   cost, total time, and Yuksel stages;
5. mandatory absolute reference/endpoint quality and ratio-to-best evidence;
6. standardized topology and persistence evidence tied to both counted endpoints.

Olhoff outer/LP/solver work, full timing decomposition, OAT sensitivities, and the complete
27-row detail remain supplementary; no evaluator is demoted to a sensitivity or omitted
from the primary decomposition. The full plot disposition, empirical
power-fit discipline, and source-data requirements are in `SCALING_AND_FIGURE_SPEC.md`;
table layouts are in `PROPOSED_TABLE_LAYOUTS.md`.

## Evidence reuse and need for new runs

Olhoff's eight available full snapshot histories make provisional offline R/A rescans
possible after thresholds are frozen, including states before later failures. The 800x100
artifact is unavailable (`RUN_ERROR`/E1 `N/A`). Successful LP solver-internal iteration
counts still require instrumentation.

Proposed and Yuksel final-campaign density trajectories are not recoverable from
checksums. Their targeted replays cover only one case each and do not provide full
nine-mesh, every-state common-quality evidence. Therefore most of the new paired R table
requires new instrumented optimization runs. Existing final endpoint summaries can
support preflight/budget checks but cannot establish earliest persistent entry.

## Hostile-review conclusions

The strongest objections and controls are catalogued in `FAIRNESS_RISK_REGISTER.md`.
The most important are:

- E1 is a comparison convention that may resemble Proposed's mass model; E2/E3 are
  mandatory and model-dependent conclusions must be labelled;
- primary R can reward a low-quality method, so achieved quality is inseparable from its
  count claim and conditional A must not be fabricated;
- a high common target can censor lower-quality methods, which is a result rather than a
  defect only if the target was truly external;
- exact binarization can penalize gray or fragmented designs; the repaired physical-area
  gate, aggregate diagnostics, raw/binary evidence, known-strict T0 diagnostic, and
  method-neutral 1x1/3x3 FE-patch OAT sensitivity expose this;
- `P=100` adds equal iterations but unequal seconds; both quantities are shown;
- Olhoff's modal requirement is asymmetric because the scientific formulation is
  asymmetric; false symmetry would be less fair;
- outer counts and OC counts look numerical but are not equivalent work.

## Sensitivity plan

Beyond the reference phase, no extra optimization is needed. Rescan stored trajectories one
factor at a time:

- **quality levels 98%, 99%, 99.5% are co-primary, not a sensitivity** (M1);
- **E1/E2/E3 endpoints are co-equal decompositions, not a sensitivity** (M4);
- persistence 50, 100, 200, at every q level;
- raw relative-volume tolerance 5e-4, 1e-3, 2e-3;
- Olhoff native gap 0.5%, 1%, 2%;
- topology 1x1 / 2x2 / 3x3 coarsest-mesh FE patch scales around the `A_sig` baseline; T0 is
  a known strict diagnostic whose outcome is stated up front, not a sensitivity (Mo6);
- the superseded horizon-relative reference at 900 / 1600 / 2000 / 3200, as a C2-closure
  diagnostic that cannot select a different reference rule.

For A, use independently supplied target uncertainty bounds. The mandatory best-observed
benchmark is reported at the same q levels as R.

The main conclusion is threshold-robust only if ordering/status does not materially change.
Do not run a combinatorial grid or select a preferred sensitivity after seeing it.

## Phase 2 and later execution sequence

The exact engineering requirements are in `IMPLEMENTATION_REQUIREMENTS.md`.

1. implementation and no-solve preflight;
2. tiny nonproduction engineering smoke tests;
3. a frozen 240x30 representative pilot exercising the **full two-phase sequence** —
   reference run, reference freeze, measurement run, scan — for state-index identity,
   observational invariance, reference/measurement separation, evaluator feasibility, and
   evidence completeness only;
4. **reference-phase production and `Q_ref` freeze at all 27 cells**;
5. full nine-resolution measurement production;
6. blinded offline acceptance freeze at all three q levels;
7. lightweight timing replays;
8. independent post-run audit.

The representative pilot is required because return-equivalent every-state recording is
new for Proposed/Yuksel and existing histories expose off-by-one/stage semantics. It is
not allowed to select thresholds, budgets, or profiles.

## Direct answers to the 30 required questions

1. **Proper result?** Solver-healthy, volume-feasible, support-connected with no
   individually significant detached component (`A_sig`), attaining fraction `q` of the
   frozen reference under **all three** evaluators, method-specifically valid, and all-pass
   for 100 consecutive method-level iterations — one endpoint per q level.
2. **One or two definitions?** Primary R as a **quality–effort family**; conditional
   secondary A only with independent `Omega_req`. The best-observed benchmark is mandatory,
   descriptive, and non-absolute.
3. **Which common evaluator?** **None is primary.** E1/E2/E3 are co-equal; acceptance
   requires `min_e [Q_e/Q_ref_e] >= q`. E1 is Proposed's own model, so calling it neutral is
   withdrawn. Native omega is not universal.
4. **Threshold choice without bias?** Report q = 98/99/99.5% co-primary rather than
   declaring one canonical tolerance; the values are the pre-existing Phase-1A baseline and
   sensitivities, frozen before production. Require an external `Omega_req` for A.
5. **Topology checks?** Exact support connectivity plus **per-component** significance at the
   FE-geometric `A_sig`; **no aggregate-area veto**; T0 is a known strict diagnostic; the
   permissive sensitivity is 1x1/3x3 FE patch scales; no fixed-fraction T2.
6. **Required connectivity?** One four-connected binary component intersects the Q4
   element footprints incident on both prescribed mid-height fixed nodes.
7. **Raw or binary sanity?** Exact-count volume-preserving binary for the hard graph;
   raw volume/grayness/topology are secondary.
8. **Olhoff modal condition?** S1 policy stage 2 and native `N=2`, gap12<=1% throughout
   the external persistence window.
9. **Proposed/Yuksel analogue?** No invented multiplicity condition; only Yuksel Stage-2
   eligibility is intrinsic.
10. **Persistence?** Same `P=100` consecutive pointwise all-pass states at every q level;
   50/200 sensitivity. P is an inherited uniform-proof-length convention, not a value
   derived from all three methods; method-specific P is prohibited.
11. **k_enter?** First state of the earliest all-pass P-window.
12. **k_cert?** Last state of that window, `k_enter+P-1`.
13. **Headline counts?** `k_enter` leads maturation; equally prominent `k_cert` gives
   conservative certification. Neither is hidden in the supplement.
14. **Proposed count?** Each completed single-loop OC update; post-loop eigensolve excluded.
15. **Yuksel count/report?** Stage 1 and Stage 2 separately plus chronological sum; only
   Stage 2 can certify. The Stage-1 field **is** carried into Stage 2 (`x = xPhys`), so the
   design trajectory is continuous and eligibility rests on objective mismatch alone.
16. **Olhoff inner iteration?** In selected LP code, `nInner=1` means one `linprog` call;
   it is not a simplex/HiGHS iteration.
17. **Olhoff inner work in main table?** None beyond outer count/LP-call footnote; only
   genuinely captured solver iteration statistics belong in supplement; never fake `nInner`.
18. **Failure versus insufficient budget?** Solver flag/nonfinite failure has precedence;
   an uncertified cell reaching its frozen `B_meas` receives the applicable `NOT_REACHED`
   subclass.
19. **Measurement horizon?** Exactly `min(max(B0,b_ref+P-1),B_ref)` after the reference
   freeze; no progress-triggered extension or algorithm/gate change.
20. **Timing to endpoints?** Matched fixed-horizon replays through enter and cert plus
   ordinary init/finalization; external gate time separate.
21. **Main timing?** Paired median result times and mean loop seconds/method-iteration,
   with Yuksel stage-specific values mandatory.
22. **Supplement timing?** Init/loop/finalize/gate, repeats/range/MAD, Olhoff
   eigen/gradient/LP and genuinely observed solver-iteration decomposition.
23. **What trajectory quantities exist?** Complete Olhoff fields/scalars at eight available
   meshes; 800x100 is unavailable/RUN_ERROR/N/A; terminal summaries for all rows; targeted
   Proposed 160 and Yuksel 800 scalar diagnostics.
24. **What needs instrumentation?** Proposed/Yuksel every-state return-equivalent fields,
   uniform cumulative time, Olhoff successful-call solver iterations, precise support
   component metrics, state mapping.
25. **Can most k* be offline now?** No. Olhoff supports a *provisional* offline rescan at
   eight meshes (its 800x100 artifact is zero bytes and `RUN_ERROR`/E1 `N/A`), but not a production endpoint, because
   every measurement trajectory must be paired with a declared reference run. Proposed and
   Yuksel cells cannot.
26. **Images?** Standard paired raw/binary accepted-state grid, common supports/orientation;
   representative four meshes main, all 27 supplement.
27. **Strongest fairness objections?** E1 model affinity, self-relative low-quality bias,
   target choice, binarization/islands, persistence cost asymmetry, Olhoff modal asymmetry,
   and incomparable count units.
28. **Small sensitivity?** OAT persistence, volume tolerance, Olhoff gap, FE-patch topology
   scale, and the superseded-horizon reference diagnostic; no new solves or Cartesian sweep.
   Quality levels and evaluators are co-primary results, not sensitivities.
29. **Representative pilot?** Yes for frozen instrumentation/integrity at 240x30, not for
   scientific tuning.
30. **What Phase 2 implements?** Isolated hashed manifests/runner, invariant observer
   extension, every-state return-equivalent storage, a **two-pass offline engine** (causal
   reference freeze, then measurement scan against the frozen triplet), the repaired topology
   gate, the q-level paired k scan and status ledger, endpoint timing replays, mandatory
   status-aware scaling figures/fits with common-support companions and LOO ranges,
   tables/images, and audit hooks.

## Human decision and challenge register

### A — must be approved before the independent delta audit

None. All 24 audit findings are accepted and dispositioned, and every constant is frozen.
The package is determinate enough to be re-challenged. **Delta-audit readiness does not
authorize implementation or production, and the author does not declare the methodology
ready for implementation.**

### B — independent delta auditor must challenge before implementation

1. Whether `A_sig = 4·A_e(160x20) = 0.01` is a defensible method-neutral significant-feature
   scale, and whether its growth to 100 elements at 800x100 is too permissive.
2. Whether the stabilization rule genuinely severs horizon dependence or merely relocates it
   into `L_ref = 500` and `epsilon_ref = 0.001`, and whether refusing any cap-based fallback
   is the right trade against losing cells to `REFERENCE_NOT_ESTABLISHED`.
3. Whether the all-evaluator minimum over attainment ratios is adequately symmetric, and
   whether its structural conservatism is acceptable.
4. Whether q = 98/99/99.5% is the smallest scientifically useful preregistered set, and
   whether the quality–effort family plus mandatory absolute quality prevents overreading
   self-relative efficiency.
5. Whether the two documented deviations from the audit's stated minimum corrections are
   acceptable: retiring `a_res` outright rather than renaming it (Mo7), and declining the
   suggested filter-footprint sensitivity in favour of FE patch scales (Mo6).
6. Whether the absence of any Proposed/Yuksel trajectory evidence for the topology gate, and
   the unavailable zero-byte 800x100 Olhoff artifact (`RUN_ERROR`/E1 `N/A`), are tolerable before production.
7. Whether `P=100`, paired `k_enter/k_cert`, `k_enter`-primary scaling with common-support
   companions and LOO ranges, and P=50/200 sensitivity fairly represent maturation and
   certification without inflating or flattening comparative scaling.
8. Whether frozen practical profiles, Yuksel Stage-2 eligibility, and Olhoff policy-stage-2
   plus `N=2`, gap12<=1% faithfully represent the methods, and whether reporting `k_gate` is
   sufficient exposure of the asymmetric gate-imposed floor.
9. Whether the deterministic per-cell horizon equation, the disclosed Yuksel Stage-1 cap
   change, censoring, log-fit minimum/range, timing replays, and the result firewall are auditable
   and symmetric.

### C — must be approved before implementation or production

1. Ratify the audited protocol/version hash, exact source/profile hashes, new result root,
   platform, compute/storage budget, and independent audit disposition.
2. Supply and provenance-lock an independent `Omega_req(mesh)` if A is desired; otherwise
   record `A_NOT_INSTANTIATED`. The best-observed benchmark is mandatory and needs no
   separate approval.
3. Authorize the no-solve checks, smoke tests, 240x30 integrity pilot, **reference-phase
   production at all 27 cells**, nine-mesh measurement production, paired timing replays, and
   storage of full trajectories, budgeting for two trajectories per cell (~75 h offline
   evaluation, ~40 GB storage). Pilot results may correct engineering defects only, never
   scientific gates or profiles.
4. Approve production unblinding only after automated identity, invariance, schema,
   resource, fit/censoring, and plot/table consistency gates pass.

## Final verdict

READY FOR INDEPENDENT DELTA AUDIT

The independent auditor that issued C1 and C2 must receive this repaired package for a delta
audit before any implementation. No implementation, optimization, production execution, or
performance-campaign modification is authorized by this document. See
`PHASE1C_AUDIT_RESPONSE.md` for the full finding-by-finding response.

## Phase 2H controlled refreeze amendment

The common evaluator is Candidate C: actual-gray E1/E2/E3 interpolation, adaptive
structural-mode search, and the unanimous `voidKE<0.5`, `voidSE<0.5`,
`densityParticipation>0.5` classifier. The exact-count binary design remains an endpoint
manufacturability/topology diagnostic and is excluded from Q, reference, and persistence.
Failure to find a structural mode fails closed as `STRUCTURAL_MODE_NOT_FOUND`.

Olhoff-LP is the principal route and Olhoff-MMA is a separately labelled secondary route;
`lp`, `mma`, and `both` selection never collapses their iteration accounts. Production is
blocked pending Candidate C precision, cross-method, and reference-length qualification.
All prior constants, hard topology gate, budget equation, mesh sequence, and blind-reporting
rules remain frozen.
