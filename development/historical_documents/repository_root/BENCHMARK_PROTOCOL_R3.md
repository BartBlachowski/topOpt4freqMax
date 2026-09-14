# Benchmark protocol R3

Status: **METHODOLOGY RESOLVED — ENGINEERING GATES OPEN — DO NOT EXECUTE**
Date: 2026-08-26
Governing audit: `BENCHMARK_FAIRNESS_AUDIT.md`
Machine-readable manifest: `examples/Performance/benchmark_protocol_r3.json`

This revision defines methodology only. It authorizes no performance campaign,
optimization experiment, parameter sweep, or algorithm change. Existing R2 and pre-fix
outputs are excluded from parameter selection and scientific conclusions. When this
document, the manifest, the resolved methodology, the required instrumentation, and exact
implementation bytes are committed together on a clean tree, that commit becomes the R3
freeze. Observed R3 rankings may not be used to alter it.

## 1. Methods and identity

| ID | Implementation | Audited SHA256 |
|---|---|---|
| `Proposed` | `analysis/ourApproach/Matlab/topopt_freq.m` | `19399780d733380993668fabddaa1d3edbe85856809193b166a77cdccf171b82` |
| `Yuksel` | `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m` | `5afc3d16b4ed6af05793df461b541ed3b2ea62a6da8836f38301a9a3917e6ba2` |
| `OlhoffDu2007Repro` | `Matlab/reproduction2007/algo/olhoffOpt.m` and Eq. (22) `innerLoopLP.m` | `4784ecf3f6b42d8af6f5a9695e2924bf1f4c924ebf9c76be9a858a2ef769e5de` and `7724753c02f84d6009c3998f758d5b3f9c5144ad39ca6f470584a2c99e089465` |

Audit base: branch `benchmark-methodology-r2`, HEAD
`cf290fc7f9daf9da27bc8224f9585a0e1657bff1`. The final campaign must use a later
clean freeze commit and inventory every entry point, dispatcher, evaluator, manifest, and
instrumentation helper. Historical `Olhoff`, `OlhoffExact`, and
`analysis/OlhoffApproach*` implementations are excluded. The frozen Du–Olhoff
reproduction and its provenance are not modified by this protocol.

## 2. Physical problem and meshes

- Plane-stress beam: `L=8 m`, `H=1 m`, `t=1 m`; square Q4 elements with 2×2
  integration and consistent mass.
- `E0=1e7 Pa`, `nu=0.3`, `rho0=1 kg/m^3`; volume fraction `0.5`; uniform initial
  design `x=0.5`; no passive regions or concentrated mass.
- Both translations are fixed at the exact mid-height node of each end; `nely` is even.
- Target is the first natural frequency. Each optimizer retains its native formulation.
- Scaling meshes: `160×20, 240×30, 320×40, 400×50, 480×60`.
- Native and harmonized-quality meshes: `160×20, 240×30, 320×40`.
- Full OAT sensitivity mesh: `240×30`.
- Targeted filter-radius replication mesh: `320×40`.

The four fixed DOFs must be asserted identical across native builders and every common
evaluator.

## 3. Native parameter profiles and provenance gate

Native means the formulation and parameter profile the method itself optimizes and
reports. It does not mean that numerical values are equivalent across methods.

### 3.1 Proposed — resolved manuscript/SS profile

```text
profile_id=proposed_manuscript_ss_oc_a0_2026-07-14
optimizer=OC; objective=compliance under F(x)=omega0^2 M(x) Phi0
reference=fully solid, frozen; mode 1 factor 1; Phi0' M0 Phi0=1
load sensitivity=omitted; per-iteration load-norm rescaling=false
pK=3; mass interpolation=linear (pM=1); no low-density mass branch
filter=density-weighted sensitivity; rmin=2.0 element
declared boundary=symmetric; effective historical operator=truncated centroid stencil
move=0.2; design bounds=[0,1]; native_tol=0.001; max_iters=2000
Emin/E0=1e-9; rho_void/rho0=1e-9
initial design=uniform at Vf; no active continuation/Heaviside projection
OC volume equality by Lagrange-multiplier bisection
```

The governing provenance is `PROPOSED_NATIVE_PROFILE_AUDIT.md`. The exact pre-R3 base
configuration was added at commit `d051985` (2026-07-14), has SHA256
`c5e8949a318dd6ac657faf034ca85b2210d3a01976c37dab1602eec8ae341047`, and remained
unchanged through the 2026-07-29 audit at `310043e`. It implements the manuscript/A0
formulation and predates the R3 methodology lineage begun on 2026-07-31. The March MMA
profile at `4582541`/`76b2894` is retained only as
`Proposed-legacy-MMA-2026-03-08`; it is not native R3 evidence.

Two limitations remain explicit. The manuscript mathematical section says the design
lower bound is `1e-3`, whereas the historical executable OC profile uses `[0,1]`. The A4
JSON declares a symmetric filter boundary, whereas the executable Proposed filter uses a
truncated centroid stencil. R3 freezes the effective historical values above; any change
requires a separately justified R3.x amendment. The current R3-branch solver also contains
the obsolete nodal-density semi-harmonic formula. No run is allowed until an engineering
gate proves exact A0 load/profile behavior. The inactive legacy `filter.heaviside` key,
obsolete `semi_harmonic_rho_source`, and dead mode-2 load are prohibited.

### 3.2 Yuksel

```text
profile_id=yuksel_published_ss_2025
optimizer=OC; p=3; eta=0.5; beta=1
filter=sensitivity; rmin=2.5 element; boundary=symmetric
move=0.2; stage1_tol=0.01; stage2_tol=0.01
stage1_max_iters=10000; stage2_max_iters=10000
Emin/E0=1e-9; rho_void/rho0=1e-9
mass factor g(x)=x for x>0.1 and x^6 for x<=0.1, with the stated floors
stage 1=midspan point-load compliance
stage 2=F=M(x)u_hat refreshed each iteration; dF/dx omitted
continuation counters present but inactive for p=3,beta=1
```

The published simply supported case uses `320×40`, sensitivity radius 2.5, move 0.2,
and 0.01 termination for each static optimization. The local function fallback
`rmin=8.75` is generic top99neo heritage, not the published simply supported profile.

### 3.3 OlhoffDu2007Repro

```text
profile_id=olhoff_du_2007_repro_fig3a_best
configuration_lineage=fig3a_best; target n=1
filter=top88 sensitivity generalized-gradient; rmin=1.3 element
filterMode=diag; move=0.005
rho_min_design=1e-3; initial rho=0.5; p=3
massInterp=Du Eq.(4): g(rho)=rho above 0.1, rho^6 at/below 0.1
tolMult=0.05; Nmax=4
innerSolver=lp; Eq.(22) equalities; offDiag=false
tolOuter=0.001; maxOuter=1600; threads=1
support=mid; axial=both; BC=SS
```

The radius 1.3 is a reconstructed, benchmark-supported value from the successful frozen
basin; it is not stated in the Du–Olhoff paper. It is constant in element units, so its
physical length is mesh dependent: `r_phys=1.3*(8/nelx)` on these square meshes. The
paper also leaves move and multiplicity tolerance unstated. These limitations remain in
the comparator label and table footnotes. Another method's material floor must never
populate `rho_min_design`.

## 4. Native stopping and harmonized diagnostic status

Native stopping remains separate:

- Proposed: finite successful OC update and `max|x_new-x_old| <= 0.001` on the raw
  design field, else `CAP_HIT` at the 2000-iteration safety budget.
- Yuksel stage 1: after at least two iterations, `max_active|Δx| < 0.01` is
  `STAGE_HANDOFF`, not convergence. Stage 2 applies the same strict threshold and is the
  final native stop. A stage cap is reported explicitly.
- OlhoffDu2007Repro: `max|Δrho| < 0.001` only after a successful LP; otherwise
  `CAP_HIT` at 1600 or `SOLVER_FAILURE`.

The common external diagnostic is evaluated only in the final native stage. Its first
accepted endpoint `k*` is the end of 10 consecutive successful iterations satisfying

\[
d_\infty(k)=\max_e|x^{phys}_e(k)-x^{phys}_e(k-1)|\le0.003,
\qquad
r_V(k)=\frac{|\bar x^{phys}(k)-0.5|}{0.5}\le10^{-3}.
\]

This is evidence of persistent stationarity and volume feasibility, not optimality and
not a universal definition of convergence. Store native stop status separately from one
of these exact harmonized statuses:

- `SATISFIED`;
- `NOT_SATISFIED_WITHIN_RECORDED_TRAJECTORY`;
- `CENSORED_BY_ITERATION_BUDGET`;
- `SOLVER_FAILURE`;
- `NOT_APPLICABLE`.

If a validated Olhoff trajectory remains move-saturated and never passes `d_infinity`,
report it as censored. Do not call it converged or failed for that reason, and do not
compute a Proposed/Olhoff or Yuksel/Olhoff speedup to common convergence.

## 5. Experiment A — controlled serial fixed-work scaling

Question: how does controlled serial cost per unit of each implementation's algorithmic
work scale, and is the fitted exponent stable across trajectory regimes?

- Paths: Proposed; Yuksel stage 1; Yuksel stage 2; Olhoff outer iteration including one
  Eq. (22) LP solve.
- On each of five meshes, record two 100-iteration windows in the same deterministic
  path invocation: `W1=11:110` and `W2=201:300`.
- `W1` is retained from R2. `W2` is preregistered because the already-frozen
  `fig3a_best` history places the Olhoff multiplicity transition at iteration 95, so W1
  straddles it and W2 samples a post-transition regime. No timing result selected W2.
- Per path/mesh: one untimed warm-up invocation and 10 measured invocations. A measured
  invocation executes through iteration 300 and returns both windows; it is not counted
  twice.
- If native stopping occurs before 300, fixed-work mode may suppress only the stop/handoff
  exit while executing the identical iteration kernel. Before campaign use, exact-prefix
  invariance must pass through the native endpoint and kernel-call/component identities
  must pass throughout both windows. Yuksel stage 1 requires its own handoff-suppression
  gate. A failed gate makes that path-window `NOT_APPLICABLE`; it does not permit a
  method-specific replacement window.
- Timed boundaries start immediately before FE/assembly and end after the accepted update.
  Exclude path setup, stage preparation, final analysis, history serialization, memory
  polling, plots, and I/O.
- Report median seconds/iteration, MAD, CV, range, raw repetitions, and component shares.

For each path and window independently fit five mesh medians by unweighted OLS:

\[
t_{iter}=C N_e^p,\qquad \log t_{iter}=\log C+p\log N_e.
\]

Report `C`, `p`, `R2`, adjusted `R2`, slope CI, residuals, and mesh count. Also report
`Delta_p=p_W2-p_W1` with a repetition-resampling 95% interval and per-mesh `W2/W1`
ratios. No universal pass/fail threshold is invented. If regimes differ, report both fits
and the regime dependence; never average them away. Yuksel stages remain separate.

## 6. Experiment B — method-native practical performance

- Three native meshes and the unresolved/resolved profiles in Section 3.
- One deterministic history-producing full run plus three measured full replays per
  method/mesh. The history run is the untimed warm-up/check and may be reused by C.
- Include native setup, all native stages/loops, and required native terminal analysis;
  exclude JSON parsing, plotting, common evaluation, writes, and memory measurement.
- Report `objective_native`, its definition/direction, `omega1_native` through
  `omega3_native`, native volume/grayness, topology, solver statuses, iterations, native
  stop class/reason, median wall time/range, and peak RSS.
- For Yuksel, always report `iter_stage1`, `iter_stage2`, and
  `iter_total=iter_stage1+iter_stage2`, plus `T_stage1`, `T_stage2`, and
  `T_Yuksel_total=T_stage1+T_stage2`. `T_Yuksel_total` is the practical method-level
  quantity; stage-specific times are diagnostics. A stage-2 speedup is never a
  whole-method speedup.
- This is not an identical-parameter table and contains no cross-method speedup.

## 7. Experiment C — harmonized stationarity and quality evaluation

Question: under the same external diagnostics, what quality and stationarity evidence
does each native trajectory provide?

- Use the three native meshes and Section 3 internal profiles. Preserve all native stage
  transitions; only final termination may be extended for diagnostic discovery after the
  exact-prefix gate.
- Apply Section 4 offline. If `k*` exists, replay exactly to it and verify the terminal
  density checksum. Use one warm-up plus 10 timed replays for a timing comparison.
- Always report the native endpoint and stop beside the harmonized status.
- Evaluate the native endpoint and `k*` (if distinct) using all Section 9 representations
  and models. A capped design may receive descriptive common-quality evaluation if the
  evaluator succeeds.

Valid comparisons when any row is censored are: fixed-work scaling from A; native
performance with native stop labels from B; common quality at explicitly named recorded
endpoints; stationarity trajectories and diagnostic values; and pairwise time-to-`k*`
only between two `SATISFIED` rows at the same mesh and timing boundary. Invalid
comparisons are any speedup involving a censored/failed/not-applicable row, treating the
budget endpoint as `k*`, ranking convergence speed from unequal recorded horizons, or an
aggregate speed ranking that silently includes censored rows.

## 8. Experiment D — targeted robustness

At `240×30`, retain the preregistered OAT levels below. Centers are shared across sweeps;
there is no Cartesian product and no result feeds back into B or C.

| Method | Parameter | Frozen levels |
|---|---|---|
| Proposed | `rmin` elements | 1.5, 2.0, 2.5 |
| Proposed | move | 0.1, 0.2, 0.3 |
| Proposed | native tolerance | 0.0005, 0.001, 0.002 |
| Yuksel | `rmin` elements | 2.0, 2.5, 3.0 |
| Yuksel | move | 0.1, 0.2, 0.3 |
| Yuksel | both stage tolerances | 0.005, 0.01, 0.02 |
| OlhoffDu2007Repro | `rmin` elements | 1.1, 1.3, 1.5 |
| OlhoffDu2007Repro | move | 0.0025, 0.005, 0.01 |
| OlhoffDu2007Repro | `tolMult` | 0.02, 0.05, 0.08 |

Add only the filter-radius rows at `320×40`, using the same three levels for each method.
The three native centers reuse B; this adds six optimization trajectories, not a second
full Experiment D.

For every radius row at both meshes record all native/common raw and binary frequencies,
volume, grayness, topology/connectivity, stop and solver status, iteration saturation,
and feature-scale proxies. Proposed additionally records its native objective, OC
bisection status, volume residual, and active-move fraction. Yuksel additionally records
stage counts, handoff, stage-2 moving-compliance/load/mode-change diagnostics, OC status,
and active-move fraction. Do not invent multiplicity labels for either method.

Olhoff additionally records `omega1..omega3`, actual absolute and relative
`omega1/omega2` gap, `N`, persistence of coalescence over the last 50 iterations,
topology/connectivity, volume, every LP status, failed-LP count, move saturation, and cap
status. The actual gap is authoritative. Native-radius performance and radius-sensitivity
evidence are distinct; no universal radius is inferred unless the evidence independently
supports one.

The common-stop grid `d_infinity={.002,.003,.005}` and
`rV={5e-4,1e-3,2e-3}` is evaluated offline from the same histories and causes no run.

## 9. Native, common raw, and binary evaluation

No FE interpolation is declared universally neutral.

### Native evaluation

Each method's own formulation produces `objective_native` and
`omega1_native..omega3_native`. These establish what the optimizer actually
optimized/reported and are never overwritten by common results.

### Common raw-density models

All models use the common Q4 mesh, consistent mass, supports, three-mode deterministic
eigensolver, tolerance `1e-8`, and residual/finiteness checks. Let `z` be the submitted
raw physical density.

- `E1_simp_linear_floor1e-6` (preregistered primary):
  `E/E0=1e-6+(1-1e-6)z^3`,
  `rho/rho0=1e-6+(1-1e-6)z`.
- `E2_yuksel_piecewise_floor1e-9`:
  `E/E0=1e-9+(1-1e-9)z^3` and
  `rho/rho0=1e-9+(1-1e-9)g(z)`, where `g(z)=z` for `z>0.1` and `z^6`
  otherwise.
- `E3_olhoff_eq4_rhomin1e-3`: set `z3=max(z,1e-3)`, then
  `E/E0=z3^3` and `rho/rho0=g(z3)` with the same discontinuous Du Eq. (4)
  piecewise law.

E1 remains primary because it is the pre-existing R2 common model, is numerically
regularized on arbitrary submitted fields, and was frozen before R3 results—not because
it is neutral or favorable to Proposed. It is explicitly model dependent. A raw-density
cross-method ordering may be called evaluator-robust only if every claimed pairwise
frequency difference has the same sign under E1, E2, and E3. No magnitude/tie threshold
is invented; print near-zero differences. If any claimed direction changes, report model
dependence and no single raw ranking. Columns are
`omega{1..3}_common_raw_E{1..3}`.

### Volume-preserving binary representation

Create a separate indicator field with exactly the target solid-element count. Sort by
raw density descending and break threshold ties by stable element index. Apply each E1–E3
model to that indicator (including E3's declared `z3` mapping), and report achieved
volume/connectivity. Columns are `omega{1..3}_common_binary_E{1..3}`. Binary evaluation
can favor already-discrete methods and penalize gray ones, so it cannot alone determine a
method ranking and is never a replacement for raw-density evaluation.

## 10. Controlled serial timing claims

Run one thread on one unchanged host; pin and record BLAS/OpenMP settings, CPU/RAM, OS,
MATLAB release/update and toolboxes, BLAS identity, power state, and run order. Keep the
host idle and on AC power. Interleave a frozen order independent of results. Report all
repetitions; replace only documented machine failures and retain both records. Use median
and MAD as primary summaries, plus CV and range.

These results support a **controlled serial computational comparison**. Single-threading
reduces resource-scheduling and hidden-parallelism differences and makes per-kernel work
more interpretable. It does not establish the fastest implementation, optimal parallel
scaling, or best multicore/GPU practical runtime. No supplementary practical-hardware
experiment is needed because R3 makes none of those claims; one becomes necessary only
if the paper adds such a claim.

Memory is measured separately in fresh processes using OS peak RSS, three repetitions,
and an empty-session baseline. Memory polling never occurs inside timing windows.

## 11. Publication table schemas

### Table A — controlled serial computational scaling

One row per method/stage/window: `method_stage`, `window_id`, radius value/unit/operator,
mesh count and element range, measured iterations, repetitions, median `t_iter` at each
mesh, MAD/CV/range, `C`, `p`, `p_95CI`, `R2`, adjusted `R2`, residual summary, component
shares, and kernel-validity status. A companion row/table reports `Delta_p`, its interval,
and per-mesh `W2/W1`. Yuksel stage 1 and 2 are distinct.

### Table B — method-native practical performance

One row per method/mesh: method, mesh, native profile ID and provenance status,
`objective_native` plus definition/direction, `omega1..3_native`, the E1–E3 common raw
and binary columns at the native endpoint, volume, grayness, topology/connectivity,
`iter_stage1`, `iter_stage2`, `iter_total`, outer/inner, `T_stage1`, `T_stage2`,
`T_Yuksel_total`, median native wall/range, peak RSS, native stop class/reason, and solver
status. Non-applicable cells are em dashes. Yuksel total is printed explicitly.

### Table C — harmonized stationarity and quality

One row per method/mesh: native endpoint iteration/class, recorded horizon, `k_star`,
harmonized criterion status, censoring reason/budget, endpoint `d_infinity` and `rV`,
E1–E3 common raw columns, E1–E3 binary columns, grayness, optimization time to `k_star`
only if satisfied, evaluator time, combined time, and eligible pairwise speedup. Speedup
cells involving any non-`SATISFIED` row are em dashes with reason.

### Table D / supplement — robustness

One row per method/mesh/varied parameter/level: native profile and fixed-settings ID,
native/common E1–E3 raw and binary frequencies, eigengaps, volume, grayness,
topology/connectivity/feature metrics, iterations/stages, descriptive wall time, native and
harmonized status, solver status, and method-specific diagnostics listed in Section 8.
Include explicit panels for the 240×30 OAT, 320×40 filter replication, and evaluator-model
sensitivity.

## 12. Failure, censoring, and exclusion rules

- Exclude no valid run because its result is unfavorable. Archive invalid repetitions and
  report the valid count; do not impute.
- Nonfinite values, factorization/eigensolver failure, invalid MMA/OC update, or LP exit
  other than success is `SOLVER_FAILURE`; failure precedence is mandatory.
- A safety budget is `CAP_HIT` natively and `CENSORED_BY_ITERATION_BUDGET` for the
  unmet common criterion. It is never convergence.
- Common quality may be reported at a capped/failed optimization endpoint only if the
  common evaluator succeeds and the row remains visibly censored/failed.
- Repro cluster truncation and multiple-mode warnings are counted and printed.
- Stale/pre-fix R2 outputs are excluded from R3 tables and parameter decisions. Existing
  timing evidence may be used only for capacity planning and structural feasibility as
  identified in the audit.

## 13. Configuration and regression contract

The JSON manifest is authoritative and rejects ambiguous shared optimization knobs.
Serialize resolved problem, method profile, provenance status, filter units/operator,
DOFs, interpolation/floors, stops/budgets, thread state, commit, hashes, run ID, window,
and evaluator model. Assert `iter_stage1+iter_stage2=iter_total`, LP
`inner=outer` on the R3 route, volume and solver consistency, and distinct native/common
namespaces. `optimization.olhoff_exact.included_in_r3=false` is non-dispatchable.

Primary scaling regression is unweighted OLS on five log-medians. Bootstrap over
repetitions within mesh supplies diagnostic percentile intervals; no mesh is removed. A
path-window with fewer than five valid mesh points is incomplete and receives no primary
cross-method scaling claim.

## 14. Campaign matrix and cost

| Work | Design | Invocations / evaluations |
|---|---|---:|
| A fixed-work timing | 5 meshes × 4 paths; one warm-up + 10 measured; both windows per invocation | 220 invocations, 400 measured timing segments |
| A memory | 20 path/mesh cells × 3 fresh processes | 60 invocations |
| B native full optimization | 3 meshes × 3 methods × (1 history + 3 timed) | 36 full runs |
| C discovery/extension | up to 3 meshes × Proposed/Yuksel; reuse B/Repro histories | at most 6 full/extended runs |
| C accepted endpoint timing | up to 9 satisfied cells × (1 warm-up + 10 measured) | at most 99 runs; censored cells add zero |
| D 240×30 OAT | 21 unique cells, 3 B centers reused | 18 new full runs |
| D 320×40 radius replication | 9 unique cells, 3 B centers reused | 6 new full runs |
| Gates | invariance/evaluator/manifest checks | approximately 12 checks |
| External FE evaluation | B native endpoints + C distinct endpoints + D endpoints, 2 representations × 3 models | 54 B calls + at most 54 C calls + 180 D calls, with 36 D-center calls reusable from B |

Upper bound: **445 optimization/path invocations** plus approximately 12 engineering
checks, or about **457 total campaign executions/checks**. The 445 include 280 short
fixed-work/memory paths and 24 new sensitivity runs. There is no evaluator Cartesian
sweep beyond the three frozen models and two frozen representations.

No new run was made for cost estimation. Existing evidence gives approximately
141–164 s for the post-fix 240×30 native-radius Repro trajectory, 0.02–0.12 s/iteration
for Proposed and 0.017–0.077 s/iteration for Yuksel over observed meshes in superseded
files used only for capacity planning, and no measured 480×60 or stage-separated upper
bound. Extending fixed-work paths to iteration 300, adding six 320×40 filter runs and
three evaluator models revises the deliberately broad serial planning range from 6–10 to
approximately **8–14 wall-clock hours**, plus validation. This is not a performance
prediction.

If this cost becomes excessive, the first reduction is the 60-run memory campaign unless
the paper retains a memory claim; next reduce memory repetitions or move non-radius OAT
dimensions to a later supplement. Do not remove E1–E3 evaluator robustness, the 320×40
radius replication, W2, or censored-row safeguards: those are the four adversarial fixes.

## 15. Revision rules

- Freeze all ranges, models, windows, statuses, and schemas before R3 results.
- Sensitivity results cannot replace native centers.
- A cap/parameter correction requires a declared R3.x revision preserving what was
  already observed. Performance ranking is never an admissible reason.
- Do not infer a universal filter radius or neutral evaluator from these experiments.

## 16. Freeze gates and Proposed pre-result rule

Methodological gate M1 is **closed** by `PROPOSED_NATIVE_PROFILE_AUDIT.md`. The selected
profile is the immutable pre-R3 A0/A4 OC profile at `d051985`, not the March MMA legacy
comparison profile. The decision applied the already-declared source order—manuscript,
explicit pre-R3 protocol, then predating implementation default—and used no R3 or stale
comparative outcome. OC versus MMA is therefore not an R3 numeric OAT dimension.

The signed audit, its source commit/config hash, and the three synchronized R3 deliverables
must be included in the eventual clean freeze commit. The executable `[0,1]` versus
manuscript `[1e-3,1]` design-bound discrepancy and the effective truncated versus declared
symmetric filter-boundary discrepancy remain disclosed limitations; neither may be changed
silently after results.

Engineering gates are: restore and regression-test the exact Proposed A0 load/profile on
the current branch; serialize and validate the effective Proposed design bound and filter
operator; Repro failure precedence; both-window fixed-work controls and prefix/kernel
invariance (including Yuksel stage-1 handoff suppression); callable E1–E3 raw/binary
evaluator validation; single-thread enforcement; fail-closed manifest validation; exact
endpoint replay; topology/connectivity metrics; synchronized table writers; and a clean
freeze commit/hash inventory.

R3 execution is forbidden until all engineering gates pass. Passing them changes the
status to **FROZEN** but changes no result-contingent choice.
