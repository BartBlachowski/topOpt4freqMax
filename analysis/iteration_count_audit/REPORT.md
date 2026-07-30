# Independent audit: mesh-dependent iteration counts in the Yuksel–Yilmaz implementation

Audit date: 2026-07-30  
MATLAB: R2025b Update 1  
Benchmark entry point: `examples/Performance/performance_comparison.m`

## Executive summary

The dominant mechanism is a three-way interaction specific to Yuksel Stage 2:

1. each outer iteration advances the mode estimate only once through
   \( \hat u_k \rightarrow F_k=M(x_k)\hat u_k \rightarrow
   U_{k+1}=K(x_k)^{-1}F_k \);
2. the OC update uses a stiffness-only compliance sensitivity and explicitly
   ignores the derivative of the changing load \(F=M(x)\hat u\);
3. convergence is declared from the single largest element-density update.

This is a coupled, lagged fixed-point iteration, not descent on one fixed
mesh-independent scalar objective. Mesh refinement changes the topology basin
and the physical regularization scale. On some meshes the mode/load feedback
therefore triggers a late topology transition; OC continues to move a small
number of elements while the global design and moving compliance are already
nearly stationary. The max-norm stopping test turns that localized tail into
hundreds or thousands of counted iterations.

The strongest case is 320x40:

- Stage 2 takes 1,372 iterations, versus 178 at 160x20 and 883 at 400x50.
- The consecutive-mode angle does not remain below 0.1 degrees until iteration
  549 (iteration 5 at 160x20 and 4 at 400x50).
- A significant binary-topology change (>0.1% of elements per 10-iteration
  snapshot) occurs as late as iteration 1,111; hole count changes until 1,301.
- At termination the maximum step is 0.002953, but the 95th-percentile step is
  only \(7.91\times10^{-6}\): a ratio of 374. The mean step is
  \(1.16\times10^{-5}\).
- The median relative change in the moving compliance over the second half is
  only \(7.65\times10^{-6}\), despite the long max-change tail.

Single-factor controls at 320x40 reduce Stage 2 as follows:

| Control | Stage-2 iterations | Reduction from 1,372 |
|---|---:|---:|
| Baseline | 1,372 | — |
| Hold physical filter radius at the 160x20 value (4 elements) | 653 | 52.4% |
| Freeze the Stage-1 mode estimate | 814 | 40.7% |
| Freeze the first Stage-2 inertial load | 616 | 55.1% |

These reductions overlap and must not be added. They show that neither the
mode update, filter scale, nor stopping norm alone is a complete explanation.
The moving-load feedback is the formulation-level source; shrinking physical
filter scale and mesh-specific topology transitions determine how strongly it
is excited; the max stopping norm amplifies it in the reported count.

This is not evidence of a broken linear solver, random nondeterminism, stale
cache, or checkerboard-dominated solution. It is best classified as an
algorithm/benchmark-protocol artefact (coupled partial-sensitivity OC plus a
mesh-dependent physical regularization and max-norm termination), not a simple
coding bug and not an intrinsic statement about all realizations of the
Yuksel–Yilmaz method.

One premise needs qualification. The current recorded artifact gives iteration
count coefficients of variation of 2.5% (Olhoff), 56.0% (Yuksel), and 41.8%
(Proposed) across all nine meshes. Proposed is stable over the first four
meshes (7.7% CV) but not over all nine. Yuksel is already unstable over the
first four (51.4% CV) and is much more expensive.

## Reconstructed execution path

The runner:

- reads `performance_comparison.json`;
- disables live/final/snapshot images;
- overrides the filter to radius 2 in element units;
- runs nine meshes, three approaches, and one sample each;
- calls `run_topopt_from_json` for every combination;
- computes total time as returned mean loop time per iteration times returned
  iteration count;
- writes `table1_performance.csv`, the paper-style TeX table, two complexity
  CSVs, and four complexity plots.

Relevant locations are `performance_comparison.m:4-27,34-44,60-116,202-275`.

`run_topopt_from_json.m:118-126` converts radius 2 elements to
\(r_\mathrm{phys}=2(8/\mathrm{nelx})\). Dispatch is:

| Method | Solver | Update | Load/mode handling | Termination |
|---|---|---|---|---|
| Olhoff | `topFreqOptimization_MMA` | MMA | eigensolve of lowest modes every iteration; stiffness and mass terms in eigenvalue sensitivity | relative omega change, RMS design change, grayness and projection-polish conditions |
| Yuksel | `top99neo_inertial_freq` | OC | Stage 1 point-load compliance; Stage 2 updates \(F=M(x)\hat u\) and one response/mode estimate every outer iteration; ignores \(dF/dx\) | `max(abs(x_new-x_old)) < 0.003` in Stage 2 |
| Proposed | `topopt_freq` | MMA (from JSON) | semi-harmonic load uses one cached solid-domain eigenpair; density weighting changes but the reference mode does not evolve | `max(abs(x_new-x_old)) < 0.003` |

Code evidence:

- Yuksel’s changing load and omitted load derivative are stated at
  `top99neo_inertial_freq.m:27-28` and implemented at `:576-637`.
- Its stop is at `:668`; Stage 1 is separately capped at 200 by
  `run_topopt_from_json.m:312-323`.
- Proposed caches the solid baseline once at `topopt_freq.m:284-317`, builds
  the semi-harmonic load from that fixed vector at `:868-895`, and uses the max
  design change at `:590-592`.
- Proposed’s semi-harmonic load sensitivity is also omitted by default
  (`topopt_freq.m:80-83,949-955`), so omitted load sensitivity alone cannot
  explain the difference. The differentiator is Yuksel's evolving mode
  estimate/full consistent mass feedback.
- Olhoff uses current eigenpairs and both stiffness and mass terms in
  `topFreqOptimization_MMA.m:272-358`; its multi-metric stop is at `:472-527`.
- Filter-type mapping is at `run_topopt_from_json.m:764-798`. Despite a common
  JSON label, the actual regularizations differ: Yuksel and Proposed use
  sensitivity filters; Olhoff uses its density filter and Heaviside projection.

No postprocessing eigensolve affects the measured loop count. Frequency-history
saving is off in the benchmark JSON, and the optional postprocessing starts
after timing in `run_topopt_from_json`.

## Result-artifact audit

### Available before this audit

- `performance_comparison.json`: tracked input, available.
- `table1_performance.csv`: available but git-ignored. Its timestamp
  (2026-07-30 03:20) predates the current runner timestamp (11:35), so it is
  formally stale relative to source. Exact current reruns reproduced all four
  audited Yuksel rows, all four Proposed rows, and the 160x20/320x40 Olhoff
  rows.
- `table1_paper_style.tex`: tracked, available, and later than the CSV; it is a
  derivative of the CSV, not raw history.
- `table1_complexity_fit.csv`,
  `table1_complexity_fit_fixedexp.csv`, and the four PNG complexity plots:
  available, ignored, and regenerated from the CSV on 2026-07-30.
- `table1_paper_style.pdf`: available and ignored, but not produced by the
  benchmark runner shown here; it is a separately compiled derivative.
- `results/*_freq_iterations.{fig,png}`: available only for selected meshes and
  dated March-May 2026. They predate the July implementation and cannot have
  been produced by the current performance runner because history saving is
  false. Their extracted lengths are inconsistent with the current table
  (for example Yuksel 320x40 has 1,610 plotted points), so they are treated as
  stale, unknown-provenance corroboration only.

### Missing before this audit

- No raw `.mat` run structures for the performance table.
- No design-change, mode-similarity, sensitivity, OC-multiplier, or topology
  history for the actual performance runs.
- No per-run console logs or environment/commit metadata tied to the ignored
  performance CSV.
- No frequency histories for all nine meshes.

### Regenerated during this audit

- Exact current MATLAB baseline diagnostics for Yuksel at 160x20, 240x30,
  320x40, and 400x50.
- Three labeled 320x40 causal controls (fixed physical radius, frozen mode,
  frozen inertial load).
- Current comparator verification for Proposed at all four coarse meshes and
  Olhoff at 160x20 and 320x40.
- Scalar histories, topology snapshots every 10 iterations, condition proxies,
  plots, and CSV summaries under `analysis/iteration_count_audit/results/`.
- Numeric extraction of the legacy `.fig` files under
  `results/legacy_fig_extracted/`, kept separate from authoritative data.

The audit logging is opt-in and default-false. Baseline runs with it enabled
reproduced the recorded counts exactly, demonstrating that the logging does
not change the baseline trajectory.

## Hypothesis results

### H1 — stopping criterion: strong amplifier, not sole root cause

Yuksel Stage 2 stops on the maximum accepted design-variable change. The same
0.003 threshold is dimensionless, but its max norm is statistically and
topologically mesh sensitive: more elements create more opportunities for one
boundary element to remain active.

| Mesh | Stage 2 | Persistent RMS < 0.003 | Persistent P95 < 0.003 | Final max/P95 |
|---|---:|---:|---:|---:|
| 160x20 | 178 | 48 | 51 | 5.2 |
| 240x30 | 540 | 269 | 241 | 42.1 |
| 320x40 | 1,372 | 832 | 542 | 373.5 |
| 400x50 | 883 | 110 | 7 | 343.8 |

At 320x40 fewer than 5% of elements control the stopping result after
iteration 542; at termination the mean and RMS steps are respectively
\(1.16\times10^{-5}\) and \(7.99\times10^{-5}\). A P95 rule with the same
numeric tolerance would reduce the count by 60.5%, but it would also ignore
real late member/hole changes, so that is a sensitivity estimate, not a
recommended replacement.

Proposed also uses a max change, proving that the norm by itself is not enough.
Its fixed reference mode and MMA trajectory do not excite the same prolonged
localized tail on the first four meshes.

### H2 — sensitivity scaling: absolute scale changes, shape changes only mildly

The mean filtered sensitivity magnitude at Stage-2 iteration 1 decreases from
\(2.02\times10^{-9}\) (160x20) to \(2.36\times10^{-11}\) (400x50), an 85-fold
change. This is real mesh scaling. OC, however, absorbs uniform scale through
its Lagrange multiplier. After volume-gradient normalization, the mean OC
driver divided by the multiplier changes only from \(4.20\times10^{-4}\) to
\(2.42\times10^{-4}\) at iteration 1.

The coefficient of variation of the normalized OC driver grows monotonically
from 0.87 to 1.17, and the sensitivity maximum becomes more separated from its
95th percentile. This promotes localized updates, but it does not explain the
320-to-400 count drop: the scaling/localization trend is monotonic while the
count is not. H2 is a secondary contributor to the max-norm tail.

### H3 — mode evolution: causal and strongly mesh dependent

The mode estimate is normalized and sign-aligned each iteration. Euclidean
cosine, angle, and MAC (cosine squared) were recorded.

- Persistent angle <0.1 degrees: iterations 5, 238, 549, and 4 for 160, 240,
  320, and 400 meshes.
- Persistent `du < 0.003`: iterations 4, 238, 549, and 2.
- At 320x40 the maximum consecutive angle is 3.50 degrees.

Freezing the Stage-1 mode estimate at 320x40 cuts Stage 2 from 1,372 to 814
(40.7%). Freezing the first inertial load cuts it to 616 (55.1%). Therefore
the lagged mode/load feedback is causal.

The stale legacy frequency plot independently shows the same event: at 320x40
omega1 falls from 150.27 rad/s at the Stage-1 boundary to 133.58 rad/s near
total iteration 662, then recovers to about 161.22; the 400x50 legacy history
does not show this collapse. Because those plots predate current source, this
is corroboration, not primary evidence.

### H4 — topology transitions: strong explanation of the residual tail

Using a 0.5 density threshold and snapshots every 10 Stage-2 iterations:

- 320x40 has 41 significant binary-change intervals and cumulative binary
  churn 0.193; 400x50 has 20 intervals and churn 0.125.
- The last significant transition occurs at iterations 1,111 and 721,
  respectively.
- Hole count changes until iteration 1,301 at 320x40, versus 771 at 400x50.
- Final hole counts are 13, 13, 19, and 31 over the four meshes. Different
  meshes are not tracing a simple nested discretization of one topology.

The topology montage shows delayed loss/merging of thin central members at
320x40. These events explain why the maximum element change can remain large
after mean changes and the moving compliance have flattened.

### H5 — OC dynamics: amplifier/conduit, not a failed bisection

The OC multiplier is smooth: its coefficient of variation over the second
half is below 0.8% in all four baselines. Volume stays within
\(4.12\times10^{-4}\) of 0.5. Ten-iteration snapshot directions are almost
always positively aligned, so there is no global two-cycle.

There are nevertheless localized aggressive steps. The 0.2 move limit remains
active somewhere through Stage-2 iterations 4, 268, 833, and 27 across the
four meshes; maximum changes above 0.01 persist through 165, 522, 1,305, and
843. OC is reacting to moving/localized sensitivities, but the multiplier
bisection itself is stable and finite-guarded.

The comparator difference is real: the benchmark forces MMA for Proposed,
while Yuksel is hard-coded OC. An exact Yuksel-with-MMA factorial control is
not implemented, so the isolated percentage due purely to optimizer choice
remains unresolved. The measured evidence rejects an OC bracket/volume bug.

### H6 — filter influence: major interacting cause

Radius 2 elements means physical radii of 0.100, 0.0667, 0.050, and 0.040 m
over the first four meshes. Thus the runner keeps stencil size fixed but
changes the continuum regularization problem.

At 320x40, using radius 4 elements to preserve the 160x20 physical radius:

- reduces Stage 2 from 1,372 to 653 (52.4%);
- restores final top-bottom symmetry error from \(3.57\times10^{-3}\) to
  \(4.52\times10^{-10}\);
- changes the final hole count from 19 to 13;
- changes omega1 from 161.23 to 159.38 rad/s.

This is strong causal evidence. It cannot alone explain why Yuksel differs
from both comparators because all receive an element-scaled radius, but their
actual filter/projection and optimizer formulations differ. Yuksel's coupled
mode/load OC loop is much more sensitive to the topology branch exposed by
the shrinking physical length scale.

### H7 — mesh artefacts: symmetry branch matters; checkerboards do not dominate

The final binary checkerboard proxy is 0-0.18% on the fine audited meshes and
no checkerboard-dominated pattern is visible. That rejects checkerboarding as
the dominant mechanism.

The 320x40 baseline does exhibit top-bottom symmetry error
\(3.57\times10^{-3}\), versus approximately \(10^{-6}\) or less at 160, 240,
and 400. The fixed-physical-radius control removes it. This is evidence of a
mesh/filter-dependent symmetry-broken topology branch, not random seeding
(the code path is deterministic).

### H8 — linear solver behavior: rejected as dominant

Yuksel Stage 2 performs a fresh direct sparse Cholesky solve
(`decomposition(...,'chol') \ F`) and no inner iterative linear solve. Extra
linear-solver iterations therefore cannot increase the outer count.

A final-stiffness spectral condition proxy increases monotonically:
\(1.78\times10^8\), \(3.78\times10^8\), \(6.41\times10^8\), and
\(9.55\times10^8\). Outer count nevertheless drops from 1,372 at 320x40 to
883 at 400x50. Conditioning raises numerical risk and per-iteration cost, but
does not track the non-monotonic outer iteration count.

### H9 — objective landscape: globally flat with localized topology motion

The second-half median relative change of the moving compliance is
\(3.07\times10^{-5}\), \(4.44\times10^{-5}\), \(7.65\times10^{-6}\), and
\(1.01\times10^{-5}\). The anomalous 320x40 case is especially flat globally
while maximum density changes remain large.

Because \(F\) changes and its derivative is omitted, this compliance is not a
fixed objective value; it is still a useful flatness indicator. The evidence
supports a broad, nearly indifferent region containing multiple thin-member
topologies. Local member/hole motion—not continued large global improvement—
dominates the tail.

## Ranked mechanisms

1. **Coupled one-step mode/inertial-load feedback with partial sensitivity.**
   Directly causal: freezing mode or load removes 41-55% of the 320x40 Stage-2
   iterations.
2. **Changing physical filter scale and the topology/symmetry branch it
   selects.** Directly causal: fixed physical radius removes 52% and changes
   topology/symmetry. This strongly interacts with item 1.
3. **Maximum-element stopping criterion applied to localized OC motion.**
   It magnifies the above dynamics into the reported count; the 320x40 final
   max/P95 ratio is 374.
4. **Flat discrete topology landscape and sensitivity localization.**
   Explains late hole/member transitions after global quantities stabilize.
5. **Optimizer choice (OC versus comparator MMA).** Plausible secondary
   interaction, but the OC multiplier and volume solve are stable; an exact
   optimizer factorial was not run.

Rejected as dominant: absolute sensitivity scale alone, checkerboards,
linear-solver iteration count/conditioning, random seeds, stale eigenpair
cache, and result aggregation/timing.

## Scientific answer

Only Yuksel has an outer loop in which the design, mass matrix, inertial load,
and mode estimate all move together while the OC sensitivity treats the load
as fixed. Refining the mesh with a fixed two-element filter changes the
physical problem and admits different thin-member/hole branches. Some meshes
(especially 320x40) then experience a late mode/topology transition; others
(400x50) settle their mode almost immediately. The OC update follows these
localized changes, and a max-element stopping rule counts them until the last
few elements move by less than 0.003.

Olhoff avoids this mechanism by optimizing current eigenvalues with current
eigenvectors, including stiffness and mass sensitivity terms, under MMA and a
multi-metric convergence rule. Proposed uses a cached solid-domain reference
mode and MMA, so it has design-dependent loading but no evolving-mode
fixed-point feedback. That is why Yuksel becomes strongly mesh dependent much
earlier and more severely.

The phenomenon is resolved at the implementation/experiment level: it is an
expected consequence of this particular coupled surrogate, filter scaling,
OC trajectory, and stopping norm. It is not demonstrated to be an unavoidable
property of the published continuum method, nor is there evidence for a
single implementation bug.
