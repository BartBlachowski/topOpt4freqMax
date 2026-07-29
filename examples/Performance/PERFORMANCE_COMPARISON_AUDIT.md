# Performance-comparison scientific and implementation audit

Audit date: 2026-07-29  
Audited entry point: `run_topopt_from_json(data)`  
Primary driver: `examples/Performance/performance_comparison.m`  
Primary configuration: `examples/Performance/performance_comparison.json`

This is an adversarial audit of the legacy cross-code experiment, not an
endorsement of its outputs. The unsuccessful `OlhoffApproachExact`
reconstruction is outside this audit. No production optimizer or manuscript
text was changed, and no full benchmark campaign was launched.

## 1. Executive verdict

**Decision: D. REMOVE.**

The experiment cannot support a manuscript-level comparison of the published
Olhoff, Yuksel, and proposed methods. The three branches do not solve the same
optimization problem; the comparator codes are explicitly local, non-faithful
implementations; the reported timing, iteration, and memory quantities are not
publication-valid; and the proposed `240x30` result reached the iteration cap
with `ch. > tol`, so it is not converged. The repository's governing revision
decision had already retired this evidence, and the active manuscript has
already removed the quantitative cross-code table.

The numbers may be retained only as clearly quarantined internal observations
about these exact software implementations. They must not be described as
method-level evidence, used to rank natural-frequency performance, or placed
back into the paper.

No rerun of the current driver can change this verdict. A future meaningful
comparison would require a new shared formulation and instrumentation; that
would be a **redesigned experiment**, not a repair of this Table 1.

## 2. Immediate critical findings

Each finding below contains the required severity, evidence, consequence,
correction, and invalidation judgment. Later sections provide the detailed
trace and supporting calculations.

| ID | Severity | Finding and evidence | Scientific consequence | Recommended correction | Existing results invalidated? |
|---|---|---|---|---|---|
| F01 | **CRITICAL** | The driver declares itself a “LEGACY CROSS-CODE TIMING DRIVER — NOT REVIEWER EVIDENCE,” says EXP1/EXP5 are retired, prohibits timing/memory/speedup/scaling claims, and says its numbers must not enter a manuscript (`performance_comparison.m:1-26`). The same file still prints “Table 1” and writes `table1_performance.csv` (`performance_comparison.m:160-173,233-264`). | The artifact contradicts its own governing provenance and is easy to misuse as reviewer evidence. This is not a comment-only defect. | Keep Table 1 absent; quarantine or rename this driver/output as an internal legacy benchmark and add a hard publication-evidence guard. | **Yes: all manuscript/table uses of this experiment.** |
| F02 | **CRITICAL** | Olhoff optimizes a lower bound on several eigenvalues with MMA, projection, continuation, and grayness; Yuksel performs a compliance warm start followed by an inertial-load fixed-point/OC loop; OurApproach minimizes compliance under a solid-reference modal surrogate and reports frequency ex post (`topFreqOptimization_MMA.m:274-404`; `top99neo_inertial_freq.m:183-219,516-633`; `topopt_freq.m:315-405,635-703,1060-1134`). | Achieved frequencies, convergence, and costs are outcomes of different mathematical problems. They cannot establish relative method performance. | If comparison is still wanted, define one common problem and implement controlled ablations in shared analysis/optimizer infrastructure. | **Yes: all cross-method rankings and method-level claims.** |
| F03 | **CRITICAL** | The proposed `240x30` run stopped at 10,000 with `ch.=0.010-0.012`, while the configured tolerance is `0.005` (`performance_comparison.json:82-86`). The driver records the capped count without a convergence gate (`performance_comparison.m:130-152`). `topopt_freq` defines `ch` as maximum design-variable change (`topopt_freq.m:786-788`). | The reported `124.2 rad/s` is a limit-hit endpoint, not a converged optimum. Printing it beside terminating runs silently changes its meaning. | Add mandatory `converged`, `termination_reason`, constraint, stationarity, and cap-rejection gates before any export. | **Yes: proposed `240x30`; it also blocks acceptance of the whole table.** |
| F04 | **CRITICAL** | The performance JSON runs the proposed branch with solid baseline, complete load sensitivity, and MMA (`performance_comparison.json:62-72,82-86`), while the manuscript's nominal proposed formulation repeatedly describes a uniform reference, omitted load derivative, and OC (`paper/main.tex:129-131,249-267,280-316,649-651`). | “ProposedApproach” in the table is not the documented paper method. Its convergence behavior cannot be attributed to that method. | Reconcile and freeze one authoritative formulation before any new experiment; encode it in a versioned manifest and test the parsed settings. | **Yes: interpretation of both proposed-method rows and comparisons against manuscript claims.** |
| F05 | **MAJOR** | Yuksel uses linear mass above `x=0.1` and `x^6` at and below it (`top99neo_inertial_freq.m:550-559`), producing an approximately `100,000`-fold density contribution jump across the cutoff (`performance_audit_gradient_results.json:21-28`). The wrapper's common linear-mass recomputation is discarded (`run_topopt_from_json.m:553-580,1035-1077`). | Yuksel's returned frequency uses a materially different and discontinuous mass model, so its frequency cannot be compared directly with the other two. | Use one documented continuous mass interpolation and independently evaluate all designs under that same model. | **Yes: Yuksel frequencies and all cross-branch frequency rankings.** |
| F06 | **MAJOR** | Filters are not equivalent. Olhoff applies density filtering plus Heaviside projection and continuation; Yuksel and OurApproach map the requested sensitivity filter to unprojected sensitivity heuristics. OurApproach uses a truncated-boundary sparse filter, whereas the comparators use symmetric padding (`run_topopt_from_json.m:237-519`; `topFreqOptimization_MMA.m:91-155,248-260`; `top99neo_inertial_freq.m:413-455,519-595`; `topopt_freq.m:211-242,727-733,1962-1972`). | Regularization, boundary influence, grayness, and update directions differ. Fixed `r=2` elements also changes physical radius from `0.10 m` at `160x20` to about `0.0667 m` at `240x30`. | Harmonize the filter operator, boundary treatment, projection, continuation, and physical radius. | **Yes: mesh comparisons, topology/frequency comparisons, and iteration comparisons.** |
| F07 | **MAJOR** | The table run did not save final designs, modes, histories, convergence reasons, constraints, or cycle states. The CSV contains no frequencies or provenance (`table1_performance.csv:1-7`), and the JSON disables frequency history (`performance_comparison.json:89-100`). | The six endpoints cannot be independently reproduced, mode-validated, or audited for grayness, localization, connectivity, and final constraints. The requested late `240x30` state is unavailable for finite differences or cycle tests. | Persist immutable configs, final `x/xPhys`, K/M metadata, modes, objective/frequency/constraint histories, MMA state, and hashes. | **Yes: final-frequency acceptance for every row.** |
| F08 | **MAJOR** | Each branch reports loop-only time, while the driver reconstructs total time as `tIter*nIter` (`performance_comparison.m:156-157`). Setup and final eigensolves are excluded differently; the fixed execution order and one sample invite JIT/cache bias (`performance_comparison.m:89-93,111-154`). A `32x4` diagnostic showed wrapper wall/loop-reconstructed times of `3.158/0.427 s`, `0.297/0.071 s`, and `0.563/0.305 s` (`performance_audit_smoke_results.json:15-20,51-56,88-93`). | The reported “Run time” is not a comparable directly measured wall time. The diagnostic ratios are illustrative only, not replacement benchmark data. | Measure end-to-end wall time with warm-up, randomized order, repetitions, environment metadata, and variance. | **Yes: all runtime, per-iteration-time, speedup, and scaling claims.** |
| F09 | **MAJOR** | `mem` is a sampled increment over an in-process RSS baseline, polled every `0.1 s`, and finalized before wrapper postprocessing (`run_topopt_from_json.m:216-225,539-545,1622-1661`). The fixed order makes later baselines include earlier MATLAB allocations. | Values such as `14-40 MB` are neither process peak RSS nor comparable “Max RAM.” Short peaks may be missed and the first method is structurally disadvantaged. | Use an external per-process peak-RSS measurement in isolated processes, with the same scope and repeated trials. | **Yes: all Max RAM and memory-efficiency claims.** |
| F10 | **MAJOR** | One Olhoff outer iteration normally includes a main and trial three-mode eigensolve; Yuksel iterations are one linear solve in each of two different stages and its reported total may include up to 200 warm-start iterations in addition to the requested stage-2 cap; OurApproach normally performs one multi-RHS linear solve and no in-loop eigensolve (`topFreqOptimization_MMA.m:274-315,453-467`; `top99neo_inertial_freq.m:362-373,436-470,544-624`; `topopt_freq.m:456-703`). | “Iterations” are not common work units and cannot rank efficiency. | Report operation counts and wall time by phase; do not compare raw iteration totals as method efficiency. | **Yes: all iteration-efficiency claims.** |
| F11 | **MAJOR** | Controlled finite differences show: (a) OurApproach's unfiltered complete derivative matches the re-evaluated full surrogate (`L2=5.18e-6`, `7.66e-6`, `7.12e-6`), but its actual sensitivity-filter direction does not (`L2=0.546`, `0.469`, `1.220`); (b) Yuksel's stiffness-only derivative matches a frozen-load objective (`L2≈3.6e-6-5.3e-6`) but differs from the full `M(x)uhat` objective (`L2≈0.454-0.510` before filtering); (c) Olhoff's design/eigen chain is accurate, but the implemented derivative with respect to `Eb` has `82.7%-89.8%` relative error (`performance_audit_gradient_results.json:53-168,585-700,1117-1232,284-360,816-892,1348-1424,554-558,1086-1090,1618-1622`). | Termination does not imply stationarity of the stated objective. The Olhoff MMA subproblem is supplied a wrong derivative; Our/Yuksel production directions are heuristic or incomplete for re-evaluated objectives. | Correct Olhoff's quotient derivative; explicitly define frozen versus full objectives; use an exact chain-rule filter for gradient-based claims or describe and validate the heuristic algorithm as such. | **Yes: stationarity/convergence claims and method-level comparisons; raw interpolation formulas themselves passed the bounded checks.** |
| F12 | **MAJOR** | The wrapper performs a common three-mode recomputation only after timing/memory but discards it; the optimizer-returned frequency remains branch-specific (`run_topopt_from_json.m:553-580`). In a small smoke run, Yuksel differed from the common linear-mass recomputation by `1.49%`, while Olhoff/OurApproach agreed to numerical precision (`performance_audit_smoke_results.json:21-31,57-67,94-104`). No saved table designs exist for the mandatory validation. | The table's frequency column mixes mass definitions and has not passed independent modal validation. | Recompute several modes for every persisted final design with one common K/M/BC definition and check residuals, ordering, localization, connectivity, volume, and grayness. | **Yes: all achieved-frequency comparisons.** |
| F13 | **MAJOR** | Yuksel stage 2 starts from its own stage-1 compliance topology; the other branches start from uniform raw density. Optimizers, lower bounds, move rules, stopping tolerances, and continuation differ (`top99neo_inertial_freq.m:53-60,166-219`; `topFreqOptimization_MMA.m:157-180`; `topopt_freq.m:159-184,744-788`). | The comparison mixes a warm-started staged algorithm with cold-start single-loop algorithms and non-equivalent stopping rules. | Treat the warm start as part of Yuksel's algorithm and compare end-to-end cost only after problem harmonization; do not interpret stage totals as equivalent iterations. | **Yes: fairness and convergence-speed claims.** |
| F14 | **MODERATE** | Several JSON keys are inactive or branch-dependent: `objective` is ignored by all three dynamic branches; comparator branches ignore load cases and optimizer; `harmonic_normalize` does not affect semi-harmonic loads; `semi_harmonic_rho_source` is parsed as obsolete/dead; the proposed `heaviside` flag is inert for a sensitivity filter (`run_topopt_from_json.m:116-175,237-519,2150-2156`; `topopt_freq.m:1145-1215,1408-1437,1962-1972,2214-2230`). A bounded toggle produced bitwise-identical proposed designs (`performance_audit_proposed_variants.json:820-1213,2413-2414`). | The configuration gives a false appearance of common control and reproducibility. | Reject unsupported fields per branch or emit an explicit parsed/effective configuration manifest with warnings. | **Yes: claims that the same JSON settings governed all methods; not every scalar endpoint solely because of this defect.** |
| F15 | **MODERATE** | Both midpoint supports restrain both `ux` and `uy` (`performance_comparison.json:55-60`). This is a two-pin constraint, not the conventional pin-plus-roller idealization usually meant by a simply supported beam. | The boundary condition may alter frequencies and must not be inferred from the label alone. | State the exact constrained DOFs and use one verified BC in all independent recomputations. | **Yes: any claim relying on a conventional simply supported BC; branches do receive the same listed supports.** |
| F16 | **MAJOR** | The active manuscript contains no Table 1 and explicitly removes cross-code performance claims (`paper/main.tex:374-379,631-644,683-692`), consistent with `SCIENTIFIC_DECISION_EXP1_EXP5.md:1-20,26-73`. However, it still contains formulation/settings claims inconsistent with this configuration and unsupported endpoint frequencies (`paper/main.tex:98,129-140,153,198-208,239-316,333-372,381-416,649-675,704-706`). | Reintroducing the legacy table would reverse the governing decision. Separately, stale methodology and endpoint text can misdescribe the active evidence chain. | Do not restore Table 1. Audit and reconcile manuscript formulation/endpoints against accepted manifests in a separate manuscript change. | **Yes: any proposed Table 1 insertion and the listed unsupported/inconsistent claims; no current table exists to delete.** |
| F17 | **MODERATE** | Current performance settings schedule no modal refresh, so repeated refresh, sign switching, and mode switching cannot explain the observed late `ch`. When refresh was explicitly enabled on `32x4`, a clean session first lacked `a4_mode_screen`; after adding its path diagnostically, the branch reached a classified B3 connectivity failure at iteration 16 (`topopt_freq.m:523-600`; `performance_audit_proposed_variants.json:13,812-817`). | Refresh is not the cause of this specific `240x30` cap, but the selectable refresh branch is not self-contained and has a separate robustness dependency. | Wire and test refresh dependencies explicitly before studying refreshed-load behavior; retain B3 as a classified failure, not an automatic recovery. | **No for the frozen-reference `240x30` causal claim; yes for claims that refresh-every-iteration is currently production-ready.** |
| F18 | **MODERATE** | Olhoff stores and returns the historically best frequency/design, not necessarily the last iterate, and uses design/relative-frequency/grayness logic beyond a single `ch` threshold (`topFreqOptimization_MMA.m:345-348,474-529,558-579`). | Its returned topology and iteration count do not describe the same endpoint semantics as the other branches. | Report final and best-so-far states separately, with the actual termination reason and constraints for each. | **Yes: endpoint and convergence-semantics equivalence.** |
| F19 | **MINOR** | The generated CSV stores method, mesh, iterations, reconstructed time, per-iteration time, and “MaxRAM,” but not frequency, convergence status, config hash, environment, or provenance (`table1_performance.csv:1-7`). | The file is not a self-auditing scientific result and can be detached from its warnings. | Replace any future result with a schema-validated manifest; rename the legacy file so it cannot be mistaken for paper evidence. | **Yes: standalone evidentiary use of the CSV.** |

## 3. Actual mathematical problem solved by each method

Let \(x\) denote element design variables, \(\rho\) the physical density,
\(K(\rho)\) and \(M(\rho)\) the assembled matrices, and \(V(\rho)\) the
volume. The common material settings requested by JSON are
\(E_0=10^7\), \(\rho_0=1\), and stiffness/mass void ratios \(10^{-6}\)
(`performance_comparison.json:43-53`). Similar symbols below do not imply
equivalent algorithms.

### Olhoff branch

The implemented design variables are \((x,E_b)\), with
\(x_i\in[10^{-3},1]\) except pinned passive elements and an auxiliary lower
bound \(E_b\) (`topFreqOptimization_MMA.m:157-176`). Physical density is

\[
\rho=P_{\beta,\eta}(Hx),
\]

where \(H\) is a symmetric density filter and \(P\) is a Heaviside projection
whose \(\beta\) ramps continuously toward 64
(`topFreqOptimization_MMA.m:91-155,232-260`). The state problems are

\[
K(\rho)\phi_j=\lambda_jM(\rho)\phi_j,\qquad j=1,2,3,
\]

with

\[
E_e=E_{\min}+\rho_e^3(E_0-E_{\min}),\qquad
\rho^m_e=\rho_{\min}+\rho_e(\rho_0-\rho_{\min}).
\]

MMA minimizes

\[
f_0(x,E_b)=-E_b+\gamma(\beta)\,\frac1{n_e}
            \sum_e4\rho_e(1-\rho_e)
\]

subject to

\[
g_j=\frac{E_b-\lambda_j/\lambda_{\rm ref}}{\max(1,E_b)}\le0,
\quad
\overline{\rho}-v_f\le0,\quad
v_f-\overline{\rho}\le0 .
\]

The last two inequalities enforce approximate volume equality. The implemented
\(\partial g_j/\partial E_b=1/\max(1,E_b)\) is wrong when \(E_b>1\), because
the denominator also depends on \(E_b\)
(`topFreqOptimization_MMA.m:383-404`). The branch normally evaluates three
modes at the current point and three more at a trial point. It returns the
historically best \(\omega_1\), not necessarily the last iterate. Its stopping
logic combines recent design change, relative frequency change, grayness, and
iteration cap (`topFreqOptimization_MMA.m:474-529,558-579`).

This is a local max-min eigenvalue formulation with added mechanisms. It does
not consume the JSON compliance objective or semi-harmonic loads.

### Yuksel branch

The design variables are \(x_i\in[0,1]\), with the requested sensitivity-filter
setting yielding \(\rho=x\). It is a two-stage algorithm:

1. Stage 1 minimizes compliance \(F_0^TK(x)^{-1}F_0\) for a fixed point load
   using OC, stopping when maximum design change is below `0.01` or at its
   stage cap.
2. Stage 2 starts from that topology. At iteration \(k\), it forms
   \[
   F_k=M(x_k)\widehat u_{k-1},\qquad
   K(x_k)u_k=F_k,
   \]
   Euclidean-normalizes and sign-orients \(u_k\), and minimizes the current
   compliance \(F_k^Tu_k\) with OC.

The stage-2 mass law is

\[
\rho^m(x)=\rho_{\min}+(\rho_0-\rho_{\min})
\begin{cases}
x^6,&x\le0.1,\\
x,&x>0.1,
\end{cases}
\]

which is discontinuous at \(0.1\)
(`top99neo_inertial_freq.m:516-582`). The production sensitivity differentiates
only stiffness while holding \(F_k\) fixed
(`top99neo_inertial_freq.m:584-595`). Stage 2 stops only when maximum design
change is below the JSON tolerance (`0.005`) or at its cap; the computed mode
change `du` is logged but not part of termination
(`top99neo_inertial_freq.m:603-628`). The final natural frequency is an
eigensolve under the branch's discontinuous mass model.

This is a staged sequential compliance/fixed-point heuristic, not direct
first-eigenfrequency maximization. It ignores the JSON load cases, objective,
optimizer choice, and Heaviside flag.

### OurApproach branch

The design variables are \(x_i\in[0,1]\). With sensitivity filtering selected,
\(\rho=x\); the Heaviside flag does not project the density. A solid reference
eigenpair is computed once:

\[
K(1)\Phi_0=\omega_0^2M(1)\Phi_0,\qquad
\Phi_0^TM(1)\Phi_0=1 .
\]

Because no `update_after_iterations` is configured, this reference is frozen.
At every iteration the active mode-1 load is

\[
F(x)=\omega_0^2M(x)\Phi_0 ,
\qquad K(x)u(x)=F(x),
\qquad c(x)=F(x)^Tu(x).
\]

The second configured load case has overall factor zero and contributes
nothing. Stiffness is SIMP with exponent 3; mass is linear
(`our_mass_interpolation.m:26-34`). With `load_sensitivity="complete"`, the raw
derivative includes

\[
\frac{dc}{dx_e}
=2u^T\frac{\partial F}{\partial x_e}
-u^T\frac{\partial K}{\partial x_e}u .
\]

That raw derivative is correct for this frozen-reference, design-dependent-load
objective. The production sensitivity filter then transforms it with the
standard density-weighted heuristic rather than an exact chain rule for a
filtered design (`topopt_freq.m:669-733`). MMA minimizes compliance subject to
volume, with move limit `0.2`. After each update, a downward volume projection
is applied. It stops when
\(\max_i|x_i^{k+1}-x_i^k|<0.005\) or the cap is reached
(`topopt_freq.m:744-788`). The printed `mma_pre` and `mma_post` are volume
residuals immediately before and after the update, not KKT or MMA residuals
(`topopt_freq.m:992-998`). The final \(\omega_1\) is an ex-post eigenvalue of
the final topology, not the optimized objective (`topopt_freq.m:1060-1134`).

### Side-by-side problem table

| Property | Olhoff | Yuksel | OurApproach |
|---|---|---|---|
| Primary objective | Maximize lower bound on lowest 3 eigenvalues, plus gray penalty | Stage 1 fixed-load compliance; stage 2 inertial-load compliance/fixed point | Semi-harmonic surrogate compliance |
| Directly maximizes \(\omega_1\)? | Approximately, through a max-min eigenvalue formulation | No | No |
| State solve in main loop | 3-mode generalized eigensolve plus trial eigensolve | One linear solve; no in-loop eigensolve | One multi-RHS linear solve; no in-loop eigensolve under this config |
| Load | No external load objective | Internal \(M(x)\widehat u\); JSON loads ignored | JSON mode-1 semi-harmonic load |
| Load refresh | Not applicable | Displacement/mode estimate updated every stage-2 iteration | Reference mode never refreshed; \(M(x)\Phi_0\) still changes with design |
| Load derivative | Not applicable | Omitted/frozen within iteration | Complete before heuristic sensitivity filter |
| Stiffness | SIMP \(p=3\), lower bound \(10^{-3}\) on design | SIMP \(p=3\), design can reach 0 | SIMP \(p=3\), design can reach 0 |
| Mass | Linear | Discontinuous \(x^6/x\) branch | Linear |
| Filter | Symmetric density filter | Symmetric sensitivity heuristic | Truncated-boundary sensitivity heuristic |
| Projection | Heaviside, continuous \(\beta\) continuation | None for requested filter | None for requested filter; flag inert |
| Optimizer | MMA with auxiliary \(E_b\) | OC | MMA |
| Initialization | Uniform raw \(x=v_f\) | Uniform stage 1; nonuniform stage-1 warm start for stage 2 | Uniform raw \(x=v_f\), but solid modal reference |
| Volume handling | Two MMA inequalities on projected mean | OC preserves current mean | MMA inequality plus post-update downward projection |
| Stopping | Multi-condition design/frequency/gray logic | Per-stage maximum design change; stage-1 tol `0.01`, stage-2 `0.005` | Maximum raw-design change `0.005` |
| Returned frequency | Best-so-far topology | Final eigensolve with branch mass law | Final ex-post eigensolve with linear mass |
| JSON `"objective":"compliance"` | Ignored | Ignored as a field; implementation happens to use compliance surrogates | Ignored as a selector; implementation happens to use compliance |

**Conclusion [F02, CRITICAL]:** the shared JSON label does not define a shared
mathematical experiment. Only Olhoff directly places eigenvalues in its
optimization constraints; the other two optimize different compliance
surrogates and report frequency afterward.

## 4. Call graph and implementation map

```text
performance_comparison.m
  └─ run_topopt_from_json(data)
      ├─ parse/validate common JSON, supports, passives, radii, RSS sampler
      ├─ approach = Olhoff
      │   └─ topFreqOptimization_MMA
      │       ├─ local filter/projection
      │       ├─ evalEigen / eigs (3 modes at current point)
      │       ├─ mmasub → subsolv
      │       └─ evalEigen / eigs (3 modes at trial point; final best eigensolve)
      ├─ approach = Yuksel
      │   └─ top99neo_inertial_freq
      │       ├─ localComplianceLoop
      │       │   └─ K \ F0 → sensitivity filter → localOcUpdate
      │       ├─ stage-1 final eigensolve / mode estimate
      │       ├─ localInertialLoop
      │       │   └─ M(x) uhat → K \ F → sensitivity filter → localOcUpdate
      │       └─ localFirstNOmegas (final eigensolve)
      └─ approach = OurApproach
          └─ topopt_freq
              ├─ initial solid K/M eigensolve
              ├─ optional refresh path → a4_mode_screen
              ├─ build semi-harmonic load
              ├─ K \ F → raw compliance sensitivity
              ├─ sensitivity filter
              ├─ mmasub → subsolv (or OC if configured)
              └─ final topology eigensolve

run_topopt_from_json then, for every branch:
  ├─ stops RSS sampling before postprocessing
  ├─ independently assembles a common linear-mass K/M and computes modes
  │   (diagnostic result is not substituted for the branch frequency)
  └─ optionally visualizes/saves
```

Implementation references:

- Wrapper parsing and dispatch:
  `tools/Matlab/run_topopt_from_json.m:44-175,216-225,237-519`.
- Wrapper memory finalization and postprocessing:
  `tools/Matlab/run_topopt_from_json.m:539-580,1035-1077,1622-1661`.
- Olhoff branch:
  `analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m`.
- Yuksel branch:
  `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`.
- Proposed branch:
  `analysis/ourApproach/Matlab/topopt_freq.m`,
  `analysis/ourApproach/Matlab/our_mass_interpolation.m`.
- Shared MMA:
  `tools/Matlab/mmasub.m:66-97`, `tools/Matlab/subsolv.m`.
- Optional proposed refresh dependency:
  `scripts/revision_v1/a4_mode_screen.m`.

The wrapper forwards only a subset of common-looking configuration:

- Olhoff receives material, radius, tolerance, move, passives, supports, and
  Heaviside settings, but not the objective, load cases, requested optimizer,
  or sensitivity-filter semantics.
- Yuksel receives common geometry/material/support settings and a mapped filter
  code, but not load cases, objective, optimizer, or Heaviside behavior.
- OurApproach receives the optimizer, load cases, complete/omitted load
  sensitivity, and semi-harmonic settings.

## 5. Configuration-equivalence matrix

| Config item | Requested/effective value | Olhoff | Yuksel | OurApproach | Equivalent? |
|---|---|---|---|---|---|
| Mesh | `160x20`, `240x30` | Yes | Yes | Yes | Yes |
| Geometry/material | `8x1`, \(E=10^7,\rho=1\), void ratios \(10^{-6}\) | Mostly forwarded | Mostly forwarded | Forwarded | Nominally |
| Supports | Both `ux,uy` at two midpoint nodes | Forwarded | Forwarded | Forwarded | Yes, but label is ambiguous |
| Passive regions | JSON definitions | Forwarded | Forwarded | Forwarded | Nominally |
| Volume fraction | `0.5` | Projected-density equality pair | OC mean target | MMA inequality/postprojection | No |
| Objective field | `"compliance"` | Ignored; eigenvalue lower bound | Ignored; hard-coded staged compliance | Ignored as selector; hard-coded compliance | No shared semantics |
| Load cases | Mode 1 factor 1; mode 2 case factor 0 | Ignored | Ignored | Consumed; only mode 1 active | No |
| Modal baseline | `"solid"` | Not applicable | Internal stage-1 result | Solid eigensolve | No |
| Modal refresh | No update interval present | Not applicable | Internal estimate every stage-2 iteration | Reference frozen | No |
| Load sensitivity | `"complete"` | Not applicable | Omitted | Complete raw derivative | No |
| Optimizer | `"MMA"` | MMA | Ignored; OC | MMA | No |
| Move | `0.2` | MMA plus additional safeguards | OC | MMA | Same scalar, different update |
| Stopping tolerance | `0.005` | Branch-specific multi-condition | Stage 1 `0.01`; stage 2 `0.005` | `0.005` max raw-design change | No |
| Max iterations | `10000` | Single loop | Stage 2 plus stage 1 up to 200 | Single loop | No |
| Filter type | Sensitivity | Converted to density filter/projection behavior | Sensitivity heuristic | Sensitivity heuristic | No |
| Boundary condition | Symmetric | Symmetric convolution | Symmetric `imfilter` | Truncated sparse stencil | No |
| Radius | 2 elements | 2-element stencil | 2-element stencil | 2-element stencil | Same element units only |
| Heaviside | `true` | Active with continuation | Inactive for requested filter | Inert for requested filter | No |
| Mass law | JSON void ratio | Linear | Discontinuous \(x^6/x\) | Linear | No |
| Starting design | Implicit uniform | Uniform raw | Uniform stage 1; warm stage 2 | Uniform raw | Stage 2 differs |
| Lower design bound | Not explicitly shared | `0.001` | `0` | `0` | No |
| Eigen normalization | Branch-specific generalized modes | Mass-normalized for sensitivities | Euclidean stage estimate; generalized final modes | Mass-normalized reference/final modes | No |
| Frequency history | Disabled | None | None | None | Equally unavailable |
| Final image | Driver disables | None saved | None saved | None saved | Equally unavailable |

The driver's statement that `radius=2` element units “keeps the filter
consistent across all resolutions” is false in physical terms
(`performance_comparison.m:50-55`). The element width is \(8/160=0.05\) m and
\(8/240=0.0333\) m, so the nominal physical radius changes from \(0.10\) m to
\(0.0667\) m. A fixed stencil is not a mesh-independent length scale.

## 6. Convergence diagnosis of `OurApproach`

### What the printed quantities mean

- `ch.` is \(\|x^{k+1}-x^k\|_\infty\), measured on the raw design variable
  (`topopt_freq.m:786-788`).
- `mma_pre` is the volume residual before the MMA update.
- `mma_post` is the volume residual after the update/projection.
- Neither `mma_pre` nor `mma_post` is an MMA residual, KKT norm, dual residual,
  or stationarity measure (`topopt_freq.m:992-998`).

The final `240x30` changes `0.010-0.012` exceed `tol=0.005`, so the cap is
unambiguously non-converged under the implemented test [F03, CRITICAL].
`topopt_freq` returns `last_change`, but not `converged`,
`termination_reason`, KKT residual, constraint summary, or sufficient state
history (`topopt_freq.m:1043-1054`). The wrapper/driver does not infer the
status before exporting.

### What can be ruled out for this configuration

- **Repeated modal refresh:** absent. No refresh interval is configured, and
  diagnostics report zero scheduled/effective refreshes.
- **Eigenvector sign switching or mode switching in the loop:** the reference
  eigenvector is computed once and frozen.
- **Heaviside continuation:** the flag is inert for this sensitivity-filter
  branch; disabling it produced exactly the same small-mesh design.
- **Active `0.2` move cap at iterations 9997-10000:** the observed changes are
  far below `0.2`.
- **A false “converged” interpretation of `mma_pre/post`:** those values are
  only volume residuals and are not used as the stopping test.

### Most defensible diagnosis

The available evidence supports **optimizer/update stagnation or a longer
cycle on a heuristic filtered direction**, under a configuration that differs
from the paper's nominal algorithm. The raw complete semi-harmonic derivative
is accurate, but the density-weighted sensitivity filter substantially changes
it and is not the derivative of the re-evaluated objective. MMA receives this
heuristic vector, while its asymptotes evolve from iteration history
(`mmasub.m:66-97`). This is sufficient to make max-design-change convergence
unreliable, but the exact period and causal sequence cannot be recovered.

It would be scientifically improper to assert a two-cycle from four scalar
`ch` values. The original run saved neither the last designs nor MMA
asymptotes. A related `160x20` complete+MMA diagnostic capped at 400 iterations
with change about `0.0181` and plateau-like behavior, but no verified period-2
cycle (`examples/Revision_v1/cr2/mma_diagnostic/output/cr2_mma_results.json:68-111`;
`cr2_mma_diagnosis.md:83-109`). That is analogous evidence, not proof about
the missing `240x30` state.

The bounded `32x4` variants also distinguish optimizer behavior: current MMA
did not repeat with period 2, whereas the OC variant did repeat to about
`10^-12` in lag-2 norm (`performance_audit_proposed_variants.json:271-409,2014-2407`).
Therefore “MMA two-cycle” is not established. Reducing the MMA move to `0.05`
merely made the cap active and did not produce convergence
(`performance_audit_proposed_variants.json:1218-1611`).

**Causal conclusion [F02/F04/F11]:** this is not evidence that a single
semi-harmonic refresh defect caused the failure. It is a non-converged run of a
different surrogate/configuration, with an inexact filtered search direction
and insufficient telemetry to resolve the late dynamics.

Minimum required telemetry is: cap/converged reason, objective and frequency,
volume/constraint and KKT metrics, \(L_\infty\) and normalized \(L_2\) design
change, move-limit activity, MMA asymptotes/duals, modal-load norm/change,
refresh count, MAC and eigenvalue gaps when refresh is enabled, and checkpoint
or stable hashes for at least the final 100 states.

## 7. Gradient-verification results

The diagnostic used central finite differences in MATLAB R2025b on a
nondegenerate `32x4` problem at:

1. uniform density;
2. a deterministic mild perturbation;
3. a clipped/downsampled `160x20` late-state proxy.

The third design is **not** the missing `240x30` late state. That mandatory
check is impossible because the production run did not persist it [F07,
MAJOR]. The proxy only tests late-like nonuniform densities. Tested indices
cover boundary/interior and low/high-density elements. Full arrays and
pointwise values are in
`performance_audit_gradient_results.json`; the runner is
`performance_audit_gradient_checks.m`.

Relative errors below are stabilized central-FD comparisons. “Raw” means
before the production sensitivity heuristic. “Full” re-evaluates every
design-dependent term; “frozen” holds the current load fixed.

| Branch/check | Uniform | Perturbed | Late proxy | Assessment |
|---|---:|---:|---:|---|
| Our raw complete vs full objective, relative \(L_2\) | `5.18e-6` | `7.66e-6` | `7.12e-6` | Pass |
| Our raw complete vs full, median pointwise | `4.68e-6` | `1.00e-5` | `1.77e-5` | Pass |
| Our production filtered complete vs full, relative \(L_2\) | `0.546` | `0.469` | `1.220` | Fail as exact gradient |
| Our production filtered complete vs full, median pointwise | `0.494` | `0.551` | `1.177` | Fail as exact gradient |
| Our volume gradient vs FD, relative \(L_2\) | `2.52e-9` | order `1e-9` | order `1e-9` | Pass |
| Yuksel raw omitted vs frozen-load objective, relative \(L_2\) | `3.57e-6` | order `1e-6` | order `1e-6` | Pass only for frozen load |
| Yuksel raw omitted vs full objective, relative \(L_2\) | `0.510` | `0.454` | `0.492` | Fail for full objective |
| Yuksel independently completed derivative vs full objective | order `1e-6` | order `1e-6` | order `1e-6` | Confirms missing term |
| Olhoff eigen/filter/projection design chain, relative \(L_2\) | `2.98e-6` | order `1e-6` | order `1e-6` | Pass |
| Olhoff volume chain, relative \(L_2\) | `8.86e-10` | order `1e-9` | order `1e-9` | Pass |
| Olhoff eigen-constraint \(E_b\) production derivative relative error | `0.827` | `0.898` | `0.870` | Fail |
| Olhoff corrected \(E_b\) derivative relative error | order `1e-11` | order `1e-10` | order `1e-10` | Pass |

Interpretation:

- OurApproach correctly differentiates its evolving linear-mass load when
  `complete` is selected. The earlier tiny V1 fixture independently reports a
  maximum raw complete-load FD error of `3.79e-8`
  (`scripts/revision_v1/v1_4_sensitivity_results.json:53`).
- The production sensitivity-filter transform is a heuristic search direction,
  not an exact derivative through a physical-density map. It therefore cannot
  be used as evidence of MMA stationarity for the actual compliance.
- Yuksel's implemented derivative is internally correct only if the load is
  frozen during differentiation. Re-evaluating \(M(x)\widehat u\) requires the
  missing \(2u^T(\partial F/\partial x)\) contribution.
- Olhoff's stiffness/mass eigenvalue derivative and filter/projection chain
  pass, but the auxiliary quotient derivative is wrong when \(E_b>1\). For
  \(q=\lambda_j/\lambda_{\rm ref}\) and \(E_b>1\),
  \(g=1-q/E_b\), hence \(\partial g/\partial E_b=q/E_b^2\), not \(1/E_b\).
- Linear mass and SIMP stiffness elemental derivatives used in the Our/Olhoff
  checks match finite differences. Yuksel's two pieces differentiate
  individually, but the interpolation itself is discontinuous at the cutoff.

These are bounded verification tests, not certification of all topologies,
mode multiplicities, or optimizer trajectories.

## 8. Semi-harmonic-load analysis

Only the first configured load case is active. The case-2 factor is zero and is
applied to its force, so its mode-2 load contributes zero RHS and zero
objective (`performance_comparison.json:25-40`;
`topopt_freq.m:1408-1437`).

For OurApproach:

- A solid-domain mode is computed before the loop
  (`topopt_freq.m:315-355`).
- With no refresh interval in this JSON, the reference \((\omega_0,\Phi_0)\)
  remains frozen (`topopt_freq.m:358-405,1145-1215`).
- The force is nevertheless design-dependent because
  \(F(x)=\omega_0^2M(x)\Phi_0\), using the current linear mass matrix.
- `semi_harmonic_rho_source="x"` is not an effective selector in this path;
  the wrapper treats the legacy key as obsolete and the local parser is not
  used to change the active density source
  (`run_topopt_from_json.m:131-160`; `topopt_freq.m:2214-2230`).
- `harmonic_normalize=true` affects harmonic loads, not this semi-harmonic
  construction.
- `load_sensitivity="complete"` includes the mass/load derivative in the raw
  gradient. Controlled FD checks confirm it for the frozen reference.

For Yuksel, the JSON semi-harmonic cases are ignored. The branch constructs its
own \(F=M(x)\widehat u\), refreshes the displacement estimate on every stage-2
iteration, Euclidean-normalizes it, and omits the design derivative of that
load. For Olhoff, the JSON loads are ignored entirely.

Thus there is no “consistent refresh across methods.” OurApproach minimizes
compliance under a frozen-reference but evolving-mass modal surrogate; Yuksel
minimizes a different sequential compliance under a changing internal load;
Olhoff solves an eigenvalue problem. OurApproach's reported final frequency is
an ex-post metric and is not algebraically its optimized objective.

The current no-refresh setting rules out load-refresh periodicity as the direct
cause of the `240x30` late changes. Enabling refresh is a different algorithmic
branch and exposed the separate dependency/B3 issue [F17, MODERATE].

## 9. Final-frequency validation

The mandatory per-design gate could not be completed because the final table
designs do not exist in `examples/Performance`: there are no `.mat` states,
topology arrays/images, modes, or final manifests. The CSV does not contain
frequency. Therefore it is impossible to:

- independently reassemble the six actual table designs;
- verify their first several eigenpairs and residuals;
- check rigid modes and ordering;
- inspect global versus localized modes and support connectivity;
- recompute volume using the density entering K/M;
- measure grayness or minimum features; or
- reproduce the user's `124.1909 rad/s` from a persisted state.

The wrapper's generic postprocessor does assemble a common SIMP-stiffness,
linear-mass system and requests three modes after the method completes
(`run_topopt_from_json.m:553-580,1035-1077`). That calculation is outside the
reported timing/memory scope and, critically, is discarded rather than
substituted for the returned branch frequency.

The bounded `32x4` smoke test checked this machinery:

- Olhoff branch vs common recomputation: relative difference `2.47e-12`.
- OurApproach vs common recomputation: `4.74e-13`.
- Yuksel vs common recomputation: `1.49e-2`, attributable to the different
  mass interpolation.

These values validate only the diagnostic states, not the unavailable table
topologies. They prove that the wrapper can report branch frequencies under
different mass models [F05/F12, MAJOR].

The exact BC must also be stated: both support nodes have both translational
DOFs fixed. No rigid modes were seen in the small diagnostic, but that does not
replace validation of the six missing final states.

**Gate result: FAIL. No reported table frequency is accepted as independently
validated.**

## 10. Timing validity

`tIter` is branch loop time divided by the branch iteration count, not an
end-to-end average:

- Olhoff starts its timer immediately before the outer loop and stops it after
  the loop (`topFreqOptimization_MMA.m:221-224,550-553`). It includes current
  and trial eigensolves but excludes setup and final best-topology eigensolves.
- Yuksel sums the separately measured stage-1 and stage-2 loop times and divides
  by their combined iterations (`top99neo_inertial_freq.m:244-251,408-488,512-628`).
  Stage initialization and final eigensolves are excluded.
- OurApproach measures only its optimization loop
  (`topopt_freq.m:456-468,1004-1006`). It excludes the solid reference
  eigensolve, setup, and final eigensolve.
- The wrapper's common independent eigensolve happens after these timers and
  after RSS sampling.

Consequently `tIter*nIter` merely reconstructs each branch's own loop timer. It
is not the direct wall-clock runtime requested by the table label. The
diagnostic direct wrapper timings listed in F08 demonstrate excluded overhead,
but were cold, fixed-order, single runs and must not be promoted as replacement
performance results.

Further validity failures:

- `nSamples=1` (`performance_comparison.m:93`), so variance is unknowable.
- Execution is always Olhoff, Yuksel, OurApproach for each mesh
  (`performance_comparison.m:89-93,111-154`), so the first branch bears JIT and
  allocation costs while later branches inherit caches.
- No warm-up protocol, randomized order, isolated process, hardware/software
  manifest, garbage-collection observation, or repeated-trial statistics exist.
- The timed work differs by branch, particularly eigensolves and warm starts.

**Finding [F08, MAJOR]:** every runtime, per-iteration time, speedup, runtime
ratio, and scaling inference from this driver is invalid. The current bounded
diagnostics intentionally do not quantify publication-grade run-to-run
variance because the experiment failed the earlier equivalence gate.

## 11. Iteration-count validity

Approximate work per reported outer iteration is:

| Operation | Olhoff | Yuksel stage 1 | Yuksel stage 2 | OurApproach, current config |
|---|---:|---:|---:|---:|
| K assembly | 2 (current + trial) | 1 | 1 | 1 |
| M assembly | 2 | only if history/final diagnostics require it | 1 | 1 |
| Linear solve | 0 separate | 1 | 1 | 1 multi-RHS solve |
| Generalized eigensolve | 2 | 0 in loop | 0 in loop | 0 in loop |
| Requested eigenpairs | 3 + 3 | 0 | 0 | 0 |
| Filter passes | forward/back chain(s) | sensitivity pass | sensitivity pass | sensitivity pass |
| Optimizer update | 1 MMA | 1 OC | 1 OC | 1 MMA |
| Load refresh | N/A | fixed point load | internal estimate every iteration | none for reference; mass-weighted force rebuilt |
| Trial analysis | yes | no | no | no |
| Continuation | beta/projection/gray logic | stage counters available | stage counters available | none effective here |

The Yuksel wrapper caps stage 1 at `min(max_iters,200)` and then permits stage 2
up to `max_iters` (`run_topopt_from_json.m:362-373`). A diagnostic with
`max_iters=12` returned 24 iterations, exactly `12+12`
(`performance_audit_smoke_results.json:51-85`). Therefore even “maximum
iterations” has different semantics.

Raw iteration count is not a scientific efficiency metric [F10, MAJOR].
Operation counts, factorization/eigensolver statistics, and end-to-end wall
time are required if computational efficiency is studied.

## 12. Memory-measurement validity

The wrapper starts a MATLAB timer that polls the current MATLAB process RSS
through `ps -o rss` every `0.1 s`. It subtracts an initial baseline and reports
the largest sampled increment stored in appdata
(`run_topopt_from_json.m:216-225,1622-1661`). Sampling ends before common
postprocessing (`run_topopt_from_json.m:539-545`).

This is:

- incremental sampled in-process RSS, not process peak RSS;
- vulnerable to missing sub-100-ms peaks;
- dependent on memory already retained by the shared MATLAB process;
- biased by fixed execution order;
- inconsistent with a “Max RAM” label; and
- scoped differently from an end-to-end run.

The current CSV pattern is consistent with the bias: the first Olhoff run
reports `173 MB`/`301 MB`, while later methods report `14-40 MB`
(`table1_performance.csv:2-7`). In the bounded cold-order diagnostic the first
method similarly reported `164.4 MB`, followed by `10.5 MB` and `19.25 MB`
(`performance_audit_smoke_results.json:15-20,51-56,88-93`). This is not an
independent validation of true memory consumption; it demonstrates the
baseline/order problem.

**Finding [F09, MAJOR]:** “Max RAM” must be removed. A future claim needs
isolated processes and an OS-level peak metric over identical end-to-end scope.

## 13. Comparator fidelity and fairness

The authoritative project decision already says the evidence is retired:

- `examples/Revision_v1/SCIENTIFIC_DECISION_EXP1_EXP5.md:1-20` assigns obsolete
  status to the cross-code performance/scaling evidence.
- Lines `26-61` explain that implementation-specific differences make the
  construct invalid.
- Lines `63-73` select A4 instead of rehabilitating EXP1/EXP5.
- `examples/Revision_v1/README.md:15-17` repeats the retirement.
- `paper/reviews/revision_plan.tex:97-121,686-698` supersedes performance-table
  work and says not to use the legacy driver.

The code confirms the documented fidelity concerns:

- `topFreqOptimization_MMA.m:3-13` calls itself a local inspired
  implementation, not a faithful reproduction. It adds three-mode max-min
  constraints, Heaviside projection, continuous continuation, grayness
  penalty, adaptive safeguards, and a post-MMA trial eigensolve.
- `top99neo_inertial_freq.m:1-30` is a local two-stage implementation. It has
  its own discontinuous mass model, warm start, fixed-point update, OC rule,
  and stopping semantics. It does not reproduce a shared/published iteration
  unit.
- The JSON does not neutralize these differences; many common-looking fields
  are ignored branch-wise.

Scientifically defensible uses are limited to:

- internal engineering observations about these exact code paths;
- profiling a single implementation after proper instrumentation;
- ablations within a common implementation; or
- a future shared-framework experiment with accurate provenance.

It is not defensible to call the current artifact a comparison of the
published Olhoff, Yuksel, and proposed methods. The unsuccessful exact
Olhoff reconstruction remains independent of this conclusion.

## 14. Manuscript/Table 1 consistency

### Current governing state

The premise that this script is currently used for an active revised-paper
Table 1 conflicts with the checked repository:

- `paper/main.tex` contains no quantitative cross-code Table 1.
- It explicitly says the cross-code headline/table was removed
  (`paper/main.tex:374-379`).
- It rejects cross-code timing/efficiency interpretation
  (`paper/main.tex:631-644`) and limits the conclusions accordingly
  (`paper/main.tex:683-692`).
- `paper/reviews/algorithms_comparison.tex:315-368` provides qualitative
  algorithm comparison, not this numerical benchmark.
- No response/rebuttal letter or supplement file containing
  `table1_performance.csv` was found by repository text/file search.

Thus **D. REMOVE** preserves the active paper's existing decision. The legacy
driver and CSV remain a provenance hazard, not current accepted paper evidence.

### Affected or inconsistent claims that still require separate reconciliation

| File/lines | Claim/problem | Severity and consequence |
|---|---|---|
| `paper/main.tex:98` | Abstract describes a fixed inertial load and no additional optimizer; the performance config uses evolving \(M(x)\Phi_0\), complete derivative, and MMA. | **MAJOR:** cannot describe the table's proposed row. |
| `paper/main.tex:129-131` | Says uniform reference and omitted load derivative; config says solid and complete. | **CRITICAL (F04):** different proposed formulation. |
| `paper/main.tex:134-140` | Claims four meshes and competitive frequencies; this driver uses two meshes and one proposed endpoint is capped. | **MAJOR:** unsupported breadth/convergence. |
| `paper/main.tex:153` | Attributes projection/gray/trial mechanisms in a way that risks conflating the local extension with a published method. | **MAJOR:** comparator provenance. |
| `paper/main.tex:198-200` | Says uniform reference and that load-design coupling is eliminated, although \(F=\omega_0^2M(x)\Phi_0\) remains design-dependent. | **MAJOR:** mathematical misdescription. |
| `paper/main.tex:208` | Gives lower density `0.001`; Our/Yuksel design variables can reach zero. | **MODERATE:** branch-specific mismatch. |
| `paper/main.tex:239-240` | Describes a low-density mass branch for the local Olhoff path; current Olhoff mass is linear, while Yuksel owns the discontinuous branch. | **MAJOR:** mass-model attribution. |
| `paper/main.tex:249-267` | Repeats uniform/omitted/external-load formulation; performance uses solid/complete/design-dependent load. | **CRITICAL (F04):** proposed row is not this formulation. |
| `paper/main.tex:272` | Refers to initial solid domain, conflicting with nearby uniform-reference wording. | **MODERATE:** internal manuscript inconsistency. |
| `paper/main.tex:280-290` | Describes sensitivity filtering and OC; performance OurApproach uses MMA. | **MAJOR:** optimizer mismatch. |
| `paper/main.tex:298-316` | Describes uniform/OC and tolerance `1e-3`; performance uses solid/MMA and `5e-3`. | **MAJOR:** stopping and formulation mismatch. |
| `paper/main.tex:333-350` | Describes void ratios `1e-9`, common linear mass/filter settings, mesh-independent radius, and tolerance `1e-3`; the driver uses `1e-6`, Yuksel mass differs, filter operators differ, physical radius changes, and tolerances differ. | **MAJOR:** fairness/reproducibility claims fail. |
| `paper/main.tex:353-372` | Reports endpoint frequencies around `174.3/160.5/159.3 rad/s`; governing decision says endpoint artifacts remain outstanding (`SCIENTIFIC_DECISION_EXP1_EXP5.md:135-141`). | **MAJOR:** unsupported numerical provenance. |
| `paper/main.tex:381-416` | Discusses frequency histories not produced by the performance JSON, which disables iteration-frequency saving. | **MODERATE:** cannot be sourced to this run. |
| `paper/main.tex:590-601` | Uses fairness language around local comparisons. | **MODERATE:** must remain explicitly local/qualitative. |
| `paper/main.tex:649-651` | Again states uniform/OC/omitted settings. | **MAJOR:** not the performance configuration. |
| `paper/main.tex:665` | Says evaluation occurs after convergence; the proposed `240x30` endpoint did not converge. | **MAJOR:** mandatory gate failure if applied here. |
| `paper/main.tex:673-675,679-681,704-706` | Does not clearly distinguish MMA performance runs from OC formulation text. | **MODERATE:** optimizer provenance ambiguity. |
| `paper/main.tex:419` | A separate clamped example explicitly uses MMA while the general methodology says OC. | **MODERATE:** broader manuscript consistency issue, not evidence for Table 1. |
| `docs/olhoff_implementation_analysis.tex:127-161` | Simply supported comparison retains stale void/mass/filter/tolerance and endpoint statements, although its header marks provenance-only status (`:1-7`) and its local-deviation discussion is accurate (`:21-77`). | **MODERATE:** stale supplement/provenance hazard. |

`paper/results_manifest.md:18-30` says only final validated results should be
used and stale outputs are forbidden, but it does not provide accepted
provenance for this legacy table. Historical review-plan text is subordinate to
the explicit superseding statements cited above.

No manuscript files were modified under the audit-first policy.

## 15. Controlled diagnostic experiments

All diagnostics used a cheap `32x4` mesh and preserved production code. They
are causal probes, not benchmark evidence.

### A. Wrapper smoke runs

`performance_audit_smoke_runs.m` invoked all actual wrapper branches with a
12-iteration cap. Results are in `performance_audit_smoke_results.json`.

| Branch | Returned iterations | Reconstructed loop time | Direct wrapper wall time | Last change/status |
|---|---:|---:|---:|---|
| Olhoff | 12 | `0.427 s` | `3.158 s` | No `converged`/reason |
| Yuksel | 24 (`12+12`) | `0.0705 s` | `0.297 s` | S1 `0.0417`, S2 `0.00892`; no status |
| OurApproach | 12 | `0.3046 s` | `0.563 s` | `0.0246`; no reason |

This confirms timer-scope and cap-semantics defects without claiming stable
performance ratios.

### B. Proposed-branch variants

`performance_audit_proposed_variants.m` ran a maximum of 80 iterations with
strict `1e-6` change tolerance. Results are in
`performance_audit_proposed_variants.json`.

| Requested diagnostic | Outcome | Interpretation |
|---|---|---|
| 1. Current configuration | Capped at 80; late lag-2 \(L_\infty\) `0.096-0.226` | Non-converged; not a verified period-2 cycle |
| 2. Truly fixed semi-harmonic RHS | Not selectable without changing production formulation | Current “frozen reference” is not fixed RHS because \(M(x)\Phi_0\) changes |
| 3. Refresh every iteration | Clean run first failed on missing `a4_mode_screen` path; after diagnostic path addition, classified B3 failure at iteration 16 | Separate refresh dependency/connectivity defect; not current-run cause |
| 4. Refresh disabled after initialization | Exactly identical to current (`max abs x difference = 0`) | Confirms current reference is frozen |
| 5. Projection disabled | Exactly identical to current | Confirms Heaviside key is inert here |
| 6. Reduced move `0.05` | Capped with final change at move limit `0.05` | Parameter tuning did not correct mechanism |
| Load derivative omitted | Capped; different trajectory | Complete/omitted choice matters but neither small run converged |
| OC instead of MMA | Capped; exact lag-2 repetition (`~1e-12` to `9e-10`) | A genuine small-mesh OC two-cycle, not evidence of an MMA two-cycle |
| 7. Alternative metrics | Computed diagnostically where states existed; no production KKT/MMA residual was exposed | Change alone is insufficient |
| 8. Restart late `240x30` with frozen history | Impossible: final `x` and MMA history were not saved; no restart interface | Required production artifact absent |
| 9. Restart with reset asymptotes | Impossible for same reason | Required production artifact absent |
| 10. Compare alleged alternating `240x30` states | Impossible: state vectors/hashes were not saved | Four scalar log lines cannot establish a cycle |

All successful variants reached the deliberately short cap. None is presented
as a corrected method. The refresh runner added only a diagnostic MATLAB path;
production wrapper behavior was not altered.

### C. Gradient checks

The three-design results in Section 7 cover raw/full/frozen objectives,
volume, stiffness, mass, filtering/projection, eigenvalue sensitivity, and the
Olhoff auxiliary constraint. The unavailable exact `240x30` state is explicitly
recorded as an acceptance failure, not replaced silently by the proxy.

### D. Full-run evidence not collected

Per the code-change policy and because equivalence already failed, no repeated
randomized timing campaign, isolated peak-memory campaign, or six-case
frequency campaign was run. Such campaigns would spend resources quantifying
an invalid construct.

## 16. Required corrections

These are prerequisites for any future experiment; they are not a prescription
to restore the current table.

1. **Containment [F01/F19, CRITICAL/MINOR].** Keep the table absent, rename and
   quarantine the legacy script/CSV, and make publication export impossible
   without an accepted manifest. Existing table uses remain invalid.
2. **Define the scientific comparison [F02/F04, CRITICAL].** Choose either a
   common direct eigenfrequency problem or an explicitly labeled surrogate
   ablation. Freeze objective, load, K/M, filter, BC, volume, initialization,
   optimizer, and stopping semantics. Existing method-level comparisons remain
   invalid.
3. **Correct implementation defects [F05/F11, MAJOR].** Remove the Yuksel mass
   discontinuity if a common mass model is intended; correct the Olhoff
   \(E_b\) quotient derivative; decide whether load dependence is frozen or
   complete and test that exact objective. Existing affected endpoints require
   recomputation in any redesigned study.
4. **Use exact or honestly labeled regularization [F06/F11, MAJOR].** A
   sensitivity heuristic must not be called an exact MMA gradient. Harmonize
   filter boundary handling, projection, continuation, and physical radius.
   Existing mesh and stationarity claims remain invalid.
5. **Make configuration effective and auditable [F14, MODERATE].** Emit a
   branch-specific requested/effective manifest and fail on ignored scientific
   fields. Existing claims of one shared JSON configuration remain invalid.
6. **Instrument termination [F03/F07, CRITICAL/MAJOR].** Persist
   `converged`, `termination_reason`, cap status, constraints, KKT/update
   residuals, final changes, histories, and checkpoints. Any cap hit must fail
   export. Proposed `240x30` remains invalid.
7. **Persist and validate endpoints [F07/F12/F15, MAJOR/MODERATE].** Save
   designs and independently recompute several modes under one common K/M/BC,
   with residual, ordering, localization, connectivity, volume, grayness, and
   feature checks. All current frequency rows remain unaccepted.
8. **Replace computational metrics [F08-F10, MAJOR].** Use direct end-to-end
   wall time, warm-up, randomized repeated isolated runs, variance, external
   peak RSS, and operation counts. All current time/memory/iteration efficiency
   fields remain invalid.
9. **Fix refresh packaging before studying it [F17, MODERATE].** Make the
   optional refresh dependency self-contained and retain explicit B3 failure
   classification. This does not rehabilitate the current table.
10. **Reconcile the paper separately [F16, MAJOR].** Preserve removal of the
    table and audit the listed formulation/settings/endpoints against accepted
    result manifests. No manuscript change was made here.

## 17. Scientific decision for Table 1

### D. REMOVE

This is the only category supported by the evidence.

- **A. RETAIN** fails because the problems, implementations, and metrics are
  not equivalent and one headline result is non-converged.
- **B. RECOMPUTE** fails because rerunning the same construct would still
  compare different problems and non-faithful codes.
- **C. REDESIGN** accurately describes what a *future new comparison* would
  require, but it is not the disposition of this Table 1. The present artifact
  cannot be repaired into method-level evidence.
- **D. REMOVE** matches the code header, the governing EXP1/EXP5 decision, and
  the active manuscript, which already contains no quantitative cross-code
  table.

Allowed residual use: clearly marked internal benchmarking of these exact local
implementations, with no published-method attribution and no reuse of the
current time/memory/frequency fields as validated evidence.

## 18. Residual risks

- **MAJOR:** The unavailable final `240x30` designs and MMA state prevent a
  definitive period/cycle diagnosis. The causal conclusion is bounded to what
  can be ruled in or out from code and cheap diagnostics.
- **MAJOR:** Finite differences were performed on `32x4` designs, including a
  proxy rather than the missing failing state. Mode multiplicity or
  ill-conditioning at the original mesh could introduce additional defects.
- **MAJOR:** The current table's six final modes, connectivity, volume,
  grayness, and minimum features remain unverified because artifacts were not
  saved.
- **MODERATE:** Optional modal refresh has a path dependency and can encounter
  B3 connectivity failure. It was not active in the audited table run.
- **MODERATE:** Platform-specific RSS polling was inspected and smoke-tested on
  the present macOS/MATLAB environment only; this is already sufficient to
  reject “Max RAM,” not to characterize other platforms.
- **MODERATE:** Manuscript searches found no active Table 1 or response-letter
  use, but generated/untracked documents outside the repository cannot be
  excluded.
- **MINOR:** Diagnostic wall times are subject to cold-start/JIT noise and are
  intentionally not publication data.

None of these residual risks weakens the removal decision. Additional evidence
could reveal more defects; it cannot make the current three problems
mathematically equivalent.

## 19. Exact files inspected and changed

### Inspected

Primary experiment and current output:

- `examples/Performance/performance_comparison.m`
- `examples/Performance/performance_comparison.json`
- `examples/Performance/table1_performance.csv`

Wrapper and shared numerical infrastructure:

- `tools/Matlab/run_topopt_from_json.m`
- `tools/Matlab/mmasub.m`
- `tools/Matlab/subsolv.m`
- `tools/Matlab/supportsToFixedDofs.m`
- `tools/Matlab/validateLoadCases.m`
- `tools/Matlab/parsePassiveRegions.m`

Three implementation branches:

- `analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m`
- `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`
- `analysis/ourApproach/Matlab/topopt_freq.m`
- `analysis/ourApproach/Matlab/our_mass_interpolation.m`
- `analysis/ourApproach/Matlab/mass_normalize_modes.m`
- `analysis/ourApproach/Matlab/orient_modes_deterministic.m`

Revision decisions, related diagnostics, and refresh support:

- `examples/Revision_v1/SCIENTIFIC_DECISION_EXP1_EXP5.md`
- `examples/Revision_v1/README.md`
- `examples/Revision_v1/REVISION_CURRENT_STATE_REPORT.md`
- `examples/Revision_v1/revision_implementation_audit.md`
- `examples/Revision_v1/A4_SPECIFICATION_V3.md`
- `examples/Revision_v1/cr2/mma_diagnostic/output/cr2_mma_results.json`
- `examples/Revision_v1/cr2/mma_diagnostic/output/cr2_mma_diagnosis.md`
- `scripts/revision_v1/a4_mode_screen.m`
- `scripts/revision_v1/v1_4_sensitivity_results.json`
- `scripts/revision_v1/authoritative_formulation_audit.md`

Manuscript/provenance files:

- `paper/main.tex`
- `paper/results_manifest.md`
- `paper/reviews/revision_plan.tex`
- `paper/reviews/algorithms_comparison.tex`
- `paper/reviews/REVISION_AUDIT.md`
- `paper/reviews/final_review_V1.tex`
- `paper/reviews/final_review_V2.tex`
- `docs/olhoff_implementation_analysis.tex`
- `docs/olhoff_audit.md`

Repository-wide text/file searches also covered `paper/`, `docs/`,
`examples/`, `analysis/`, `scripts/`, and `tools/` for “Table 1,”
“performance comparison,” runtime, speedup, memory, scaling, Olhoff, Yuksel,
ProposedApproach, and `table1_performance.csv`. No active response/rebuttal
letter containing this artifact was found.

### Added diagnostic/audit files

- `examples/Performance/performance_audit_gradient_checks.m`
- `examples/Performance/performance_audit_gradient_results.json`
- `examples/Performance/performance_audit_smoke_runs.m`
- `examples/Performance/performance_audit_smoke_results.json`
- `examples/Performance/performance_audit_proposed_variants.m`
- `examples/Performance/performance_audit_proposed_variants.json`
- `examples/Performance/PERFORMANCE_COMPARISON_AUDIT.md`
- `examples/Performance/PERFORMANCE_COMPARISON_REMEDIATION_PLAN.md`

### Production/manuscript files changed

None. In particular, the three optimizer implementations, wrapper, primary
driver/configuration, CSV, and manuscript were not edited. Pre-existing
worktree changes under
`analysis/OlhoffApproachExact/experiments/faithful_reconstruction/` were not
part of this audit and were left untouched.
