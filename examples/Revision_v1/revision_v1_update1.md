# Revision_v1 Update 1: Completion Plan

## Objective

Update the revision implementation so that every reviewer demand is either:

1. supported by completed, converged, traceable evidence; or
2. resolved by an explicit correction, qualification, or retraction in the
   manuscript and response letter.

Generated evidence must determine the final claims. A failed, capped,
mode-invalid, or unconverged run must not be reported as confirming a claim.

## Mandatory Scientific Decision: Authoritative Formulation

The revised implementation shall retain the design-dependent inertial load
documented in the manuscript:

\[
\mathbf f_j(\mathbf x)=(\omega_j^{(0)})^2
\mathbf M(\mathbf x)\boldsymbol\Phi_j^{(0)}.
\]

The reference frequency \(\omega_j^{(0)}\) and mode
\(\boldsymbol\Phi_j^{(0)}\) are computed once on one declared reference
design and then frozen. The mass matrix \(\mathbf M(\mathbf x)\) continues to
evolve with the design. Consequently,
\(\partial\mathbf f_j/\partial x_e\ne 0\).

The proposed approximation omits this nonzero load-sensitivity contribution;
it must not describe the contribution as mathematically zero. Its practical
effect must be evaluated by the CR2 experiment.

Implementation actions:

- choose one reference design, recommended as the fully solid domain, and use
  it in every MATLAB/Python path and JSON configuration;
- replace the current `omega0` factor with `omega0^2`;
- remove the additional `rho_nodal(x)` scaling because design dependence is
  already represented by `M(x)`;
- use one documented mode normalization and mass-weighted MAC definition;
- make the omitted/full sensitivity selection an explicit configuration
  option;
- update equations, pseudocode, comments, examples, and response text to match.

**Gate A0:** MATLAB and Python must produce matching reference eigenpairs,
loads, objectives, and analytical sensitivities for a small deterministic test.
Require relative disagreement `<= 1e-8` for reference eigenfrequencies,
load-vector entries, objectives, and sensitivities, with an absolute `1e-12`
fallback for quantities numerically equal to zero. Mass-normalize eigenvectors
and apply a consistent phase/sign convention before comparison; compare modal
subspaces rather than individual vectors when eigenvalues are repeated or
numerically clustered.
No production experiment may run before this gate passes.

## Workstream 1: Fail-Loud Experiment Infrastructure

Extend every solver result with:

- success flag and termination reason;
- iteration count and cap;
- final design change and constraint residual;
- objective, frequency, feasibility, and grayness histories;
- tracked-mode index and MAC history where applicable;
- measured initialization, optimization-loop, postprocessing, and total
  wall-clock times;
- configuration hash, source commit, random seed, and software/hardware data.

Update `run_all_revision_experiments.m` so that it returns failure if any
mandatory experiment:

- throws an exception;
- returns an empty result or required `NaN`;
- reaches an iteration cap without satisfying the declared convergence rule;
- loses the target mode under the declared MAC criterion;
- omits a required MAT, CSV, image, diary, or manifest artifact.

Each experiment must write to its own output directory. Preserve complete
stack traces and prevent correlation files or mode plots from being silently
overwritten.

**Gate I1:** an intentionally failing smoke experiment must make the master run
fail and identify the exact failed acceptance condition.

## Workstream 2: Solver Verification

Before long runs, add small-mesh regression checks for:

- reference eigenpairs and normalization;
- construction of \((\omega_0)^2M(x)\Phi_0\);
- weighted multi-mode objective assembly;
- omitted and complete load-sensitivity variants;
- central finite-difference sensitivity checks;
- mass-weighted MAC and mode reordering;
- MATLAB/Python parity.

For the complete-gradient variant, require a declared finite-difference
tolerance, recommended relative error \(\le 10^{-5}\). The omitted-gradient
variant is not expected to match the complete finite-difference derivative;
its effect is evaluated through converged optimization outcomes.

**Gate V1:** all regression checks pass before any active stage (S1, EXP2, EXP2b,
EXP3, A4) is regenerated.

## Workstream 3: Recover the Clamped-Beam Experiments

### Exp2: full clamped-beam analysis

1. Reproduce and capture the current failure with its full stack trace.
2. Fix the failure and run
   \(\alpha=\{1,0.75,0.5,0.25,0\}\) with the authoritative formulation.
3. Require convergence under a predeclared design, feasibility, and mode-validity
   rule; capped runs remain failures.
4. Export initial frequencies, complete spectra below tracked modes, MAC
   matrices, selected mode indices, convergence histories, grayness,
   topologies, and requested mode shapes.
5. Diagnose the \(\alpha=0.75\) result without assuming monotonicity.
6. For every case, perform the A5 a-posteriori check: state whether the tracked
   mode is the structure's lowest mode. Explain that the method targets a
   reference mode shape and does not necessarily maximize the fundamental
   frequency.
7. If the already planned external-method comparison is retained, use the
   local `analysis/OlhoffApproach` implementation and label it "local
   Olhoff-inspired implementation." Prefer the \(\alpha=1\) single-mode case
   unless an equivalent two-mode objective already exists for that comparator.
   Do not use `OlhoffApproachExact` and do not infer agreement with the
   published Du--Olhoff optimum.

### Exp3: mesh convergence

Run the same authoritative formulation at 200x25 and 400x50 with identical
physical geometry, interpolation, filtering, stopping rules, and mode-tracking
logic. Report frequency, gain, MAC, selected mode, grayness, constraint
residual, and topology differences. Declare the numerical convergence criterion
before inspecting results; if it is not met, report the lack of mesh convergence.

**Gate E23:** Exp2 and Exp3 produce complete artifacts with no capped runs.
Any gain below the MAC threshold is marked invalid rather than presented as a
verified modal improvement.

## Workstream 4: Separate CR2 and A4 Studies

These studies answer different questions and must not be conflated.

### CR2: omitted load sensitivity

Using the authoritative evolving-mass load, compare:

- Variant A: \(M(x)\) load with the load-sensitivity term omitted;
- Variant B: the same load with the complete
  \((\omega_0)^2(\partial M/\partial x_e)\Phi_0\) contribution.

Verify Variant B with central finite differences. Run both variants to
convergence from identical initial conditions. Compare topology, objective,
tracked frequencies, mode validity, grayness, feasibility, iteration history,
and sensitivity differences. Conclude that omission is negligible only if the
converged evidence supports that statement.

### A4: eigenpair refresh approximation

On the specified simply supported 400x50 case, run

\[
N=\{1,5,10,50,\infty\},
\]

where \(N=\infty\) is the permanently frozen reference. Use MAC plus frequency
continuity to track the same physical mode after refresh. Save refresh events,
mode indices, MAC, frequencies, objective, feasibility, and convergence
histories.

Treat \(N=1\) as a fully refreshed reference variant, not automatically as the
canonical Yuksel--Yilmaz method. Claim equivalence only if the complete
algorithms are shown to match. If \(N=1\) or another variant oscillates or does
not converge, report that result and do not use its endpoint as an accuracy
reference.

**Gate E4:** CR2 comparisons are converged and use the same load formulation;
A4 comparisons are converged and track the same physical mode. Otherwise the
corresponding reliability claim must be qualified or retracted.

## Workstream 5: Rebuild Performance Evidence — RETIRED

**Retired 2026-07-13 by `SCIENTIFIC_DECISION_EXP1_EXP5.md`.** EXP1 and EXP5 are
removed from the reviewer evidence chain, so there is no performance evidence to
rebuild. This workstream is **void in its entirety**: no solver instrumentation,
no benchmark mode, no warm-up/repeat timing campaign, no timing or memory
standard deviations, no Yuksel iteration-discrepancy investigation, and no
scaling fit. **Gate P1 no longer exists.**

The defect is construct validity, not instrumentation. The local comparators are
not faithful reference implementations: `OlhoffApproach` performs a trial
eigensolve after every MMA update — roughly doubling its eigensolves per outer
iteration — and adds a Heaviside projection, a continuation schedule, and a
grayness penalty absent from the published method, while the local
Yuksel-inspired code does not reproduce the published iteration counts. A timing,
memory, or scaling comparison against either measures implementation choices, not
methods, at any level of instrumentation. Comparator telemetry is therefore
**deliberately not implemented**.

The affected reviewer demands (runtime decomposition, timing/memory standard
deviations, source and bound of the speedup, ω₁ per method and mesh, the Yuksel
180–200 iteration discrepancy, and the O(nₑ^1.3) scaling correction) are
**resolved by retraction**, which the Definition of Complete explicitly permits.
The manuscript makes no cross-code performance claim; the efficiency argument
survives only as an operation count (one eigensolve at initialization versus one
per design iteration).

`exp1_perf_table.m` and `exp5_scaling.m` are preserved unmodified under
`archive/obsolete_evidence/exp1_exp5/` and are denied by name in the runner's
preflight P2. The frozen-eigenpair accuracy and cost question that EXP1 proxied
is answered within a single implementation by **A4**.

## Workstream 6: Low-Mode and Grayness Diagnosis

Rerun the capped building cases to convergence. For every suspicious low-MAC
mode, export:

- mode shape;
- elementwise kinetic and strain energy;
- fraction of energy in low-density regions;
- localization metric;
- density support and selected-mode MAC.

Classify modes from localization and energy evidence, not MAC alone. Report grayness for
every final topology and explain the gray regions in the affected figures.

**Mass-interpolation axis: exhausted (2026-07-14, see
[`MASS_INTERPOLATION_DECISION.md`](../../MASS_INTERPOLATION_DECISION.md)).** The declared
method uses **linear** mass interpolation with `ρ_min` regularization and no low-density
correction branch. Linear, `pmass=6`, and Du–Olhoff Eq. 4b have all been tested on the same
400×50 case; **none removes the localized-mode family**, and the flagged modes carry
essentially **no kinetic energy in the low-density material**. Any further mitigation
comparison (RAMP, Heaviside, void-mass penalization) must therefore not be premised on the
mass model being the cause. Current evidence associates the modes with **disconnected solid
components** — stated as evidence, not as an established mechanism. **This decision schedules
no new experiment.**

Retain the claim that the method avoids spurious low-density modes only if this
evidence supports it. Otherwise replace it with a bounded limitation and explain
the role of a-posteriori mode tracking.

**Gate S1:** every suspicious mode used in the paper has a documented physical
classification; unsupported absence claims are removed.

## Workstream 7: Manuscript and Response Update

Regenerate all affected tables and figures from accepted artifacts. Then:

- correct or qualify the clamped-beam \(\alpha=0.75\) monotonicity claim;
- reframe novelty as a frozen-reference simplification of Yuksel--Yilmaz;
- state clearly that load sensitivity is nonzero but omitted;
- distinguish sensitivity omission from eigenpair freezing/refresh;
- add the A5 lowest-mode check and explain mode-shape targeting;
- correct load, baseline, MAC, filter, and sensitivity equations;
- correct manuscript Eq. 9 explicitly so that MAC uses the documented
  mass-weighted inner product implemented by the revision experiments;
- reconcile OC/MMA descriptions and the actual contribution scope;
- distinguish literature methods from local implementations and identify
  `OlhoffApproach` as the sole local Olhoff-inspired comparator;
- state that the `OlhoffApproachExact` reconstruction attempt is archived and
  is not reviewer evidence;
- replace unsupported speedup, gap, convergence, complexity, and
  no-spurious-mode claims;
- report convergence, feasibility, grayness, MAC validity, and mesh evidence;
- explain design-dependent-load difficulties and symmetric/asymmetric-domain
  applicability;
- correct `Phi` versus `phi` notation;
- review the non-standard `quasi-static` terminology;
- reconcile void-material values (`1e-9` versus `1e-6`) and stopping tolerances
  (`1e-3` versus `2e-3`);
- verify the Huang comparison and all figure/table cross-references;
- complete the application, topology-method, Rayleigh, MMA, filter,
  load-sensitivity, and design-dependent-load references;
- enumerate the supplementary material precisely.

The response letter must address every reviewer item individually and cite the
exact manuscript location and supporting artifact. Adverse results must trigger
narrative rewriting, not selective omission. The 8.6% gap-to-published-optimum
and canonical 7.1x speedup narratives are withdrawn and cannot be regenerated
from the archived reconstruction. The monotonicity and 4.61x building-gain
narratives remain evidence-contingent under their existing gates.

## Workstream 8: Reproducibility Package

Create:

- a script/config/result/figure/table manifest;
- clean-run instructions and expected artifact checks;
- a saved hardware/software report and complete run log;
- a supplementary-material inventory;
- an immutable release tag and commit hash;
- a license and permanent archive/DOI;
- checksums for accepted result artifacts;
- a machine-readable completion report.

## Execution Tiers

### Tier 1: mandatory blockers

- authoritative formulation and parity tests;
- fail-loud runner and convergence telemetry;
- successful, converged Exp2 and Exp3;
- valid CR2 and A4 studies;
- low-mode classification for every published case;
- manuscript claim reconciliation and response letter;
- artifact manifest, commit, license, and permanent release.

### Tier 2: mandatory if the associated claim is retained

- RAMP/Heaviside mitigation comparison;
- claims of absence of spurious modes;
- strong frozen-accuracy or monotonicity claims;
- the 4.61x building-gain headline, if retained.

If Tier 2 evidence cannot be completed, remove or narrow the associated claim.
Canonical Du--Olhoff reproduction is not a Tier 2 task; that workstream is
closed and archived.
Do not substitute capped or partial results.

### Tier 3: useful supplementary additions

- additional meshes beyond the required two;
- additional refresh intervals;
- broader interpolation and optimizer comparisons;
- extra mode visualizations not required for reviewer traceability.

Tier 3 work must not delay correction of Tier 1 blockers.

## Compute Strategy

1. Run unit and small-mesh smoke tests first.
2. Run one production configuration per experiment and inspect convergence and
   artifacts before launching a sweep.
3. Run the scientific gates (S1, EXP2, EXP2b, EXP3, A4, CR2). There is no timing
   study: EXP1 and EXP5 are retired (Workstream 5).
4. Preserve failed runs and logs for diagnosis, but exclude them from accepted
   evidence.

## Definition of Complete

The revision is complete only when:

1. every mandatory experiment passes the master runner;
2. no reported result is capped, empty, unconverged, or mode-invalid;
3. every numerical claim maps to a configuration, source commit, and accepted
   result artifact;
4. every modal gain satisfies the declared MAC criterion;
5. every comparator is accurately labeled and evaluated under matched rules;
6. timing columns represent directly measured quantities;
7. adverse findings are incorporated into the manuscript;
8. every reviewer demand is `FULFILLED` with evidence or explicitly resolved
   by correction, qualification, or retraction;
9. an independent repeat audit reports no remaining unsupported claim and no
   unresolved `PARTIAL` or `NOT FULFILLED` item.
