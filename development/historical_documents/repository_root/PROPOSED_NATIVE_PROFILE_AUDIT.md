# Forensic provenance audit: Proposed native profile for R3

Audit date: 2026-08-26
Audit branch: `benchmark-methodology-r2`
Audit HEAD: `cf290fc7f9daf9da27bc8224f9585a0e1657bff1`
Scope: documentary and source-history audit only; no optimization, performance run, parameter sweep, or algorithm modification was performed.

## 1. Executive verdict

A unique, pre-R3, scientifically defensible **simply-supported-beam Proposed profile** does exist, but it is not the March 2026 legacy performance profile currently copied into R3.

The profile is the OC, single-mode, frozen-solid-reference configuration declared in `examples/Revision_v1/A4_SPECIFICATION_V3.md` and encoded in `examples/Revision_v1/a4_ss_400x50_base.json` at commit `d051985fd7a2314a4a458e7bfba18d956954eebf` on 2026-07-14. Its configuration SHA256 is `c5e8949a318dd6ac657faf034ca85b2210d3a01976c37dab1602eec8ae341047`, unchanged at the 2026-07-29 pre-R3 performance audit (`310043e`). It predates the R3 methodology branch work begun on 2026-07-31.

The decision is independent of benchmark performance:

- the first Proposed implementation used OC;
- the current formal manuscript algorithm and algorithm supplement use OC bisection;
- the July authoritative-formulation gate defines the exact load as
  `F(x) = omega0^2 M(x) Phi0`, with a solid, frozen, mass-normalized reference eigenpair and omitted load sensitivity;
- the July A4 preregistration fixes OC, `rmin=2` elements, move `0.2`, floors `1e-9`, linear mass, tolerance `1e-3`, cap `2000`, no Heaviside/continuation, and uniform initialization;
- the March MMA configuration is explicitly a legacy cross-code experiment and was later judged not to represent the manuscript's nominal Proposed method.

This is **WP2 Case C with a unique R3 primary**: OC and MMA are legitimate numerical backends for the core quasi-static compliance formulation, and MMA generated some historical experiment-specific results, but only OC matches the formal algorithm, original implementation lineage, and pre-R3 A4 profile. Changing the backend does not change the frozen-load compliance objective, but it materially changes the update trajectory, volume enforcement, stopping behavior, and the paper's “no MMA solver required” implementation/cost claim. MMA is therefore not interchangeable in a native-performance benchmark.

The resolution has three disclosed limitations:

1. The current R3-branch solver SHA256 `19399780...f171b82` still implements the older `rhoDof .* [omega0 M0 Phi0]` semi-harmonic path, not the July authoritative formula. R3 may not run until the authoritative behavior is restored and regression-verified; this audit does not authorize that code change.
2. The July executable profile uses design bounds `[0,1]`, while the manuscript mathematical section states a design lower bound `1e-3`. R3 freezes the historically executable value `0`, discloses the manuscript mismatch, and must not confuse it with the stiffness or mass floors.
3. The A4 JSON says filter boundary `symmetric`, while the Proposed MATLAB filter actually uses a truncated centroid stencil at domain boundaries. R3 must serialize the effective operator and cannot claim symmetric padding until implementation and declaration agree.

These are engineering/manuscript-consistency gates, not a remaining optimizer/profile-choice ambiguity.

## 2. Evidence hierarchy

The requested hierarchy was applied in order. A source was not treated as authoritative merely because it was later or executable.

### WP0 freeze record

Before audit edits, the branch was `benchmark-methodology-r2` at
`cf290fc7f9daf9da27bc8224f9585a0e1657bff1`. The tree was already dirty:

- modified: `.gitignore`, `examples/Performance/benchmark_results.json`,
  `examples/Performance/performance_comparison.m`,
  `examples/Performance/table1_paper_style.tex`, and
  `tools/Matlab/run_topopt_from_json.m`;
- untracked: `BENCHMARK_PROTOCOL_R3.md`, `BENCHMARK_FAIRNESS_AUDIT.md`,
  `examples/Performance/benchmark_protocol_r3.json`,
  `DIAGNOSTIC_REPRO2007_BENCHMARK.md`, `MIGRATION_REPRODUCTION2007_REPORT.md`, and
  the `Matlab/reproduction2007/` tree (reported by Git under `Matlab/`).

All pre-existing modifications/untracked files were preserved. This audit created only
`PROPOSED_NATIVE_PROFILE_AUDIT.md` and, because the evidence produced a clear profile,
updated the three explicitly authorized untracked R3 methodology files. No `AGENTS.md` was
present in the workspace.

### Current evidence inventory

The full text search covered Proposed/ourApproach, OC/MMA, performance comparison,
stopping/convergence, reference/normalization, and material floors. The current records
material to Proposed provenance are:

- implementations: `analysis/ourApproach/Matlab/topopt_freq.m` and
  `analysis/ourApproach/Python/topopt_freq.py`; historical path `ourApproach/*` before the
  repository reorganization; `tools/Matlab/mmasub.m` for the auxiliary MMA backend;
- runners/contracts: `tools/Matlab/run_topopt_from_json.m`,
  `tools/Python/run_topopt_from_json.py`, `tools/Matlab/validateLoadCases.m`, and
  `docs/topopt_config.schema.json`;
- performance/R3 configuration and drivers:
  `examples/Performance/performance_comparison.json`,
  `examples/Performance/performance_comparison.m`,
  `examples/Performance/benchmark_protocol_r3.json`, `BENCHMARK_PROTOCOL_R3.md`, and
  `BENCHMARK_FAIRNESS_AUDIT.md`;
- protocol/audit records: `examples/Performance/PLAN_two_table_redesign.md`,
  `examples/Performance/STOP_RULE_AUDIT.md`,
  `examples/Performance/ledger/freeze_record.json`,
  `examples/Performance/ledger/protocol_ledger.json`, and the determinism,
  extension-invariance, history-logging, and instrumentation validation JSON files;
- revision configurations/drivers: `examples/Revision_v1/ss_beam*.json`,
  `clamped_beam*.json`, `ablation_*.json`, `exp1_perf_table.m`,
  `exp2_clamped_beam.m`, `exp3_mesh_convergence.m`,
  `exp4_sensitivity_ablation.m`, `exp5_scaling.m`, and
  `run_all_revision_experiments.m`;
- case configurations that demonstrate experiment-specific Proposed variants:
  `examples/ClampedBeam/BeamTopOptFreq.json`,
  `examples/ClampedHingedBeam/ClampedHingedBeamTopOptFreq.json`,
  `examples/HingedBeam/BeamTopOptFreq.json`, and
  `examples/Building/BuildingTopOptFreq.json`;
- manuscript/revision evidence: ignored `paper/reviews/algorithms_comparison.pdf`, with
  `paper/main.tex`, `paper/reviews/algorithms_comparison.tex`,
  `scripts/revision_v1/authoritative_formulation_audit.md`,
  `examples/Revision_v1/A4_SPECIFICATION_V3.md`,
  `examples/Revision_v1/a4_ss_400x50_base.json`,
  `MASS_INTERPOLATION_DECISION.md`, and
  `examples/Performance/PERFORMANCE_COMPARISON_AUDIT.md` recovered from Git history.

Other search hits concerned Yuksel, Olhoff, generic OC/MMA libraries, or unrelated
reproduction work; they were inventoried for namespace leakage but were not treated as
Proposed-setting authorities. Binary `.mat`, image, and PDF contents were not searched as
text; their associated configs/manifests and recoverable manuscript sources were audited.

| Class | Source record | Date / commit | Relevant evidence | Intended method or one experiment? | Finding |
|---:|---|---|---|---|---|
| 1 | `paper/main.tex` recovered from Git | 2026-07-14, `d051985` | Uniform initialization; one frozen eigenpair; `F=omega0^2 M(x)Phi0`; omitted `dF/dx`; density-weighted sensitivity filter; OC bisection; `max|Delta x|<1e-3`; linear mass, stiffness `p=3`; no MMA/continuation required | intended method | Highest-level current mathematical definition; internally conflicts on uniform versus solid reference and on `[1e-3,1]` versus executable `[0,1]` |
| 1 | `paper/reviews/algorithms_comparison.tex` / PDF | 2026-07-13, `8938332`; PDF SHA256 `ff0c88f...5484a8` | Algorithm 3 uses OC, omitted load sensitivity, density filter, and `max|Delta x|`; explicitly notes that A0 still had to resolve physical versus then-current semi-harmonic load | intended method | Strong OC evidence; load formula was not yet closed on July 13 |
| 2 | `scripts/revision_v1/authoritative_formulation_audit.md`, Gate A0 decisions, implemented source | audit lineage 2026-06-19 to 2026-07-14; implementation `d051985` | Resolves `F=omega0^2 M(x)Phi0`, solid frozen reference, mass normalization/deterministic phase, no nodal-density double scaling, no load-norm rescaling, explicit omitted/complete sensitivity | explicit pre-R3 formulation freeze | Closes the July 13 A0 ambiguity before R3 |
| 2 | `examples/Revision_v1/A4_SPECIFICATION_V3.md` and `a4_ss_400x50_base.json` | 2026-07-14, `d051985`; config SHA256 `c5e894...1047` | “one and only” base: OC, mode 1, solid reference, omitted sensitivity, floors `1e-9`, linear mass, `rmin=2` element, move `.2`, tolerance `.001`, cap `2000`, no Heaviside, uniform `Vf` start | pre-R3 single-factor preregistration | Unique complete SS profile; unchanged through `310043e` on July 29 |
| 3 | `examples/Performance/performance_comparison.json` | 2026-03-06 `4582541`; changed 2026-03-08 `76b2894` | MMA, legacy semi-harmonic, solid reference, floors `1e-6`, `rmin=2`, move `.2`, tolerance `.004` then `.003`, cap `2000` then `10000` | one legacy cross-code experiment | Exact provenance for old performance rows, not the formal method |
| 3 | `examples/Revision_v1/ss_beam.json` | 2026-05-29, `2fb35be` | OC, legacy semi-harmonic, initial reference, floors `1e-9`, `rmin=2`, move `.2`, tolerance `.001`, cap `2000` | one revision SS experiment | Transitional profile; close numerically to A4 but predates A0 load/reference correction |
| 3 | clamped/building revision configurations and manifests | February-June 2026; authoritative artifacts visible by `d051985` | MMA, solid reference, floors `1e-6`, move `.2`, commonly tolerance `.001`, legacy then corrected load lineage | case-specific manuscript experiments | Shows MMA was a legitimate backend for those examples, not the universal Proposed identity |
| 4 | original `ourApproach/Matlab/topopt_freq.m` | 2026-02-12, `e32d3e8` | OC only; physical harmonic load with current `M(x)`; uniform start/reference; omitted load sensitivity; `[0,1]`; move `.2`; tolerance `.01`; cap `2000` | original implementation | Establishes OC and the core physical-load idea before MMA existed |
| 4 | current/default Proposed implementation | audited HEAD; historical default retained from `7871381` onward | optimizer default OC; move `.2`; tolerance `.01`; cap `2000`; `[0,1]`; linear mass | generic default | Corroborates OC/move/cap only; generic tolerance/radius are not the SS profile |
| 5 | `examples/Performance/PERFORMANCE_COMPARISON_AUDIT.md` | 2026-07-29, `310043e` | Calls legacy experiment non-publication evidence; finds its Proposed row uses MMA and a formulation different from the nominal paper method; notes inactive normalization/Heaviside keys | pre-R3 audit | Negative evidence against promoting March MMA to native; no result is used to choose OC |
| 5 | `examples/Performance/ledger/freeze_record.json` and `protocol_ledger.json` | 2026-07-31, `b5ed5c3` | Says Proposed native profile was not frozen because the R3 branch lacked the manuscript | early R3 ledger | Correct for that branch inventory, but incomplete globally: the diverged `develop` lineage contains the missing pre-R3 evidence |
| 6 | R3 candidate profile | 2026-08-26 | Copies March 8 MMA settings | later choice | Rejected as the native profile; it was never methodologically affirmed |

The `develop` commits `8938332`, `d051985`, and `310043e` are not ancestors of the audited R3 HEAD; their merge base is `b98cc963` (2026-06-18). They remain valid repository history and are precisely the evidence the July 31 branch-local ledger failed to see.

## 3. Historical chronology

| Date | Commit | Change | Documented reason / classification | Relative to known comparisons |
|---|---|---|---|---|
| 2026-02-12 | `e32d3e8` | First Proposed MATLAB implementation: OC and `F=omega1^2 M(x)Phi1` | original method implementation | before MMA and before the March comparison |
| 2026-02-22 | `7871381` | Adds selectable OC/MMA; default remains OC | commit “MMA Multi works!”; enables multi-load/clamped experiments | before March comparison; auxiliary capability |
| 2026-02-23 | `7ef99a0` | Adds `semi_harmonic`: `rho_nodal(x) .* [omega0 M0 Phi0]`, solid/initial baseline option | “Beam and building works”; described as an explicit baseline fix | formulation drift from the original physical load |
| 2026-03-06 | `4582541` | Creates legacy performance configuration: MMA, solid legacy semi-harmonic, floors `1e-6`, `rmin=2`, `.004`, `2000` | performance-comparison configuration | comparison work had begun; experiment-specific |
| 2026-03-08 | `76b2894` | `.004 -> .003`, `2000 -> 10000`, inactive Heaviside false -> true | “Frequency history plots created”; no methodological rationale | after comparative workflow existed; problematic as a “native” source |
| 2026-05-29 | `2fb35be` | SS revision profile uses OC, initial legacy reference, floors `1e-9`, `.001`, `2000` | revision experiments | supersedes March settings for SS intent, but load remained transitional |
| 2026-06-19 to 2026-06-30 | `2f3389b`–`47737e` | Revision campaign and frozen-solver artifacts; clamped/building primarily MMA | revision implementation/results | case-specific results already known; not used to choose R3 |
| 2026-07-13 | `8938332` | Consistency audit/supplement formalizes OC algorithm but leaves A0 load alternative explicit | methodology consistency audit | pre-R3 |
| 2026-07-14 | `d051985` | Implements A0 physical load and adds authoritative A4 OC base config/specification | methodological correction and one-factor preregistration | pre-R3; no R3 result exists |
| 2026-07-29 | `310043e` | Adversarially removes legacy cross-code comparison from publication evidence | scientific/implementation audit | sees old results but uses them only to invalidate, not tune, the legacy profile |
| 2026-07-31 | `b5ed5c3` | R3-lineage ledger says Proposed profile unavailable because manuscript absent on branch | branch-local inventory gap | start of current R3 design lineage |
| 2026-08-26 | current | R3 candidate copies March MMA profile | current methodological blocker | no R3 run authorized or inspected |

No evidence of misconduct is implied. The chronology shows normal experimental and revision drift; it also shows why a profile name and hash are necessary.

## 4. OC versus MMA

### Core formulation versus numerical backend

The core Proposed formulation is frozen-eigenpair quasi-static compliance:

1. initialize `x_e=Vf`;
2. compute a mass-normalized reference eigenpair once on the declared reference domain;
3. at each iteration assemble `F(x)=omega0^2 M(x)Phi0`;
4. solve `K(x)u=F(x)` and minimize compliance using stiffness-only sensitivity (the nonzero load derivative is deliberately omitted);
5. filter sensitivities, enforce volume, and stop on maximum design change;
6. perform a terminal eigensolve for achieved frequency.

OC and MMA solve the density-update subproblem. They do not redefine the quasi-static objective, but they are not observationally interchangeable:

- OC uses multiplicative updates and a Lagrange-multiplier bisection enforcing volume equality each iteration.
- MMA has asymptote history, different constraint scaling, a move clip, and in this repository a conditional post-update volume projection.
- Iteration counts, convergence dynamics, per-iteration work, and memory differ.
- The manuscript explicitly claims a standard-compliance extension requiring no MMA solver. An MMA benchmark would contradict that implementation/cost claim even if its objective were identical.

### Finding

The historical classification is **Case C**:

- OC is manuscript-native and R3-primary.
- MMA is a legitimate auxiliary/backend variant used in clamped/building and legacy performance experiments.
- The old March MMA profile must be named, if ever retained, `Proposed-legacy-MMA-2026-03-08`; it may not be called `Proposed-native-R3`.
- Solver choice should not be hidden in a numerical robustness row or selected by performance.

## 5. Parameter provenance matrix

| Parameter | Provenance chain | R3 native value | Confidence / qualification |
|---|---|---|---|
| Optimizer | OC-only original `e32d3e8` -> OC default after selectable MMA `7871381` -> May SS OC `2fb35be` -> manuscript/Algorithm 3 OC -> A4 OC | `OC` | CONFIRMED for manuscript/SS native profile |
| Objective/load | original `omega0^2 M(x)Phi0` -> obsolete nodal-density semi-harmonic `7ef99a0` -> A0 audit and `d051985` correction | compliance under `F=omega0^2 M(x)Phi0` | CONFIRMED; current R3 code is stale |
| Reference design | original uniform initial -> February/March solid option -> May initial -> manuscript internally mixed -> A0/A4 solid | fully solid reference, frozen | CONFIRMED by explicit pre-R3 A0 decision; for uniform no-passive SS, mode shape is invariant but `omega0` and reported gain reference differ |
| Eigen normalization | original mass normalization -> A0 adds deterministic orientation | `Phi0' M0 Phi0=1`; largest-magnitude free DOF oriented positive | CONFIRMED; no per-iteration norm rescaling |
| Load normalization | March `harmonic_normalize=true`; May/A4 false; code applies the flag only to `harmonic`, not legacy/current `semi_harmonic` | false / prohibited for authoritative path | CONFIRMED; the March true value was inactive for semi-harmonic and is not a real alternative |
| Load sensitivity | omitted in original/manuscript/A4; complete variants explicitly ablations | omitted, although mathematically nonzero | CONFIRMED; must be labelled approximation |
| Modal factors | March stores mode 1 factor 1 and a dead mode 2 factor 0; May/A4 use one mode | one active mode: mode 1, factor 1 | CONFIRMED; remove dead mode-2 entry |
| Filter type | original and all SS profiles use density-weighted sensitivity filtering | sensitivity filter | CONFIRMED |
| Filter radius | original fallback `.05 m`; March/May/manuscript SS/A4 converge on `2` elements | `2.0` element units | STRONGLY SUPPORTED for SS profile; physical radius changes with mesh |
| Filter boundary | configs say symmetric; source builds a truncated centroid stencil | effective truncated stencil; declaration mismatch must be fixed or relabelled | DISCLOSED ENGINEERING LIMITATION |
| Move | original hard-coded `.2`; configurable but unchanged in March, May, A4 | `0.2` | CONFIRMED |
| Design lower bound | original/current OC and MMA source use `[0,1]`; manuscript says `[1e-3,1]`; A4 config supplies no override | `0.0` effective executable bound | STRONGLY SUPPORTED executable value; manuscript mismatch disclosed |
| Stiffness floor | original `Emin/E0=1e-9`; March `1e-6`; May SS, manuscript SS, A4 `1e-9` | `1e-9` | CONFIRMED for SS; not the design lower bound |
| Mass floor | original `rho_min/rho0=1e-6`; March/clamped `1e-6`; May SS, manuscript SS, A4 `1e-9` | `1e-9` | CONFIRMED for current SS profile; not the design lower bound |
| Mass interpolation | original/current implementation linear; July mass decision corrects manuscript's temporary `p_M=3` drafting error | linear, `p_M=1`, no low-density branch | CONFIRMED |
| Stiffness penalization | all sources | `p_K=3`, fixed | CONFIRMED; no continuation |
| Heaviside/projection | March key toggled but inactive; manuscript/A4 explicitly none | none | CONFIRMED; unsupported key prohibited |
| Initialization | original, manuscript, May/A4 | uniform `x_e=Vf` (R3: `.5`) | CONFIRMED; distinct from solid reference |
| Native stop | original/max raw change `.01`; March `.004/.003`; May/manuscript/A4 `.001` | strict loop continuation while `max|x^k-x^{k-1}| > 1e-3`; native convergence at `<=1e-3` after successful update | CONFIRMED for current SS profile; raw `x`, not common `xPhys` persistence |
| Iteration cap | original/March-6/May/A4 `2000`; March-8 `10000` lacked method rationale | `2000` safety cap | STRONGLY SUPPORTED; `CAP_HIT`, never convergence or methodological stopping condition |
| OC internal control | original and current OC routine | bounds `[0,1]`, move `.2`, lambda bracket `[0,1e9]`, relative bracket tolerance `1e-3`, at most 200 bisections | implementation-defined numerical subsolver; serialize at freeze |
| Volume | OC lineage | equality enforced by OC bisection over active elements | CONFIRMED for no-passive SS |

Equal numbers in other methods do not imply equal mathematical tests. In particular, Proposed native tolerance is on raw design-variable change; R3's harmonized diagnostic remains a separate ten-iteration test on physical density and volume residual.

## 6. Reconstructed historical profiles

### P0 — `Proposed-original-physical-OC-2026-02-12`

`e32d3e8`: OC; uniform `x=Vf` both start and reference; physical current-mass load `omega0^2 M(x)Phi0`; mass-normalized mode; omitted load derivative; sensitivity filtering; `rmin=.05 m` fallback; move `.2`; design `[0,1]`; `Emin/E0=1e-9`; `rho_min/rho0=1e-6`; linear mass; tolerance `.01`; safety cap `2000`; no continuation.

This is coherent and establishes identity, but its generic radius/tolerance and mixed floors are superseded for the manuscript SS case.

### P1 — `Proposed-legacy-semi-harmonic-MMA-2026-03-08`

`4582541` plus `76b2894`: MMA; obsolete load `rhoDof(x) .* [omega0 M0 Phi0]`; solid reference; `rho_source=x`; `harmonic_normalize=true` (inactive for this load type); omitted semi-harmonic load sensitivity by default; sensitivity radius `2` elements; move `.2`; design `[0,1]`; stiffness/mass floors `1e-6`; linear mass; tolerance `.003`; cap `10000`; an inert Heaviside key; uniform start.

This is a coherent legacy performance-experiment profile, not the native method. Its tolerance/cap changes occurred during performance-history work and cannot be promoted by outcome.

### P2 — `Proposed-revision-SS-transitional-OC-2026-05-29`

`2fb35be:examples/Revision_v1/ss_beam.json`: OC; obsolete semi-harmonic formula; initial/uniform reference; one mode; floors `1e-9`; linear mass; radius `2` elements; move `.2`; tolerance `.001`; cap `2000`; normalization false (inactive); uniform start; no Heaviside.

This profile supplies the SS numerical controls later retained by A4 but not the final A0 load/reference definition.

### P3 — `Proposed-manuscript-SS-OC-A0-2026-07-14` (`Proposed-native-R3`)

`d051985` A0 implementation plus the immutable A4 base config: OC; exact physical current-mass load; solid frozen mass-normalized reference; one mode; omitted load sensitivity; floors `1e-9`; linear mass; radius `2` elements; move `.2`; tolerance `.001`; cap `2000`; uniform start; no continuation/Heaviside/load-norm rescaling; effective design bounds `[0,1]`.

This is the only complete profile that simultaneously postdates the explicit formulation correction, predates R3, and is declared as a one-factor baseline rather than assembled retrospectively.

### Case-specific MMA profiles

The clamped-beam/building line uses MMA, solid reference, commonly floors `1e-6`, radius `2`, move `.2`, and case-specific tolerances. These profiles generated important numerical examples. They demonstrate backend legitimacy but do not override the SS/manuscript algorithm profile.

## 7. Manuscript/result correspondence

| Evidence/result family | Matching profile | Confidence | Basis and limitation |
|---|---|---|---|
| Original Proposed demonstration | P0 | CONFIRMED | source itself contains the complete implementation |
| March legacy performance driver, timings, iteration endpoints and topology images | P1 | CONFIRMED | driver reads that exact JSON; later audit quarantines the evidence |
| May SS revision intent | P2 | CONFIRMED as configuration; UNKNOWN for every surviving endpoint | config and experiment scripts are explicit; individual artifact-to-commit hashes are incomplete on the audited branch |
| Current manuscript mathematical algorithm | P3 core | CONFIRMED | equations/pseudocode plus A0 resolution; manuscript still contains lower-bound/reference wording inconsistencies |
| July A4 `N=inf` baseline/intended SS revision evidence | P3 | CONFIRMED as preregistered profile | exact config/hash and factor-drift validator; no A4 result is used here |
| Clamped/building manuscript topologies/tables | case-specific MMA profiles | STRONGLY SUPPORTED | manuscript explicitly states MMA and matching settings; some artifacts predate the A0 load correction, so formula-level provenance must be checked before calling each endpoint authoritative |
| Reported legacy cross-code performance headline | P1, not P3 | CONFIRMED and scientifically retired | cannot support a native R3 claim |
| Revision results currently intended for publication | mixed: P3 formulation, case-specific OC/MMA backends | PLAUSIBLE to STRONGLY SUPPORTED by case | exact result manifests must remain the authority; no single solver backend generated all manuscript figures |

Numerical similarity was not used to upgrade any confidence level.

## 8. Post-hoc drift assessment

| Change | Classification | R3 treatment |
|---|---|---|
| Adding MMA in `7871381` | experimental capability for multi-load/case work | retain as named auxiliary backend |
| Replacing physical load with nodal-density semi-harmonic in `7ef99a0` | methodological drift, initially documented as a baseline fix | historical only; superseded by A0 |
| March 8 tolerance `.004 -> .003` and cap `2000 -> 10000` | benchmark convenience/history-production change; no mathematical rationale in commit | not native and not used as R3 center |
| March inactive `harmonic_normalize=true` and Heaviside toggle | configuration drift/dead controls | prohibit in authoritative manifest rather than infer meaning |
| May switch to OC, floors `1e-9`, tolerance `.001` | revision alignment with SS manuscript | retained in P3; predates A0 exact load correction |
| July linear-mass correction | mathematical/methodological correction supported by source audit | retain; not performance tuning |
| July A0 physical-load correction | mathematical correction after explicit static audit | retain; exact pre-R3 implementation exists |
| July A4 base profile | preregistered one-factor experiment baseline | decisive profile source; independent of R3 ranking |
| July 29 legacy performance audit | adverse validation/removal of unsupported evidence | use only to reject legacy evidentiary status, never to select a faster endpoint |

The March profile is particularly vulnerable to post-hoc criticism because its cap/tolerance changed after comparison machinery and outputs existed. P3 is still a revision-era declaration made with prior results in the repository, but its rationale is mathematical/manuscript consistency and one-factor experimental control, not observed ranking.

## 9. Recommended R3 Proposed profile

The exact profile ID is `proposed_manuscript_ss_oc_a0_2026-07-14`. The label in tables may be `Proposed-native-R3`, but the full ID and provenance must be serialized.

```yaml
profile_id: proposed_manuscript_ss_oc_a0_2026-07-14
source_commit: d051985fd7a2314a4a458e7bfba18d956954eebf
source_config: examples/Revision_v1/a4_ss_400x50_base.json
source_config_sha256: c5e8949a318dd6ac657faf034ca85b2210d3a01976c37dab1602eec8ae341047
optimizer:
  type: OC
  design_bounds: [0.0, 1.0]
  move_limit: 0.2
  volume_update: lagrange_multiplier_bisection_equality
  lambda_bracket: [0.0, 1.0e9]
  lambda_relative_tolerance: 1.0e-3
  lambda_max_bisections: 200
formulation:
  objective: compliance
  load_type: proposed_frozen_reference_inertial
  load_formula: F(x) = omega0^2 * M(x) * Phi0
  reference_design: fully_solid
  reference_refresh_interval: 0
  reference_mode: 1
  reference_mode_factor: 1.0
  eigenvector_normalization: Phi0_transpose_M0_Phi0_equals_1
  deterministic_phase: largest_magnitude_free_DOF_positive
  per_iteration_load_norm_rescaling: false
  load_sensitivity: omitted
interpolation:
  stiffness: Emin/E0 + (1-Emin/E0)*x^3
  stiffness_floor_ratio: 1.0e-9
  stiffness_penalization: 3.0
  mass: rho_min/rho0 + (1-rho_min/rho0)*x
  mass_floor_ratio: 1.0e-9
  mass_exponent: 1.0
  low_density_mass_branch: none
filter:
  type: density_weighted_sensitivity
  radius: 2.0
  radius_units: element
  declared_boundary_in_source_config: symmetric
  effective_historical_operator: truncated_centroid_stencil
  heaviside_projection: false
initialization:
  design: uniform_at_problem_volume_fraction
continuation: none
native_stop:
  field: raw_design_x
  test: max_element_abs_change_lte_1e-3_after_successful_update
  tolerance: 1.0e-3
  maximum_iterations: 2000
  cap_role: safety_budget
  cap_hit_is_convergence: false
```

The profile intentionally does not contain `semi_harmonic_rho_source`, `harmonic_normalize=true`, a dead mode-2 load, MMA asymptotes, MMA volume projection, or a Heaviside key.

Before execution, R3 engineering validation must prove that the frozen current solver implements this block. A code port is outside this audit's authorization.

## 10. Alternative profiles

- **P1 / March MMA:** retain only as a quarantined legacy implementation profile. It is neither formal-manuscript-native nor a suitable robustness center.
- **P2 / May OC:** historically important transitional SS profile. It supplies correct SS controls but uses the obsolete load/reference semantics; P3 supersedes it.
- **Original P0:** establishes OC/core lineage, but its generic tolerance/radius and mass floor are not the later manuscript SS specification.
- **Case-specific MMA:** legitimate for reproducing the clamped/building figures that explicitly used MMA. It is not primary in an SS method-native R3 table.
- **Report both OC and MMA as co-primary:** rejected. It would double the campaign, obscure the manuscript-native cost claim, and answer no necessary R3 question.

OC versus MMA should not be added automatically to Experiment D. A future, separately named “optimizer-backend robustness” study could be scientifically meaningful if the question is solver dependence of the frozen-load surrogate and both arms use the exact same A0 load, floors, filter, stop, and evaluation. It is unnecessary for the present R3 fairness question and would not repair provenance.

## 11. R3 implications

### Required documentation changes

The three R3 deliverables are updated by this audit to:

- replace `proposed_candidate_2026-03-08` with `proposed_manuscript_ss_oc_a0_2026-07-14`;
- close methodological gate M1 with this dated provenance record;
- set OC, physical A0 load, solid frozen reference, one active mode, floors `1e-9`, linear mass, radius `2` elements, move `.2`, native tolerance `.001`, and cap `2000`;
- remove obsolete density-source, load-normalization, MMA-volume-projection, and dead mode-2 settings;
- recenter the Proposed tolerance OAT on `.001` using the predeclared factor-two levels `.0005, .001, .002`;
- change Proposed diagnostics from MMA/projection to OC/bisection/volume-feasibility diagnostics;
- retain engineering `DO_NOT_EXECUTE` status until the current branch implements and verifies the resolved profile.

### Configuration contract

Keep these namespaces separate:

- `problem.*`: geometry, element, solid material constants, BCs, target volume, mesh, passive regions, target mode. Uniform initialization may be stated here as a problem-independent starting field but must be copied explicitly into the Proposed resolved manifest.
- `optimization.proposed.*`: optimizer, A0 load/reference, sensitivity omission, interpolation floors/laws, design bound, filter/operator/radius, move, native stop/cap, initialization, and absence of continuation.
- `measurement.*`: E1-E3 external evaluators, eigensolver settings, timing boundaries W1/W2, thread/environment controls, replay/checksums, status taxonomy, and table fields.

Proposed must not inherit Yuksel's piecewise mass law/stage tolerances or Olhoff's design floor, multiplicity controls, LP settings, filter, or move.

### Experiments A-E and timing windows

- **A:** OC is the Proposed fixed-work kernel. W1/W2 remain method-neutral measurement windows; prefix/kernel identity must be validated after the A0 implementation is restored.
- **B:** use the exact P3 named profile. Report cap hits and native raw-design stopping without cross-method speedup.
- **C:** preserve the native `.001` raw-`x` stop and separately apply the existing common physical-density persistence criterion. Do not replace one with the other.
- **D:** keep filter/move/tolerance OAT only; center on P3. Do not add MMA unless a new solver-robustness question is preregistered.
- **E1-E3:** unchanged and external; P3's own `1e-9` floors do not change E1's preregistered role.
- **W1/W2:** unchanged; measurement policy may not alter the OC kernel.

## 12. Reviewer-risk assessment

Could a hostile reviewer reasonably claim P3 was selected to improve Proposed's R3 ranking? They could note that P3 was formalized after legacy comparative results existed. The stronger claim that it was selected *because of* R3 performance is not supported: P3 predates R3, is an immutable single-factor base, matches the original and formal OC lineage, and its exact load was selected by a static mathematical consistency audit. No R3 result exists and no comparative value was used here.

| Setting | Plausible criticism | Provenance defense | Residual limitation |
|---|---|---|---|
| OC | MMA was used in old tables and some manuscript figures; OC may be faster | original code, current formal algorithm, A0/A4 preregistration all select OC before R3 | manuscript examples remain backend-mixed; do not generalize OC provenance to clamped/building reproduction |
| `rmin=2` | chosen topology-dependent radius; physical length shrinks with mesh | appears in March, May, manuscript SS, and A4 before R3 | filter operator/boundary mismatch and element-unit mesh dependence must be explicit |
| move `.2` | larger move could reduce iterations | invariant from original code through every relevant profile | OC dynamics may still be move-sensitive; D reports `.1/.2/.3` without replacing center |
| design floor `0` | contradicts manuscript `1e-3` and may favor compliance | exact original/A4 executable bound; separated from material floors | manuscript correction or an explicitly new R3.x profile is required if authors intend `1e-3` |
| material floors `1e-9` | lower floors could affect topology/frequency | May SS, current SS manuscript, A4 base; not selected from outcome | clamped/building use `1e-6`; common E1-E3 results remain model dependent |
| normalization | “false” may change load scale and optimizer | authoritative path mass-normalizes `Phi0` and applies no iterative norm rescaling; A0 explicitly removes hidden scaling | legacy boolean was inactive for semi-harmonic, so old JSON alone was misleading |
| tolerance `.001` | stricter stop increases runtime and could hurt Proposed | May SS, manuscript and immutable A4 base all predate R3 | cap may censor it; R3 must report rather than loosen after observation |
| cap `2000` | cap may truncate Proposed selectively | original/default, May and A4 use `2000`; treated only as safety budget | manuscript does not define it as methodology; cap hit is censoring, never convergence |
| solid reference | chosen after initial-reference results | explicit A0 decision and A4 preregistration; uniform no-passive SS has identical eigenvector shape under scalar interpolation | reference eigenvalue and gain normalization differ, and passive-region cases are not invariant |
| omitted `dF/dx` | mathematically incomplete | original code, manuscript and A4 define the omission as the approximation | must be disclosed; complete-sensitivity ablations do not redefine native profile |

The defense is adequate without claiming that criticism is impossible. The remaining risk is mitigated by the exact profile ID, negative legacy classification, mandatory E1-E3 evaluation, native/common separation, and no post-result amendment rule.

## 13. Final methodological status

The Proposed methodology/profile choice is resolved. R3 remains prohibited from execution until the current branch's solver, runner, effective filter contract, resolved-config serializer, and regression tests are shown to implement the P3 block exactly. The executable design-bound/manuscript mismatch and filter-boundary mismatch must remain visible in the freeze record. No benchmark code was changed and no optimization was run in this audit.

**PROPOSED PROFILE RESOLVED WITH DISCLOSED LIMITATION**
