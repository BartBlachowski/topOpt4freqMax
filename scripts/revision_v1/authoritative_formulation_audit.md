# Authoritative Inertial-Load Formulation Audit

## Scope and authority

This is a static source audit. No executable code, configuration, manuscript,
or test was changed.

The authority for required behavior is
`examples/Revision_v1/revision_v1_update1.md`, supported by
`scripts/revision_v1/IMPLEMENTATION_MAP.md`. The authoritative proposed-method
load is

\[
f_j(x) = (\omega_j^{(0)})^2 M(x)\Phi_j^{(0)},
\]

where the reference eigenfrequency and mass-normalized reference eigenmode are
computed once on the fully solid reference design and then frozen, while
`M(x)` is assembled from the evolving physical density. Therefore
`df_j/dx_e != 0`. The proposed variant omits that nonzero contribution; the
complete CR2 variant includes
`(omega_j^(0))^2 (dM/dx_e) Phi_j^(0)`.

The audit searched MATLAB, Python, JSON, TeX pseudocode/manuscript text, Markdown
plans, and formulation-bearing comments for `omega0`, `omega0^2`, `M(x) Phi0`,
`rho_nodal(x)`, reference eigenpairs, MAC, normalization, and load-sensitivity
selection. Historical reviewer reports and generated result artifacts are
evidence inputs, not implementation authorities; they are listed separately.

## Executive findings

1. **Both proposed solvers implement the wrong semi-harmonic load.** MATLAB and
   Python cache `omega0 * M0 * Phi0` and multiply it componentwise by projected
   nodal density. This is neither the authoritative formula nor the manuscript
   formula.
2. **The MATLAB and Python sensitivity interfaces differ.** MATLAB has a
   boolean `semi_harmonic_load_sensitivity`; Python always omits the
   semi-harmonic contribution and its wrapper does not parse that option.
3. **The MATLAB `harmonic` path is closest to the required load**, assembling
   `omega^2 * M(x) * Phi`, but it has separate refresh semantics, optional load
   norm rescaling, and always includes the partial mass sensitivity. It cannot
   be used as the authoritative proposed path without separating those choices.
4. **Python does not implement `harmonic_baseline=solid`.** The MATLAB wrapper
   and solver do; the Python wrapper and solver do not.
5. **Postprocessing reference modes do not match optimization reference modes.**
   MATLAB postprocessing always computes them on uniform `x=Vf`, even when the
   load eigenpair is computed on the solid domain. Exp3 then labels those
   uniform-field frequencies as solid-domain frequencies.
6. **MAC defaults are unsafe.** Explicit revision configurations use squared,
   mass-weighted MAC, but the legacy shorthand and schema default to unsquared
   `mass_inner_product`. Python has no corresponding MAC postprocessing path.
7. **Mode normalization is only partially explicit.** Proposed-solver
   eigenpairs are mass-normalized, but no deterministic phase convention is
   applied. MATLAB postprocessing relies on the generalized eigensolver's
   scaling and applies only a sign convention.
8. **The JSON schema disagrees with both wrappers.** It advertises baseline
   value `current` instead of solver value `initial`, rho source `x_filtered`
   instead of `xphys`, omits `harmonic_baseline` and the sensitivity selector,
   and describes `semi_harmonic` as `omega^2 M phi` although execution differs.

## MATLAB executable locations

### `analysis/ourApproach/Matlab/topopt_freq.m`

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| Lines 5-7 | Header describes cached `(M0,Phi0,omega0)` without stating the actual projected-density formula. | Header must state `omega0^2*M(x)*Phi0`, frozen reference eigenpair, evolving mass, and selectable omitted/complete sensitivity. | Yes | Replace the header comment and remove the term “fixed harmonic load” where it obscures whether only the eigenpair or the load is fixed. |
| Lines 70-83 | Parses `harmonic_normalize`, `semi_harmonic_baseline`, `semi_harmonic_rho_source`, `harmonic_baseline`, and boolean `semi_harmonic_load_sensitivity`. Defaults permit multiple formulations. | One solid reference design; no rho projection; no load-norm rescaling; explicit `omitted|complete` sensitivity mode independent of refresh. | Yes | Replace rho-source parsing with `load_sensitivity` enum parsing; fix reference baseline to `solid` for revision runs; default load normalization off or remove it from the authoritative path; retain a separately named refresh setting. |
| Lines 219-238 | Allocates `M0`, `Phi0`, `omega0`, `baseVec`, and nodal projection cache for semi-harmonic loads. | Cache only the reference eigenpair and metadata. Assemble `M(x)*Phi0` from current `M` each iteration. | Yes | Remove `semiHarmonicBaseVec` and the nodal-projection dependency from the proposed load path; retain `omega0` and `Phi0`; expose them in diagnostics/parity output. |
| Lines 240-268 | Legacy no-load-case path computes an eigenpair on initial `x=Vf`, mass-normalizes it, and later uses current `M`; load sensitivity is omitted because configured-load sensitivity is bypassed. | No silent alternate formulation. Revision behavior must use solid reference and explicit sensitivity mode. | Yes | Route legacy invocation through one normalized authoritative load-case configuration or reject it for revision evidence; do not retain implicit initial-reference semantics. |
| Lines 284-318 | Builds configurable solid/initial `K0,M0`, mass-normalizes modes, then stores `omega0*(M0*Phi0)`. | Compute the reference eigenpair on solid design, but do not use `M0` in the iterative load. The scalar is `omega0^2`. | Yes | Keep the solid eigenproblem; replace base-vector construction with cached `omega0_sq=omega0.^2` and `Phi0`. At iteration load assembly compute `omega0_sq(k)*(M*Phi0(:,k))`. |
| Lines 321-368 | `harmonic` cache can use initial or solid baseline. Solid baseline precomputation is MATLAB-only. | Reference-design selection must be identical in MATLAB/Python and fixed to solid for the proposed revision. | Yes | Extract common reference-eigenpair initialization used by proposed and refresh studies; remove baseline ambiguity from revision configs. |
| Lines 386-469 | Correctly assembles current `M(x)`, but also computes `rhoNodal` solely for semi-harmonic load/sensitivity. | Current `M(x)` is the only source of load design dependence. | Yes | Delete rho-source projection from the authoritative path. Keep the projection helper only for unrelated features, if any. |
| Lines 477-521 | Passes both current `M` and projected-density/base-vector state into load and sensitivity functions. | Pass current `M`, frozen `Phi0`, `omega0^2`, and sensitivity mode. | Yes | Redesign function signatures so load formula and sensitivity mode share the same authoritative data. |
| Lines 746-788 | `_localCurrentModesFromSubmatrices` mass-normalizes each mode using `phi' M phi=1`; no deterministic phase convention or clustered-subspace handling. | Mass-normalized reference modes with deterministic sign/phase for Gate A0; clustered modes compared as subspaces. | Yes | After mass normalization, orient each real mode by its largest-magnitude free DOF. Return cluster metadata for parity tests; do not force vector equality inside repeated eigenspaces. |
| Lines 790-908 | `harmonic` uses `omega^2*M(x)*Phi`; optional `harmonic_normalize` rescales load norm. `semi_harmonic` uses `rhoDof .* baseVec`. | Authoritative load exactly `omega0^2*M(x)*Phi0`, without rho multiplication or hidden norm rescaling. | Yes | Replace the semi-harmonic branch with the authoritative formula. Either remove `harmonic_normalize` from this path or hard-fail if enabled in revision configs. Preserve load-case scalar factors only. |
| Lines 910-979 | `harmonic` includes `2 U^T omega^2(dM/dx)Phi`; semi-harmonic optionally differentiates `rho_nodal*baseVec`. | Same load for both CR2 variants; omitted mode returns no load term, complete mode includes the mass derivative. | Yes | Replace the boolean and projected-density derivative with an enum branch: `omitted -> continue`; `complete -> vec=omega0^2*Phi0`, then reuse element `dM/dx` contraction. Ignore `domega/dx,dPhi/dx` because the reference eigenpair is frozen. |
| Lines 1094-1125 | Allows `solid|initial` baseline and `x|xphys` rho source. | Solid reference only for revision; no rho source in load. | Yes | Remove `_localSemiHarmonicRhoSourceVector` from the proposed path. Retain baseline helper only if non-revision studies need it; revision configs must reject non-solid baseline. |
| Lines 1127-1340 | Debug formula, FD load perturbation, and direct sensitivity helpers verify `rho_nodal*omega0*M0*Phi0`. | Debug and FD tools must verify `omega0^2*M(x)*Phi0` and its complete derivative. | Yes | Rewrite formula checks to compare assembled current-mass load; perturb element density, rebuild `M(x+/-h)`, central-difference compliance, and compare against complete analytical sensitivity. Report omitted and complete contributions separately. |
| Lines 1660-1701 | Parsers validate the obsolete baseline/rho choices. | Schema and both languages must accept the same authoritative options. | Yes | Remove rho parser and add strict sensitivity enum parser; align baseline validation with the final single reference-design decision. |

### `tools/Matlab/run_topopt_from_json.m`

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| Lines 98-114, 399-420 | Reads/passes baseline, rho source, harmonic baseline, and MATLAB-only boolean load sensitivity. | Parse one cross-language reference-design key and `load_sensitivity: omitted|complete`; no rho source. | Yes | Add strict enum parsing, pass identical fields to the solver, reject obsolete revision keys, and include the new keys in elastic2D dynamic-key rejection. |
| Lines 504-533 | Computes postprocessing “initial” modes separately and correlates them with topology modes. | Correlation reference modes must be the same solid-domain reference modes used to build the load. | Yes | Pass the solver's cached reference modes into postprocessing or call a shared reference-mode routine with `x_ref=solid`; record reference-design provenance in CSV labels. |
| Lines 1131-1173 | `localSolveReferenceModes` hard-codes uniform `xRef=volfrac`; this conflicts with solid load baselines. | Fully solid reference design, including passive void/solid handling identical to solver initialization. | Yes | Accept explicit reference density/passive sets or consume solver-returned modes. Remove the hard-coded scalar `volfrac` reference for revision outputs. |
| Lines 1259-1307 | Sorts generalized modes and applies deterministic sign, but does not explicitly mass-normalize them. | Explicit `phi' M phi=1`, then deterministic sign; clustered-mode subspace treatment in tests. | Yes | Normalize every returned mode with `Mff` before sign orientation; assert norm within tolerance and return normalization diagnostics. |
| Lines 1777-1838 | Correlation default is unsquared `mass_inner_product`; legacy `write_correlation_table=true` silently selects it. | One mass-weighted squared MAC definition. | Yes | Change default to `mac`; deprecate or reject `mass_inner_product` for revision evidence; make the legacy shorthand select `mac`. |
| Lines 1890-1922 | Implements mass-weighted cosine `C0` and squared MAC `C0^2` using final topology `Mfull`. | Eq. 9 mass-weighted squared MAC, with the mass matrix explicitly identified. | No formula change | Keep `mac` branch. Document that `Mfull=M(x*)`; add tests for scaling/sign invariance and parity with Python. Remove reliance on the unsquared branch in revision artifacts. |
| Lines 2055-2060 | Dynamic-key list omits `harmonic_baseline` and `semi_harmonic_load_sensitivity`. | All formulation-only keys rejected for elastic2D. | Yes | Replace obsolete keys with the final reference/sensitivity keys and include refresh settings. |

### MATLAB support and tests

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `tools/Matlab/projectQ4ElementDensityToNodes.m:83-85` | Comment preserves `Pavg` for semi-harmonic sensitivity. | No nodal-density derivative in the authoritative load. | Yes, comment/path usage | Remove the formulation claim; retain `Pavg` only if another feature uses it, otherwise retire the helper after dependency search. |
| `tools/Matlab/validateLoadCases.m:7-15,140-190` | Accepts separate `harmonic` and `semi_harmonic`; refresh belongs only to harmonic. | One authoritative frozen-reference load with independent refresh and sensitivity choices. | Yes | Define unambiguous semantics for the retained type. Prefer keeping `semi_harmonic` for the proposed method and adding explicit refresh/sensitivity fields; reserve comparator types for comparator implementations. |
| `examples/check_edof_and_harmonic_sensitivity_optionB.m:1-5,31,39-145` | Tests the harmonic mass derivative and factor scaling; calls it “Option B.” | CR2 test must exercise the authoritative proposed load under both sensitivity modes. | Yes | Rename/rewrite as an authoritative-load regression; assert identical loads between omitted/complete modes and distinct gradients, then central-difference the complete gradient. |
| `examples/test_multi_load_cases_ourApproach.m:30-40` | Mixes harmonic and obsolete semi-harmonic loads in one smoke test. | Test authoritative load aggregation without invoking two inconsistent formulas. | Yes | Use the authoritative type for modal loads; add a separate refresh-path test rather than mixing types. |

## Python executable locations

### `analysis/ourApproach/Python/topopt_freq.py`

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| Lines 4-6, 111-119 | Header/docstring describes cached `(M0,Phi0,omega0)` and rho-source options. | Document the authoritative evolving-mass formula and explicit sensitivity mode. | Yes | Replace stale documentation and add the new configuration contract. |
| Lines 151-153 | Parses load normalization, semi baseline, and rho source; has no sensitivity selector or harmonic solid-baseline selector. | Same fields and defaults as MATLAB. | Yes | Parse `reference_design=solid`, `load_sensitivity=omitted|complete`, and refresh interval; remove rho source; disable load normalization for authoritative runs. |
| Lines 284-316 | Builds solid/initial `M0`, mass-normalizes reference modes, caches `omega0*M0*Phi0`. | Cache `omega0^2` and `Phi0`; use current `M(x)` during load assembly. | Yes | Remove `semi_M0` from iterative load state after eigensolve; store `semi_omega0_sq` and mass-normalized, phase-oriented `semi_phi0`. |
| Lines 318-342 | Legacy path computes mass-normalized mode on initial `x=Vf`. | No implicit alternate reference design. | Yes | Route through authoritative configuration or reject for revision evidence. |
| Lines 344-347, 420-439 | Harmonic cache is initialized only from current iteration; Python lacks MATLAB's solid baseline path. | Cross-language solid reference and identical refresh schedule. | Yes | Add shared solid-reference initialization and refresh behavior matching MATLAB exactly. |
| Lines 445-452 | Projects element density to `rho_nodal`. | No projected-density multiplier. | Yes | Delete from authoritative load path. |
| Lines 470-504 | Harmonic computes `omega^2*M*phi` with optional norm rescaling; semi-harmonic computes `rho_dof*base_vec`. | Exact authoritative load without norm rescaling. | Yes | Replace semi-harmonic branch with `omega0_sq*(M@phi0)`; remove/forbid normalization in revision runs. |
| Lines 520-541 | Calls sensitivity helper without a sensitivity-mode argument. | Explicit omitted/complete behavior. | Yes | Pass the parsed enum and frozen eigenpair into `_load_sensitivity`. |
| Lines 753-787 | Mass-normalizes modes but has no deterministic phase convention; exceptions are silently swallowed. | Gate A0 mass normalization, phase convention, and fail-loud behavior. | Yes | Orient by largest-magnitude free DOF; return/raise eigensolve failure; expose clustered-mode metadata for subspace comparison. |
| Lines 806-823 | Allows obsolete baseline and rho-source options. | Solid reference; no rho source. | Yes | Remove rho helper and align reference validation with MATLAB/schema. |
| Lines 826-875 | Harmonic includes mass derivative; semi-harmonic always `continue`s and comment says sensitivity is “zeroed.” | Nonzero derivative deliberately omitted or completely included by option. | Yes | Add enum branch; use `vec=omega0_sq*phi0` for complete derivative; comment must say “nonzero term omitted,” never “zero.” |

### Python wrapper and validation

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `tools/Python/run_topopt_from_json.py:102-121` | Dynamic-key registry includes obsolete semi baseline/rho source but not sensitivity/reference/refresh keys. | Same dynamic keys as MATLAB. | Yes | Replace with final authoritative keys and reject them for elastic2D. |
| `tools/Python/run_topopt_from_json.py:475-510,526-548` | Parses load cases, normalization, baseline, and rho source; does not parse sensitivity option or harmonic baseline. | Cross-language identical config. | Yes | Parse strict enums and pass them unchanged; remove rho-source pass-through. |
| `tools/Python/run_topopt_from_json.py` (no MAC implementation) | Accepts modal postprocessing keys in the dynamic-key set but does not compute the MATLAB correlation matrix. | One documented mass-weighted squared MAC in both languages for Gate A0/V1 parity. | Yes | Implement/shared-test `MAC=(phi^T M psi)^2/[(phi^T M phi)(psi^T M psi)]`, using final topology `M(x*)`, or explicitly route Python outputs through a shared postprocessor. |
| `tools/Python/load_cases.py:6-18,105-130` | Mirrors current harmonic/semi-harmonic type split and harmonic-only refresh. | Same unambiguous type/refresh semantics as MATLAB. | Yes | Update normalized fields and validation for independent reference, refresh, and sensitivity choices. |
| `tools/Python/nodal_projection.py` | Exists to support projected-density loads. | Not part of authoritative inertial load. | Conditional | Remove only after confirming no non-authoritative feature depends on it; otherwise retain as a generic utility with no proposed-method claim. |

## JSON schema and configuration locations

### Schema

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `docs/topopt_config.schema.json:154-171` | Correctly describes `semi_harmonic` text as `omega^2 M phi`, but executable semi-harmonic code does not match; `update_after` is harmonic-only. | Schema and code must describe one authoritative formula with independent refresh. | Yes | Define exact formula, frozen reference semantics, sensitivity enum, and refresh semantics; avoid “fixed mode shape” if refresh is enabled. |
| `docs/topopt_config.schema.json:415-431` | `harmonic_normalize` says “unit energy” although code preserves first observed Euclidean load norm; baseline enum is `solid|current` while solvers use `solid|initial`; rho enum is `x|x_filtered` while solvers use `x|xphys`. | No hidden load rescaling; one solid reference; no rho source. | Yes | Remove/deprecate normalization and rho fields for authoritative runs; correct enums; add `reference_design`, `load_sensitivity`, and refresh schema. |
| `docs/topopt_config.schema.json:491-514` | Calls shorthand a MAC table, but default metric is unsquared `mass_inner_product`. | Default and shorthand must be squared mass-weighted MAC. | Yes | Set default to `mac`; describe final-topology mass matrix; mark `mass_inner_product` legacy/non-acceptable for revision evidence. |

### Revision_v1 configurations

| Locations | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `ablation_semi_harmonic.json:3-4,13-16,44-47` | Uses obsolete projected-density semi-harmonic formula, solid baseline, omitted sensitivity by absent boolean. | CR2 Variant A: authoritative evolving-mass load, sensitivity explicitly omitted. | Yes | Keep solid reference; remove `semi_harmonic_rho_source`; add `load_sensitivity:"omitted"`; ensure no load normalization. Rewrite name/notes. |
| `ablation_semi_harmonic_with_loadsens.json:3-4,13-16,44-47` | Same obsolete formula with boolean projected-density derivative. | CR2 Variant B: identical load, complete mass derivative. | Yes | Add `load_sensitivity:"complete"`; remove rho source/old boolean; rewrite derivative note as `omega0^2*dM/dx*Phi0`. |
| `ablation_harmonic_frozen_solid.json:3-4,13-16,44-45` | Already computes the closest formula with solid frozen eigenpair and no norm scaling, but sensitivity is implicitly always included. | Same authoritative load with sensitivity and refresh independently explicit. | Yes | Migrate to final authoritative type/config; set `load_sensitivity` explicitly and `refresh_interval:"infinity"`; do not treat it as a separate formulation. |
| `ablation_harmonic_periodic_solid.json:3,13-16,44-45` | Correct current-mass formula with refresh 50 and complete partial sensitivity. | A4 N=50 using same load and selected sensitivity policy as all A4 variants. | Yes | Migrate to common type; set refresh 50 and the A4 sensitivity policy explicitly so only refresh differs. |
| `clamped_beam_200x25.json:19-35,81-83,110-114` and `clamped_beam_400x50.json` same ranges | Obsolete semi-harmonic formula; solid baseline; rho source; load normalization true; explicit squared MAC. | Authoritative two-mode load, solid reference, omitted sensitivity for proposed method, no norm rescaling, squared MAC. | Yes | Remove rho source; set omitted mode; set/remove normalization false; retain `metric:"mac"`; rewrite notes. |
| `clamped_beam_harmonic_eq7_200x25.json:19-36,83-84,111-115` and `...400x50.json` same ranges | Uses correct current-mass formula and frozen solid modes but `harmonic_normalize:true`, so actual loads are rescaled and not Eq. 7. | No rescaling; no duplicate “Eq7 vs semi” formulation after unification. | Yes | Set normalization false/remove key; migrate into controlled sensitivity or refresh variants; rewrite “exact Eq.7” claim. |
| `ss_beam.json:13-16,44-46` | Obsolete formula and uniquely uses `baseline:"initial"`. | Solid reference and authoritative omitted-sensitivity load. | Yes | Set solid reference, remove rho source, add omitted sensitivity, rewrite provenance. |
| `ss_beam_harmonic_frozen.json:3-4,13-16,44` and `ss_beam_harmonic_periodic.json` same ranges | Current-mass formula but default initial reference and load normalization true; frozen/periodic semantics differ. | Solid reference, no normalization, common load/sensitivity, refresh only difference. | Yes | Add explicit solid reference, disable normalization, set sensitivity policy, and express refresh as `infinity`/50 under common configuration. |

### Other example configurations

| Locations | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `examples/Building/BuildingTopOptFreq.json:34-48,72-92,95-104` | Semi-harmonic modes, implicit solid baseline, normalization true, legacy unsquared correlation shorthand. | Explicit solid authoritative two-mode load, no rescaling, omitted sensitivity, squared MAC. | Yes | Add final reference/sensitivity keys, disable normalization, replace shorthand with explicit `metric:"mac"`. |
| `examples/ClampedBeam/BeamTopOptFreq.json:25-38,62-86,89-98` | Obsolete semi-harmonic formula, explicit rho source, normalization true, legacy unsquared correlation. | Authoritative formula and squared MAC. | Yes | Same migration as revision clamped configs; replace shorthand. |
| `examples/ClampedHingedBeam/ClampedHingedBeamTopOptFreq.json:25-38,62-86,89-98` | Same as ClampedBeam. | Same authoritative formula. | Yes | Same migration. |
| `examples/Performance/performance_comparison.json:25-38,62-86,94-97` | Obsolete semi-harmonic formula and normalization true. | Benchmark exact authoritative load without diagnostics/rescaling. | Yes | Add final explicit keys, remove rho source, disable normalization. |
| `examples/HingedBeam/BeamTopOptFreq.json:25-35,63-79,83-91` | Load is self-weight only, but stale semi-harmonic debug/baseline/rho and harmonic normalization keys remain. | Non-inertial example must not imply inertial formulation. | Yes | Remove all inertial-only optimization keys; no formula migration needed. |
| `examples/Building/BuildingTopOptSelfWeight.json:34-42,72-96` | Self-weight only but contains `harmonic_normalize:true`. | No inertial-only option in self-weight config. | Yes | Remove the unused key. |

## Experiment scripts and formulation-bearing comments

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `examples/Revision_v1/exp2_clamped_beam.m:6-45,60-70,275-300` | Treats semi-harmonic and harmonic Eq. 7 as distinct formulations; comments correctly define squared MAC but incorrectly say current paper Eq. 9 is unweighted; interpretation can declare accuracy from endpoint differences. | One authoritative formulation; Eq. 9 is already mass-weighted in current `paper/main.tex`; conclusions require converged/mode-valid evidence. | Yes | Remove obsolete two-formulation comparison, update comments, and run only controlled sensitivity/refresh differences. Delete unconditional “small difference confirms” text. |
| `examples/Revision_v1/exp2b_building.m:39-57` | Forces true MAC, but comments claim paper Eq. 9 is unweighted and hard-code solid reference frequencies. | Eq. 9 and implementation must match; reference values must come from accepted solid-reference artifacts. | Yes | Correct stale Eq. 9 comment; compute/read reference frequencies rather than hard-code them; ensure explicit MAC config. |
| `examples/Revision_v1/exp3_mesh_convergence.m:4-26,30-67,210-235` | Compares two formulations; helper named `localSolidEigenpairs` actually extracts uniform `x=Vf` postprocessing modes. | One formulation at two meshes and actual solid-domain reference frequencies. | Yes | Remove formulation dimension, use solver reference outputs/shared solid eigensolver, rename helper, and eliminate CSV-label probing. |
| `examples/Revision_v1/exp4_sensitivity_ablation.m:1-35,49-74,280-317,321-425` | CR2 A/B validates projected-density derivative; C/D uses current-mass formula; direct ratio manually reconstructs obsolete `omega0*M0*Phi0`. | CR2 must compare omitted/complete derivatives of the same authoritative load; A4 separately varies refresh. | Yes | Replace all four setup descriptions/configs; rebuild direct ratio with current `M(x)` and `omega0^2*dM/dx*Phi0`; central-difference Variant B; remove fixed 2% validation assertion unless predeclared and convergence-qualified. |
| `examples/Revision_v1/run_all_revision_experiments.m:47,128,719` | Labels outputs as semi-harmonic versus Eq. 7. | One authoritative formulation with CR2/A4 dimensions. | Yes | Rename summaries and artifact checks after experiments are restructured. |
| `examples/weightedTopologyResultsHelper.m` | Does not implement formula; reruns whichever JSON is supplied and moves correlation artifacts. | Must preserve authoritative configuration/provenance. | Minor | No formula change; add manifest/provenance validation when implementation phase begins. |

## Pseudocode, manuscript, schema comments, and plans

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `paper/main.tex:129-136` | States evolving `M(x)`, fixed eigenvector, omitted sensitivity; reference is called initial uniform. | Formula is correct; reference must be solid and sensitivity described as nonzero but omitted. | Yes, text | Replace “initial uniform” with declared solid reference and explicitly distinguish omission from mathematical zero. |
| `paper/main.tex:238-259` (load/sensitivity equations) | Eq. load uses `omega0^2*M(x)*Phi0` and gives nonzero derivative, but reference line 241 says uniform `x=Vf`; justification later treats load as external. | Solid reference; nonzero derivative deliberately omitted and evaluated by CR2. | Yes, text | Correct reference design; retain equations; qualify external-load treatment as an optimization approximation, not a physical identity. |
| `paper/main.tex:263-269` (multi-mode equation) | Correct formula but says eigenpairs are from initial solid domain, inconsistent with line 241. | One solid reference statement everywhere. | Text reconciliation | Retain formula and make all preceding/following reference terminology identical. |
| `paper/main.tex:284-303` (algorithm summary) | Pseudocode initializes `x=Vf`, solves on `K0,M0` without declaring that reference matrices are solid, and omits sensitivity without flagging it as nonzero. | Explicit solid reference and sensitivity policy. | Yes | Separate design initialization from reference-domain eigensolve; annotate omitted nonzero term and CR2 complete variant. |
| `paper/main.tex:310-320` (Eq. 9 MAC) | Current source already gives squared mass-weighted MAC. Mass matrix identity is not explicitly tied to final topology. | Same formula, explicitly `M(x*)`, consistent normalization. | Minor | Keep equation; define `M=M(x*)` and reference-mode embedding; remove stale comments elsewhere claiming Eq. 9 is unweighted. |
| `paper/main.tex:634-636,657-659` | Repeats correct evolving-mass formula but again says uniform-density reference; states pure static sensitivity. | Solid reference and explicit approximation language. | Yes | Correct reference design and say the nonzero load derivative is omitted. |
| `paper/reviews/revision_plan.tex:128-173` | Correctly diagnoses old mismatch but offers alternatives, including `omega0^2*M0*Phi0`, that are superseded by update1. | Evolving `M(x)` decision is final. | Yes, historical plan alignment | Mark A0 superseded and replace alternatives with the update1 authoritative decision. |
| `paper/reviews/revision_plan.tex:297-334` | Refresh discussion recognizes formula/sensitivity confounding and warns N=1 is not Yuksel. It says “zero-load-sensitivity approximation.” | Nonzero sensitivity omitted; same formula across refresh variants. | Yes, wording | Replace “zero-load-sensitivity” with “omitted nonzero load-sensitivity”; retain N=1 warning. |
| `paper/reviews/revision_plan.tex:650-673` | Says frozen load is externally specified and density gradient “set to zero”; describes current semi/harmonic split. | The load is design-dependent; the contribution is omitted by policy. | Yes | Rewrite after code unification; remove claims tied to old type names. |
| `paper/reviews/revision_plan.tex:991-1055` (Algorithm 3) | Load pseudocode is correct and assembles current `M(x)`, but line 1044 says load sensitivity is “set to zero.” Reference domain is merely “chosen.” | Solid reference; nonzero derivative omitted. | Yes, small | Name solid domain and change comment to “nonzero load-sensitivity contribution intentionally omitted.” |
| `paper/reviews/algorithms_comparison.tex` Algorithm 3/notation | Duplicates the revision-plan algorithm and mass-orthonormal notation. | Same authoritative wording and normalization. | Yes if retained | Apply the same reference/sensitivity corrections; ensure only one maintained pseudocode source or generate one from the other. |
| `docs/olhoff_implementation_analysis.tex:93-112,195-209` | Describes proposed method with correct evolving mass but uniform-density reference and omitted sensitivity. | Solid reference and explicit nonzero omission; regenerated results later. | Yes | Correct formulation prose; do not update numerical claims until accepted reruns exist. |
| `examples/Revision_v1/revision_v1_update1.md:14-54,85-100,137-175,242-253` | Correct authority, Gate A0 tolerances, CR2/A4 separation, and Eq. 9 requirement. | Remains authority. | No | No change. Note that current `paper/main.tex` Eq. 9 already appears compliant; independent audit should verify compiled manuscript numbering. |
| `scripts/revision_v1/IMPLEMENTATION_MAP.md:27-35,100-108,158-167,233-242` | Correctly maps the required formulation changes. | Remains implementation map. | No substantive change | Update file/line references after implementation only. |

## Comparator implementations: inspect but do not rewrite as the proposed method

These locations use related terms but implement different algorithms. They must
remain isolated and accurately labeled; applying the proposed formula to them
would invalidate comparator fidelity.

| Location | Current behavior | Required behavior | Change? | Exact proposed change |
|---|---|---|---|---|
| `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m:2-8,27-28,552-585` | Stage 2 uses `F=M(x)*u_hat`, Euclidean-normalizes the evolving displacement estimate, phase-aligns it, and omits `dF/dx`. | Preserve Yuksel comparator semantics; do not call it the authoritative frozen-reference load. | No formula change | Add comparator provenance/labeling only. Any N=1 refreshed proposed variant must not be called Yuksel unless full equivalence is established. |
| `analysis/YukselApproach/Python/solver.py:403-445,674-743` | Python counterpart uses `F=M(x)*u_hat`, Euclidean normalization and phase alignment, omitted load derivative. | Same as MATLAB comparator and published-method interpretation. | No authoritative change | Verify MATLAB/Python comparator parity separately; do not merge with Gate A0 proposed-method implementation. |
| `analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m:199-207,274-355` and `analysis/OlhoffApproach/Python/solver.py:195,356-449` | `omega0` names initial diagnostic frequencies; eigenpairs are recomputed for bound optimization. They do not construct the proposed inertial load. | Keep as comparator behavior. | No | Do not replace diagnostic `omega0` merely because it matches the search token. Label comparator accurately. |
| `analysis/OlhoffApproachExact/Matlab/compute_elem_sensitivity.m` and `compute_generalized_gradients.m` | Use mass-normalized eigenvectors and exact eigenvalue sensitivities, not compliance-load sensitivity. | Preserve canonical/exact comparator implementation. | No | Reuse only normalization test ideas; do not transplant proposed-load changes. |

## Historical/non-authoritative records

The following files contain many occurrences of the audited terms but are
review evidence or prior audits, not executable specifications:

- `paper/reviews/final_review_V1.tex`
- `paper/reviews/final_review_V2.tex`
- `paper/reviews/REVISION_AUDIT.md`
- `examples/Revision_v1/revision_implementation_audit.md`
- `docs/olhoff_audit.md`
- `analysis/OlhoffApproachExactOpus/**`

They should remain immutable evidence unless a separate documentation-cleanup
task explicitly asks to annotate them. Statements copied from them into current
pseudocode, comments, or response text must use the update1 authority.

## Required implementation order

1. Finalize one JSON contract: solid reference, authoritative load type,
   independent refresh interval, and `omitted|complete` sensitivity.
2. Update schema and both wrappers so invalid/obsolete combinations fail.
3. Update MATLAB and Python reference eigenpairs, normalization/phase, and load
   assembly.
4. Update omitted/complete mass sensitivity in both languages.
5. Align postprocessing reference modes and squared mass-weighted MAC.
6. Add Gate A0 parity and central-FD tests.
7. Migrate JSONs and experiment scripts/comments.
8. Update pseudocode/manuscript text only after executable parity passes.

Until steps 1-6 pass, no existing `Revision_v1` result is evidence for the
authoritative formulation.

