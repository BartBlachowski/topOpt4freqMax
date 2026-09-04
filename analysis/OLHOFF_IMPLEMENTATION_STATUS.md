# Du–Olhoff implementations in this repository — status and dispatch policy

This repository contains **seven** distinct realizations of the Du & Olhoff
(2007) eigenfrequency-maximization algorithm family. Several of them share
function names (`olhoffOpt`, `model2D`, `assemble2D`, `eigSolve`, `genGrad`,
`innerLoop`, `prepFilter`, `applyFilter`, `mmasub`, `subsolv`), and MATLAB
resolves by path order, so *which* one executes is a property of the path, not
of the call. Getting it wrong produces a run that looks fine and is
scientifically void — that has happened in this project before.

This file is the authority on which is which.

---

## 1. Conference-active implementation

| | |
|---|---|
| path | **`analysis/OlhoffM4Reconstruction/`** |
| solver core | `analysis/OlhoffM4Reconstruction/+frozen/{algo,fem,filter,mma_published,mma}` |
| entry points | `olhoffm4_config.m`, `olhoffm4_run.m`, `olhoffm4_paths.m` |
| imported from | `/Users/piotrek/Programming/Matlab/Olhoff` (see `IMPORT_MANIFEST.json`) |
| benchmark label | **Du–Olhoff reconstruction (M4)** |
| status | **CONFERENCE-ACTIVE.** The only implementation the conference benchmark may execute. |

Its frozen realization: genuine nested MMA sub-optimization
(`innerSolver='mma'`, `innerVar='drho'`, published Svanberg constants, full
(25d) coupling, `tolInner = 0.05`); **M4** multiplicity treatment
(`multRule='subspace'`, `subN = 2`, no threshold classifier); fixed **physical**
filter radius `R = 0.06·b` with `rminEl` derived per mesh; outer stop
`‖Δρ‖₂ < 0.05·√(N_e/3200)`, i.e. a mesh-independent per-element RMS tolerance of
`8.838835e−04`, with the settled-move guard; S2 staged move ladder
`[0.04 0.02 0.01 0.005]` on the legacy `beta` stall signal; single-threaded.

**It must not be called "Olhoff 2007."** It is a reconstruction: numerical
continuation and inner-convergence details are incompletely specified in the
original publication. See `olhoffm4_caveat.m`.

### Why the core sits under `+frozen/`

`genpath` skips folders whose name begins with `+`, **and every folder beneath
them**. Repository scripts (`examples/Revision_v1/*.m`) call
`addpath(genpath(<repo>/analysis))`. A plain subfolder would put this copy of
`mmasub.m`, `subsolv.m`, `olhoffOpt.m` and the rest ahead of `tools/Matlab` and
would break the isolation guarantee that
`Matlab/reproduction2007/runner/repro2007_verify_isolation.m` checks. Under
`+frozen/` the core is invisible to `genpath` and reachable only through
`olhoffm4_paths()`, which asserts the implementation's identity before
returning.

---

## 2. Classification of every other Olhoff implementation

| path | class | what it is | may the conference benchmark reach it? |
|---|---|---|---|
| `Matlab/reproduction2007/` | **historical / reference, still used** | The clean-room Du–Olhoff 2007 reproduction (Eq. 22 LP route + a paper-literal MMA route). Still dispatched by `tools/Matlab/run_topopt_from_json.m` under the approach key `OlhoffDu2007Repro`, and used by several `analysis/iteration_efficiency_*` and `analysis/olhoff_*` audits. Deliberately outside `analysis/` so no `genpath(analysis)` sweep reaches it; guarded by `repro2007_paths()`. | **NO** |
| `analysis/olhoff_stabilization_audit/` | **superseded** | The `olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1` profile: `r_min = 1.3` elements, `move = 0.005 → 0.0025`, and a **fixed 1600-outer-iteration work horizon that is explicitly not native convergence**. This is what the previous benchmark driver dispatched for its Olhoff column. Retained as evidence. | **NO** |
| `analysis/OlhoffApproach/` | **superseded** | The original bound-formulation MMA implementation (`topFreqOptimization_MMA.m`), plus a Python port. Still referenced by `tools/Matlab/run_topopt_from_json.m` for the legacy `Olhoff` approach key. Not the reconstruction. | **NO** |
| `analysis/OlhoffApproachExact/` | **historical / reference** | The "exact Olhoff 2014" line: separate FE, multiplicity detection and generalized-gradient code (`topopt_freq_exact.m`). Reachable through `run_topopt_from_json` for its own approach key. | **NO** |
| `analysis/OlhoffApproachExactOpus/` | **historical / reference** | An independent clean-room re-derivation from the paper (Python + experiments). Not wired into any MATLAB dispatch. | **NO** |
| `analysis/OlhoffRegularized/` | **superseded** | A globalized variant built on the `reproduction2007` primitives, with its own audit tree. Self-contained; nothing outside it calls it. | **NO** |
| `analysis/OlhoffReproduced2007/` | **historical / reference** | A thin runner exposing `reproduction2007` on the Yuksel benchmark geometries. Calls `repro2007_paths()`; adds no algorithm of its own. | **NO** |
| `analysis/olhoff_fixed_budget_audit/`, `analysis/olhoff_native_convergence/`, `analysis/olhoff_nested_mma_route_audit/`, `analysis/olhoff_practical_convergence_audit/` | **historical audit evidence** | Audit trees around the `reproduction2007` implementation. No solver of their own that the benchmark should call. | **NO** |

Nothing in this table is deleted. Each is preserved for provenance; each is
fenced off from the conference benchmark.

---

## 3. How the fence is enforced

Three independent mechanisms, all fail-closed:

1. **`analysis/OlhoffM4Reconstruction/olhoffm4_forbidden_paths.m`** enumerates
   every path above. `olhoffm4_assert_dispatch.m` refuses to return unless every
   function the import owns resolves *inside* the import root and *outside*
   every forbidden path. `olhoffm4_paths()` calls it before any solve, and also
   proves that `mmasub` is the **published** Svanberg copy rather than the
   `asfound` variant.

2. **`examples/Performance/conference_bench/confbench_preflight.m`** checks,
   before the first solve, that (a) no forbidden directory is on the MATLAB
   path, and (b) no Olhoff-family function name resolves into a forbidden tree.
   A failure stops the benchmark with nothing solved.

3. **The driver adds no forbidden path.**
   `examples/Performance/performance_comparison.m` adds only
   `tools/Matlab`, `analysis/three_method_parametric_study`,
   `examples/Performance/conference_bench` and
   `analysis/OlhoffM4Reconstruction`. It never calls
   `analysis/olhoff_stabilization_audit/run_stabilization_case.m`, never reads
   `final_campaign_profile.json`, and never dispatches through
   `run_topopt_from_json` for the Olhoff column.

Additionally, every run records the resolved `.m` file of each owned function
into `benchmark_manifest.json`, so the artifacts say which code produced the
numbers rather than which code was intended to.

---

## 4. The previous driver

`examples/Performance/performance_comparison.m` was replaced. The previous
version is preserved verbatim at
`examples/Performance/legacy_r3/performance_comparison_r3.m`; see the README
beside it. Its Olhoff column dispatched the **superseded** stabilization
profile, so its Olhoff rows are not conference numbers.
