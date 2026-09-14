# Conference performance benchmark — preflight

**This document is the current authorization.**
`examples/Performance/FINAL_CAMPAIGN_PREFLIGHT.md` authorized the R3 /
"final campaign" era benchmark and is **superseded**. It is preserved as
historical evidence with a banner saying so; only that banner was added.

| | |
|---|---|
| driver | `examples/Performance/performance_comparison.m` (rewritten, 436 lines; the previous 1528-line driver is at `legacy_r3/performance_comparison_r3.m`) |
| helpers | `examples/Performance/conference_bench/` |
| Olhoff implementation | `analysis/OlhoffM4Reconstruction/` (imported; see `IMPORT_MANIFEST.json`) |
| repo HEAD at run | `d0c78ca66f72b6041db908f808bd342621f5ced6`, branch `benchmark-methodology-r2` |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1, MACA64, Apple Accelerate BLAS (ILP64) |
| threads | `maxNumCompThreads = 1` for every measured run |
| memory | **not measured, not reported** |

---

## 1. Active configuration

The driver's whole control surface is one block at the top of the file. The
shipped default is the nine-resolution campaign, present as an editable literal:

```matlab
cfg.resolutions = [
    160   20
    240   30
    320   40
    400   50
    480   60
    560   70
    640   80
    720   90
    800  100
];
```

There is no manifest indirection, no `isFinalCampaign`, no protocol mode, no
environment variable, and no helper that can change the active mesh list. The
160×20 preflight below was produced by editing **only** that matrix to
`[160 20]` — a one-hunk diff against the shipped file, verified by `diff`.

`scientific_evidence` and `performance_campaign` are **derived** from `cfg`, not
declared beside it, so they cannot contradict what ran:

```
scientific_evidence  = isempty(cfg.maxOuterOverride) && all(N_e >= 3200)
performance_campaign = scientific_evidence && isequal(cfg.resolutions, the nine meshes)
```

3200 elements is 160×20, this project's documented mesh-resolution floor.

**Configuration precedence.** One direction only:

```
top-level cfg  ->  validated method configs  ->  run  ->  manifest records
```

The method configs are read from the artifacts that froze them
(`profile_freeze_manifest.json` for Proposed and Yuksel, `olhoffm4_config.m` for
the Du–Olhoff reconstruction) and are printed in full before launch. The
manifest is **output**: nothing reads it back.

---

## 2. Verified: the imported Du–Olhoff (M4) reconstruction

Source: `/Users/piotrek/Programming/Matlab/Olhoff` (not a git repository; state
identified by SHA-256 of every file in its solver tree, recorded in
`SOURCE_SHA256.txt`). 23 files imported; **22 byte-identical**, one carrying a
declared timing-instrumentation patch (`patches/olhoffOpt.timing-instrumentation.diff`:
one `hist` field, one `tic`, one `toc`).

Realization checks, all PASS at every active mesh:

| property | verified value |
|---|---|
| genuine nested MMA | `innerSolver='mma'`, `innerVar='drho'`, `mmaVariant='published'`, `offDiag=1`, `maxInner=500`, `minInner=5` |
| M4 multiplicity, frozen subN | `multRule='subspace'`, `subN=2`, no threshold classifier |
| fixed physical filter | `rminPhys=0.06`, `rminEl=NaN` derived at run time → 1.2 at 160×20; `filterMode='all'` |
| `tolInner` | `0.05` |
| outer RMS stopping semantics | `‖Δρ‖₂ < 0.05` at 160×20, i.e. per-element RMS `< 8.838835e−04`, constant across meshes; `outerGuard='settledmove'`; cap 400 |
| S2 continuation realization | `moveFamily='S2'`, `move₀=0.04`, ladder `[0.04 0.02 0.01 0.005]`, window 10, tol 5e−3, **`s2Signal` absent ⇒ legacy `beta` signal** (the design-driven `drms` trigger was measured and **not adopted**) |
| single thread, diagnostics off | `threads=1`, `diag=0` |

**Equivalence at 160×20** (`analysis/OlhoffM4Reconstruction/evidence/import_equivalence_160x20.json`),
bitwise on the raw IEEE-754 bytes of the final density field:

| run | implementation | `diag` | outer | Σ inner MMA | ω₁ | ρ SHA-256 | wall |
|---|---|---|---|---|---|---|---|
| A | audited **source** `algo/olhoffOpt.m` | true | 91 | 2241 | 169.49522702153845 | `22aa4447bc5f2855…` | 124.1 s |
| B | **import** `+frozen/algo/olhoffOpt.m` | true | 91 | 2241 | 169.49522702153845 | `22aa4447bc5f2855…` | 125.6 s |
| C | **import**, benchmark config | false | 91 | 2241 | 169.49522702153845 | `22aa4447bc5f2855…` | 120.8 s |

`A ≡ B` bitwise (ρ, ω, history, outer count): the import and its timing patch
are inert. `B ≡ C` bitwise: switching the per-iteration diagnostic recorder off
— the **only** benchmark-vs-audit configuration difference, made so the recorder
is not timed — changes nothing. The full configuration diff between the audited
struct and the benchmark struct is exactly `{diag, name, mmasubPath}`; 51 of 51
scientific fields are identical.

---

## 3. Verified: no superseded implementation can run

`analysis/OLHOFF_IMPLEMENTATION_STATUS.md` classifies all seven realizations.
Three independent fail-closed mechanisms, all PASS:

1. `olhoffm4_paths()` prepends only the folders this import owns, then
   `olhoffm4_assert_dispatch` proves each of the **20 owned function names**
   resolves inside `analysis/OlhoffM4Reconstruction` and outside every path in
   `olhoffm4_forbidden_paths.m`; it additionally proves `mmasub` is the
   **published** Svanberg copy. The resolved file of every name is written into
   `benchmark_manifest.json`.
2. The preflight checks that no forbidden directory is on the MATLAB path and
   that no Olhoff-family name (`olhoffOpt`, `model2D`, `assemble2D`, `eigSolve`,
   `genGrad`, `innerLoop`, `prepFilter`, `applyFilter`, `multRule`,
   `moveControl`, `deltaLambda`) resolves into a forbidden tree.
3. The driver adds no forbidden path, never calls `run_stabilization_case`,
   never reads `final_campaign_profile.json`, and never dispatches the Olhoff
   column through `run_topopt_from_json`.

Mechanics self-test **T2** proves the gate actually fails closed: with
`Matlab/reproduction2007/algo` deliberately prepended, `olhoffm4_assert_dispatch`
raises `olhoffm4_assert_dispatch:WrongImplementation`, and the guard then
resolves correctly once installed.

The core lives under `+frozen/` because `genpath` skips `+`-folders and
everything beneath them; a plain subfolder would shadow `tools/Matlab/mmasub.m`
under the `addpath(genpath(analysis))` used by `examples/Revision_v1/*.m` and
break the isolation guarantee `repro2007_verify_isolation.m` checks.

---

## 4. Method-native computational accounting

`timing_schema.json` is written beside every result set. Total wall time is the
common quantity; counts and component times are method-native and are **not**
mathematically identical across methods.

| | Count 1 | Count 2 | Time 1 | Time 2 |
|---|---|---|---|---|
| **Proposed** | reference eigenanalysis **solves** (always 1 — a solve, never an iteration) | SIMP iterations | eigenanalysis + preparation | SIMP loop |
| **Yuksel** | Stage-1 iterations | Stage-2 iterations | Stage-1 loop | Stage-2 loop |
| **Du–Olhoff (M4)** | outer iterations | cumulative nested MMA iterations | outer work **excluding** the nested MMA solve | nested MMA total |

`N_outer + N_inner` is never reported as a generic iteration count.

**Timing boundaries.** `total_wall_time_s` is a caller-side `tic/toc` around the
solve and nothing else. Outside it: CSV/JSON/MAT writing, plotting, table
formatting, LaTeX generation, the common E1/E2/E3 evaluator and topology
rendering. Inside the solver:

- Olhoff — a per-outer-iteration timer (`hist.tOuter`, the declared patch) plus
  the pre-existing `hist.tInner` around the nested MMA loop.
  `outer_time_excluding_inner_s = Σt_outer − Σt_inner`, so the nested solve
  cannot be double-counted. Nesting is asserted per iteration
  (`t_inner ≤ t_outer`, `t_eig + t_grad + t_inner ≤ t_outer`, `Σt_outer ≤ T_call`).
- Proposed — a dedicated timer around the single reference eigenanalysis
  (`stage1_reference_eigen_time_s`), asserted to be a sub-interval of Stage-1
  preparation, with the solve count asserted `== 1`.
- Yuksel — the solver's own disjoint stage timers; `N₁+N₂ = N_total` and
  `T₁+T₂ = optimization_loop_time` are both asserted.

**Two residuals, and they are different things.**
`timing_accounting_residual_s = T_total − (T₁+T₂+T_overhead)`; its components are
nested measured intervals, so it catches a mis-derived component.
`independent_crosscheck_residual_s` is the caller-side total minus the solver's
**own** self-reported wall time — two separate measurements of one interval —
so it catches a mis-nested timer that the identity cannot see. Predeclared
tolerances (1e−6 s / 1e−9 relative; 0.05 s / 5 % respectively) and the flags
`TIMING_ACCOUNTING_FAIL` / `TIMING_CROSSCHECK_FAIL` are exported with the values.

**Memory is out** of the contract: no RAM column in the primary table, detailed
table, CSV, LaTeX, scaling fits or preflight. The RSS sampler that
`run_topopt_from_json` would otherwise run at 10 Hz *inside* the timed loop is
now switched off explicitly (`benchmark.measure_memory = false`); the deprecated
detailed-CSV column reads `NOT_MEASURED`, never `0`.

> Reliable, method-independent peak-memory measurement was not available in the
> MATLAB environment; memory was omitted rather than reported with inconsistent
> semantics.

---

## 5. Mechanics smoke — PASS

`examples/Performance/conference_benchmark/smoke/`, meshes 40×6 and 60×8 with a
6-iteration outer budget. `scientific_evidence = false`,
`performance_campaign = false`. **Not citable as a result.**

All three methods ran, dispatched correctly, produced complete accounting, and
wrote every artifact. Timing residuals `0.000e+00` on all six rows. The scaling
fit **refused itself**: *"this run is not a complete performance campaign; a
scaling exponent must not be fitted to smoke or preflight data."*

`confbench_selftest.m` — 7/7 PASS (`smoke/mechanics_selftest.json`):

| id | test | result |
|---|---|---|
| T1 | a raising solver yields `RUN_ERROR`, not a crash | PASS |
| T2 | dispatch gate refuses a shadowing superseded implementation | PASS |
| T3 | all three methods produce the same 22-field record set; struct array builds | PASS |
| T4 | timing identity passes, and `TIMING_ACCOUNTING_FAIL` fires on a deliberately inconsistent record | PASS |
| T5 | no memory column in any exported table | PASS |
| T6 | the frozen Olhoff configuration rejects an odd `nely` | PASS |
| T7 | preflight refuses a mesh above 160×20 without acknowledgement | PASS |

---

## 6. Scientific preflight at 160×20 — PASS

`examples/Performance/conference_benchmark/preflight_160x20/`, run
2026-09-04T14:12:11+02:00. `scientific_evidence = true`,
`performance_campaign = false`. 29/29 preflight checks PASS. Warm-up at 48×6,
discarded. Nothing was tuned.

| Method | Count 1 | Count 2 | Time 1 [s] | Time 2 [s] | Total [s] | ω₁ | Status |
|---|---|---|---|---|---|---|---|
| Proposed | 1 | 107 | 0.154470 | 1.477544 | 1.836471 | 109.050082 | NATIVE_CONVERGED |
| Yuksel | 121 | 123 | 1.245253 | 1.602205 | 3.160339 | 157.278427 | NATIVE_CONVERGED |
| Du–Olhoff reconstruction (M4) | 91 | 2241 | 3.151542 | 123.406393 | 126.606877 | 169.495227 | NATIVE_CONVERGED |

**Proposed.** Stage-1 solve count = 1 (asserted). Stage-1 preparation
0.154470 s, of which the reference eigenanalysis alone is 0.034892 s (asserted a
sub-interval). SIMP: 107 iterations in 1.477544 s. Overhead 0.204457 s.
Identity residual `0.000e+00`.

**Yuksel.** 121 + 123 = 244 (identity asserted). Stage 1 1.245253 s, Stage 2
1.602205 s, and `T₁+T₂` equals the reported loop time (asserted non-overlapping).
Overhead 0.312881 s. Identity residual `4.441e−16` s.

**Du–Olhoff reconstruction (M4).** Imported implementation confirmed at run
time. 91 outer iterations; 2241 cumulative nested MMA iterations;
**24.63 inner iterations per outer**; outer-exclusive 3.151542 s (of which
eigenproblem 2.856397 s, sensitivities 0.229082 s, bookkeeping 0.066063 s);
nested MMA 123.406393 s; **inner time share 97.47 %**; per inner iteration
0.055068 s; overhead 0.048942 s. Converged natively at
`‖Δρ‖₂ = 2.358e−02 < ε = 5.0e−02` with the move limit settled at 0.01 (ladder
stage 3), multiplicity 2, no non-converged inner solve, volume 0.49999901.
Identity residual `0.000e+00`.

**Timing residuals.** Identity: `0.000e+00`, `4.441e−16`, `0.000e+00` s.
Independent cross-check: `6.430e−04`, `9.295e−04`, `1.600e−04` s. No flag raised.

**Gross-regression check** against the previous campaign
(`examples/Performance/final_campaign/`), for the two methods whose
implementation did not change:

| | previous campaign | this preflight |
|---|---|---|
| Proposed iterations / ω₁ native | 107 / 109.05008231141127 | 107 / 109.0500823 |
| Proposed common-evaluator ω₁ (E1) | 153.67521021173187 | 153.675210 |
| Yuksel stages / ω₁ native | 121 + 123 / 157.27842695834786 | 121 + 123 / 157.278427 |
| Yuksel common-evaluator ω₁ (E1) | 157.16681742808953 | 157.166817 |

Identical. The Olhoff row is deliberately **not** comparable: the previous
campaign's Olhoff column was the superseded fixed-1600-iteration stabilization
profile (ω₁ = 167.3355), and this one is the imported M4 reconstruction. Its
value matches the audited source bit for bit, and its ladder history, terminal
grayness (0.13402) and gap₁₂ (1.454 %) reproduce the audited 160×20 run.

Total wall times fell against the previous campaign for the dispatched methods
(Proposed 2.22 → 1.84 s, Yuksel 9.26 → 3.16 s), which is expected: the 10 Hz
`ps`-forking memory sampler no longer runs inside the timed loop.

---

## 7. Caveats that travel with every Du–Olhoff number

Written into `benchmark_results.json` metadata, `BENCHMARK_NOTES.md`, the CSV
header comments and the LaTeX caption:

> Du–Olhoff timings and iteration counts refer to the frozen reconstruction used
> in this study. Some continuation and inner-solver details are not uniquely
> specified by the original publication; therefore these values should be
> interpreted as representative measurements of this reconstruction rather than
> exact historical implementation timings.

The row is published, not censored. What is **not** claimed: an exact historical
runtime, an exact historical iteration count, or that the frozen continuation
policy is uniquely defined by Du & Olhoff.

**A limit the source audits impose, recorded here rather than discovered later.**
`audit_s2_design_continuation/REPORT.md` (verdict `S2_LADDER_ITSELF_DEFECTIVE`)
established that this method's **outer iteration count is an artifact of the
move-limit continuation trigger**: the same solver, filter, M4 and outer
tolerance give 91/104/131 outer iterations under one admissible trigger and
86/54/59 under another, with no change to the physics. That report states *"no
scaling exponent may be fitted to either set."* A fitted exponent for the
Du–Olhoff column would therefore describe **this reconstruction's continuation
schedule**, not the published method; `confbench_caveats.scaling` says so and it
is exported with any fit. This does not affect Proposed or Yuksel, and it does
not affect the measured total wall times, which are what the table is for.

That report also carries `PERFORMANCE_COMPARISON_INTEGRATION = NOT_AUTHORIZED`.
That verdict governs its own scope — whether the *design-driven `drms` trigger*
could be adopted. It was not adopted; the benchmark runs the legacy `beta`
signal, i.e. the realization those verdicts were issued against. The present
task's instruction is explicit that the measured Olhoff results are to be
published with the caveat attached rather than suppressed, and that is what is
done here.

---

## 8. Artifacts produced

`examples/Performance/conference_benchmark/<run label>/`:

```
conference_performance_table.csv      primary, method-native
conference_performance_table.tex      the same table, for the slide
conference_performance_detailed.csv   explicit method-specific field names, full precision
benchmark_results.json                every record
benchmark_manifest.json               exactly what ran, with hashes
timing_schema.json                    what each count and time means
BENCHMARK_NOTES.md                    the caveats
benchmark_records.mat                 raw records, including design fields
```

The preflight refuses to start if the output directory would land on
`examples/Performance/benchmark_results.json`,
`examples/Performance/table1_performance.csv` or
`examples/Performance/final_campaign/`. Nothing earlier was overwritten.

---

## 9. Verdict

Every mechanics test passes; the import is proved equivalent to the audited
source at scientific scale; the fail-closed dispatch is proved to fail closed;
all three methods converge natively at 160×20 with exact timing identities; and
the two methods whose implementations did not change reproduce the previous
campaign exactly.

The one remaining guard is deliberate: `cfg.confirmLongCampaign = false` in the
shipped driver, so pressing Run today stops at the preflight rather than
launching a multi-hour campaign. Launching the nine meshes requires setting that
flag — a decision for the operator, not for this preflight.

# CONFERENCE_BENCHMARK_PREFLIGHT_PASS

# NINE_MESH_PERFORMANCE_CAMPAIGN = NOT_AUTHORIZED

*Not authorized **by this task**, which is explicitly prohibited from launching
240×30 or larger campaign meshes and from launching the nine-mesh campaign. The
benchmark itself is ready: the preflight passes, and the only thing standing
between this state and the campaign is the operator setting
`cfg.confirmLongCampaign = true`.*
