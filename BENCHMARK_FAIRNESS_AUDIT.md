# Benchmark fairness audit: R3 adversarial revision

Audit date: 2026-08-26
Audited branch: `benchmark-methodology-r2`
Audited HEAD: `cf290fc7f9daf9da27bc8224f9585a0e1657bff1`
Protocol: `BENCHMARK_PROTOCOL_R3.md`
Manifest: `examples/Performance/benchmark_protocol_r3.json`

This is a methodology and documentation audit only. No performance campaign, new
optimization, parameter sweep, or algorithm modification was performed. Existing
`examples/Performance/table1_*`, `benchmark_results.json`, and
`performance_log.txt` remain **STALE / SUPERSEDED** for scientific use because they
contain pre-fix trajectories. Where explicitly stated, already-existing timing records
are used only for capacity planning, and frozen histories are used only to establish
algorithmic-regime or execution feasibility. Neither use selects a parameter by outcome.

## 1. Executive assessment

The adversarial revision resolves the common-evaluator and common-stopping semantics,
adds the missing 320×40 radius replication, guards scaling against the known Olhoff
multiplicity transition, and makes Yuksel totals and the controlled-serial claim
unmistakable. `PROPOSED_NATIVE_PROFILE_AUDIT.md` also closes methodological gate M1.

The resolved Proposed profile is the OC, A0 physical-load, single-mode SS baseline declared
and implemented on the pre-R3 `develop` lineage at `d051985` on 2026-07-14. Its immutable
A4 config SHA256 is `c5e8949a...1047`, unchanged through the July 29 audit at `310043e`.
This selection uses mathematical/manuscript provenance and a pre-R3 single-factor protocol,
not any comparative outcome. The March MMA profile is now explicitly a legacy experiment
profile. R3 remains prohibited until engineering gates make the current branch implement
and serialize the resolved profile exactly.

### Disposition of the four principal objections

| Finding | Status | Revision and remaining limitation |
|---|---|---|
| A1 — common evaluator may favor Proposed | **RESOLVED** | Native results are separate; common raw results use frozen E1, E2, and E3 method-family interpolation models; binary fields are a separate representation. E1 is primary for continuity with the pre-existing protocol and numerical regularity, not neutrality. A raw ordering is evaluator-robust only if unchanged across all three. |
| A2 — common threshold structurally censors Olhoff | **RESOLVED** | Experiment C is now harmonized stationarity and quality evaluation with exact status/censor semantics. A move-saturated Olhoff row is censored, not converged or failed, and cannot enter speed-to-common-convergence ratios. |
| A3 — native filter radii are attackable | **MITIGATED WITH DISCLOSED LIMITATION** | Native-radius performance is labelled as such; radius evidence uses frozen OAT ranges at 240×30 and a filter-only replication at 320×40. The three native values are not called equivalent and no universal radius is inferred. Olhoff 1.3 remains a reconstructed basin-supported value. |
| A4 — sensitivity only at 240×30 may favor Olhoff | **RESOLVED** | The exact radius levels are replicated at Yuksel's principal 320×40 mesh without duplicating the rest of Experiment D. |

### Disposition of weaker objections

| Objection | Status | Control |
|---|---|---|
| Volume-preserving binarization favors discrete designs | **RESOLVED** | Binary is separate from raw density, carries a grayness warning, and cannot alone rank methods. |
| W1 intersects an Olhoff multiplicity regime change | **RESOLVED** | Add W2=201:300, fit each window independently, report `Delta_p`, and never average differing regimes. |
| Yuksel stage-2 time can be mistaken for total cost | **RESOLVED** | All schemas require stage 1, stage 2, and explicit sums for time and iteration; only total is method-level practical performance. |
| Single-thread time could be read as best practical runtime | **RESOLVED** | Claims are limited to controlled serial computation; no multicore/GPU or fastest-implementation claim is made. |
| Proposed native profile is author-favorable | **RESOLVED WITH DISCLOSED LIMITATION** | The pre-R3 A0/A4 OC profile is primary. The executable `[0,1]` versus manuscript `[1e-3,1]` design bound and effective truncated versus declared symmetric filter boundary remain visible; no benchmark result selected the profile. |

## 2. Repository evidence and scope

The working tree was already dirty at the previous audit and the three R3 files were
untracked; all unrelated user changes were preserved. The relevant evidence hierarchy
was searched in the requested order:

1. current manuscript/revision material in `paper/reviews/`;
2. frozen earlier experiment/protocol configurations;
3. implementation defaults and their introduction commits;
4. revision documentation and configuration files;
5. git history and frozen Du–Olhoff provenance.

Key immutable evidence includes:

- Proposed implementation SHA256 `19399780…f171b82`;
- current algorithm-comparison supplement SHA256 `ff0c88f8…5484a8`, PDF creation
  2026-07-13;
- authoritative A0 implementation and A4 single-factor specification committed at
  `d051985` on 2026-07-14;
- A4 base configuration SHA256
  `c5e8949a318dd6ac657faf034ca85b2210d3a01976c37dab1602eec8ae341047`, unchanged
  through the pre-R3 performance audit at `310043e` on 2026-07-29;
- benchmark configuration first committed at `4582541` on 2026-03-06 and updated at
  `76b2894` on 2026-03-08;
- revision experiment profile committed at `2fb35be` on 2026-05-29;
- Yuksel paper SHA256 `c2b45479…851cf4`;
- Du–Olhoff paper SHA256 `b4dc8153…378bc4`;
- frozen reproduction provenance SHA256 `958c4b4f…a7aaa`.

The successful reproduction remains `OlhoffDu2007Repro`, `fig3a_best`, Eq. (22) LP,
with frozen regression identity. Nothing here upgrades it to recovered author code or
changes its byte-frozen algorithm.

## 3. Proposed native-setting provenance audit

### 3.1 Source-level chronology

| Source | Date / commit | What it establishes | Predates R3? |
|---|---|---|---|
| Initial implementation defaults, `topopt_freq.m` | 2026-02-12, `e32d3e8` | OC only; `Emin/E0=1e-9` from `Emin=1e-2,E0=1e7`; `rho_void/rho0=1e-6`; move .2; tolerance .01; cap 2000; uniform initial/reference design; mass-normalized eigenvector; physical `omega0^2 M(x)Phi0` load; omitted load sensitivity. Radius is an external argument; fallback is 0.05 physical. | yes |
| Benchmark profile introduced, `performance_comparison.json` | 2026-03-06, `4582541` | MMA, E/rho floors `1e-6`, solid baseline, density source `x`, normalize true, sensitivity radius 2.0 element, move .2, tolerance .004, cap 2000, mode-1 factor 1/mode-2 factor 0. | yes |
| Benchmark profile update | 2026-03-08, `76b2894` | Changes tolerance `.004 -> .003`, cap `2000 -> 10000`, and toggles a Heaviside key that is inactive on this sensitivity-filter dispatch. | yes |
| Revision-v1 simply-supported profile | 2026-05-29, `2fb35be` | OC, E/rho floors `1e-9`, initial baseline, normalize false, radius 2.0 element, move .2, tolerance .001, cap 2000. | yes |
| Current manuscript algorithm supplement | PDF created 2026-07-13 | Describes the Proposed quasi-static loop with an OC bisection update, sensitivity filter, frozen reference eigenpair, and omitted load sensitivity; it records that A0 still had to resolve the exact load. | yes |
| A0 authoritative formulation and executable correction | 2026-07-14, `d051985` | Fixes `F=omega0^2 M(x)Phi0`, solid frozen mass-normalized reference, deterministic phase, no nodal-density double scaling or load-norm rescaling, and explicit omitted/complete sensitivity. | yes |
| A4 “one and only” SS base profile | 2026-07-14, `d051985`; unchanged at `310043e` | OC, one mode, A0 load, solid reference, floors `1e-9`, linear mass, radius 2.0 element, move .2, tolerance .001, cap 2000, no Heaviside/continuation, uniform start. | yes |
| Pre-R3 legacy performance audit | 2026-07-29, `310043e` | Quarantines the March MMA comparison and states that it is not the manuscript's nominal Proposed profile. | yes |
| R3 candidate | 2026-08-26 | Initially copied the March benchmark profile; replaced by the July A0/A4 profile in this audit. | no (this is R3 documentation correction) |

The chronology is independent of benchmark quality. The July A0/A4 record is decisive
because it is complete, internally named as the authoritative one-factor baseline, and
predates R3. The July 31 R3-lineage ledger missed it because `develop` diverged at merge
base `b98cc963`; absence from the current branch is not absence from repository history.

### 3.2 Parameter-by-parameter finding

| Resolved value | Provenance | Contradictory legitimate value(s) | Finding |
|---|---|---|---|
| optimizer `OC` | original implementation; manuscript; May SS; A4 | MMA in March/clamped/building experiment profiles | resolved for manuscript/SS native; MMA remains a named auxiliary backend |
| `F=omega0^2 M(x)Phi0` | original physical load; A0 audit and implementation at `d051985` | obsolete nodal-density semi-harmonic path on current R3 branch | resolved methodology; engineering port/regression gate open |
| solid frozen reference, one mode | A0 and A4 | original/May initial reference; March dead mode-2 entry | resolved by explicit pre-R3 A0 decision |
| mass-normalized `Phi0`, no load-norm rescaling | A0/A4 | March `harmonic_normalize=true`, inactive for legacy semi-harmonic | resolved; obsolete boolean prohibited |
| omitted load sensitivity | original, manuscript, A4 | complete ablation variants | resolved; omission remains disclosed approximation |
| `pK=3`, linear mass `pM=1` | all executable lineage plus July mass decision | transient manuscript draft `pM=3` | resolved mathematical correction |
| sensitivity filter, `rmin=2.0` elements | March, May, manuscript SS and A4 | original generic `.05 m` fallback | resolved for SS; physical radius is mesh dependent |
| effective truncated boundary stencil | Proposed source | A4 JSON declares symmetric | resolved as historical effective behavior with disclosed implementation/declaration limitation |
| move `0.2` | original/default, March, May, A4 | none material | resolved |
| design bound `[0,1]` | original and A4 executable source | manuscript says `[1e-3,1]` | freeze effective `0`; manuscript inconsistency disclosed and distinct from material floors |
| tolerance `.001` | May SS, manuscript and A4 | `.004/.003` March; `.01` generic default | resolved for current SS profile |
| cap `2000` | original/default, March-6, May, A4 | `10000` March-8 | resolved as safety budget; cap hit is not convergence |
| `Emin/E0=1e-9`, `rho_min/rho0=1e-9` | May SS, manuscript SS, A4 | March/clamped `1e-6`; original mass floor `1e-6` | resolved for SS; case-specific alternatives remain documented |
| uniform `x=Vf` initialization | original, manuscript, A4 | none material | resolved and distinct from solid eigenpair reference |
| no continuation/Heaviside | manuscript and A4 | inactive March key | resolved; unsupported key prohibited |

### 3.3 Pre-result resolution rule applied

`PROPOSED_NATIVE_PROFILE_AUDIT.md` applies the pre-result rule and is the dated rationale.
The explicit manuscript/A0 definition prevails for formulation and OC; the immutable A4
pre-R3 protocol supplies manuscript-silent SS controls; predating implementation defaults
supply the effective design bound and OC internals. Rejected alternatives and nonperformance
rationales are preserved there. No benchmark output broke a tie.

## 4. Native radius provenance and interpretation

| Method/value | Exact provenance | Paper/default/local distinction | Unit limitation |
|---|---|---|---|
| Proposed `2.0` | March `4582541`, May SS `2fb35be`, manuscript SS, and immutable A4 base `d051985` | resolved pre-R3 manuscript/SS value; effective operator is truncated despite the config's symmetric label | constant element units, so `r_phys=2*(8/nelx)` |
| Yuksel `2.5` | published simply supported 320×40 case, paper p. 3205/Section 6.1; local `run_simply_supported.m` | published case value; local generic function fallback 8.75 is not used | constant element units in R3; published physical interpretation is mesh-specific |
| Olhoff `1.3` | frozen `fig3a_best` reconstruction and `Matlab/reproduction2007/PROVENANCE.md` | paper does not specify radius; inferred from successful reconstruction basin | constant element units; `r_phys=1.3*(8/nelx)` |

These radii use different filter operators and histories and are not mathematically
equivalent. B reports native-radius performance. D reports radius sensitivity at both
240×30 and 320×40. The protocol never backs out one supposedly fair radius from results.

## 5. Evaluation-model decision

Native evaluation answers what each method optimized/reported:
`objective_native` and `omega{1..3}_native`. Common evaluation answers what happens
when the same density representation is placed in one of three frozen external models.
Neither substitutes for the other.

Let `g(z)=z` above 0.1 and `z^6` at/below 0.1. The complete small set is:

| ID | Stiffness | Mass | Role/bias disclosure |
|---|---|---|---|
| E1 | `1e-6+(1-1e-6)z^3` | `1e-6+(1-1e-6)z` | preregistered primary; pre-existing, regular on arbitrary fields, Proposed-like |
| E2 | `1e-9+(1-1e-9)z^3` | `1e-9+(1-1e-9)g(z)` | Yuksel-family interpolation |
| E3 | with `z3=max(z,1e-3)`: `z3^3` | `g(z3)` | reconstructed Olhoff-family interpolation/admissible lower bound |

Everything else—mesh, Q4 integration, consistent mass, supports, modes, eigensolver start,
tolerance, and status checks—is common. E1 stays primary to preserve a choice made before
R3 results and because it is numerically regular, not because it is neutral. Reporting E2
and E3 is mandatory, not optional sensitivity. If the cross-method ordering changes,
the conclusion is model dependent and there is no single common-raw ranking.

Operationally, “evaluator-robust” requires the sign of every claimed pairwise frequency
difference to agree under E1, E2, and E3. R3 invents no magnitude/tie cutoff; near-zero
differences are printed and interpreted cautiously. Any sign reversal makes that claim
model dependent.

For binary evaluation, stable rank/tie-breaking creates exactly the target number of
solid elements. The same E1–E3 models evaluate that separate indicator. This explicitly
tests representation sensitivity but cannot establish ranking alone because thresholding
can reward a method that is already discrete and alter a gray method more severely.

Exact namespaces are `omega*_common_raw_E1..E3` and
`omega*_common_binary_E1..E3`; an unlabeled `omega_common` is invalid.

## 6. Harmonized stationarity and censoring

Experiment C no longer claims universal time-to-convergence. It asks: under the same
external diagnostics, what quality/stationarity evidence does each trajectory provide?

The frozen rule remains `d_infinity<=.003`, relative volume residual `<=.001`, for 10
consecutive successful final-stage iterations. Retaining it preserves pre-registration;
reinterpreting it as a diagnostic avoids pretending that Olhoff's move .005 and another
method's stopping norm are interchangeable.

Exact harmonized statuses are:

1. `SATISFIED`: first valid persistent window exists;
2. `NOT_SATISFIED_WITHIN_RECORDED_TRAJECTORY`: a valid finite recorded trajectory ends
   without the criterion and without a protocol budget censoring the intended observation;
3. `CENSORED_BY_ITERATION_BUDGET`: observation stops at a declared budget before the
   criterion is seen;
4. `SOLVER_FAILURE`: a required solve/update fails; failure takes precedence;
5. `NOT_APPLICABLE`: no scientifically valid criterion exists for that record/path.

A move-saturated Olhoff budget endpoint is censored. It is not converged; the threshold
miss alone is not solver failure. Valid evidence is common quality at the labelled cap,
the `d_infinity/rV` history, native stop evidence, and A's fixed-work scaling. Invalid
evidence is time-to-`k_star`, a speedup against P/Y, or convergence-speed ordering from
unequal horizons. Pairwise timing remains valid only when both rows are `SATISFIED` on the
same mesh and boundary.

The already-frozen native-radius `fig3a_best` 240×30 history remains move-saturated at
`d_infinity=0.005` through its 1600-iteration budget. If replay validation confirms that
identity for its R3 history row, its exact common status is
`CENSORED_BY_ITERATION_BUDGET`; no Proposed/Olhoff or Yuksel/Olhoff common-convergence
speedup exists for that row. Other meshes are classified from their own trajectories.

## 7. Fixed-work regime audit

The frozen reproduction regression documents the `fig3a_best` multiplicity transition
at iteration 95. Therefore the old `W1=11:110` mixes the simple-mode and coalesced regimes.
The concern is real even without looking at new timing.

Adopt `W2=201:300` because it:

- has the same 100-iteration length as W1;
- is wholly after the already-known iteration-95 Olhoff transition;
- lies far below the Olhoff native budget 1600;
- can be executed by Proposed and Yuksel final-stage kernels using the existing concept of
  stop-only extension, subject to a longer prefix/kernel gate;
- can execute Yuksel stage 1 only through a dedicated fixed-work mode that suppresses its
  handoff exit while leaving the compliance kernel unchanged.

The static stop audit places the Yuksel stage-1 handoff test at the end of the loop body,
shows that history arrays are sized by the iteration budget, and shows continuation is
inactive in this profile. This establishes structural feasibility of a 300-iteration
fixed-work kernel path; the dedicated prefix/component-call test remains mandatory before
the path is scientifically valid.

Old histories show that some native paths end before iteration 300; they are used here
only to establish the need for extension, never to choose a parameter. Existing extension
evidence validates final-stage prefixes but does not validate the Yuksel stage-1 handoff
suppression or every path through iteration 300. Those are engineering gates. If a gate
fails, the relevant W2 row is `NOT_APPLICABLE`; a method-specific substitute window is
forbidden.

Each method/stage/window receives its own `C,p,R2` fit. Report
`Delta_p=p_W2-p_W1`, bootstrap interval, per-mesh window ratios, and residuals. There is
no unsupported universal acceptance threshold. A visible/statistically detectable
regime difference is reported as such; exponents are not pooled.

## 8. Filter robustness on two meshes

At 240×30 the original OAT stays frozen:

| Method | Radius levels (elements) | Other retained OAT dimensions |
|---|---|---|
| Proposed | 1.5, 2.0, 2.5 | move; native tolerance centered on `.001` (`.0005,.001,.002`) |
| Yuksel | 2.0, 2.5, 3.0 | move; both stage tolerances |
| OlhoffDu2007Repro | 1.1, 1.3, 1.5 | move; multiplicity tolerance |

At 320×40, repeat only the three radius levels for each method. This is nine unique
conditions, of which three native centers reuse B, so it adds six trajectories. It does
not duplicate move/tolerance/multiplicity OAT and is not a tuning grid.

Common response metrics on both meshes are native and E1–E3 raw/binary frequencies,
volume, grayness, topology/connectivity/features, solver/stop status, and saturation.
Proposed adds native objective, OC-bisection/volume status and move activity. Yuksel adds both
stage counts/handoff, moving compliance/load/mode-change, OC status and move activity.
Neither receives an artificial multiplicity `N`. Olhoff records omega1–3, actual omega1/2
relative gap, `N`, last-50 coalescence persistence, LP statuses, topology/connectivity,
volume, failed LPs, move saturation and cap status.

Native-radius performance is a B result. Radius-sensitivity is D evidence. A good
off-center result cannot retroactively replace a native center.

## 9. Yuksel timing and iteration semantics

The implementation is sequential and its stages are not interchangeable work units:

\[
T_{Yuksel,total}=T_{stage1}+T_{stage2},\qquad
iter_{total}=iter_{stage1}+iter_{stage2}.
\]

Table A gives separate stage scaling rows and never pools them using an observed stage
mix. Tables B/C always print both stages and their total. Practical method-level claims
use `T_Yuksel_total`; stage-specific times are diagnostic. A stage-2 ratio cannot be
captioned, discussed, or computed as whole-method speedup.

## 10. Controlled serial claim

One-thread execution is justified because it constrains compute resources, reduces
hidden parallel scheduling differences, and makes algorithmic kernel comparisons more
interpretable. The resulting claim is **controlled serial computational comparison**.
It does not show the fastest achievable implementation, best production deployment,
parallel scalability, or modern multicore/GPU runtime.

A supplementary practical-runtime campaign is unnecessary for the current paper claim
and would introduce hardware/backend optimization as another research question. It is
required only if the manuscript later makes a practical parallel/hardware-superiority
claim.

## 11. Publication tables frozen before results

### Table A — controlled serial computational scaling

Required columns: method/stage; W1/W2; radius/operator/unit; mesh/element count; measured
iterations and repetitions; per-mesh median/MAD/CV/range; `C`; `p`; slope interval;
`R2`; adjusted `R2`; residual summary; component shares; validity. A companion block
contains `Delta_p`, interval, and W2/W1 ratios.

### Table B — method-native practical performance

Required columns: method/mesh; native-profile ID; provenance status; native objective,
definition and direction; omega1–3 native; E1–E3 common raw and binary frequencies at
native endpoint; volume/grayness/topology; stage1/stage2/total iterations; outer/inner;
stage1/stage2/Yuksel-total time; native wall median/range; RSS; native stop/reason; solver
status. It contains no cross-method speedup.

### Table C — harmonized stationarity and quality

Required columns: method/mesh; native endpoint/class; recorded horizon; `k_star`;
criterion status; censor reason/budget; endpoint `d_infinity,rV`; E1–E3 raw and binary
frequency columns; grayness; eligible optimization/evaluator/combined times; eligible
pairwise speedup. Every ineligible speedup is an em dash plus reason.

### Table D / supplement — robustness

Required columns: method/mesh/parameter/level; fixed-profile ID; native and all common
representations; gaps; volume/grayness; topology/connectivity/features; stages/iterations;
descriptive wall; native/common status; solver status; the method-specific metrics in
Section 8. Panels identify 240×30 OAT, 320×40 radius replication, and E1–E3 evaluator
dependence.

## 12. Campaign size and cost

| Category | Count |
|---|---:|
| Fixed-work timing | 220 path invocations: 20 cells × (1 warm-up + 10 measured), each yielding W1 and W2; 400 measured segments |
| Fixed-work memory | 60 fresh-process invocations |
| Native full optimization | 36 runs: 9 cells × (1 history + 3 measured) |
| C discovery/extension | at most 6 runs; reuse B and Repro histories |
| C accepted-endpoint timing | at most 99 runs if all 9 cells satisfy; zero repeat timing for censored cells |
| 240×30 sensitivity | 18 new runs plus 3 reused centers |
| 320×40 radius replication | 6 new runs plus 3 reused centers |
| Engineering/preflight checks | approximately 12 |
| External common FE calls | 54 at B endpoints; up to 54 distinct C endpoints; 180 for 30 D endpoint sets, with 36 center calls reusable |

The conservative upper bound is **445 optimization/path invocations** plus approximately
12 engineering checks, or about **457 total executions/checks**. The 445 include 280
short timing/memory paths and 24 new sensitivity trajectories. The added evaluator work
is endpoint-only, not optimization reruns.

Existing capacity evidence only: post-fix native-radius Repro 240×30 is roughly 141–164 s;
superseded files place Proposed around .02–.12 s/iteration and Yuksel around
.017–.077 s/iteration over observed meshes; the 480×60 and separated-stage upper costs
are unmeasured. Accounting for paths through iteration 300, six new radius runs and three
endpoint evaluator models gives an intentionally broad **8–14 serial wall-clock hour**
planning range plus validation. No run was executed to refine it and it predicts no
ranking.

If reduction is needed, remove or shrink the 60-run memory campaign unless a memory claim
remains, then defer non-radius OAT dimensions. Do not remove the E1–E3 set, 320×40 radius
replication, W2, or censoring safeguards.

## 13. Configuration and engineering requirements

The manifest must fail closed on legacy shared knobs and enforce method namespaces,
radius units/operator, separate design/material lower bounds, exact implementation
identity, stop/failure precedence, stage/outer count identities, thread state, and unique
native/common result names.

Engineering blockers are:

1. make LP/solver failure take precedence over any zero-increment stop label;
2. implement W1/W2 segment timing for four paths without changing their kernels;
3. pass prefix and component-call invariance through iteration 300, including dedicated
   Yuksel stage-1 handoff suppression;
4. implement and validate common E1–E3 evaluation for raw and binary representations;
5. enforce one thread for all methods and capture the environment;
6. implement fail-closed manifest validation and exact resolved-config serialization;
7. implement C endpoint replay/checksum and the five exact criterion statuses;
8. implement topology/connectivity and method-specific D diagnostics;
9. update table writers to the frozen schemas and Yuksel sums;
10. create a clean freeze commit and complete hash inventory.

These tasks implement an already-defined protocol. They do not authorize changes to any
optimization algorithm.

## 14. Methodological status

M1 is **closed** by `PROPOSED_NATIVE_PROFILE_AUDIT.md`. The synchronized profile ID is
`proposed_manuscript_ss_oc_a0_2026-07-14`; the March MMA candidate is rejected as a legacy
experiment profile. The decision is a pre-result documentation correction based on the
pre-R3 A0/A4 record, not a performance-tuned change.

Engineering gates remain open. Most importantly, the current R3 branch must restore and
regression-test the A0 physical load, serialize the effective design bound/filter operator,
and pass the existing instrumentation/evaluator/freeze gates. This status authorizes no
algorithm change or campaign.

## 15. Final hostile-reviewer test

> Could a hostile reviewer still reasonably argue that any remaining protocol choice
> systematically favors Proposed, Yuksel, or Olhoff?

- **Proposed:** the strongest remaining criticism is that the resolved A0/A4 declaration
  was made after legacy comparisons existed and that its design-bound/filter declarations
  still disagree with effective code. The defense is that the exact OC profile predates R3,
  was frozen for a one-factor experiment, matches the original/formal algorithm, and was
  selected without outcome evidence. Mandatory E2/E3 results prevent the Proposed-like E1
  evaluator alone from determining a raw conclusion.
- **Yuksel:** its published 320×40 context is now represented in radius robustness, and
  its total two-stage cost cannot be replaced by stage 2. E2 is included without being
  made uniquely primary.
- **Olhoff:** native 1.3 remains reconstruction-supported rather than paper-specified, and
  move-saturated censoring limits what can be said. These are disclosed limitations, not
  hidden advantages. W2 prevents its known multiplicity transition from being averaged
  invisibly into the only scaling exponent; E3 exposes its evaluation model.

A hostile reviewer can still dispute the scientific relevance of native radii,
single-thread execution, the chosen model family, or the disclosed Proposed implementation
mismatches. They cannot reasonably claim the profile was silently selected from R3 ranking:
its source commit/hash, alternatives, limits, invalid comparisons, and revision rules are
frozen in advance. Execution remains forbidden until engineering identity is proven.

**READY AFTER ENGINEERING FIXES — DO NOT EXECUTE**
