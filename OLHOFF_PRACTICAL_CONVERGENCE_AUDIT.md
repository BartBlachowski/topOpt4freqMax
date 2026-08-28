# Olhoff practical-convergence and stopping-criterion audit

## 1. Executive verdict

The current `maxOuter=1600` behavior is not a native convergence result.  It is
a safety-budget exhaustion caused by a stopping test that asks for
`max(abs(drho)) < 0.001` while successful LP steps almost always contain an
element at the `0.005` move limit.  The authoritative 240x30, `rmin=1.3`, LP
trajectory is move-saturated in **1600/1600** outer iterations.  Its volume is
feasible and its first frequency is already within 0.178% of the best validated
late value at iteration 400, yet the native design-infinity-norm test cannot
fire.

The evidence does **not**, however, justify replacing 1600 with a finalized
native rule yet.  The saved trajectories contain complete spectral, volume,
LP-status and maximum-update histories, but only final density fields.  They do
not contain per-iteration RMS density updates, moving-element fractions, or
arbitrary topology checkpoints.  Those missing quantities are exactly what is
needed to distinguish a harmless move-saturated local update from continuing
global design drift.  Scalar-only rules also give demonstrable false positives
before delayed mode-ordering events.

The practical iteration scale supported by the authoritative trajectory is:

- 99% of the late-reference `omega1` is reached and then retained at iteration
  225;
- 99.5% is reached and then retained at iteration 321;
- the residual late-reference loss is 0.178% at 400 and 0.125% at 800;
- 99.9% is never subsequently retained by every recorded pre-update state,
  because the LP realization settles into a persistent two-cycle of about
  0.124% amplitude.

Thus the evidence supports an **order of a few hundred iterations**, not 1600,
for the authoritative 240x30 run if a 0.2--0.5% residual spectral uncertainty is
scientifically acceptable.  It does not support one universal iteration count
or a robust online detector across mesh/configuration changes.

## 2. Evidence inventory

### WP0 freeze record

- Branch: `benchmark-methodology-r2`
- HEAD at audit start: `cb6353feb941f12b2aaa927e622649e1ccc926f7`
- Working tree at audit start: dirty.  Existing modified/untracked benchmark,
  runner, equivalence, and diagnostic work was preserved.  This audit added
  only this report and `analysis/olhoff_practical_convergence_audit/`.
- During the audit, additional changes to generated performance artifacts were
  observed in the shared worktree; they were not edited or reverted here.

The SHA-256 provenance check matched **all 61/61** entries in
`Matlab/reproduction2007/SOURCE_SHA256.txt`, including the documented
`CLAUDE.md -> SOURCE_CLAUDE.md` and `results/* -> baseline/*` mappings.  The
frozen numerical directories `algo/`, `fem/`, `filter/`, and `mma/` remain
unchanged.  Modified `runner/` files are repository integration code and are
excluded from that manifest by construction.

The authoritative original and migrated artifacts are byte-identical:

| Artifact | SHA-256 |
|---|---|
| `lp240_rmin1.3.mat` | `7f452b3f09268404bcdd4d83b0ebec36576e563e132b790798aecdf58e7c303c` |
| `FIG4_definitive.mat` | `6188febf91c8d3f843a2aab734d84059751337f3f878c5f8a62ed991466d7133` |
| `FINAL_lp_240x30.mat` | `696f8cfb98f90f596a455124ab33aa139f8fff406c3557f5948144b01d6239ef` |

The frozen numerical tree hash independently recorded by the benchmark-path
equivalence proof is
`d0fcc873310aeea504a84bc6b93f484b073ceecc06685cf304e1df73f82a8747`.

### Trajectories used

The full configuration and validity inventory is machine-readable in
`analysis/olhoff_practical_convergence_audit/results/artifact_inventory.csv`.
The principal evidence is:

| Role | Mesh | `rmin` (elements) | move | horizon | LP failures | saved history |
|---|---:|---:|---:|---:|---:|---|
| authoritative reproduction | 240x30 | 1.3 | 0.005 | 1600 | 0 | spectrum, `N`, max update, volume, LP status |
| radius controls | 240x30 | 1.1, 1.5, 1.8, 2.2 | 0.005 | 1600 each | 0 | same scalar history |
| cross-resolution/radius controls | 160x20 | 1.1, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0 | 0.005 | 1600 each | 0 | same scalar history |
| Fig. 4 move control | 240x30 | 1.3 | 0.02 | 400 | 0 | same scalar history + final density |
| Fig. 4 move endpoints | 240x30 | 1.3 | 0.01, 0.02, 0.03, 0.05 | 100 | 0 | prefix history + final density |
| move family | 160x20 | 3.0 | 0.001--0.02 | 400--4000 | 0 | same scalar history |
| current R3 profile | 160x20--400x50 | 2.0 | 0.005 | 627--1600 | one at 400x50 | exact checkpoints and final state only |

The current R3 paths are bit-identical to direct clean-room calls at all four
meshes.  The 160x20, 240x30, and 320x40 rows hit the cap at 1600.  The 400x50
row fails its LP at iteration 627 and is not a valid convergence/timing row.

No existing clean-room artifact contains a full density trajectory.  Therefore
RMS updates, moving-element fractions, and density/topology distance to an
arbitrary stopping iteration are unavailable retrospectively.  They are not
silently imputed.

## 3. Why current runs reach 1600

In the frozen `olhoffOpt.m`, after a successful subproblem solve,

\[
\rho^{k+1}=\operatorname{clip}(\rho^k+\Delta\rho^k),\qquad
d_\infty^k=\max_e|\Delta\rho_e^k|.
\]

The only native convergence break is the strict test

\[
d_\infty^k < \texttt{tolOuter}.
\]

With the performance configuration, `tolOuter=0.001` and `move=0.005`.
Therefore any single element attaining the move bound prevents convergence.
For the authoritative trajectory,

\[
d_\infty^k=0.005\quad\text{for all }k=1,\ldots,1600.
\]

The status logic is:

1. `linprog` succeeds only when `flag==1` and returns a step.
2. If it fails or returns no solution, `innerLoopLP` sets
   `drho=zeros(Ne,1)` and `innerConv=false`.
3. The frozen loop then sees `d_inf=0 < tolOuter` and breaks.  Without the
   runner classifier this looks like convergence even though it is failure.
4. The runner correctly applies
   `SOLVER_FAILURE > CONVERGED > CAP_HIT`: final subproblem failure is
   `SOLVER_FAILURE`; otherwise a taken native break is `CONVERGED`; otherwise
   `nOuter>=maxOuter` is `CAP_HIT`.

The current 400x50 profile demonstrates the failure path exactly: iteration
627 has LP flag 0, `innerConv=false`, and `d_inf=0`; its status is correctly
`SOLVER_FAILURE`, not convergence.

The evidence cannot determine how many elements dominate `d_inf` late in the
authoritative run because the elementwise updates were not saved.  It does
prove that the maximum is disconnected from practical spectral stationarity.
Consequently, **`max|Delta rho| < tolOuter` is not a useful practical
convergence test for this LP realization**.  It should remain telemetry, not a
native stopping component.

The complete authoritative iteration table is
`analysis/olhoff_practical_convergence_audit/results/trajectory_metrics.csv`,
and the overview plot is
`analysis/olhoff_practical_convergence_audit/results/authoritative_240x30_diagnostics.png`.

## 4. Late-stage objective gain

For retrospective diagnostics only, the authoritative reference is the maximum
validated `omega1` over the final 10% of the trajectory plus the final
post-update state: **170.471473 rad/s**.  It is close to, but deliberately not
defined as, the terminal value 170.470909 rad/s.

| iteration | `omega1` | retained | later gain (rad/s) | late-reference loss |
|---:|---:|---:|---:|---:|
| 50 | 105.0888 | 61.646% | 65.3827 | 38.354% |
| 100 | 150.3845 | 88.217% | 20.0870 | 11.783% |
| 150 | 162.6144 | 95.391% | 7.8571 | 4.609% |
| 200 | 167.7998 | 98.433% | 2.6717 | 1.567% |
| 250 | 169.0811 | 99.184% | 1.3903 | 0.816% |
| 300 | 169.4684 | 99.412% | 1.0030 | 0.588% |
| 400 | 170.1676 | 99.822% | 0.3039 | 0.178% |
| 800 | 170.2576 | 99.875% | 0.2138 | 0.125% |

The earliest iterations that reach and subsequently retain each fraction are
146 (95%), 179 (97.5%), 225 (99%), and 321 (99.5%).  No recorded pre-update
state reaches and subsequently retains 99.9%, because the late two-cycle
continues through iteration 1600.  These points are retrospective value-loss
measures, not convergence declarations.

For the current `rmin=2` R3 profile, exact checkpoints show:

| mesh | gain 200 -> last valid checkpoint | gain 400 -> last valid checkpoint | interpretation |
|---|---:|---:|---|
| 160x20 | +0.0309 (0.019%) to 1600 | exactly 0 to 1600 | 1200 repeated iterations add no recorded `omega1` |
| 240x30 | +1.1185 (0.670%) to 1600 | +0.00359 (0.0021%) to 1600 | essentially all gain complete by 400 |
| 320x40 | +1.3744 (0.817%) to 1600 | +0.2936 (0.174%) to 1600 | small but nonzero late gain |
| 400x50 | +1.1067 (0.656%) to 400 | later state invalid | LP failure at 627; no terminal reference |

This independently confirms that 1600 is excessive, but also shows mesh
dependence in the meaningful tail.

## 5. Design/topology evolution

Only one exact same-trajectory topology checkpoint pair exists: the 240x30,
`rmin=1.3`, `move=0.02` 100-iteration artifact is bit-identical over its scalar
prefix to the 400-iteration Fig. 4 run.  Comparing their post-update densities:

- normalized/mean L1 difference: 0.01138;
- RMS L2 difference: 0.04782;
- 13.57%, 5.61%, and 2.81% of elements differ by more than 0.01, 0.05, and
  0.10, respectively;
- binary disagreement at density 0.5: 0.958%;
- both thresholded designs have one left-to-right connected main component;
- `omega1` at 100 is 170.87496 versus 170.74489 at 400, so the later density
  polishing does not improve the first frequency.

This is evidence that objective stationarity can coexist with continuing gray
density evolution.  It is not an authoritative `move=0.005` topology result and
is not used to set a tolerance.  The requested 100/150/200/... topology series
for the authoritative run does not exist and cannot be reconstructed from its
saved MAT file.

## 6. Multiplicity evolution

The authoritative run first enters `N>=2` at iteration 95.  Twenty consecutive
bimodal states are first observable at iteration 114; a relative first gap below
1% for 20 consecutive iterations is first observable at 126.  At those points,
8.70% and 7.27% of late-reference `omega1` improvement still remain.  Entering
the bimodal basin is therefore necessary but not sufficient for practical
convergence.

The final authoritative gap is 0.232% and the last 50 states are all `N=2`.
Nevertheless `omega3` still undergoes a large mode-ordering episode around
iterations 340--390, and changes from 288.0 near the first attractive scalar
stop at 397 to about 285.2 at the validated final design.  A practical rule must
observe more than `N` or the first eigengap.

Other valid long histories demonstrate why persistence alone is insufficient:

- 240x30 `rmin=1.8` appears scalar-stable and bimodal, then `omega1` drops to
  126.04 at iteration 338 before recovering;
- 240x30 `rmin=1.5` later drops to 83.02 at iteration 389 and finishes simple;
- the current `rmin=2` profile finishes `N=1` at 160x20 and 240x30, `N=2` at
  320x40, and becomes violently mode-order unstable before the 400x50 LP
  failure.

The current performance profile therefore does **not** support a general claim
of true bimodality.  It is explicitly an `rmin=2` operating point, not the
`rmin=1.3` paper-reproduction profile.

## 7. Candidate convergence diagnostics

The offline analysis evaluated:

\[
r_{\omega,1}(k)=\frac{|\omega_1^k-\omega_1^{k-1}|}{|\omega_1^k|},
\qquad
r_{\omega,q}(k)=\frac{|\omega_1^k-\omega_1^{k-q}|}{|\omega_1^k|},
\]

for `q=5,10,20`, plus a more conservative window envelope

\[
B_{\omega,q}(k)=
\frac{\max_{j=k-q+1}^k\omega_1^j-
      \min_{j=k-q+1}^k\omega_1^j}
     {|\omega_1^k|}.
\]

`N`, eigengap, `omega2`, `omega3`, volume residual, LP status, and
`max|Delta rho|` were tracked.  Objective tolerances from `1e-2` to `1e-5`,
windows 5/10/20, modal requirements from none through bimodal gaps of
5%/2%/1%/0.5%, and persistence 5/10/20 were replayed.

RMS density update and moving-element fraction were not evaluated because the
required elementwise history is absent.  This is an evidence limitation, not a
reason to substitute the maximum update.

The authoritative late state is a two-cycle: the one-step relative change stays
near 0.124%, while even lags 10 and 20 become nearly zero.  Therefore
`r_omega,10` or `r_omega,20` alone can report false stationarity by comparing
the same phase of a cycle.  The window envelope detects the cycle but needs a
scientific amplitude tolerance.

Volume is not discriminating: its relative residual is about machine precision
throughout the useful trajectories.  It remains a feasibility prerequisite,
not the convergence signal.

## 8. Offline stopping replay

Representative authoritative results with 10-iteration persistence and a
stable `N>=2`, gap <=1% modal state are:

| objective test | tolerance | fires | loss at fire | worst later retained fraction |
|---|---:|---:|---:|---:|
| endpoint `q=20` | 0.1% | 265 | 0.584% | 99.265% |
| endpoint `q=20` | 0.05% | 408 | 0.172% | 99.826% |
| endpoint `q=20` | 0.02% | 419 | 0.0459% | 99.826% |
| envelope `q=20` | 0.5% | 237 | 0.732% | 99.129% |
| envelope `q=20` | 0.2% | 397 | 0.0562% | 99.821% |
| envelope `q=20` | 0.1% | never | -- | -- |

The apparently strongest simple shadow rule,

```text
B_omega,20 <= 0.002
AND N stable and >= 2
AND gap <= 0.01
AND relative volume residual <= 1e-8
FOR 10 consecutive iterations
```

fires at 397 on the authoritative trajectory.  It is **not recommended as a
native rule**: it fires at 237 on the 160x20 `rmin=1.5` trajectory before a
later 1.9% objective deterioration, at 491 on 160x20 `rmin=1.2` with 0.55%
late-reference loss, and does not fire on every intended bimodal trajectory.
Looser variants fire before the 25--51% delayed collapses in the 240x30 radius
controls.  Increasing persistence from 10 to 20 does not prevent all such
false positives.

All replay results, including non-firing combinations, are in
`analysis/olhoff_practical_convergence_audit/results/stopping_rule_replay.csv`.
Candidate-stop topology distance is marked unavailable rather than fabricated.

## 9. Move-limit interaction

Both the stopping rule and conservative move contribute to the observed count:

- the stopping rule is primary: the 240x30 `move=0.02` run is still at the move
  bound in 399/400 iterations and also hits its cap;
- the move controls the trajectory timescale: corresponding 160x20 families
  were budgeted approximately as 400 (`move=0.02`), 800 (`0.01`), 1600
  (`0.005`), 3000 (`0.002`), and 4000 (`0.001`), and nearly all remain
  move-saturated;
- those endpoints do not share one basin, so their counts cannot be divided by
  move and called equivalent convergence times.

The high-move evidence verifies the safety concern.  At 240x30 `move=0.03`,
the history reaches `omega1=7.48` at iteration 55 and spends nine iterations
below 50; `move=0.05` reaches 16.55 at iteration 26 and spends seven iterations
below 50.  Their last 50 iterations are bimodal only 8% and 0% of the time.
The saved final thresholded states happen to reconnect, so final-only
connectivity is not evidence that the trajectory remained connected.

`move=0.02` is the largest existing setting with a smooth connected Fig. 4
trajectory, but changing the native performance move is not justified by this
audit.  It would change the operating trajectory and needs its own prospective
validation.  `move=0.005` remains defensible for reproduction quality, not as a
reason to retain the current stopping test.

## 10. Cross-resolution behavior

The same candidate cannot be replayed at 320x40 or 400x50 because the full
histories were discarded after the equivalence calculations.  Exact
cross-resolution checkpoints nonetheless show that one iteration count is not
robust:

- 160x20 `rmin=2` is spectrally unchanged from 400 through 1600;
- 240x30 changes only 0.0021% in `omega1` from 400 to 1600 and loses bimodality;
- 320x40 improves 0.174% from 400 to 1600 and stays bimodal;
- 400x50 becomes mode-order unstable and fails at 627.

Among existing full long histories near the paper-reproduction radius, the
99.5%-retained iteration is 321 for authoritative 240x30 `rmin=1.3`, but 1091
for 160x20 `rmin=1.2`.  This difference is partly late cycling/basin behavior,
not a monotone mesh-scaling law.  The audit therefore cannot produce a
resolution-robust native stopping table without the minimal telemetry runs in
section 14.

## 11. Comparison with Yuksel's fixed 200 iterations

Yuksel's 200 iterations remain a fixed work budget, not a convergence claim.
They are not used to calibrate any tolerance here.

At iteration 200, the authoritative reproduction retains 98.43% of its
late-reference `omega1`; 2.672 rad/s (1.567%) remains.  It does not retain 99%
until 225 or 99.5% until 321.  In the current R3 profile, 200 is almost terminal
at 160x20, but leaves about 0.67% and 0.82% to the 1600 checkpoint at 240x30 and
320x40.  The result is classification **D: iteration count is strongly
mesh/configuration dependent**.  For the authoritative 240x30 trajectory,
Yuksel's budget truncates the run before a <=0.5% practical-loss region.

## 12. Implications for the performance comparison

Using the existing R3 timing artifact without rerunning the campaign:

| Intended statement | Classification | Evidence |
|---|---|---|
| Olhoff iterations are relatively expensive | **SUPPORTED** | At 160--320, existing Olhoff per-iteration time is about 2.4--5.1x the comparator times. |
| Olhoff uses relatively few iterations | **NOT SUPPORTED** by current rows; **plausible but unvalidated** under a practical stop | Current valid rows all report 1600. A shadow stop near 400 would be fewer than Yuksel at 240/320 but more than Proposed. |
| Olhoff gives the highest `omega1` | **SUPPORTED** on valid 160--320 existing rows | Its final first FE frequency exceeds both comparators on those rows. |
| Olhoff gives true bimodality/highest spectral quality | **PARTIALLY SUPPORTED** | The faithful `rmin=1.3` reproduction is bimodal, but the actual `rmin=2` performance rows finish simple at 160/240 and bimodal at 320. |
| Yuksel is computationally intermediate | **PARTIALLY SUPPORTED** | Its total time lies between Olhoff and Proposed at 240/320; it is slightly faster than Proposed at 160, and 400 has no valid Olhoff row. |
| Proposed has the cheapest iterations | **NOT SUPPORTED** | Existing Yuksel per-iteration time is lower at 160--400. |
| Proposed has the cheapest total trajectory | **PARTIALLY SUPPORTED** | It is lowest at 240--400, but Yuksel is lower at 160. |
| Proposed accepts lower `omega1` and no exact bimodality | **SUPPORTED** on existing valid rows | Its first frequency is lower and its first gap is much larger than the faithful bimodal Olhoff result. |

Even an estimated 400-iteration Olhoff run would remain materially more
expensive in total than Proposed at 240x30 and would not establish the desired
narrative at every resolution.  The intended interpretation must therefore be
reported as a tradeoff with qualifications, not assumed.

## 13. Recommended native stopping policy, if justified

No final native stopping policy is justified by the available evidence.

The required status architecture is nevertheless clear:

```text
if subproblem failed:
    SOLVER_FAILURE
elseif objective-envelope stable
   AND intended modal basin stable
   AND no recent mode-order instability
   AND volume feasible
   AND global design update small/sparse
   for a persistence window:
    CONVERGED
elseif iteration == maxOuter:
    CAP_HIT
else:
    RUNNING
```

`max|Delta rho|` should be retained only as diagnostic telemetry.  The
objective-envelope shadow rule in section 8 is a development candidate, not a
policy.  An RMS/sparsity design component and a mode-order stability component
must be derived from prospective data before tolerances are declared.

The benchmark operating point must also be resolved independently: if the
performance narrative claims faithful Du--Olhoff bimodality, use and validate
the paper-reproduction radius family near 1.2--1.3 elements; if it retains the
shared `rmin=2` profile, drop the general true-bimodality claim.  A stopping rule
cannot repair that configuration distinction.

## 14. Minimal prospective validation plan

Do not run the final nine-resolution campaign.  The smallest defensible next
experiment is:

1. Freeze one native Olhoff operating profile (`rmin` choice included) before
   examining outcomes.
2. Add an audit-only history sink outside the frozen numerical tree, or an
   explicitly provenance-reviewed observer hook, that records each `drho`,
   density checkpoint, LP flag, and spectrum without changing numerical steps.
   Prove bit identity of all existing scalar histories against the frozen path.
3. Run exactly one shadow-stop trajectory at each development mesh 160x20,
   240x30, 320x40, and 400x50, with the current validated implementation and a
   safety cap.  Continue after the shadow criterion so later loss and false
   positives remain observable.
4. Record RMS update, fractions above density changes 0.001/0.005/0.01,
   20-step density displacement, binary disagreement/connectivity, `N`, first
   gap, `omega3`, volume, and LP status.
5. Accept a rule only if its offline-predicted and prospectively observed
   trigger agree, no later material spectral/topological transition occurs,
   connectivity and intended multiplicity are preserved, and the 400x50 LP
   failure is never mislabeled.
6. Only after those four shadow runs pass, perform four live-stop confirmations.

This is four development trajectories plus, conditionally, four inexpensive
confirmation runs—not a parameter sweep or final performance campaign.

## 15. Remaining uncertainties

- Elementwise late updates were not saved, so the hypothesis that only a small
  number of elements dominate `d_inf` is plausible but unquantified.
- Authoritative topology checkpoints at move 0.005 do not exist.
- Full current-profile histories at 320x40 and 400x50 were computed transiently
  but not archived, preventing cross-resolution replay.
- Deterministic delayed mode-order transitions show that finite scalar
  persistence is not by itself proof of basin stability.
- The meaning of a scientifically negligible loss (0.1%, 0.2%, 0.5%, etc.)
  remains a reporting choice; the audit quantifies rather than chooses it.
- `rmin=1.3` reproduces the Du--Olhoff bimodal result but is not the current
  shared performance profile; `rmin=2` is the current profile but does not
  preserve bimodality consistently.
- The 400x50 LP failure may be avoided by a valid earlier stop, but failure
  avoidance is not evidence that the earlier state was converged.

Machine-readable evidence is under
`analysis/olhoff_practical_convergence_audit/results/`; the generating script is
`analysis/olhoff_practical_convergence_audit/run_offline_audit.m`.  The script
loads existing artifacts only and does not call the optimizer.

## Final verdict

**1600 CONFIRMED EXCESSIVE, BUT STOPPING RULE NOT YET ROBUST**
