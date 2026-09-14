# Native convergence detector audit for the bimodal Du--Olhoff reproduction

## Outcome

No detector tested here satisfies the acceptance criteria. A simple period-two,
modal-event-aware hybrid is a convincing detector on the 240x30 development
trajectory, firing at iteration 812, but the locked configuration never fires
on any of the 160x20, 320x40, or 400x50 hold-outs by iteration 1600. It was not
retuned after hold-out inspection. Consequently, the prospective stopping
experiment was not run: WP13 explicitly permits it only after non-stopping
cross-resolution validation succeeds.

This is a scientifically useful negative result. The 240x30 trajectory has a
well-defined stationary period-two regime, while the other three trajectories
continue to show objective/design recurrence violations and late modal events.
Their healthy, connected, bimodal terminal states do not prove that an online
stationarity event occurred before the cap.

## Frozen scope and provenance (WP0)

The audit began on branch `benchmark-methodology-r2` at
`cb6353feb941f12b2aaa927e622649e1ccc926f7`. The working tree was already dirty;
all pre-existing modified and untracked benchmark/equivalence work was
preserved. This audit added only `OLHOFF_NATIVE_CONVERGENCE_DETECTOR.md` and
`analysis/olhoff_native_convergence/`. It did not edit the imported numerical
implementation under `Matlab/reproduction2007/{algo,fem,filter,mma,runs}` or any
Proposed/Yuksel implementation.

The 61 imported files were re-hashed against `SOURCE_SHA256.txt`, applying the
documented content-preserving migration mapping `CLAUDE.md -> SOURCE_CLAUDE.md`
and `results/* -> baseline/*`. All 61 matched. A direct un-mapped `shasum -c`
is not the correct migration check because those seven paths were deliberately
relocated.

The frozen trajectory profile was:

| quantity | value |
|---|---:|
| domain / volume | 8x1 / 0.5 |
| development mesh | 240x30 |
| hold-outs | 160x20, 320x40, 400x50 |
| `rminEl` | 1.3 FE widths at every resolution |
| move | 0.005 |
| `rhomin` | 0.001 |
| SIMP / mass interpolation | `p=3` / option `4` |
| multiplicity tolerance | `tolMult=0.05` |
| filtering | `filterMode=diag` |
| subproblem | Eq. (22) LP, `innerSolver=lp` |
| eigensolver / threads | deterministic-start `eigs` / 1 |
| supports | case a, mid-height, axial restraint at both ends |
| maximum outer iterations | 1600 |
| legacy outer test | `max(abs(drho)) < 0.001` |

`rminEl=1.3` was kept in element units. It was not converted to 1.3 physical
metres or held as a fixed physical radius under refinement.

## Observation without trajectory changes (WP1--WP2)

`olhoffOptTelemetry.m` is an audit-only mirror outside the frozen tree. It calls
the unchanged model, assembly, eigensolver, gradients, filter, and LP routines.
Its additions observe already computed values, keep density snapshots, and can
apply a wrapper-level detector after an unchanged update.

For the authoritative 240x30 run, the mirror was compared against
`baseline/lp240_rmin1.3.mat`. Density, final eigenvalues/frequencies,
`nOuter`, mode classification, and every non-timing history field were exactly
equal (`isequaln`; maximum absolute difference zero). Thus telemetry did not
change any frozen numerical decision.

The telemetry includes all requested spectral values, `N`, every relative
gap, cross-iteration modal assurance/order changes, `beta`, max/RMS/mean design
updates, moving fractions at five declared thresholds, move-bound and near-bound
fractions, volume, LP flag and scaled residuals, inner/eigen/finite-state health,
and every post-update density state. The resulting compressed MAT files are
4--12 MB each, so retaining every density state was not excessive for these
four diagnostic runs.

## Why the old criterion fails

On all four meshes, `max(abs(drho))` was exactly 0.005 at every one of the 1600
iterations: its minimum, final value, and move limit were identical. At the
last iteration, only 14--28% of elements were at the move bound, so a localized
or alternating subset controls the maximum even when macroscopic evolution is
small. The test `max(abs(drho)) < 0.001` therefore contains no stopping
information for these trajectories. Reaching 1600 means `CAP_HIT`, not
intrinsic Olhoff convergence.

The authoritative trajectory also never approaches a zero one-step RMS update.
It settles into an alternating LP cycle with one-step `d_rms` about 0.00232 and
about 21.5% of elements moving by more than 0.001. In contrast, its same-phase
two-step density RMS falls to about 0.0001--0.0003 and its thresholded topology
becomes almost invariant. This distinction motivated phase-aware candidates;
an absolute one-step RMS or “small moving fraction” rule would reject a useful
stationary cycle forever.

## Retrospective labels and detector families (WP3--WP8)

The retrospective label is validation machinery only. It was never exposed to
the online detector. For a hypothetical fire at `k`, horizons 50, 100, and 200
required:

- phase-balanced objective deviation no greater than 0.1%; the reported table
  also exposes 1%, 0.5%, and 0.25% accuracy levels;
- persistent `N=2` and gap no greater than 1%;
- no subsequent multiplicity/mode-order event;
- thresholded topology turnover no greater than 0.5%;
- healthy LP/eigensolver/finite state and feasible volume.

Normalized density L1/L2 distances, binary-topology disagreement, support-to-
support four-connectedness, and largest-component fraction were also evaluated
against the continued state. Connectivity and future evidence were not used to
make the online decision.

Eight small, interpretable candidates were developed only on 240x30. All
families retained common bimodality, recent-modal-event, feasibility, and solver
health guards.

| candidate | evidence | fire | H50 | H100 | H200 | classification |
|---|---|---:|:---:|:---:|:---:|---|
| O loose | objective phase/block | 441 | pass | fail | fail | false positive |
| O mid | objective phase/block | 446 | pass | fail | fail | false positive |
| O strict | objective phase/block | 833 | pass | pass | pass | true on development |
| D loose | design recurrence | 441 | pass | fail | fail | false positive |
| D mid | design recurrence | 807 | pass | pass | pass | true on development |
| D strict | design recurrence | 862 | pass | pass | pass | true on development |
| H balanced | objective + design + event guard | 812 | pass | pass | pass | true on development |
| H conservative | stricter hybrid | 867 | pass | pass | pass | true on development |

The loose rules are the delayed-instability demonstration: a 50-step check
would accept them, but 100/200-step topology evolution rejects them. At their
fires, binary topology still differs from the terminal state by 1.21% and
density L2 distance is about 0.05. Persistence alone is therefore not enough
when its underlying signals are permissive.

The selected hybrid configuration was frozen in
`native_convergence_config.json` before any hold-out was inspected:

```text
period = 2
objective block = 20
stationarity window = 40
persistence = 20 consecutive candidate states
relative drift between adjacent 20-step objective block means <= 1e-4
max same-phase objective recurrence over the window <= 1e-4
max same-phase density RMS over the window <= 1.25e-3
max same-phase 0.5-topology turnover over the window <= 7e-4
N = 2 and gap12 <= 1e-2 over the last 40 iterations
no recent mode-order/multiplicity event
relative volume residual <= 1e-8
LP, eigensolver, and finite-state health required
```

This uses information available at `k` only. It has no terminal target, future
horizon, resolution-specific constant, or reference to 200 iterations. The
20-step persistence acts on conditions that already contain 40 steps of
history. Hysteresis was not added: it did not address the development false
positives, and there was no evidence that extra state would improve hold-out
generalization.

On 240x30 the balanced hybrid fires at 812. The post-update state has
`omega=(170.4710,170.8666,285.2271)`, `N=2`, gap 0.2321%, connected topology,
0.0139% binary disagreement with the continued terminal topology, and
phase-specific absolute omega1 loss of 0.000078%. The phase-balanced terminal
loss is 0.00117%. Its estimated runtime is 80.4 s versus 152.1 s for 1600,
saving 788 iterations and approximately 71.8 s on this run.

The conservative sensitivity candidate fires at 867. Its phase-balanced
terminal loss is similarly tiny, but stopping on that particular alternating
phase gives a 0.126% instantaneous omega1 loss. This is additional evidence
that a prospective detector for an alternating trajectory must define which
phase-specific state is returned, not merely when a phase-balanced envelope is
stationary.

## Locked cross-resolution validation (WP9, WP11--WP12)

Both predeclared hybrids were replayed without retuning:

| mesh | role | balanced fire | conservative fire | last modal event | last-200 max gap | result |
|---:|---|---:|---:|---:|---:|---|
| 160x20 | hold-out | never | never | 1591 | 1.079% | cap hit |
| 240x30 | development | 812 | 867 | 392 | 0.297% | development-only pass |
| 320x40 | hold-out | never | never | 1544 | 1.158% | cap hit |
| 400x50 | hold-out | never | never | 1505 | 1.105% | cap hit |

This is not merely failure of the persistence counter. On all three hold-outs,
the balanced detector's objective phase-recurrence and density phase-RMS tests
passed for 0% of eligible iterations. No complete raw candidate state occurred.
In the final 200 iterations, max same-phase objective recurrence remained
0.36--0.50% and max same-phase density RMS remained 0.0043--0.0051, far above
the locked 0.01% and 0.00125 thresholds. Modal events also occurred very late.

All four runs had zero solver failures, relative terminal volume error below
`2.3e-15`, `N=2` at the cap, final gaps of 0.146--0.232%, and a single
four-connected solid component spanning the supports. Thus the hold-outs are
not failed solutions or simple-mode basins. They are healthy bimodal cap-hit
trajectories for which this experiment cannot establish an earlier stationary
event. A final small gap by itself is insufficient because the gap reopened
above 1% and ordering events occurred during the recent history.

The answer to “does one detector work without retuning?” is therefore no. It
would be invalid to relax the phase/design tolerances using these hold-outs and
then count the result as hold-out validation. A new detector may be developed
from this evidence, but it requires fresh unseen validation trajectories.

## Failure precedence and status taxonomy (WP10--WP11)

The wrapper evaluates health before convergence. Eight adversarial replay tests
all passed: healthy stationary state accepted; LP failure with an artificial
zero step, inner failure, eigensolver warning, non-finite state, simple-mode
stationarity, reopened gap, and infeasible volume all rejected.

The useful status order is:

```text
SOLVER_FAILURE > CONVERGED_BIMODAL > CAP_HIT > RUNNING
```

`STATIONARY_SIMPLE_MODE` may be reported as a separate scientific outcome, but
it is neither solver failure nor successful bimodal convergence. Explicit
`N=2` plus an actual gap guard is necessary here because bimodality defines the
successful reproduction; stable omega1 alone cannot establish it.

## Prospective stopping decision (WP13)

No prospective stop was run. The selected detector failed three of three
hold-outs, so it did not satisfy the prerequisite for active termination. The
machine-readable prospective CSV records `NOT_RUN` and the gate reason for all
four meshes. Consequently, offline-versus-prospective equality, prospective
runtime savings, and prospective stopped-state quality remain unvalidated.

## Native counts, performance interpretation, Yuksel, and move acceleration

There is no validated native iteration count from this work. The only defensible
counts are:

- 812 for a promising development-only detector on 240x30;
- `CAP_HIT=1600` for all three hold-outs under the locked detector;
- not “Olhoff requires 1600 iterations.”

Observed full-run mean wall time per iteration was 0.0423, 0.0951, 0.1799, and
0.2986 s for 160x20 through 400x50; total full-run times were 67.7, 152.1,
287.9, and 477.7 s. These confirm that multiplicity-aware iterations become
expensive with resolution, but the failed validation does not yet support a
performance-table claim that Olhoff uses “relatively few” native iterations.
The 240x30 development result is promising evidence, not a publishable global
stopping policy.

Only retrospectively, Yuksel's fixed Dynamic-Code budget of 200 is materially
below the 812 development fire and below every unresolved hold-out cap. The
observed behavior is resolution-dependent; no detector tolerance was chosen or
changed in response to 200.

Move acceleration was intentionally not mixed into this study. The evidence
does justify a separate future experiment with move 0.01/0.015/0.02, but only
after a new detector protocol and fresh hold-outs are designed. Larger moves
may accelerate early evolution and may also damage connectivity, so this audit
provides no authorization to change the frozen `move=0.005` performance path.

## Direct answers to the required questions

1. **Why does old max-update stopping fail?** At least one element takes the
   0.005 move on every iteration of every mesh, including stationary-looking
   alternating regimes; the statistic never approaches 0.001.
2. **Best observables?** Phase-balanced objective drift, same-phase density
   recurrence, thresholded topology turnover, recent eigengap/order events,
   and solver/volume health. One-step RMS and moving fraction remain large in
   the useful 240x30 cycle.
3. **Must bimodality be explicit?** Yes: persistent `N=2` and a measured small
   gap are necessary, though not sufficient.
4. **How long must stability persist?** On development, 20 consecutive states
   over a 40-step history (roughly 59 iterations of evidence) survived 200-step
   look-ahead. No mesh-independent duration was validated.
5. **Can apparent convergence destabilize?** Yes. Loose rules at 441/446 pass
   H=50 and fail H=100/200; hold-outs have modal events after iteration 1500.
6. **What prevents false positives?** Phase recurrence, design/topology guards,
   modal-event exclusion, persistence, and health precedence prevent the known
   development false positives. They do not yet generalize into a useful stop.
7. **One detector without retuning?** No.
8. **Native iteration counts?** Development-only 812; all hold-outs cap-hit at
   1600 under the locked rule. No validated native count exists.
9. **Objective sacrificed?** On development, 0.000078% phase-specific absolute
   loss for the balanced candidate (0.00117% by phase-balanced mean). No
   cross-resolution stopped-state loss is available.
10. **Computation saved?** Development-only: 788 iterations and about 71.8 s
    (47%). No validated hold-out savings.
11. **Versus Yuksel 200?** Materially above 200 and unresolved on hold-outs;
    this comparison was made only after locking and validation.
12. **Proceed to move acceleration?** Only as a separately designed future
    study, not as part of the present unvalidated stopping campaign.

## Reproducible artifacts

All outputs are under `analysis/olhoff_native_convergence/results/`:

- `native_convergence_candidates.csv`: development family grid, firing states,
  future labels, density distances, and classification;
- `native_convergence_false_positives.csv`: common/candidate stops at all three
  horizons and four objective accuracy levels;
- `native_convergence_cross_resolution.csv`: locked detector replay and quality;
- `native_convergence_holdout_blockers.csv`: per-condition non-firing evidence;
- `native_convergence_prospective_validation.csv`: explicit not-run record;
- `native_convergence_config.json`: pre-hold-out frozen primary/sensitivity
  configurations;
- `native_convergence_precedence_tests.csv`: adversarial status-gate tests;
- four compressed development/hold-out MAT files and seven requested diagnostic
  plot types (spectra, gap/N, RMS, moving fraction, max-vs-RMS, delayed evolution,
  and firing iteration comparison).

The scripts that generated them and the reusable future-blind detector are in
`analysis/olhoff_native_convergence/`. No nine-resolution campaign was run.

**NO ROBUST NATIVE DETECTOR FOUND**
