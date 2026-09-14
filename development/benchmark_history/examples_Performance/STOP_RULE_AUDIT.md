# Static stop-rule audit

Plan section 6.2. Audit date: 2026-07-31, against commit `cdf8cde` plus the WP3
extension-mode changes.

The requirement: for each solver, document every use of its tolerance, maximum
iteration value and stop flag, and verify that bypassing termination changes
**only whether another iteration is entered** — not an update, a continuation
decision, a stage handoff, a random stream, or any solver state before the native
stopping point.

A source-code argument is not sufficient on its own (section 6.3 runs the paired
test). This document is what that test is checking against.

---

## Olhoff–Du — `analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m`

| Use | Location | Role | Touched by extension mode |
|---|---|---|---|
| `cfg.conv_tol` | termination check, both branches | compared against `rel_change_obj` and `change_x` | no — the comparison still runs and still sets `stop_reason` |
| `grayness < 0.05` | termination check | third conjunct of the native stop | no |
| `polish_left <= 0` | termination check, final-β branch | minimum polish at maximum β | no |
| `cfg.maxiter` | `for it = 1:maxiter` | safety budget | no — still bounds the run |
| `gray_tol` | β advance | **continuation trigger**, not termination | no — left active |
| `beta_interval`, `beta_list` | β advance | **continuation schedule** | no — left active |
| `move_reduce`, `move_safe`, `Nsafe` | move adaptation | trust-region state | no |
| `reduce_counter` | eigensolver failure path | robustness | no |

**Verdict.** The native stop is a single `break` reached from two mutually
exclusive branches. Extension mode gates only those two `break` statements and
records `nativeStopIter` on first entry. Every conjunct is still evaluated, so
`stop_reason` is set at exactly the same iteration as in a native run. β
continuation, move adaptation and the eigensolver retry are all outside the gated
region.

One point of care, verified by reading: `move` is mutated by the β-advance blocks
and by the trial-rejection guard, both of which sit **above** the termination
check in the loop body. Extending therefore cannot alter `move` at or before the
native stopping iteration.

---

## Yuksel–Yilmaz — `analysis/YukselApproach/Matlab/top99neo_inertial_freq.m`

| Use | Location | Role | Touched by extension mode |
|---|---|---|---|
| `stage1Tol` | `localComplianceLoop` break | **stage handoff**, not final termination | **no — deliberately left active** |
| `stage2Tol` | `localInertialLoop` break | final native termination | yes — this break only |
| `stage1_maxit` | stage-1 loop bound | safety budget | no |
| `maxit` | stage-2 loop bound | safety budget | no |
| `penalCnt`, `betaCnt` | `cnt(...)` each iteration | continuation | no — and both are inert here, see below |
| `move` | OC update | fixed at 0.2 | no |

**Verdict.** This method has two identical-looking stop tests and only the second
is a termination. Stage 1's test decides when the compliance stage hands its mode
estimate to the inertial stage, which plan section 4.3 explicitly requires to stay
active in the discovery pass. Extension mode gates the stage-2 break alone.

Continuation is inert in this configuration and does not need gating:
`cnt = @(v,vCnt,l) v+(l>=vCnt{1})*(v<vCnt{2})*(mod(l,vCnt{3})==0)*vCnt{4}` with
`penalCnt = {1,1,25,0.25}` and `betaCnt = {1,1,25,2}` both carry the factor
`(v < 1)`, while `penal = 3` and `beta = 1`. Both increments are therefore
identically zero. This is why the freeze record lists a continuation floor of 0
for this method and why its only transition marker is the stage handoff.

---

## Proposed — `analysis/ourApproach/Matlab/topopt_freq.m`

| Use | Location | Role | Touched by extension mode |
|---|---|---|---|
| `convTol` | `while` condition | final native termination | yes — disjunct added |
| `maxIters` | `while` condition | safety budget | no — still bounds the run |
| `move` | MMA clamp | fixed per run | no |

**Verdict.** The termination is in the loop condition rather than a `break`, so
extension mode adds a disjunct: `while (change > convTol || extend) && loop < maxIters`.
The native run exits at the top of the iteration *after* `change` falls to the
tolerance, so `nativeStopIter` is recorded as the iteration in which `change`
first satisfies the test — the last iteration a native run executes. The
checkpoint mirrors the condition exactly and carries no extra guard.

---

## Shared risks checked

**Random streams.** None of the three consumes the global random stream; that is
established by `determinism_validation.json` (section 6.1) and is the reason the
extended and native runs can be expected to agree bitwise at all.

**Loop-carried state written after the stop test.** In all three solvers the
termination check is the last thing in the loop body, so nothing that would be
skipped by an early `break` is read before it on the same iteration.

**Preallocation sized by the iteration budget.** `freqIterOmega`, the history
recorder and `objectiveHistory` are all sized from `maxiter`/`maxIters`, not from
the tolerance, so extending does not resize an array that a native run would size
differently.

**What this audit cannot establish.** That the floating-point trajectory is in
fact identical. Reading the source shows the *control flow* is unchanged; only the
paired test of section 6.3 shows the arithmetic is. The two are reported together.
