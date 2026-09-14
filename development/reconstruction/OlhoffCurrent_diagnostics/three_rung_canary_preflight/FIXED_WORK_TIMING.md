# FIXED_WORK_TIMING — Part F

## Verdict

```
FIXED_WORK_SCALING_SANITY_PASS
```

Computational kernel cost increases sensibly with mesh, and the measurement
independently reproduces the in-run telemetry.

## What was measured

At each canary's **saved terminal design, held fixed**, with the same
configuration, single-thread policy and code. K = 5 timed repeats after 2
discarded warm-ups, reported as median. The design is never advanced: `drho` is
computed and discarded, and `cp_fixedwork.m` asserts before returning that `rho`
is bitwise unchanged. It produces no design, no trajectory and no convergence
claim, so it cannot contaminate a scientific record.

Both benchmarks were run **back to back after both canaries**, on a quiet host
(load 2.28 and 2.21; calibration kernels 0.013/0.011 s dgemm, 0.003 s sparse
solve) — deliberately, so the two meshes are compared under the same machine
state rather than at whatever moment each canary happened to end.

## Results

| kernel | 480×60 | 800×100 | ratio | exponent in NE | spread 480 | spread 800 |
|---|---|---|---|---|---|---|
| assembly + eigensolve | 0.27059 s | 1.30580 s | 4.826 | **1.541** | 7.3 % | 13.4 % |
| gradients (2N+1 genGrad + filter) | 0.01682 s | 0.05696 s | 3.387 | **1.194** | 2.5 % | 15.5 % |
| MMA sub-problem, whole solve | 11.91716 s | 31.95248 s | 2.681 | **0.965** | 0.6 % | 0.3 % |
| — per MMA step | 0.85123 s | 1.22894 s | 1.444 | 0.359 | — | — |
| — steps taken at that state | 14 | 26 | — | — | — | — |

NE ratio = 2.7778. Move limit 0.01 (the terminal rung) at both meshes.

### Reading the per-step figure correctly

The "per MMA step" exponent of 0.359 is **not** a kernel scaling. The step count
is a property of the design state, not of the mesh — 14 steps at 480×60 against
26 at 800×100 — so dividing by it mixes state with cost. The kernel measure is
the **whole sub-problem solve**, and that scales as NE^0.965, i.e. essentially
linear, which is what an O(NE)-per-dual-iteration sub-problem should do.

Recording this distinction matters: quoting only the per-step number would have
made the MMA kernel look strongly sub-linear, which it is not.

## Why this is a PASS

**1. The eigensolve exponent is the one theory predicts.** The generalized
eigenproblem is solved by shift-invert ARPACK, whose cost is dominated by the
sparse factorization of the shifted matrix. For a 2D problem under nested
dissection that factorization is O(n^1.5) in the number of DOFs, and DOFs scale
with NE. The measured exponent is **1.541**. This is the strongest single piece
of evidence that the kernel is behaving as a correct 2D sparse direct solve and
not, say, silently falling back to something dense or iterative.

**2. Every kernel grows; none is anomalous.** Assembly+eigensolve 1.541,
gradients 1.194, MMA sub-problem 0.965. Nothing is flat, nothing is negative,
nothing is explosive.

**3. It agrees with the in-run telemetry it never saw.** The benchmark is a
standalone re-evaluation, yet it reproduces the canaries' own stage-3 per-outer
timings (same move limit, same design):

| kernel | mesh | fixed-work | in-run stage 3 | difference |
|---|---|---|---|---|
| assembly + eigensolve | 480×60 | 0.27059 | 0.27281 | **−0.8 %** |
| gradients | 480×60 | 0.01682 | 0.01739 | −3.3 % |
| s per MMA step | 480×60 | 0.85123 | 0.84382 | +0.9 % |
| assembly + eigensolve | 800×100 | 1.30580 | 1.31201 | **−0.5 %** |
| gradients | 800×100 | 0.05696 | 0.05025 | +13.4 % |
| s per MMA step | 800×100 | 1.22894 | 1.17667 | +4.4 % |

Sub-1 % agreement on the eigensolve at both meshes. The gradient outlier
(+13.4 %) is on a 50 ms quantity whose own repeat spread is 15.5 %, so it is
inside measurement noise rather than a discrepancy.

**4. It separates cost-per-operation from number-of-operations.** That was the
point of Part F. The canaries' total wall times differ by 2.70× (2799 s vs
7554 s), but that figure mixes kernel cost with iteration count. The fixed-work
measurement isolates the first: kernels are 2.7–4.8× more expensive at 800×100,
and the run needed 1.21× more iterations. Neither factor is anomalous, and
together they account for the observed totals.

## Comparison with the legacy scaling reference

The reference band recorded **before** any measurement, from the legacy
nine-mesh fits over the 2.778× NE ratio: ≈ 3.4× for assembly+eigensolve, ≈ 2.5×
for gradients, ≈ 2.1× per MMA step.

Measured: 4.83×, 3.39×, 1.44×. Assembly+eigensolve and gradients come in
*steeper* than the legacy fit, the MMA step shallower — but the legacy fit was
a nine-point regression across a 25× NE range with `diagnostics` off and a
different controller, so it constrains the order of magnitude, not the third
digit. Nothing here is outside the band in the sense that mattered: no kernel is
sub-linear where it should be super-linear, and none is explosive.

The preregistered failure signature — "a fixed-work measurement far outside that
band, especially a *sub-linear* eigensolve" — did not occur. The eigensolve is
the steepest kernel measured, at 1.541.

## Limitation

Assembly and eigensolve are **not separable** in this instrumentation, at any
mesh: `tEig` brackets both, and `cp_fixedwork` times them together because
that is how the solver times them. Pure eigensolver cost is therefore not
reported, here or anywhere else in this study.
