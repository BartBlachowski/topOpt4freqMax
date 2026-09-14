# WP16B — Negative-verdict remediation analysis

The optimizer was not modified and must not be. The mechanism is not a defect in the
Du–Olhoff method; it is an interaction between an observational storage cast and a
**discontinuous evaluator**. Options are presented, not chosen.

## The mechanism, stated precisely

The frozen E2/E3 mass law is discontinuous at x = 0.1:

    g(x) = x^6 for x <= 0.1, else x        g(0.1^-) = 1e-6,  g(0.1^+) = 0.1

Move-limit arithmetic drives densities onto 0.099999999999999644729 and stalls there.
`single(0.0999999999999996447) = 0.10000000149011611938 > 0.1`, so the element silently
crosses the branch and its mass rises by 1e5. Nothing here is a rounding-magnitude
problem: a 2.9e-8 density change produces a 2.27e-2 frequency change because the
evaluator is discontinuous at exactly the value the optimizer parks on.

## Options

| # | Option | Protected source changes? | Optimizer maths changes? | Frozen methodology changes? | Storage cost | Risk |
|---|---|---|---|---|---|---|
| 1 | Store new Olhoff trajectories in double via a caller-side sidecar written by a wrapper around the runner | no | no | no — satisfies the default rule | 2x snapshots (e.g. 720x90 x 1601 = 830 MB/run) | low; the wrapper needs the double state, which only `res.rho` exposes, so it requires one run per checkpoint unless combined with option 2 |
| 2 | Checkpoint-limited reruns to reconstruct double states on demand | no | no | no — section 4 already mandates a checkpoint identity test | none | O(B^2) solver cost; 3200-state reconstruction at 96x12 is ~10 h, and production-scale is prohibitive |
| 3 | Narrowly authorized observer hook inside the protected runner, emitting the double state write-only (the pattern Phase-2A added to `topopt_history_record.m`) | **yes** — changes a protected hash | no | no, but requires re-freezing the protected-source hash | none beyond storage | out of scope for Phase 2B; needs explicit review authorization |
| 4 | Accept single storage but re-derive E2/E3 from a continuous mass law | no | no | **yes** — changes a frozen evaluator | none | rejected: E1/E2/E3 are frozen co-primary evaluators |
| 5 | Accept single storage and treat E2/E3 as diagnostic | no | no | **yes** — changes co-primary status | none | rejected: contract marks all three `co_equal_primary` |

## Assessment

Option 1 is the only one that satisfies the frozen rule without touching protected source
or methodology, but on its own it cannot capture intermediate states, because the
unmodified runner exposes a double state only through `res.rho` at its cap. Combining
option 1 with option 2 is correct but expensive at production scale.

Option 3 is the minimum-cost complete fix and is exactly the instrumentation Phase 2A
already applied to the other two methods' recorder. It requires a protected-source change
and therefore an explicit authorization decision outside Phase 2B.

## A separate observation for the reviewer, outside Phase-2B scope

The discontinuity at x = 0.1 affects the *double* path too. Any perturbation of order
1e-16 near that value moves E2/E3 by 2%. That is a conditioning property of the frozen
evaluator, not of single storage, and it means E2/E3 are not continuous functions of the
design at densities the optimizer demonstrably parks on. Phase 2B makes no
recommendation about it; it is recorded because it bears on how much any E2/E3-derived
quantity can be trusted at the 1% level.
