# PREREGISTRATION AMENDMENT 1 — resolver metadata in the M1 launch preflight

Written 2026-09-13, after the first M1 preflight and **before** the M1 run was launched.
No trajectory, same-state evaluation or scientific result existed at this point.
Base preregistration SHA-256: `ae998b0ff10bcbd5bde552e284626ae43929cc459885577ca2dbfc7f053dc28f`.

## Observation

The leaf-by-leaf comparison of the stored S480x60 configuration with the resolved M1
configuration (`scripts/sd_m1_config.m`) returned seven differing leaves:

| leaf | S480x60 | M1 | preregistered as allowed |
|---|---|---|---|
| material.stiffness.model | pedersen | simp | yes |
| material.mass.model | eq2 | eq4b | yes |
| runtime.diagnostics | false | true | yes |
| runtime.name | S480x60 | M1_480x60_simp4b_adaptive | yes |
| provenance.preset | duOlhoffAdaptivePedersen | duOlhoffAdaptiveMove | **not listed** |
| provenance.overrides | (list incl. mesh, name, verbose) | (list incl. mesh, name, verbose, move.initial, diagnostics) | **not listed** |
| provenance.resolvedAt | 2026-09-13T11:51:29 | 2026-09-13T16:43:22 | **not listed** |

## Why these three leaves do not violate the intent of §5

`provenance.*` is written by `olh.config.resolve` to record *how* a configuration was built.
It cannot be equal between two resolutions (the timestamp alone differs), and a
different preset name with different overrides is the only committed way to express the
M1 alignment. `grep provenance` over the executed source path (`architecture/olhoffSolve.m`,
`algo/*.m`, `fem/*.m`, `+olh/+move/limit.m`) finds no read of it (one comment in
`fem/massScale.m`). Every *scientific* leaf other than the two preregistered material fields
is identical, including `move.initial = 0.1`, ε = 0.15000000000000002, cap 400 and
`runtime.singleThread = true`.

## Amendment

§5 "M1 launch preflight": the allowed-difference set is extended by the non-scientific
resolver metadata leaves `provenance.preset`, `provenance.overrides`,
`provenance.resolvedAt`. No other rule, threshold, verdict criterion or budget changes.
