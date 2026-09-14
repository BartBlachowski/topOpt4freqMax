# TREATMENT_CONFIG — the one treatment run

## Scientific configuration: the control's, unchanged

The treatment passed the control's stored `cfg` struct to the study driver without
modification (config hash `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e`).
Every field is listed row by row in `evaluations/single_factor_diff.json`. Summary:

| group | value |
|---|---|
| domain | 8 × 1, 480 × 60 Q4, consistent mass, simply supported at mid-height, axial restraint both ends |
| material | E = 1e7, ν = 0.3, density 1; SIMP p = 3 fixed; mass eq. 4b, q = 1, low-density exponent 6, cutoff 0.1 |
| design | ρ₀ = 0.5 uniform, ρ_min = 1e-3, volume fraction 0.5 |
| eigen | target mode 1, maxCluster 4 (Jcalc = 5), `eigs` with deterministic start vector, tol 1e-12 |
| multiplicity | `subspace`, N = 2 fixed; diagonal offsets on; off-diagonal coupling on; tolerance 0.05 (multJ warning only) |
| filter | Sigmund sensitivity filter, R = 0.06 physical (3.6 elements), applied to every f_sk and f_JJ |
| projection | off |
| move | ladder [0.04 0.02 0.01], continuation signal `stageExhaustion` (frozen E = A OR B, W = P = 20, Wnp = 10) |
| stop | rule `stageExhaustion`, ε = 0.15 (l2), terminal declaration only at the last rung |
| runtime | maxOuter 1600, single numerical thread, diagnostics on |
| inner (inert under treatment) | MMA published, tol 0.05, min 5, max 500 |

## Treatment descriptor

```
treat.innerSolver      = 'exactSOCP'
treat.crossEvery       = 1      % Amendment 1: non-gating degeneracy telemetry every iteration
treat.checkpointEvery  = 25     % process-interruption resume only
treat.stopAfter        = 0      % no early stop
treat.expectOmega1     = control hist.omega(:,1)   % iteration-1 bitwise identity assertion
treat.progress         = true   % one log line per outer iteration
```

## Inner solve (per outer iteration)

1. Eligibility E1–E8 (`cs_socp_eligible.m`). On failure, `SOCP_UNSUPPORTED_CASE_HIT` and stop.
2. Assembly of the exact conic form of production problem (25) (`fp_problem.m`,
   byte-identical to the certified reference). E9–E12 are checked at x = [0; 1].
3. `coneprog` in x-coordinates, OptimalityTolerance = ConstraintTolerance = 1e-10,
   MaxIterations 500. Attempt 1 uses LinearSolver `schur`; attempt 2 uses `augmented`,
   only if attempt 1 is not certified.
4. Independent certificate C1–C8 (`cs_socp_certify.m`): four dual-feasible
   candidates, weak-duality gap within ±1e-8, production-row feasibility ≤ 1e-8,
   complementarity and stationarity bars. If no attempt is certified,
   `SOCP_CERTIFICATE_FAILURE` and stop.
5. E9–E12 at the accepted point. On failure, `SOCP_EQUIVALENCE_FAILURE` and stop.
6. The accepted increment is the clipped certified x(1:NE). Production's own
   update `rho = min(1, max(rhomin, rho + drho))` applies it.
7. Telemetry, and the cross-solver diagnostic (augmented, non-gating).

## Launch record

`run/launch.json`: launched 2026-09-13T11:43:07+0200, MATLAB 25.2.0.2998904
(R2025b), pid 1109, `+impl` `edbfe47e…52cb`, preregistration `5b6186f2…6ddd`,
amendment `298efd23…a340`. The one-run lock is
`evidence/c480_socp_causal_run/LAUNCHED.lock`. The launch command was
`nohup caffeinate -i matlab -batch "addpath('scripts'); cs_run_treatment('launch')"`.
