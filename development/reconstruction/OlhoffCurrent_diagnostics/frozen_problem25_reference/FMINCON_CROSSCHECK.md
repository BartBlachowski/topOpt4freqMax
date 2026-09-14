# FMINCON_CROSSCHECK — Part 8

Independent nonlinear route: `fmincon` on the **literal four-row production
problem** — rows 1–2 through the production `deltaLambda`/`ddlam` as nonlinear
constraints, rows 3–4 as the exact affine rows, production box — with exact
analytic gradients. It never sees the cone form. Scripts `fp_fmincon.m`,
`fp_fmincon_sqp.m`, `fp_gradcheck.m`; data `evaluations/fmincon_*.json/.mat`.

## Gradient check (disclosed step issue)

Preregistered: central differences at S1 along 5 seeded directions, step
`1e−6·move = 1e−8`, bar 1e−5. At that step rows 2 and 4 show relative errors
1.4e−4 and 3.8e−3 — but the error **scales as 1/h** (row 4 is exactly linear,
its FD error is pure roundoff/h), the 1/h signature the prior audits used to
identify FD noise. Sweeping the step:

| step | row 1 | row 2 | row 3 | row 4 |
|---|---|---|---|---|
| 1e−8 (preregistered) | 7.0e−6 | 1.4e−4 | 1.4e−6 | 3.8e−3 |
| 1e−7 | 1.5e−7 | 8.1e−6 | 1.2e−7 | 2.0e−4 |
| 1e−6 | 2.8e−10 | 5.6e−9 | 4.9e−12 | 6.4e−8 |
| 1e−5 | 7.0e−9 | 1.7e−8 | 2.0e−12 | 1.1e−9 |

Minimum error per row 2.8e−10, 5.6e−9, 4.9e−12, 6.4e−8: all gradients pass at
a roundoff-appropriate step. The preregistered step was mis-chosen for
central differences (eps^{1/3} ≈ 6e−6 is the right scale). Independently, the
equivalence tests showed `ddlam` equals the closed-form gradient to 6e−15 and
`fmincon` reached first-order optimality 9e−12, which a wrong gradient would
not permit.

## Cross-check A — interior-point, exact Lagrangian Hessian (matrix-free)

`HessianMultiplyFcn` with the rank-2 operator of `SOCP_DERIVATION.md`,
`SubproblemAlgorithm = 'cg'`, tolerances 1e−10/1e−10/1e−14.

| start | exitflag | iters | `bs` | `bs − bs_ref` | first-order opt | min e₂−e₁ on path | ‖drho−drho_ref‖₂ | cosine | wall |
|---|---|---|---|---|---|---|---|---|---|
| S0 zero | 1 | 38 | 1.0017918952 | −2.34e−7 | 9.0e−12 | 7 428.6 | 0.0234 | 0.99973 | 25 s |
| S1 P19 | 2 | 39 | 1.0017918919 | −2.37e−7 | 6.2e−10 | 7 430.9 | 0.0235 | 0.99973 | 25 s |
| S2 M500 | 1 | 39 | 1.0017918973 | −2.32e−7 | 8.9e−12 | 7 407.3 | 0.0233 | 0.99973 | 25 s |
| S3 M5000 | 1 | 38 | 1.0017918961 | −2.33e−7 | 1.2e−11 | 7 421.0 | — | — | 17 s |
| S4 ½·xmax | 1 | 38 | 1.0017918947 | −2.34e−7 | 2.5e−11 | 7 460.0 | 0.0233 | 0.99973 | 23 s |
| S5 ½·xmin | 1 | 39 | 1.0017918994 | −2.29e−7 | 8.9e−12 | 6 964.4 | 0.0230 | 0.99974 | 25 s |
| S6 seeded random | 2 | 39 | 1.0017918971 | −2.32e−7 | 6.8e−10 | 7 460.0 | 0.0239 | 0.99972 | 25 s |

Seven starts, seven results within 7.5e−9 of each other and **2.3e−7 below the
certified optimum** — inside the preregistered 1e−6 agreement bar and on the
correct side of the certificate (nothing exceeds the bound). The deficit is
the interior-point barrier: at 1e−6·width almost every coordinate is
classified "interior", at 1e−4·width ~4 500 still are; the objective lost by
holding ~4 500 near-degenerate coordinates 1e−4 of the way inside their box is
of exactly this size. Constraint violation 0 at every solution.

Image functionals: `a` agrees with the reference to 6e−3 (of 48.16), `b` to
2e−3 (of −0.104), volume Σdrho to 1.8e−7; `c` differs by +1.8 to +3.1 (of
93.9). The `c` direction (raising λ₂ alone) changes `e₁` only at second order,
so the optimal set is nearly flat along it — this is the preregistered
"objective agreement with primal non-uniqueness", not a disagreement. The
cosine 0.9997 and ‖·‖₂ distance 0.023 (2.3 % of ‖drho_ref‖) reflect that
flatness plus the barrier offsets; the maximum single-element difference is
0.93·move on isolated near-degenerate elements.

KKT with `fmincon`'s own multipliers (`lambda.ineqnonlin/ineqlin/lower/upper`):
stationarity normalized RMS 1.7e−11 … 8.7e−9 (PASS level), primal and dual
PASS, but box complementarity 1.4e−6 … 4.3e−6 against the 1e−6 bar ⇒
`REFERENCE_PROBLEM25_KKT_INCONCLUSIVE` for every fmincon point. That is the
interior-point floor, as anticipated in the preregistration's warning, and it
is why the conic point with its certificate — not a fmincon point — is the
reference.

Eigenvalue separation never fell below 6 964 (S5's path) — the cone apex was
never approached; no nonsmooth point was crossed.

## Cross-check B — interior-point, L-BFGS Hessian

Only the S0 run completed before finalization: exitflag 1, 662 iterations,
`bs = 1.0017913912` (−7.4e−7 from the reference, inside the 1e−6 bar),
first-order optimality 7.5e−11, 5 634 s wall. At ~1.5 h per start the
remaining five L-BFGS starts were not completed; the process was stopped at
finalization so that the manifest is stable. Disclosed in `PROVENANCE.md`.

## Cross-check C — SQP

`fmincon` `sqp` from S1 printed iteration 0 and never finished its first
dense QP subproblem in 34 min 46 s at 20.4 GB resident; the 30-minute
`OutputFcn` cap cannot fire inside a QP, so the process was killed. Recorded
as **not numerically viable at n = 28 801**, not as a disagreement
(`evaluations/fmincon_sqp.json`).

## Verdict contribution

`fmincon` methods agree with the conic reference in objective (max
|Δbs| = 7.4e−7 ≤ 1e−6) from every start including the three MMA states, and
find nothing above the certificate. No evidence of multiple local solutions;
Case D is excluded independently of the convexity certificate.
