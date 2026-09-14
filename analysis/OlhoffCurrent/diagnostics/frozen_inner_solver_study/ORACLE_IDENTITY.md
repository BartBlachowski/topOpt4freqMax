# Oracle identity

FROZEN_SOLVER_ORACLE_PASS

All 90 reference-manifest entries match. Authoritative endpoint rho386:
`0a498a7d6ab0565b29c15ff9364060d937d4df10aa661fc90d02c038ce6e4a60`. The subproblem was constructed at rho385:
`9b1443e508a6e4ecb5288d30167c7258d837be127aa3a9a42d8cc092240f3ded`. These are different states, deliberately distinguished.
Outer=386, stage=3, mesh=480x60, move=.01; config and implementation tree match.
The construction context is the hashed, previously independently reconstructed
context, checked against the authoritative trajectory. No new FE solve was used.

Production re-evaluation gives beta=26921.772248325255,
bs=1.001792128848. The independently recomputed dual gap is
9.153e-12; exact certificate KKT passes.
Across 24 fixed test points, maximum
value discrepancy is 2.309e-14,
maximum relative gradient discrepancy 2.208e-16.
The production and conic feasible sets remain equivalent.

Both fresh SOCP repetitions reproduce the oracle vector bit-for-bit. They use
the certificate multipliers, since coneprog's returned duals remain inaccurate.
See `evaluations/oracle_identity.json` and `SOCP_COST.json` for both dual sets.
