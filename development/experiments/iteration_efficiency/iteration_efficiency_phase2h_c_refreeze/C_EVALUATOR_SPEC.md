# Candidate C common evaluator — frozen Phase 2H specification

Candidate C evaluates the actual gray density field. It is a neutral, offline,
post-hoc evaluator and is never called by a native optimizer.

For E1, stiffness is `1e7*(1e-6+(1-1e-6)*rho^3)` and mass is
`1e-6+(1-1e-6)*rho`. For E2, stiffness is
`1e7*(1e-9+(1-1e-9)*rho^3)` and mass is
`1e-9+(1-1e-9)*g(rho)`. For E3, `rho_eff=max(rho,1e-3)`, stiffness is
`1e7*rho_eff^3`, and mass is `g(rho_eff)`. In both cases
`g(r)=1e5*r^6` for `r<=0.1`, otherwise `g(r)=r` (Olhoff Eq. 4a).

For each ordered eigenmode, compute diagnostics over the evaluator-specific region
`rho_eff<=0.1`. A mode is structurally valid if and only if all three strict tests pass:

- `voidKE < 0.5`;
- `voidSE < 0.5`;
- `densityParticipation > 0.5`.

IPR is recorded but nonbinding. The selected value is the frequency of the lowest
unanimously valid mode. The classifier identifier is
`candidate_c_unanimous_v1`; ties at a threshold fail the strict test.

The exact-count binary projection is retained only as an endpoint manufacturability and
topology diagnostic. It is excluded from Q, the reference floor, persistence, and status.

