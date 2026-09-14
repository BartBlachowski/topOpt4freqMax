# WP1/WP2 — Amended common evaluator definition
OFFLINE_AMENDMENT_VALIDATION — NO_NEW_OPTIMIZATION

## WP2 — which evaluators require amendment

Traced separately. `study_evaluate_design.m` `solve_modes` lines 39–58.

| | E1 | E2 | E3 |
|---|---|---|---|
| native association | Proposed | Yuksel | Olhoff |
| old stiffness | `1e7*(1e-6+(1-1e-6)*z^3)` | `1e7*(1e-9+(1-1e-9)*z^3)` | `1e7*max(z,1e-3)^3` |
| **new stiffness** | **unchanged** | **unchanged** | **unchanged** |
| old mass | `1e-6+(1-1e-6)*z` | `1e-9+(1-1e-9)*g4(z)` | `g4(max(z,1e-3))` |
| **new mass** | **unchanged** | `1e-9+(1-1e-9)*g4a(z)` | `g4a(max(z,1e-3))` |
| amendment required | **no** | **yes** | **yes** |
| code path | line 41–42 | line 44–48 | line 50–55 |

with

    g4 (x) = x^6        for x <= 0.1,   x  for x > 0.1     [Du & Olhoff Eq. (4)]
    g4a(x) = 1e5 * x^6  for x <= 0.1,   x  for x > 0.1     [Du & Olhoff Eq. (4a)]

**E1 is untouched**: its mass law is linear and carries no branch, so it was never exposed
to the defect. Only the E2/E3 low-density mass branch changes.

**E2 and E3 are not collapsed.** They continue to differ in exactly the ways they differed
before: E2 uses an additive void floor `1e-9` on both stiffness and mass and no density
clamp; E3 uses no additive mass floor and instead clamps `z3 = max(z,1e-3)` before both the
stiffness cube and the mass law. Sharing the same low-density mass *variant* does not make
them the same evaluator, exactly as sharing Eq. (4) did not before.

Everything else in the evaluator is byte-identical: mesh construction, Q4 element matrices,
assembly indices, midheight pinned supports, `eigs` options, deterministic start vector,
eigenvalue sorting, the exact-count binary projection with index tie-break, and every
topology and volume diagnostic.

## WP1 — branch-point analysis

| quantity | Eq. (4) — pre-amendment | Eq. (4a) — post-amendment |
|---|---|---|
| left limit `g(0.1-)` | 1.000000e-06 | 1.000000e-01 |
| value `g(0.1)` | 1.000000e-06 | 1.000000e-01 |
| right limit `g(0.1+)` | 1.000000e-01 | 1.000000e-01 |
| absolute jump | 9.9999e-02 | 6.939e-17 (one ULP) |
| multiplicative jump | 1.0e+05 | 1 + 7e-16 |
| **continuity** | **discontinuous** | **C0 continuous** |
| low-side derivative at 0.1 | `6*x^5` = 6.0e-05 | `6*1e5*x^5` = **6.000000** |
| high-side derivative at 0.1 | 1 | **1.000000** |
| **C1?** | no | **no — slope jumps 6 → 1** |

Verified numerically: `1e5*(0.1)^6 = 0.10000000000000003`, which differs from `0.1` by
2.776e-17, i.e. one double ULP — exact to representable precision.

**Recorded honestly: the amendment removes the finite jump. It does not make the
interpolation globally smooth.** Eq. (4a) is C0 but not C1; the one-sided derivatives at
0.1 are 6 and 1. Du & Olhoff offer Eq. (4b) (`c1 = 6e5, c2 = -5e6`) for C1 continuity, which
this amendment does **not** adopt, because the defect established by Phase 2C is a *finite
jump in the function value*, not a slope discontinuity, and the minimum correction principle
forbids taking more than the evidence requires. A slope discontinuity in a post-hoc
evaluator that is never differentiated has no established effect on any frozen decision.

## Implementation

`+ie2d/study_evaluate_design_eq4a.m` — a copy of the frozen evaluator differing in exactly
two functional lines:

    E2:  g(low) = z(low).^6;    ->  g(low) = 1e5 * z(low).^6;
    E3:  g(low) = z3(low).^6;   ->  g(low) = 1e5 * z3(low).^6;

The frozen `analysis/three_method_parametric_study/study_evaluate_design.m` is **unmodified**
(SHA-256 `22a1b974c251dbe7baa6499a5aca11e6bde68469b6d2e80fd4274c9447f31343`, still matching
the contract's `quality.source_sha256`). Applying the amendment to the normative evaluator is
a post-audit action (see `PHASE2A_IMPLEMENTATION_IMPACT.md`).

`massScale.m` case `'4a'` and `mass_interp.m` mode `du2007_c0` were **not** reused directly:
they carry NATIVE optimizer semantics and are on the Olhoff reproduction path. Coupling the
common post-hoc evaluator to native optimizer code would create exactly the dependency this
amendment exists to sever. The equation is therefore implemented inline with an explicit
source citation, and WP4 tests it against the same constants those files use.
