# Verified candidate definitions

This reconstruction comes from the frozen MATLAB evaluator, the Phase-2F modal engine,
the completed survey archive, and the exact-count topology implementation. It does not
substitute the prompt's shorthand for executable semantics.

## A — frozen evaluator

- Field: actual clipped gray density.
- Stiffness: E1 `1e7(1e-6+(1-1e-6)x^3)`; E2
  `1e7(1e-9+(1-1e-9)x^3)`; E3 `1e7 max(x,1e-3)^3`.
- Mass: E1 linear with its floor; E2/E3 Du–Olhoff Eq.(4), `g=x^6` at
  `x<=0.1`, otherwise `g=x` (with E2's floor and E3's effective-density clamp).
- Solver/selection: MATLAB `eigs`, three modes, algebraically lowest mode; no
  eigenvector diagnostic, escalation, or modal-invalid status.
- Hard gate: separate exact-count connectivity/island gate; it does not validate a mode.

## B — continuous mass, algebraic-lowest

- Field and stiffness: same gray field and E1/E2/E3 stiffness conventions.
- Mass: E1 unchanged; E2/E3 continuous Eq.(4a), `g=1e5*x^6` at `x<=0.1`,
  otherwise `g=x`.
- Selection: algebraically lowest mode, with no validity rule. Phase-2F computed 12
  modes diagnostically, but extra modes do not change B's definition.
- Failure semantics: no explicit modal-invalid result.

## C — adaptive lowest valid structural mode

- Field/stiffness/mass: actual gray field; E1 unchanged; E2/E3 Eq.(4a); E3 diagnostics
  use `rho_eff=max(x,1e-3)` consistently with its pencil.
- For every mode require all three physical conditions:
  1. `voidKE(rho_eff<=0.1) < 0.5`;
  2. `voidSE(rho_eff<=0.1) < 0.5`;
  3. density-weighted kinetic participation `sum(KE_n*rho_eff) > 0.5`.
- A mode is structurally valid only when all three conditions hold. Select the
  algebraically lowest valid mode. IPR is stored as a nonbinding localization cross-check;
  it has no mesh-invariant natural cutoff.
- Adaptive schedule: `3 -> 6 -> 12 -> 24 -> 48 -> ...`; continue until a valid mode is
  found or the eigensolver/resource limit is reached. A limit returns
  `STRUCTURAL_MODE_NOT_FOUND`; the highest computed mode is never substituted.
- Hard gate: unchanged and logically separate. Both modal evaluation and the pointwise
  gate must succeed.

The Phase-2F survey's operational shortcut used `voidKE<0.5` alone. This audit found 24
selected modes contradicted by both strain energy and density participation. The verified
Candidate C is the already-investigated three-diagnostic concept above, not that shortcut.

## D — exact-count binary

- Field: stable descending-density projection with increasing-index tie break and exactly
  `round(0.5*N)` solid elements.
- E1/E2/E3 pencil: same conventions evaluated on the binary field (Phase-2F used six modes).
- Selection: algebraically lowest binary mode; no adaptive/modal validity rule.
- Hard gate: applied to the same binary topology, but checks connectivity and component
  area rather than dynamic adequacy.
