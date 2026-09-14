# NEXT_ACTION — Part 16

The preregistered decision tree (A → SOCP generality; B → filter attribution;
C → filter study; D → outer globalization) has **no branch for E**. The evidence
failure has to be resolved before the causal question can be asked again.
**No rerun of this treatment is authorized.** This run is closed.

## Single highest-information next experiment

**A zero-density-update frozen apex-certificate study at the rejected outer-15 state.**

The state is fully reproducible from the declared evidence
(`evidence/c480_socp_causal_run/C480x60_socp_state.mat`). The post-hoc re-posing
reproduced λ, λ_J and dOff bitwise and the rejected primal point to 1e-14 in bs.

Questions, in order:

1. **Is the rejected apex point optimal to the preregistered 1e-8?** Build a dual
   witness that exploits the known structure instead of a derivative-free search:
   - fix the primal active set (bound elements: lower q ≥ 0, upper q ≤ 0; interior:
     q = 0) and the apex;
   - solve the resulting small conic feasibility/least-violation problem in
     (p, μ, ν) with ‖p‖ ≤ μ (5 unknowns);
   - then evaluate the weak-duality bound exactly.

   The post-hoc exact-dual solve reached 8.6e-6 while its own interior-point method
   stalled; the structured version should be far better conditioned.
2. **Validate the apex certificate on known answers.** Use a synthetic apex problem
   with an analytically known optimum, plus the frozen oracle state, and preregister
   bars. Bars may be gain-relative if that is justified *before* seeing new treatment
   data.
3. **Decide whether a primal refinement is needed.** One option is re-solving with the
   apex imposed as a linear equality (a − c = d₂, b = 0), which is an LP-plus-box
   problem that interior-point methods terminate on cleanly. Show it gives the same
   optimum.
4. **Only if 1–3 pass:** preregister ONE new C480 exact-SOCP treatment. It is a new
   experiment, not a continuation or rerun of this one: same single factor, the
   validated apex certificate, and all other bars unchanged.

This is the cheapest experiment that can remove the obstacle that stopped this study.
It needs no FE solve beyond one frozen evaluation, and no topology run.

## Explicitly not next

- **Rerunning or resuming this treatment:** forbidden by the no-rerun rule.
- **Falling back to MMA, GCMMA or damped steps "to get past" the apex:** that
  changes the factor.
- **A filter-formulation study:** still deferred, because no inner-solver attribution exists.
- **Additional meshes (400 / 800) or the nine-mesh campaign:** still blocked.
- **Outer-globalization work:** not indicated. The 14 observed steps showed
  healthy realization, and no instability was observed.
