# Phase 4 — Decision: is a new clean-room implementation required?

## Decision

**No full new optimizer reimplementation is warranted as a deliverable right now.** The primary deliverables (a trustworthy paper-derived specification and a discrepancy analysis) are complete and have been **empirically validated** at the paper's own first gate. A from-scratch optimizer would re-derive components that the spec already *confirms* are correctly implemented in `OlhoffApproachExact`, and would not, by itself, resolve the one genuinely open question (which basin the optimum lives in).

The only code produced in Phase 4 is the minimal **FE verification** ([`fe_verify.py`](fe_verify.py)) needed to make this decision evidence-based.

## Why (evidence chain)

1. **Spec is validated at the FE gate.** Independent reconstruction reproduces the published initial frequencies SS 68.7 / CS 104.1 / CC 146.1 to <1% (p=3, mid-height SS pin, ~48×6–64×8 mesh). See [experiments/initial_frequency_verification.md](../experiments/initial_frequency_verification.md). The spec's FE/interpolation core is therefore trustworthy.

2. **The existing `OlhoffApproachExact` is equation-faithful.** Every numbered equation in the spec — including the hard multiple-eigenvalue subeigenvalue problem (eq. 18-19, 25), which the spec flagged as the biggest ambiguity — has a correct counterpart in the code (comparison §1). A clean-room rewrite of these would legitimately reproduce the *same* equations (the mission permits reusing decisions "confirmed by the paper-derived specification" — and they are confirmed).

3. **The benchmark mismatch is not in the physics.** It is in the unspecified **path-control** choices (no trust region on the full-step increment ⇒ N=1 bang-bang limit cycle; comparison §4 C1-C2) and, at the formulation level, the **disconnection / basin** question (Phase 2 §3; comparison §4 C3). Neither is fixed by re-architecting; both are fixed (or at least diagnosed) by *adding* the missing controls and *measuring* connectivity.

## What a reconstruction *would* contribute (if the user wants benchmark reproduction)

A reconstruction is justified only if the goal shifts from "validate the spec" (done) to "reproduce the optimum `456.4` bimodal connected design." That is a **path-control + basin** investigation, not an architecture exercise. The minimal, paper-honest plan:

1. **Reuse the spec-confirmed core** (FE Q4, SIMP eq. 1-3, mass models eq. 4/4a/4b, M-orthonormal eigensolve, generalized gradients eq. 19, subeigenvalue inner problem eq. 18/25, sensitivity filter). These are not architectural debt — they are paper-derived and verified.
2. **Add the two missing path controls** the paper omitted but implicitly relied on:
   - a per-outer-iteration **trust region** on `Δρ` (or `α<1` damping) so the frozen-gradient linearization stays valid;
   - optional **`p:1→3` continuation** (paper-stated; low priority for *frequency* match per finding D1, but may stabilize the path).
3. **Instrument the basin**: report a connectivity measure each iteration; compare the converged topology against Fig. 3c. If the optimizer prefers the disconnected two-block CC design, that is a *formulation* outcome (Phase 2), and reporting it honestly is more valuable than forcing a match.
4. Run in **Python** (runnable/validatable in this environment; the existing exact code is MATLAB and cannot be executed here to check against the benchmark).

This is a focused experiment of perhaps a few hundred lines, **not** a parallel framework. It is deliberately **not** built now because: (a) the mission's primary deliverable is the spec + analysis, which are done; (b) it would not be "clean-room of architecture" so much as "the confirmed core + two knobs"; and (c) it is a meaningful new effort that should be explicitly requested rather than assumed.

## Recommendation

- **Treat Phases 1–3 + the FE verification as the deliverable.** The spec is trustworthy and the discrepancy is fully characterized and evidence-backed.
- **If benchmark reproduction is desired**, green-light the focused Python reconstruction above. Its success criterion is not "match 456.4 at any cost" but "with paper-faithful settings + minimal trust region, does the optimum converge to the connected bimodal Fig. 3c, or to the feasible disconnected basin?" — which is the real open scientific question this whole exercise surfaced.
