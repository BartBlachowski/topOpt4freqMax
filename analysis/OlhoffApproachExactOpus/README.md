# OlhoffApproachExactOpus — clean-room reconstruction of Du & Olhoff (2007)

Independent, paper-first reconstruction and discrepancy analysis of Du & Olhoff (2007)
"Topological design of freely vibrating continuum structures…" (Struct Multidisc Optim 34:91-110)
+ erratum (34:545). This is **not** a refactor of `OlhoffApproachExact`; the algorithm was
re-derived from the PDFs before any code was inspected.

## Deliverables (read in order)

1. **[specification/du_olhoff_2007_spec.md](specification/du_olhoff_2007_spec.md)** — *primary deliverable.*
   Every equation, loop, variable, constraint, stopping criterion, benchmark, assumption,
   ambiguity, and undocumented detail, each classified **[E]** explicit / **[I]** implied / **[N]** not specified.
   Erratum folded in (Δ in eq. 25d/26f/26g; Fig. 2 caption fix).
2. **[paper_reconstruction/algorithm.md](paper_reconstruction/algorithm.md)** — paper-only pseudocode of Fig. 1,
   with the unspecified gaps marked rather than filled.
3. **[notes/disconnection_analysis.md](notes/disconnection_analysis.md)** — Phase 2: does the *formulation*
   permit disconnected topologies / localized modes / disconnected optima / path-dependence? (Yes to all;
   only localized modes have a dedicated safeguard.)
4. **[comparisons/paper_vs_implementation.md](comparisons/paper_vs_implementation.md)** — Phase 3: spec vs.
   the existing `OlhoffApproachExact` MATLAB code — exact matches, [N] choices, deviations, and a ranked
   root-cause analysis of the benchmark mismatch (optimizer-side, not physics-side).
5. **[experiments/initial_frequency_verification.md](experiments/initial_frequency_verification.md)** +
   **[implementation/fe_verify.py](implementation/fe_verify.py)** — Phase 4 evidence: independent FE
   reproduces the published initial frequencies (SS 68.7 / CS 104.1 / CC 146.1) to <1%, and shows they are
   the **p=3 penalized** values with a **mid-height-pin** SS support.
6. **[implementation/PHASE4_DECISION.md](implementation/PHASE4_DECISION.md)** — Phase 4 decision: a full new
   optimizer is **not** required as a deliverable; the spec is validated and the discrepancy characterized.
   A focused path-control + basin experiment is the only justified next code, if benchmark reproduction is wanted.
7. **[notes/branch_selection_investigation.md](notes/branch_selection_investigation.md)** — Phase 5: *why* the paper
   (connected Fig. 3c) and the reconstruction (disconnected two-block) choose different branches. Hypothesis ranking,
   quantified morphology, basin map (bifurcation at iter 2–3), MMA determinant-subproblem audit, literature, and the
   minimal discriminating experiment set. Backed by [experiments/morphology_analysis.py](experiments/morphology_analysis.py).

8. **[experiments/exp1_meshrefinement_result.md](experiments/exp1_meshrefinement_result.md)** — Exp-1 executed (real
   MATLAB solver, unchanged settings, mesh only 40×5→80×20): the disconnected two-block design becomes a **connected
   X-braced full-span truss** (Fig. 3c topology class) at 80×20. Confirms mesh coarseness drove the disconnection;
   caveat: disconnected designs remain higher-ω under du2007_c1, and the run ends N=1, ω₁=314 < 456.4 (not converged).

## Phase 5 headline

The disconnected branch is **genuinely higher-frequency only under (40×5 coarse mesh) AND (du2007_c1 near-massless-void
mass)** — under physical/linear mass the same designs collapse to ω₁=16–58 (up to 37× lower), and at 40×5 no *connected*
design exceeds ~288 while disconnected reach 462. So the reconstruction's disconnection is most likely a
**discretization + mass-model artifact**; Fig. 3c is the legitimate optimum of the properly-resolved problem. The
single highest-value test is **one finer-mesh (80×20) run**. This *refines* the prior on-record conclusion that
"disconnected is genuinely superior, not numerics."

## Headline conclusions

- The paper's equations are unambiguous *after the erratum*; the reproduction risk lives entirely in the
  **[N]-classified numerical choices** (mesh, filter radius, mass model, multiplicity tolerance, continuation
  schedule, MMA params, trust region).
- The existing `OlhoffApproachExact` is an **equation-faithful** transcription (including the hard
  multiple-eigenvalue subeigenvalue problem) whose **default settings make it diverge** (full-step increment ⇒
  N=1 bang-bang limit cycle). The benchmark gap is **optimizer-side**, independently confirmed by the FE check.
- The formulation **permits — and for the clamped-clamped case may favour — a disconnected two-block optimum**
  distinct from the paper's connected Fig. 3c. Matching the figure is partly a *basin* question the paper does
  not pin down.
