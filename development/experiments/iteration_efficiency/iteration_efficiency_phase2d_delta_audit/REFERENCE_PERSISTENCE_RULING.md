# WP12 / WP13 — Ruling on the reference / persistence gap
READ_ONLY_INDEPENDENT_DELTA_AUDIT

## The two propositions, kept separate

    Q1  Is Eq. (4a) scientifically suitable for the common post-hoc evaluator?
    Q2  Has float32 Olhoff trajectory storage been qualified under Eq. (4a)?

They are answered separately below. This WP rules on the **reference/persistence evidence
gap only**. A different finding (D1) governs the overall verdict and is not part of this
ruling.

## The framework questions

**1. Does Eq. (4a) alter only the pointwise evaluator mapping?**
Yes. Verified at file, call-path and numerical level (`IMPLEMENTATION_DELTA_AUDIT.md`). No
rule in `reference_phase.m`, `scan_persistence.m` or `measurement_budget.m` changed.

**2. Are reference and persistence deterministic downstream functions of the pointwise
quality sequence?**
Yes, and strictly so. `reference_phase(Q, H0)` takes only the 3-column quality matrix and
the hard-gate vector; `scan_persistence(passMatrix, P)` takes only a boolean matrix;
`measurement_budget(B0, b_ref, P, B_ref)` takes only scalars. Independently confirmed by
re-implementing all three and reproducing the stored Phase-2B outputs exactly
(`b_ref` 2200/2100, `k_enter` 233/315/609 and 232/309/524, `k_cert`, `Q_ref`, `B_meas`).

**3. Has the amended pointwise mapping been shown numerically stable over all available
relevant states?**
Yes, and independently. 236 genuine paired states, 9 branch-straddle states, 1600
production states, 8 mesh final states. Amended E2/E3 perturbation is indistinguishable
from the branch-free E1 control in every experiment (ratio 1.012 on the trajectory).

**4. Could a tiny remaining pointwise perturbation nevertheless move `b_ref`, `k_enter` or
`k_cert` because a trajectory lies arbitrarily close to a threshold?**
This is the question Phase 2D answered by assertion. It is answerable with evidence, and this
audit answered it. Worst-case interval propagation on the only reference-length trajectory in
the repository gives a **critical relative Q perturbation of 5.16e-05** (binding case,
q = 0.99), against an amended float32 perturbation of **5.60e-08** — a **922× margin** under
adversarial sign assumptions. Zero of 3200 states sit within twice the amended perturbation
of an acceptance threshold. Under Eq. (4) the same margin was **exceeded by 439×** on 48.7%
of states, which is exactly why Phase 2B failed.

**5. Is there existing evidence about minimum decision margins?**
Yes — now there is. `WP12_BREF_BLOCK_MARGINS.csv`, `WP12_ACCEPTANCE_MARGINS.csv`,
`WP12_CRITICAL_PERTURBATION.csv`. The b_ref decision on the 96x12 trajectory sat
**1.749e-04** (gain units) from flipping; the tightest acceptance margin over 3200 states was
**5.80e-06**. Both are three to four orders above the amended perturbation.

**6. Does methodology refreeze require proving storage-precision equivalence, or merely
defining a scientifically sound evaluator?**
The frozen contract already separates these. `trajectory_storage.single_precision_permission`
is conditioned on a *separate* qualification artifact, and `production_preflight.m` enforces
it with `checks.olhoff_lossless_trajectory`, which requires
`validation_outputs/olhoff_new_trajectory_precision_qualification.json`. **That file does not
exist**, verified directly. Storage qualification is therefore a production gate, not a
refreeze precondition, by the methodology's own construction.

**7. Is the single-vs-double qualification explicitly scheduled as a separate post-refreeze
requirement?** Yes — Phase 2D states it (item 40) and the preflight gate enforces it.

**8. Would requiring a new 3200-update optimization now turn an evaluator methodology audit
into an optimization experiment?** Yes, for the *amendment* question. It would not for the
*storage* question, where it is unavoidable.

**9. Would refreezing without any end-to-end evidence leave a known freeze-critical semantic
path untested?** The reference/persistence path is exercised end-to-end and validated — under
Eq. (4). What is untested under Eq. (4a) is the *numerical* transfer, and that is now bounded
at 922×. The *semantic* path is unchanged by construction.

**10. Can the missing evidence be obtained without altering the methodology decision itself?**
Yes: one 96x12 (or comparable) reference-length run with double density snapshots retained.
It is a measurement, not a methodology choice.

## RULING

    REFERENCE_PERSISTENCE_GAP =
        B — NON-BLOCKING FOR REFREEZE, MANDATORY IN THE NEW POST-REFREEZE
            PRECISION QUALIFICATION

Justification, tied to findings:

- The amendment alters the pointwise mapping only; reference and persistence are
  deterministic functions of that mapping and their **rules** are provably untouched.
- The residual pointwise perturbation is bounded by measurement at 5.60e-08 (float32) and
  ≲1e-12 (double), in both cases indistinguishable from the branch-free E1 control.
- The frozen decisions on the one reference-length trajectory available tolerate 5.16e-05
  before any of them can move. The margin is **922×** under worst-case sign assumptions.
- The methodology itself places storage qualification behind a separate, currently-failing
  preflight gate. Refreeze does not assert storage adequacy; it defines the instrument.

**Residual uncertainty, stated plainly.** The margins were measured on the pre-amendment Q
sequence. They characterise the scale at which the frozen rules operate (1e-4 to 1e-5), not
the amended trajectory's specific margins. This audit does **not** claim end-to-end
equivalence has been demonstrated; it claims the gap is bounded and correctly deferred.

## Answers to Q1 and Q2

**Q2 — float32 Olhoff storage under Eq. (4a): NOT YET QUALIFIED.** No end-to-end evidence
exists and the preflight gate correctly blocks production. Requirements are specified in
`PRECISION_REQUALIFICATION_REQUIREMENTS.md`.

**Q1 — Eq. (4a) as the common evaluator: NOT SUITABLE**, but for a reason unrelated to this
WP. The reference/persistence gap does not block refreeze. Finding **D1** does. See
`PHASE2D_DELTA_AUDIT.md`.
