# CAUSAL_VERDICT — Parts 12, 13, 14 and 17

## Verdicts (mechanical, per AUDIT_PREREGISTRATION.md §11–12)

```
C480_CONTROL_EVIDENCE_PASS
C480_SOCP_SINGLE_FACTOR_PREFLIGHT_PASS
C480_FULL_RUN_SOCP_COVERAGE_FAIL
C480_SOCP_CAUSAL_RESULT_INCONCLUSIVE
OUTER_MODEL_REALIZATION_INCONCLUSIVE
SOCP_INNER_SOLVER_CANDIDATE_REJECTED
FILTER_FORMULATION_STUDY_STILL_DEFERRED
PERFORMANCE_CAMPAIGN_STILL_BLOCKED
```

Computed by `scripts/cs_analyze.py::verdicts` → `evaluations/analysis.json → verdicts`.

## Why each verdict

**Causal: E, `C480_SOCP_CAUSAL_RESULT_INCONCLUSIVE`.** Precedence rule 1: a
fail-closed termination (`SOCP_CERTIFICATE_FAILURE`, outer 15) occurred. This is the
preregistered "genuine evidence failure" case. The treatment has no endpoint (stage 1,
move 0.04, 14 accepted iterations), so none of A/B/C/D can be evaluated.
The rule trace in `analysis.json` also shows TWI = true and OBJ_COLLAPSE = true, from
ω₁ 148.03 vs 163.93 and M_nd 48.1 vs 26.3. Those flags only record that a 14-iteration
design is less evolved than a 386-iteration one. They do **not** indicate an
outer-globalization failure and are not used, because E takes precedence.

**Realization: INCONCLUSIVE.** 14 eligible steps, below the preregistered 20. The
descriptive statistics are excellent (median r = 1.033, no negative realized gain,
Σact/Σpred = 1.04), but cover only early stage 1.

**Coverage: FAIL.** 15 of 15 states were eligible N = 2 SOCPs; 14 certified,
1 not. The run did not end through the frozen controller or the cap.

**SOCP candidate: REJECTED.** Preregistered rule: coverage FAIL ⇒ REJECTED. What is
rejected is **the preregistered certified-SOCP inner solver as specified**, i.e. its
certificate procedure, which cannot certify the cone-apex (double predicted eigenvalue)
optimum that the sub-problem reached at outer 15. The evidence does **not** show a
wrong primal step:

- both backends returned the same feasible point with exit flag 1;
- a post-hoc exact-dual computation bounds its suboptimality by 8.6e-6 (≤ 0.03 % of
  the predicted gain);
- all 14 certified steps were well realized.

That diagnostic is excluded from the verdict and does not reopen it.

**Filter gate: STILL_DEFERRED.** The preregistered rule issues NOW_JUSTIFIED only for
causal B or C. No causal attribution to the inner solver was obtained, so no residual
grayness can yet be assigned to the filter.

## The main scientific question

> Does the broad gray C480 endpoint persist when every supported Du–Olhoff
> sub-problem (25) is solved to its certified global optimum?

**Unanswered.** H1 (inner solver is the major cause), H0 (grayness persists) and H2
(outer destabilization) were not tested to an endpoint. The run established four
things before it stopped:

1. **Exact steps are bang-bang, dynamically.** 13 of 14 accepted increments moved
   ≥ 99 % of elements to a bound. Effectively every gray element took the full ±0.04
   every step.
2. **The first-order model held for these steps.** r = 0.90–1.31, λ₁ rose every step,
   no reversal cycling. Coherence was declining: cos 0.84 → 0.22, core sign reversal
   up to 57 %.
3. **The exact trajectory was much faster than MMA.** At outer 14: ω₁ 148.0 vs 133.4
   (+10.9 %), M_nd 48 % vs 72 %, broad core 0.36 vs 0.70.
4. **It drove the two lowest modes to near-coalescence in 14 steps.** gap12 went
   2.68 → 0.014. The next exact sub-problem optimum then lies at the multiple-eigenvalue
   cone apex, where the preregistered certificate is unattainable.

Point 4 matters beyond this run. Max–min eigenvalue optimization seeks multiple
eigenvalues, and the apex is where it arrives. **A certified exact-SOCP inner solver
cannot be used for this problem until it has a validated certificate at the apex.**
