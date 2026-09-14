# Direct SOCP promotion gate

SOCP_INNER_SOLVER_CANDIDATE_JUSTIFIED

N=2 equivalence, every actual reconstruction constraint, frozen certificate
reliability, and deterministic repeat are established. The current C480
configuration fixes N=2 for the whole run. N=1 and the diagonal equality route
reduce to LP; full coupling for N>2 requires an SDP under the conditions proved
in SOCP_GENERALIZATION.md. No general-N SDP implementation has been validated.

The treatment must fail closed: on N!=2, inconsistent offsets, nonlinear volume without a proved equivalent representation, asymmetric/inconsistent coefficients, or a failed primal/dual certificate, stop before accepting any outer density update. The analytical N=1 LP and general-N SDP cases define the required dispatch but are not silently treated as validated software. At a repeated predicted sub-eigenvalue, require a valid generalized conic KKT certificate or stop. Do not switch to an uncertified MMA result or drop a row to keep the treatment running. The unchanged production solver remains available outside this proposed treatment as the existing compatibility path; it is not a fidelity-certified fallback within the experiment. This explicitly bounded rejection behavior is the fallback for unsupported states.

This gate does not authorize a universal SOCP replacement or any production
repair in this task. Every prospective solve must be accepted using verified
exact constraints/KKT and a valid dual bound, not coneprog's exitflag alone.
