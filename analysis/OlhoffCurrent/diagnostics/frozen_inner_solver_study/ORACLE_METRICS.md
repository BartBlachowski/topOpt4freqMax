# Oracle metrics and exact-problem KKT

The denominator of gain recovery is bs_oracle-1, where beta_zero=lamref. A
negative recovery is permitted: the approximate solver can return beta below
the initial bound. Objective recovery alone does not establish feasibility.
The scaled beta gap, physical beta gap, recovery, d2, dinf, raw distances,
cosine, sign, bound and active-set metrics follow the frozen preregistration.

The primary sign mask is |drho_oracle|>=.9*move. All-sign agreement is reported
separately because many correct density-floor/ceiling increments are much
smaller than move. Bound agreement uses 1e-6 times each fixed box width and
is conditional on an oracle-active box bound. Full active agreement includes
interior entries. Gray and mid masks use only the frozen rho385.

For candidate x and nonlinear multipliers mu from its approximate solve,
re-evaluate every row and its gradient with production deltaLambda. Define
q=df0+J'*mu. The physical-box multiplier witness is

    xi_e=max(q_e,0), eta_e=max(-q_e,0).

For density coordinates it makes stationarity identically zero. It is NOT
an optimality certificate on its own. The decisive residual is the product of
these multipliers and the candidate's slack to the ORIGINAL problem box,
together with exact row complementarity and feasibility. Beta is interior in
all relevant candidates; its box multipliers are set to zero, so q_beta must
vanish independently. The primary scalar KKT is the max of normalized density
stationarity RMS, |q_beta|, normalized box complementarity, and raw row
complementarity. The separate primal and dual bars remain mandatory.

Original subsolv xsi/eta belong to its artificial alfa/beta box. Their
stationarity on the true problem is also reported, but transferring them to
physical bounds is not mathematically justified without checking the changed
slacks. The projected-box natural residual is another complementary diagnostic,
with q scaled by move/sRow0 and sRow0=RMS(F11/lamref).

The stored oracle dual witness supplies a global objective bound. Evaluating
f(x)-dual_bound is an objective-gap certificate only for feasible x. Its
nonnegative bound-complementarity decomposition is useful for spatial
localization. It is never confused with a fitted solver-specific multiplier.

New runs record metrics at each accepted approximation. Audit-only metric
calls are counted separately from algorithm function/gradient calls. GCMMA
trial evaluations use the production evaluator (which also computes gradients
internally); reported algorithm gradient requests and physically performed
combined evaluations must both be considered when comparing work. Reference
metric calls include additional oracle/KKT evaluations; their wall overhead
is shown separately from solve time, not charged as optimizer iterations.

Retained 1000/5000 points are re-evaluated with these same definitions. Their
original aggregate wall time is retained, but per-checkpoint timings are
unknown and are not linearly interpolated. Saved asymptote widths are reported
without inventing unavailable pre-step centres. No sparse checkpoint plot
implies that intermediate full-state metrics were measured.
