# Frozen inner-solver fidelity preregistration

Status: frozen before any new solver experiment. Only source inspection and
reference-file hash checks preceded this document. All outputs are audit-only.
No topology solve, outer density update, accepted rho update, controller
transition, production edit, or physics/filter/move change is authorized.
The production volume expression may be evaluated read-only to retain bitwise
arithmetic; no candidate density is stored or used in FE analysis.

## Oracle and identity gate

Verify all 90 files in the reference DATA_MANIFEST, source tree, authoritative
trajectory, rho386 endpoint and rho385 construction state. Re-evaluate the
retained oracle with production deltaLambda and the exact constraint arithmetic,
recompute its stored weak-duality witness and exact KKT, and check conic equality
at zero, oracle, retained MMA points and 20 seeded box points. Require the prior
KKT bars, scaled certified gap <=1e-8, value equality <=1e-12 and relative gradient
equality <=1e-10. Stop on failure. No FE/density update is needed to authenticate
the hashed frozen context independently reconstructed by the reference study.

## Source facts known before experiments

innerLoop preserves xold1/xold2, low/upp and it within the fixed inner solve.
Reset occurs on entry to a new outer subproblem. innerLoopRho additionally changes
coordinates and fails to pass dOff; it cannot serve as a persistence-only test.
The active mma_published copy has asyinit=.5, move=.5, asyincr=1.2, asydecr=.7,
asymptote distance clamp [.01,.2] times box width, raa0=1e-5, epsimin=1e-7.
The downloaded official smoptit.se archive has upper distance clamp 10.
The inactive historical copy has asyinit=.01 (and other changes).
These facts, not outcomes, determine the tests below.

## Test order and fixed variants

1. Identity gate; source/semantic inventory; GCMMA validation on official toy
   and analytic constrained quadratic (these are solver validation, no topology).
2. B0_CURRENT_REPEATED_MMA: production innerLoop at its own tolerance, then
   instrumented identical expressions through 500 calls. Check 19,50,100,500.
   Require bitwise production drho/beta and nInner=19, bitwise prior 500. The
   expensive prior 1000/5000 trajectory is reused only after hash authentication
   and checkpoint re-evaluation, labelled retained replay, never fresh work.
3. S1_PERSISTENT: same expressions and state transferred across an artificial
   function boundary at call 19 through call 50. Counter is global to this fixed
   problem. Must match B0 bitwise. This is the only possible persistence-only
   intervention here, since persistence already exists. No prior outer state is
   imported. It cannot test cross-outer warm starts under the present lock.
4. S2_ASYINIT_001: change only asyinit .5 to historical .01; clamp/move untouched.
   This preregistered historical negative control is not a proposed canonical
   correction; production already equals the published .5 initialization.
5. S3_CANONICAL_CLAMP: change only both maximum asymptote distances .2 to 10.
6. S4_SUBSOLV_ACCURACY: change only epsimin 1e-7 to 1e-12. One accuracy A/B,
   not a tolerance sweep. 1e-12 is below the raw complementarity budget implied
   by sRow0*move*1e-6 (approximately 6e-12). Inner tolInner is unchanged/offline.
7. S5_UNIT_BOX: affine reparametrization of drho to [0,1] using its exact fixed
   box width; bs remains beta/lamref. Transform all values/gradients/bounds and
   state consistently. No density coordinates or density updates are formed.
8. G0_UNSAFE: official GCMMA approximation/asymp/raa initialization with
   conservative acceptance disabled; G1_GCMMA: identical with official
   concheck/raaupdate enabled. Only G0 vs G1 isolates the safeguard. Neither
   comparison against B0 isolates globalization (multiple differences).
9. G2_GCMMA_ACCURATE: G1 with only epsimin=1e-12, to distinguish accuracy floor
   from conservativeness. concheck allowance remains 1e-7, same as G1, so this
   changes only approximate subproblem accuracy.
10. S34_CLAMP_ACCURACY: S3 plus S4, the fourth cell of a prespecified 2x2
    factorial. Its joint result is an interaction result, not a single-factor
    primary cause. No other factor changes.
11. Direct SOCP: twice rerun prior selected x-form/schur/1e-10 configuration,
    independently evaluate certificate, compare determinism, assembly/solve/
    certification costs separately. No configuration search.

Independent processes may overlap after B0 reproduction; logical analysis order
above is fixed. All fixed-state methods start at [0;1]. Initial x, rho, F, dOff,
fJJ, box, constraints, c=1000,a=0,d=0 and spectral/volume scaling are locked,
except the explicitly named single factor. Raw solver auxiliary dual warm starts
are not introduced. Production stopping diagnostics are recorded offline.

## Work budgets and stopping

Each nontrivial variant: 500 approximation solves including rejected conservative
trials; <=1800 seconds solver wall time, checked between calls; <=16 corrections
per approximation. Never accept an unverified nonconservative trial in G1/G2.
No tolInner sweep. No convergence stop before 100 calls; after 100, stop only
if all oracle fidelity bars hold for five consecutive accepted approximations.
No extension based on favorable outcomes. B0 always completes its 500-call budget
unless numeric/identity failure. Failed or invalid tests are inconclusive,
not repaired by changing scientific factors. Implementation bugs can be fixed
with explicit logs before a valid experiment; all changes disclosed.
Checkpoint metrics at 1,5,10,19,20,50,100,200,500 and available 1000/5000;
all-iterate scalar metrics for new runs. Save state/duals at checkpoints.

## Metrics and thresholds

beta_zero=lamref (evaluate zero); gainRecovery=(bs-1)/(bsOracle-1).
Report raw beta, beta gap, normalized gain gap, raw norms, d2=norm(drho-drhoO)/
norm(drhoO), dinf=max(abs(drho-drhoO))/move, cosine, sign agreement where
abs(drhoO)>=.9*move, and all-sign agreement separately. Bound agreement uses
1e-6*boxWidth, same signed box bound as oracle, denominator oracle-bound entries;
active-set agreement includes interior and exact nonlinear rows (abs(c)<=1e-8).
Gray=0.1<rho<0.9; mid=0.4<=rho<=0.6. Void rho<=.1, solid rho>=.9;
gray shell=gray outside mid, gray core=mid. No mask based on updated density.

KKT uses exact production gradients at the candidate, with nonnegative nonlinear
multipliers and implied box multipliers max(q,0),max(-q,0), q=df0+J'*mu.
Stationarity alone is tautological with these implied multipliers and is NEVER
a success measure: exact box and row complementarity, feasibility, and beta
stationarity are mandatory. Also retain the actual subsolv multipliers and
report their stationarity and artificial-bound mismatch. Primary scalar KKT is
the maximum of normalized RMS stationarity, absolute beta stationarity,
normalized box complementarity, and absolute row complementarity (each unscaled
bar stated below); projected box stationarity is an additional diagnostic.
Reference dual witness gap at any point is reported separately and is a
suboptimality certificate only if the point is feasible.

Fidelity PASS requires: abs(bs-bsO)<=1e-8, gainRecovery>=.999, d2<=.01,
dinf<=.1, sign agreement>=.99, bound agreement>=.99, max exact constraint<=1e-8,
box violation<=1e-10, dual negativity>=-1e-10, RMS stationarity/sRow0<=1e-6,
max stationarity/sRow0<=1e-5, abs(beta stationarity)<=1e-6,
max row complementarity<=1e-6 and max box complementarity/(sRow0*move)<=1e-6.
Use sRow0=RMS(F11/lamref). Report all failures; never relax a bar.

Material improvement at equal 100 and 500 calls: >=.10 higher gain recovery
AND >=20% lower d2 AND no worsening of primal feasibility beyond 1e-8; both
checkpoints required for STRONG causal evidence (or earlier sustained fidelity).
At one checkpoint only: MODERATE. Improvement in only one primary outcome:
WEAK ASSOCIATION. Worsening uses symmetric bars. Identical within 1e-10 in
primary metrics: no material effect. Mixed/missing evidence: inconclusive.
No causal attribution to untargeted factors or solely to final small step.

Offline stops: first production relative step<.05 after 5; first objective gap,
KKT and oracle distance bars individually; joint fidelity; feasible iterate;
active sets unchanged for 20 consecutive iterates. Report first and persistence,
Pearson/Spearman relStep vs d2/gap, false convergence and reversals. First hits
in sparse retained checkpoints are upper bounds, never exact stopping times.

## Selection and gates

Choose at most one future candidate; require valid baseline, validated method,
exact-problem KKT and fidelity or quantified reliable approach, deterministic
repeat, no topology-specific tuning, mathematical legitimacy and feasible cost.
For SOCP promotion, prove N=1 LP, N=2 SOC and conditional general-N affine PSD
representation, distinguish offsets consistent/inconsistent with lam, all
spectral rows, next-mode and offDiag=false equalities, and nonlinear volume.
A general-N SDP formula alone is not a validated fallback implementation.
If generalization/fallback or GCMMA validation is unresolved, withhold future
causal run rather than silently changing problem (25). No run is executed here.
Filter study remains deferred and performance campaign blocked unless specific
contrary proof is established. Later exploratory variants must be marked
POST-HOC DIAGNOSTIC and excluded from causal verdicts.
