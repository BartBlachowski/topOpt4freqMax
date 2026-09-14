from pathlib import Path
import json,hashlib,re
S=Path(__file__).resolve().parents[1];E=S/'evaluations';M=json.loads((S/'METRICS.json').read_text())
assert M['complete'],'Wait for all preregistered experiments before final reporting.'
V=json.loads((E/'verdict_decisions.json').read_text())
D=M['methods'];T=M['structure'];B=json.loads((E/'barrier_diagnostic.json').read_text());P=json.loads((E/'provenance_before.json').read_text())
def wr(n,s):(S/n).write_text(s.strip()+'\n')
def num(x):return 'not observed' if x is None else f'{x:.6g}'
def pct(x):return f'{100*x:.3f}%'
def last(n):return D[n]['last']
def table(ns):
 s='| Method | Calls | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Fidelity |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
 for n in ns:
  h=last(n);s+=f"| {n} | {h['calls']} | {pct(h['gainRecovery'])} | {num(h['d2'])} | {num(h['dinf'])} | {pct(h['signAgreement'])} | {pct(h['boundAgreement'])} | {h['kkt']:.3e} | {'PASS' if h['fidelity'] else 'FAIL'} |\n"
 return s
r=json.loads((E/'B0_RETAINED_5000.json').read_text())['history'];cps={h['iter']:h for h in r}
a=last('B0_CURRENT_REPEATED_MMA');b=cps[19];z=cps[5000];g=last('G2_GCMMA_ACCURATE');s4=last('S4_SUBSOLV_ACCURACY');s34=last('S34_CLAMP_ACCURACY')
wr('ORACLE_IDENTITY.md',f'''# Oracle identity

{V['oracle']}

All 90 reference-manifest entries match. Authoritative endpoint rho386:
`{M['oracle']['rho386']}`. The subproblem was constructed at rho385:
`{M['oracle']['rho385']}`. These are different states, deliberately distinguished.
Outer=386, stage=3, mesh=480x60, move=.01; config and implementation tree match.
The construction context is the hashed, previously independently reconstructed
context, checked against the authoritative trajectory. No new FE solve was used.

Production re-evaluation gives beta={M['oracle']['beta']:.12f},
bs={M['oracle']['bs']:.12f}. The independently recomputed dual gap is
{M['oracle']['certified_gap']:.3e}; exact certificate KKT passes.
Across {M['oracle']['equivalence']['points']} fixed test points, maximum
value discrepancy is {M['oracle']['equivalence']['max_value_error']:.3e},
maximum relative gradient discrepancy {M['oracle']['equivalence']['max_gradient_relerr']:.3e}.
The production and conic feasible sets remain equivalent.

Both fresh SOCP repetitions reproduce the oracle vector bit-for-bit. They use
the certificate multipliers, since coneprog's returned duals remain inaccurate.
See `evaluations/oracle_identity.json` and `SOCP_COST.json` for both dual sets.
''')
cptext='| Iteration | Source | Gain recovery | d2 | dinf | Sign agreement | Bound agreement | Exact KKT | Relative step |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
for k in [19,50,100,500,1000,5000]:
 h=cps[k];cptext+=f"| {k} | {'fresh + authenticated reference' if k<=500 else 'authenticated retained replay'} | {pct(h['gainRecovery'])} | {h['d2']:.6f} | {h['dinf']:.6f} | {pct(h['signAgreement'])} | {pct(h['boundAgreement'])} | {h['kkt']:.3e} | {h['relStep']:.3e} |\n"
wr('BASELINE_REPLAY.md',f'''# Exact baseline reproduction

The production innerLoop itself returned bitwise-identical drho and beta at
nInner=19 under its unchanged tolerance. The independently instrumented replay
matches at 19 and matches the earlier 500-call endpoint bit-for-bit. A distinct
persistence-boundary execution also matches call 50 bit-for-bit.

{cptext}

Fresh B0 work: 500 calls plus the separate 19-call production reproduction;
solver wall for the 500 calls: {D['B0_CURRENT_REPEATED_MMA']['out']['solverWall']:.3f}s.
The reference 5000-call replay cost 5984.418s in its original study. It was not
rerun in full: all 113 retained checkpoints were hash-authenticated and freshly
re-evaluated, avoiding a redundant roughly 100-minute control. Per-checkpoint
retained timings are unavailable. This limitation was preregistered.

At 5000, objective and design distances have decreased from production-19, but
only {pct(z['gainRecovery'])} of gain is recovered and d2 remains {z['d2']:.6f}.
No oracle-active bound is reached within the declared bound tolerance. Full
active agreement is {pct(z['activeAgreement'])}, accounted for by oracle-interior
entries, not recovery of the saturated optimum. No convergence claim follows.

All checkpoint quantities, duals and asymptotes available in the original data
are in MAT/JSON; MASTER_METRICS.csv supplies the common scalar definitions.
''')
wr('PERSISTENT_STATE_TEST.md',f'''# Persistence test

{V['persistence']}

Source inspection proves all required MMA approximation history persists within
B0. S1 serializes x, xold1, xold2, low, upp and the counter at call 19, restores
them, and continues the same fixed problem. Its call-50 vector matches B0
bit-for-bit. The state transfer is an identity, not an improvement.

Across new outer problems production resets this state. That is a separate,
untested warm-start question and cannot explain a reset between calls that does
not occur. The Newton workspace inside subsolv is reset every approximate solve,
as in the official algorithm; approximate-solve accuracy is studied separately.
innerLoopRho cannot isolate persistence because it changes coordinates and omits
dOff. See the pre-experiment semantic diff in RECONSTRUCTION_CHOICES.md.
''')
wr('ASYMPTOTE_INITIALIZATION_TEST.md',f'''# Asymptote initialization

Production already uses the official .5 initialization for the first two calls.
The historical .01 value is verified in the inactive as-found copy. S2 changes
only that value; it does not import the other historical settings.

{table(['B0_CURRENT_REPEATED_MMA','S2_ASYINIT_001'])}

The historical value changes the trajectory but does not repair oracle fidelity.
It cannot be the nonstandard initialization of the active frozen path, which
already uses .5. Initial low=xmin and upp=xmax are overwritten on calls 1 and 2.
No parameter search was performed.
''')
wr('COORDINATE_SCALING_TEST.md',f'''# Coordinate scaling

S5 uses t_e=(drho_e-lo_e)/(hi_e-lo_e), with t in [0,1]. Beta remains beta/lamref.
All physical values, derivatives, fixed box bounds and state are transformed
consistently; the stopping step is evaluated in physical increment coordinates.
No candidate density coordinates are formed.

{table(['B0_CURRENT_REPEATED_MMA','S5_UNIT_BOX'])}

The final S5 solver time is {D['S5_UNIT_BOX']['out']['solverWall']:.3f}s for 500
calls, compared with {D['B0_CURRENT_REPEATED_MMA']['out']['solverWall']:.3f}s in B0.
This is a large observed numerical-work effect, subject to shared-host timing
limits. It does not meet the registered joint fidelity or strong-effect bars.
The unchanged 1e-7 approximate-solve target still imposes its accuracy floor.

Mathematically the MMA reciprocal approximation transforms consistently under
this positive affine map: gradients acquire a box-width factor, asymptote
intervals lose it, and the 1/width regularizer compensates. The code's 1e-5
minimum-width guard is inactive on both representations. Finite Newton stopping,
residual norms, initialization safeguards and floating-point cancellation are
not generally invariant. Thus an affine reformulation can change numerical
work and the finite-accuracy path without changing problem (25). A separate
beta/objective or constraint-row scaling intervention was not registered and is
NOT ISOLATED. No extra scaled variant was added after seeing the results.
''')
wr('ASYMPTOTE_UPDATE_TEST.md',f'''# Asymptote update policy

Both source copies use the 1.2/.7 sign-based history update and the .01 minimum
asymptote distance. The active copy caps maximum distance at .2 widths; official
Svanberg source caps it at 10. Its name and README did not disclose this remaining
local modification. S3 changes exactly those two symmetric cap expressions.

{table(['B0_CURRENT_REPEATED_MMA','S3_CANONICAL_CLAMP','S4_SUBSOLV_ACCURACY','S34_CLAMP_ACCURACY'])}

The 2x2 comparison separates cap and approximate-solve accuracy. S34 is a joint
intervention; it cannot alone assign causality to either factor. All four edges
and the predefined material-effect bars appear in CAUSAL_ATTRIBUTION.md.
The outer move box is unchanged throughout. A maximum MMA asymptote distance
is different from the physical increment move limit.
''')
stoptext='| Method | Relative step | Objective gap | KKT | Distance | Feasible | Active stable 20 | Joint fidelity |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for n,s in M['stopping'].items():
 stoptext+='| '+n+' | '+' | '.join(str(s[k]) if s[k] is not None else 'never observed' for k in ['first_production','first_objective','first_kkt','first_distance','first_feasible','first_active_stable','first_fidelity'])+' |\n'
st=M['stopping']['B0_CURRENT_REPEATED_MMA'];stlong=M['stopping']['B0_RETAINED_5000']
wr('STOPPING_RULE_ANALYSIS.md',f'''# Offline stopping diagnosis

{V['stopping']}

{stoptext}

Sparse retained first hits are only upper bounds on event time, not exact
iteration timestamps. Active stabilization is unavailable between retained
checkpoints. KKT first hits here use the preregistered scalar bars; feasibility,
objective/design distance and all remaining KKT bars are required by joint fidelity.

B0 first stops at 19, with d2={b['d2']:.6f} and gain recovery={pct(b['gainRecovery'])}.
Its fresh-500 relative-step/d2 Pearson correlation is
{st['pearson_relStep_d2']:.6f}, Spearman {st['spearman_relStep_d2']:.6f};
these trend correlations do not validate optimality. Every relative-stop hit
in B0 fails joint fidelity. The retained 5000 endpoint has step={z['relStep']:.3e}
but d2={z['d2']:.6f}, KKT={z['kkt']:.3e}, and bound agreement=0.
B0's gain decreases {st['gain_decreases']} times over 500 new iterations;
small relative steps coexist with nonmonotone objective progress.

For absolute oracle objective gap, fresh-B0 Pearson correlation with relative
step is {st['pearson_relStep_absGap']:.6f} and Spearman is
{st['spearman_relStep_absGap']:.6f}. The production step criterion holds at
{st['persistence']['production']['observed_hits']} of 500 recorded iterates,
with a longest consecutive streak of
{st['persistence']['production']['longest_consecutive_iterations']}; joint
fidelity holds at zero. METRICS.json records hit counts, terminal status and
longest consecutive streaks for every diagnostic and method. Sparse retained
records cannot establish consecutive-iteration persistence and are marked null.

Lowering tolInner alone did not solve the known 5000-call replay and is not
shown capable of meeting fidelity. No broad tolerance sweep was run. The fixed
subsolv barrier floor provides an independent accuracy limitation, so merely
increasing the number of repetitions is not a demonstrated cure. This is not a
claim that every scalar trend correlation is zero.
''')
wr('GCMMA_TEST.md',f'''# GCMMA and conservative acceptance

{V['gcmma']}

The repository had no GCMMA routines. The official GPLv3 distribution was
obtained from [Svanberg's site](https://www.smoptit.se/), frozen with SHA-256,
and retained audit-only with its license. Numerical subsolv expressions match
the local copy; the upstream version adds a diagnostic print near iteration
limits. Toy reproduction and an analytic constrained quadratic validate the
interface. The printed nine-iteration toy sequence alone has KKT 1.447e-5;
continuing it reaches 5.420e-7 and passes the validation bar.

{table(['B0_CURRENT_REPEATED_MMA','G0_UNSAFE','G1_GCMMA','G2_GCMMA_ACCURATE'])}

G0 and G1 isolate only the conservative acceptance/correction switch. Their
500-call trajectories agree exactly and G1 requires zero corrections. The
maximum true-minus-approximation value is -4.803e-9: all its accepted trials
are conservative even without the 1e-7 allowance. B0's corresponding maximum
is -2.861e-7. Therefore lack of a conservative safeguard is not an observed
failure mechanism on these paths.

G2 changes only approximate-solve epsimin to 1e-12; concheck remains at 1e-7.
It reaches {pct(g['gainRecovery'])} recovery, d2={g['d2']:.6f},
dinf={g['dinf']:.6f}, and KKT={g['kkt']:.3e}. Its strict fidelity result is
{'PASS' if g['fidelity'] else 'FAIL'}. Correction and approximation iterations,
including every raa value and trial check, are retained in the JSON/MAT files.
The tight solve emits ill-conditioned-system warnings; exact-problem residuals,
not suppressed warnings or relative steps, determine the verdict.

The comparison of native GCMMA against B0 changes cap and regularization policy
as well as method; it is not an isolated globalization comparison. No global
convergence claim beyond this frozen test is made.
''')
costtable='| Repeat | Assembly s | Solve s | Certification s | Iterations | Certified gap | Bitwise oracle |\n|---|---:|---:|---:|---:|---:|---|\n'
for c in M['socp']:costtable+=f"| {c['replicate']} | {c['assembly_s']:.4f} | {c['solve_s']:.4f} | {c['certification_s']:.4f} | {c['iterations']} | {c['certificate']['aligned']['gap']:.3e} | {c['bitwise_oracle']} |\n"
wr('SOCP_ORACLE_COST.md',f'''# Direct SOCP computational scale

{costtable}

The x-coordinate/schur/1e-10 configuration is exactly the one chosen in the
reference study; no new tuning was performed. Both repeats satisfy every
preregistered fidelity criterion. Assembly excludes loading the authoritative
trajectory and building the shared model/filter objects.

The process containing both repetitions plus read-only structural diagnostics
peaked at 1,375,125,504 bytes RSS (1.281 GiB), measured by `/usr/bin/time -l`.
This is total MATLAB process memory, not an isolated cone-factorization estimate.
Per-repetition assembly-object byte estimates are in SOCP_COST.json.

coneprog exitflag remains -7. Its returned dual residual is not acceptable;
the separately verified conic dual witness supplies a valid bound and exact KKT
certificate. Both returned-dual and certificate residuals are recorded. Success
is based on the latter's checked feasibility/complementarity/gap, not the exitflag.

These are indicative costs on a shared host with concurrent audit processes.
They are not a nine-mesh performance benchmark. No solver performance tuning.
''')
boundtable='| Category | Count |\n|---|---:|\n'+''.join(f'| {n} | {c} |\n' for n,c in zip(T['bound']['names'],T['bound']['counts']))
wr('BOUND_STRUCTURE.md',f'''# Exact active-bound structure

{boundtable}

At tolerance 1e-6 times local box width, {pct(T['bound']['any_fraction'])} of
variables are at some bound, while {pct(T['bound']['move_fraction'])} are at
specifically ±move. Density-floor and density-ceiling bounds account for 19,090
variables. Thus the earlier ~33.7% move statistic and ~99.94% total saturation
measure different sets; neither is a contradictory count.

Coincidence uses equality of raw physical and move bounds to 1e-14 and has zero
members here. Interior has 16 members. Of 8,272 strict-gray elements, 12 are
interior at the declared tolerance; of 3,422 mid-gray elements, 6 are interior.
The smallest absolute gray oracle increment is 0.0009775965, so the task's
statement that EVERY gray increment is at full move is not literally supported
by the retained oracle. Its hash and certificate still pass. Threshold-dependent
activity is a description, never a substitute for KKT complementarity.
''')
wr('ORACLE_SIGN_STRUCTURE.md',f'''# Exact sign condition and first-mode approximation

For the oracle's exact nonlinear multipliers, let

    g_e = v_min' F_e v_min / lamref,
    q_e = -mu1*g_e - mu2*g_second,e - mu3*fJJ_e/lamref + mu4/Vtot.

The gradient includes both off-diagonal terms and the eigenvector induced by
the affine SOC coupling. Exact KKT gives q_e - xi_e + eta_e=0, xi,eta>=0.
Consequently q_e>0 requires the lower available box bound; q_e<0 requires the
upper available bound; an interior coordinate requires q_e=0. A zero cost can
also occur at a bound. The available bounds may be density-limited, not ±move.
This is necessary and, together with primal/dual feasibility and complementarity
on the convex problem, sufficient. No nonlinear coupling was discarded.

The oracle multipliers are approximately mu=[1,0,0,0.729113644725], with
volume threshold {T['threshold']['volume_threshold']:.12g}. Its SOC direction is
[-0.999999999615,-2.77583885322e-5]. Effective sensitivity differs from F11/lamref
by {T['threshold']['effective_vs_F11_rms']:.6g} relative L2. All classified lower
and upper oracle bounds have the correct reduced-cost sign. There are 14,314
positive and 14,486 negative reduced costs; only 22 have magnitude below
1e-6 times the maximum. Small nonzero numerical costs at interior-like points
are handled by the certified complementarity tolerance, not asserted exact zeros.

A greedy solution of the first-mode linear functional with the same volume/box
was evaluated using the full production spectral equations. It loses only
{T['first_mode_rule']['gainLoss']:.9g} of oracle gain (about 0.06%), reproducing
the preceding ~6e-4 observation. Yet d2={T['first_mode_rule']['d2']:.6f} and
dinf={T['first_mode_rule']['dinf']:.6f}; small objective error can conceal a few
full sign reversals. This is an interpretation aid, not a certified replacement
for the SOC. Fields and maps are retained in structure.mat.
''')
lt='| Iter | Class | N | Sign error (all signs) | Bound error | Share squared distance | Reduced-cost loss | Raw KKT RMS | Sensitivity RMS amplification | Amplification-distance Spearman |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for q in T['localization']:
 for c in q['classes']:lt+=f"| {q['iter']} | {c['name']} | {c['n']} | {pct(c['sign_error'])} | {pct(c['bound_error'])} | {pct(c['share_squared_distance'])} | {c['reduced_cost_contribution']:.6g} | {c['kkt_raw_rms']:.4g} | {c['sensitivity_rms_amplification']:.4f} | {c['amp_disagreement_spearman']:.4f} |\n"
wr('FAILURE_LOCALIZATION.md',f'''# Spatial localization against the oracle

{lt}

At 5000, gray shell plus gray core contain
{pct(sum(c['share_squared_distance'] for c in T['localization'][-1]['classes'][1:3]))}
of squared design error; void contributes only
{pct(T['localization'][-1]['classes'][0]['share_squared_distance'])}. The disagreement
is broad, with a strong gray design-distance component, despite the earlier
void-dominated raw residual normalization. Void and solid regions still account
for substantial effective objective loss. All oracle-active bound agreement is
zero for B0 at the strict bound tolerance, across density classes.

Filtered first-mode sensitivity RMS amplification in void is 30.6899x, matching
the prior observation. The within-void Spearman correlation of amplification
ratio with absolute oracle disagreement is negative ({T['localization'][-1]['classes'][0]['amp_disagreement_spearman']:.4f}),
so these data do not support a simple positive amplification-causes-disagreement
story. Elementwise ratios are unstable near zero raw sensitivities and remain
associations. No filter intervention was made.

Reduced-cost class loss means sum(q_oracle,e*(drho_candidate,e-drho_oracle,e)),
a supporting-dual decomposition. It is not asserted to be an additive nonlinear
spectral objective decomposition; cone curvature and row slack are separate.
Raw exact-gradient KKT, complementarity RMS, large-increment sign errors,
class-normalized distances, and first-mode linear losses are in structure.json.
''')
wt='| Method | Approx. solves | Algorithm constraint evaluations | Gradient requests | Solver wall s | KKT | Recovery | d2 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for n,d in D.items():
 if 'RETAINED' in n or n=='S1_PERSISTENT':continue
 h=d['last'];wt+=f"| {n} | {h['calls']} | {h['nonlinearEvaluations']} | {h['gradientEvaluations']} | {h['solverWall']:.3f} | {h['kkt']:.3e} | {pct(h['gainRecovery'])} | {h['d2']:.6f} |\n"
wr('WORK_NORMALIZED_COMPARISON.md',f'''# Work-normalized comparison

{wt}

All terminal approximation comparisons receive at most 500 solves, including
rejected GCMMA trials. Equal-call values at 100 and 500 determine causal labels;
no method wins from receiving an extra two orders of magnitude of work. The
retained B0-5000 history is contextual long-run evidence, not an equal-budget
competitor. SOCP uses a different cone interior-point algorithm; its internal
iterations are not mmasub calls. Assembly, solve, certification and total process
memory are separately reported in SOCP_ORACLE_COST.md.

The exact production evaluator computes gradients even when GCMMA requests
only trial values. Thus GCMMA's actually computed algorithm gradients equal its
constraint evaluations (accepted bases plus trials), while the gradient-request
column follows the algorithm's logical interface. Audit metrics require four
additional combined evaluations per new MMA iteration and three per accepted
GCMMA iteration; final CSV/JSON count these separately. Raw run logs originally
counted one metric event per iterate, corrected by source-based accounting in
fi_outputs.py. No solver sequence depends on this accounting correction.

Only direct SOCP reaches all preregistered oracle fidelity bars. It is therefore
the only measured method eligible for 'cheapest reaching fidelity'; a cheaper
inaccurate endpoint is not ranked as a success. Concurrent host load limits wall
time interpretation. Figures 15,16,21 show function-work and call-work curves.
''')
ct='| Control → treatment | Factor | Evidence by frozen bars | Recovery change at target 100 / 500 | d2 reduction at target 100 / 500 | Actual call pairs |\n|---|---|---|---|---|---|\n'
for c in M['comparisons']:
 v=c['checkpoints'];label=c['classification']
 if c['treatment']=='S34_CLAMP_ACCURACY' and c['control']=='B0_CURRENT_REPEATED_MMA':label='JOINT INTERVENTION; not a single-factor cause'
 if c['factor']=='safeguard':label='EVIDENCE AGAINST'
 ct+=f"| {c['control']} → {c['treatment']} | {c['factor']} | {label} | "+' / '.join(f"{q['recovery_change']:+.4f}" for q in v)+' | '+' / '.join(f"{q['d2_reduction']:+.1%}" for q in v)+' | '+' / '.join(str(q['actual_calls']) for q in v)+' |\n'
wr('CAUSAL_ATTRIBUTION.md',f'''# Causal attribution

{V['primary_cause']}

{V['causal_explanation']}

{ct}

The registered strong-effect rule requires >=.10 higher recovery AND >=20%
lower d2 at both equal-work checkpoints, with acceptable feasibility. If a
time-capped method lacks call 500, the terminal pair uses the last common call
for descriptive comparison; that pair cannot supply the missing registered
500-call confirmation for STRONG evidence. An
objective-only or distance-only improvement cannot meet it. A missing 500-call checkpoint (e.g. a time-capped run) cannot support STRONG
attribution; the actual matched call counts are shown. The joint S34
comparison is not assigned single-factor causality. Comparisons along its
2x2 edges hold the other factor fixed; their interpretation is conditional.

{V['ranked_causes']}

Mechanistic check: at call 19 the approximate bound complementarity products
cluster tightly around 1e-7. Their sum is {B['sum_artificial_box_comp']:.9g},
{B['sum_comp_over_gain']:.4f} times the entire certified scaled beta gain.
One such product normalized by sRow0*move is {B['normalized_barrier_scale']:.7f},
where the fidelity bar is 1e-6 (raw {B['raw_comp_required']:.3e}). The artificial
bounds lie inside the physical box, so reusing their multipliers without checking
physical slacks understates the original-problem residual. The retained first approximate solve even returns a worse model objective
than its feasible starting point (1.0711e-5 including auxiliary terms), despite
conservativeness at the candidate. These arithmetic facts explain why small iterate steps are not evidence of solving the fixed NLP.
The accuracy intervention establishes causality where its registered bars pass;
the arithmetic alone does not establish that it is the sole cause.
''')
wr('SUBSOLV_ACCURACY_TEST.md',f'''# Approximate-solve accuracy mechanism

The production primal-dual Newton solver reduces its barrier parameter from 1,
stopping at epsimin=1e-7. At production call 19, both artificial bound products
xi*(x-alfa) and eta*(beta-x) have medians approximately 1e-7. Their sum is
{B['sum_artificial_box_comp']:.10g}; the oracle's total scaled objective gain is
only {B['certified_beta_gain']:.10g}. The sum is an approximate-subproblem
central-path gap contribution, not the exact NLP's total duality gap. It exposes
an accuracy budget poorly matched to this small frozen objective gain.

The exact-problem normalized complementarity bar of 1e-6 corresponds here to
raw {B['raw_comp_required']:.3e}, far below 1e-7. The S4 accuracy choice 1e-12 was
registered from this scale before solver outcomes; it was not tuned to topology.

{table(['B0_CURRENT_REPEATED_MMA','S4_SUBSOLV_ACCURACY','S3_CANONICAL_CLAMP','S34_CLAMP_ACCURACY','G1_GCMMA','G2_GCMMA_ACCURATE'])}

Comparing S4 with B0 isolates accuracy under the current cap. Comparing G2 with
G1 isolates accuracy under native GCMMA's cap and regularization. Comparing
S34 with S3 isolates accuracy at cap 10 with production regularization. The
factorial's other edge isolates the cap after accuracy is tightened. Their
predefined evidence grades appear in CAUSAL_ATTRIBUTION.md.

Even an accurately solved MMA approximation need not yet solve the original
nonlinear problem: approximation curvature and limited motion can still leave
a large oracle design distance. This separates approximate-subproblem accuracy
from exact-problem convergence and from a conservative-acceptance safeguard.
No density, beta objective scaling, gradient filter or physical box was changed.
The variant names refer to requested accuracy; near-singular-system warnings
mean the requested tolerance is not itself a certificate. Exact NLP KKT and
oracle metrics decide success.

S34 also has a visible finite-accuracy setback: gain recovery falls from
0.997148 at call 210 to 0.647486 at call 211, with exact KKT rising to 0.237262,
while primal feasibility is preserved. It subsequently recovers. G2's recorded
gain is monotone over its 500 calls, but still fails design/KKT fidelity. No
Newton-state trace isolates the precise cause of S34's single setback; it must
not be hidden by reporting only the favorable terminal objective.

The retained first-call model gives an additional direct check: x=0, bs=1,
y=z=0 is feasible in that MMA approximation with objective -1. The returned
point has total approximate objective worse by 1.0711e-5, while true objective
is worse by 9.6343e-6. Every model component is conservative at the returned
point (maximum true-minus-approximate=-5.7687e-7). Thus failure of approximate
objective descent is demonstrated before any global-NLP attribution. See
evaluations/initial_model_diagnostic.json; no additional solve was used.
''')
wr('INNER_SOLVER_SELECTION.md',f'''# Selected future candidate

{V['candidate']}

{V['selection_explanation']}

The selection is for a future experiment only. No production adapter was
installed. No drho was applied. A method can be mathematically legitimate and
still fail the stringent finite-work fidelity test; GCMMA is not called globally
convergent on the strength of one frozen state.
''')
wr('SOCP_PROMOTION_GATE.md',f'''# Direct SOCP promotion gate

{V['socp_gate']}

N=2 equivalence, every actual reconstruction constraint, frozen certificate
reliability, and deterministic repeat are established. The current C480
configuration fixes N=2 for the whole run. N=1 and the diagonal equality route
reduce to LP; full coupling for N>2 requires an SDP under the conditions proved
in SOCP_GENERALIZATION.md. No general-N SDP implementation has been validated.

{V['fallback']}

This gate does not authorize a universal SOCP replacement or any production
repair in this task. Every prospective solve must be accepted using verified
exact constraints/KKT and a valid dual bound, not coneprog's exitflag alone.
''')
wr('C480_CAUSAL_RUN_GATE.md',f'''# Future C480 causal-run gate

{V['c480_gate']}

{V['c480_explanation']}

The future control is the validated three-rung C480 canary. Treatment keeps mesh,
initialization, filter, move, controller, p/q, projection and multiplicity settings
identical; only the inner problem-(25) solver changes. Compare omega1, Mnd, gray,
mid and broad-gray-core fractions, complete trajectory/controller events,
cumulative inner work, physical and filtered KKT, and topology. Exactly one
future treatment is contemplated, not a campaign. No topology run occurs here.
''')
wr('FILTER_STUDY_GATE.md',f'''# Filter-study gate

{V['filter_gate']}

The preceding non-integrability/non-conservativity result is unchanged. Solver
accuracy and approximation policy materially affect this frozen subproblem, so
mixing a filter intervention into the next causal test would confound attribution.
No density-filter run or sensitivity-filter change was performed. Pedersen (2000)
is background for the filter and localized-mode issue, not authority to change it.
''')
wr('PERFORMANCE_STATUS.md',f'''# Performance status

{V['performance_gate']}

The inner-solver fidelity issue is scientifically material. No evidence shows
it irrelevant, so no nine-mesh campaign can resume on these results. Frozen
method costs are reported solely to size the proposed solver experiment.
''')
wr('PROVENANCE.md',f'''# Provenance and limits

Preregistration SHA-256: `{hashlib.sha256((S/'AUDIT_PREREGISTRATION.md').read_bytes()).hexdigest()}`.
It was written before any new solver experiment. Upstream archive SHA-256:
`{P['upstream_sha256']}`. Source: https://www.smoptit.se/GCMMA-MMA-code-1.5.zip.
Its licensing files are retained; no email was sent. Repo HEAD: `{P['head']}`.

Zero topology runs, zero outer updates, zero accepted rho changes, zero
controller transitions, zero production/config/reference modifications. All
new outputs are inside this diagnostic directory. The newly added Pedersen PDF
outside the audit was supplied by the user. The final protected-file hash audit
covers {len(P['protected'])} original files, including every tracked regular
file, the reference study and the authoritative trajectory. It is recorded in
integrity_final.json; FINAL_SHA256.txt hashes final deliverables.

MATLAB R2025b at /Applications/MATLAB_R2025b.app; one numerical thread per process.
Python/numpy/scipy/matplotlib/h5py generate tables and scientific plots. No new
package installation, FE eigensolve, optimization mesh, or performance campaign.
Independent solver processes share the host; wall times are not controlled
benchmark timings. Only audit-local launch permissions were escalated after the
sandboxed MATLAB launcher silently exited. The official archive download needed
network permission. No approval rejection occurred.

## Disclosed implementation and procedural details

- The initial audit logger failed after one instrumented MMA call because
  MATLAB rejected an assignment into an empty fieldless struct. Production-19
  had already reproduced. Explicit initialization fixed the logger; the valid
  500-call run restarted from zero. Aborted log is retained. No solver formula
  or threshold changed because of that failure.
- GCMMA's nine printed toy iterations reproduce the published rounded values,
  but their KKT residual is 1.447e-5. The validation continued to its KKT bar;
  it did not loosen the bar. The initial failed nine-iteration assertion log is
  retained. This does not invalidate the subsequent analytic validation.
- The GCMMA validation completed while the independent baseline was running,
  before any frozen GCMMA experiment. Some independent prescribed tests overlap
  in wall time; the report's causal comparisons follow the registered factors.
- The previous 5000-call replay is authenticated/re-evaluated rather than wholly
  recomputed, as preregistered. New B0 extends to 500 and proves bitwise fidelity.
- Oracle-witness logging was refined from a conservative rounded prior bound
  to its explicit formula; this changes only a diagnostic at roundoff level.
- Raw per-iterate metric-event counts were converted to actual production-
  evaluator calls in final outputs; algorithm work and audit overhead are
  distinguished. This affects accounting only.
- S4 hit its registered 1800-second budget after 455 calls (1803.726s,
  checked between calls). The 500-call checkpoint is missing and is not
  manufactured or used for a strong-effect claim.
- Tight subsolv accuracy produces near-singular-system warnings; logs retain
  them. They are not suppressed or mistaken for successful convergence.
- The user supplied Pedersen (2000) after preregistration. It prompted no new
  variants. Barrier and first-call model arithmetic checks and exact gray-bound counts are read-only
  mechanistic diagnostics, not extra optimization experiments.
- The prompt's 'every gray element' bound claim is corrected from the actual
  oracle. None of its hashes, objective, constraints or certificates failed.

Production volume arithmetic temporarily evaluates the sum of rho and the
increment to preserve exact bits. No candidate density vector is saved, accepted,
or passed to FE/model/controller functions. innerLoopRho is inspected but never
executed. All resulting increment arrays are evidence only.
''')
answers=[
('Is the SOCP oracle identity intact?','Yes; all reference hashes and fresh objective/constraint, weak-duality, KKT and equivalence checks pass.'),
('Was rho updated anywhere?','No. Zero accepted or outer rho updates; only the unchanged production volume expression is evaluated read-only.'),
('Was any topology optimization run executed?','No. Zero topology runs. Toy mathematical programs validate GCMMA only.'),
('Was production iteration 19 reproduced exactly?','Yes: drho, beta and nInner=19 reproduce bit-for-bit, and fresh call 500 matches the earlier replay.'),
('Does original repeated MMA approach the oracle in objective?',f'It makes partial, nonmonotone progress: {pct(b["gainRecovery"])} recovery at 19 and {pct(z["gainRecovery"])} at 5000; it does not reach fidelity.'),
('Does it approach in design space?',f'Partially: d2 decreases from {b["d2"]:.6f} to {z["d2"]:.6f}, far above .01. No convergence is established.'),
('Does it approach the oracle active set?','Not to the declared tolerance: same-bound agreement remains zero at 5000; sign agreement alone is insufficient.'),
('Does relative step correlate with oracle distance?',f'There is trend correlation (fresh-500 Spearman {st["spearman_relStep_d2"]:.4f}), but every B0 stop hit fails fidelity. It is not a valid optimality proxy.'),
('Can lowering tolInner alone solve the problem?','Not demonstrated; the 5000 replay still fails. A fixed approximate-solve accuracy floor remains. No tolerance sweep was run.'),
('Which state resets between MMA calls?','The internal subsolv Newton primal/dual/slack workspace is freshly initialized. MMA approximation history is not reset between inner calls.'),
('Which state is preserved?','xold1, xold2, returned low/upp, the current increment/beta iterate and the increasing inner counter. These reset only upon a new outer problem.'),
('Are these choices specified by Du & Olhoff?','The paper specifies the frozen increment subproblem and use of MMA, but not numerical state policies, constants, scaling or stopping thresholds.'),
('Does persistent history materially improve convergence?','The frozen sequence already has it. S1 is a bitwise identity; cross-outer warm-start effects remain untested.'),
('Does asymptote initialization materially matter?',f'Historical .01 changes the path but ends at {pct(last("S2_ASYINIT_001")["gainRecovery"])} recovery. Active production already uses canonical .5; it is not a missing correction.'),
('Does asymptote update materially matter?',V['asymptote_answer']),
('Does GCMMA reach the oracle?','Neither native G1 nor accurate G2 meets all registered fidelity bars.'),
('How close does it get?',f'G2: recovery {pct(g["gainRecovery"])}, d2={g["d2"]:.6f}, dinf={g["dinf"]:.6f}, exact KKT={g["kkt"]:.3e} at {g["calls"]} calls.'),
('Cheapest method reaching oracle fidelity?','Direct SOCP is the only measured method that reaches every bar: about 45.5s solve plus separately measured assembly/certification.'),
('Where does current MMA disagree?',f'Across void, gray shell, gray core and solid. Gray carries {pct(sum(c["share_squared_distance"] for c in T["localization"][-1]["classes"][1:3]))} of squared increment error at 5000.'),
('Is disagreement void dominated?','No in oracle design distance. Void/solid contribute substantial reduced-cost loss, but the density-space discrepancy is broader.'),
('Does void amplification correlate with disagreement?','Void RMS amplification is 30.69x; within-void amplification-versus-distance Spearman is negative (~-0.594). This is association, not filter causality.'),
('What accounts for 99.94% saturation?','9,404 lower-density + 4,906 lower-move + 4,788 upper-move + 9,686 upper-density bounds; 16 interior, zero coincident.'),
('Why only ~33.7% move dominated?','That statistic counts only the 9,694 ±move-bound variables, excluding 19,090 density-limited active bounds.'),
('Is the oracle approximately a threshold rule?','Exactly a coupled reduced-cost KKT rule; approximately filtered F11 versus volume threshold. The first-mode-only rule loses ~0.000600 of gain but still misses strict design fidelity.'),
('Can the N=2 SOCP generalize?','Yes conditionally: the consistent-offset cluster constraint is an affine PSD inequality; N=1 is LP and diagonal equality paths are LP.'),
('What about N>2?','General full coupling gives an SDP, not a universal SOCP. Inconsistent offsets/nonlinear volume need separate proof; no production SDP is claimed.'),
('Is direct SOCP a legitimate candidate?',V['socp_gate']+'; scope and fallback are explicit in SOCP_PROMOTION_GATE.md.'),
('Which single choice has strongest causal evidence?',V['strongest_choice']),
('Which solver should be carried forward?',V['candidate']),
('Is one future corrected C480 run justified?',V['c480_gate']+'. None was executed.'),
('Is the filter study deferred?',V['filter_gate']),
('Is the nine-mesh campaign blocked?',V['performance_gate']),
('Were production files untouched?','Yes; protected-file and implementation-tree hashes are rechecked at finalization.')]
body='BOTTOM LINE\n\n'+V['bottom_line']+'\n\n'+table([n for n in D if n not in ['S1_PERSISTENT','B0_RETAINED_5000']])+f'\nCertified SOCP: beta={M["oracle"]["beta"]:.9f}, all fidelity bars PASS, both repetitions bitwise-identical.\n\n'
body+='## Required questions\n\n'+'\n\n'.join(f'{i}. **{q}** {a}' for i,(q,a) in enumerate(answers,1))
body+='\n\n## Verdicts\n\n'+'\n'.join('- '+V[k] for k in ['oracle','persistence','stopping','gcmma','socp_gate','primary_cause','c480_gate','filter_gate','performance_gate'])
body+='\n\n## Evidence and limitations\n\nThe final MATLAB audit re-evaluates 202 saved checkpoints with zero scalar-metric discrepancy and maximum conic/production constraint discrepancy 1.008e-14. G0/G1 increments and nonlinear duals agree bit-for-bit. All recorded fresh approximate iterates are primal feasible, but none passes joint fidelity. S34 has a transient gain-recovery drop at call 211; its favorable terminal objective does not establish reliable convergence.\n\nSee [PROVENANCE.md](PROVENANCE.md) for the preregistered reuse of the 5000 replay, logger/validation fixes, numerical warnings and cost limits. [CAUSAL_ATTRIBUTION.md](CAUSAL_ATTRIBUTION.md) gives the isolated comparisons and evidence grades. [MASTER_METRICS.csv](MASTER_METRICS.csv) and [METRICS.json](METRICS.json) contain common metrics; evaluations retains checkpoints and every conservative trial. No experimental variants were added after preregistration. The supplied Pedersen (2000) paper is useful background for localized modes and filtering, but supplies no missing MMA accuracy or state prescription; see [PEDERSEN_2000_CONTEXT.md](PEDERSEN_2000_CONTEXT.md).\n\n'
body+='## Figures\n\n'+ '\n'.join(f'- [{p.stem}](figures/{p.name})' for p in sorted((S/'figures').glob('FIG_*.png')))
wr('REPORT.md',body)
print('Wrote reports; primary verdict:',V['primary_cause'])
