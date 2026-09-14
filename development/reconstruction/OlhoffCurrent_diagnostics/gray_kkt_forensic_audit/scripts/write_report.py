from identity import *
import csv
G=json.loads((OUT/'evaluations/geometry.json').read_text());S=json.loads((OUT/'evaluations/stationarity.json').read_text());I=json.loads((OUT/'evaluations/identity.json').read_text());C=json.loads((OUT/'evaluations/cluster_robustness.json').read_text());T=json.loads((OUT/'evaluations/timeline.json').read_text())
N=['400','480','800']
def put(name,text):(OUT/name).write_text(text.rstrip()+'\n')
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(str(x) for x in r)+' |\n' for r in rows)
def f(x):return f'{x:.6g}' if isinstance(x,(int,float)) else str(x)
geomtable=table(['mesh','Mnd %','gray % / area','mid % / area','broad core % / area','max depth / R'],[[n,f(G[n]['Mnd_percent']),f"{100*G[n]['gray_fraction']:.3f}% / {G[n]['gray_area']:.5f}",f"{100*G[n]['mid_fraction']:.3f}% / {G[n]['mid_area']:.5f}",f"{100*G[n]['broad_core_fraction']:.3f}% / {G[n]['broad_core_area']:.5f}",f"{G[n]['max_depth']:.5f} / {G[n]['max_depth']/.06:.3f}"] for n in N])
kkttable=table(['mesh','gray RMS, all-interior dual','gray RMS, best gray dual','gray p95, best gray dual','broad RMS, best gray dual','filtered gray RMS, best gray dual'],[[n,f(S[n]['classes']['gray']['raw_reduced_normalized']['RMS']),f(S[n]['classes']['gray']['raw_grayfit_normalized']['RMS']),f(S[n]['classes']['gray']['raw_grayfit_normalized']['p95']),f(S[n]['classes']['broad']['raw_grayfit_normalized']['RMS']) if n!='400' else 'empty',f(S[n]['classes']['gray']['filtered_grayfit_common_scale']['RMS'])] for n in N])
verdicts=['GRAY_FORENSICS_EVIDENCE_PASS','FINAL_STATE_SENSITIVITY_VALIDATED','GRAY_REGIONS_NOT_KKT_STATIONARY','MULTIPLICITY_SECONDARY_OR_TRANSIENT','PROJECTION_CANARY_EXPERIMENT_PREMATURE','P_CONTINUATION_REOPENING_NOT_JUSTIFIED','PERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE']
assert all(S[n]['FD']['validated_raw'] for n in N)
assert all(S[n]['classes']['gray']['raw_grayfit_normalized']['RMS']>.1 for n in N)
causes=[
['1. sensitivity filter','STRONG EVIDENCE','Direct physical-gradient/subproblem-gradient mismatch; gray variation std reduced to 6.27% and 14.67% at 400/480; raw sign flips 43.26%/20.28%. Filtered gray-fit residual 0.022/0.049 versus raw 0.355/0.334.','At 800 the single active mode is not flattened; near-cluster dual treatment matters. No filter A/B was run.','Strong for the first-order mismatch; not a complete causal proof of gray-patch generation.'],
['2. MMA / local optimality failure','MODERATE EVIDENCE','Substantial physical KKT residuals remain after global convergence; inner stopping is relative step change, no retained KKT certificate.','Filtered local residual at 400/480 is much smaller; nothing proves a bug in published MMA or failure of the last actual inner solve.','Cannot separate inner accuracy from solving a filtered surrogate without final inner dual/iterate evidence.'],
['3. multiplicity / subspace treatment','MODERATE EVIDENCE','800 exact simple-mode gray RMS 1.163 reduces to optimistic two-mode bound 0.127 raw, 0.050 filtered; terminal gap12=3.49e-5.','480 has broad patches and a well-separated first mode. J warnings end at 16/113; not terminal.','Near-cluster local optimality is material at 800; warning-induced path causality is unidentifiable.'],
['4. topology bifurcation / alternate basin','WEAK EVIDENCE','400-to-480 changes from interface bands to two broad end patches; 800 has merged connected gray network.','Different meshes alone do not demonstrate alternative basins or a bifurcation.','No same-formulation basin comparison or matched-field FE test.'],
['5. relaxed SIMP formulation (stationary gray optimum)','EVIDENCE AGAINST','Interior gray reduced gradients fail a necessary stationarity condition with any nonnegative volume multiplier.','Relaxed formulations can in principle admit gray stationary states; this audit does not exclude other stationary designs.','No global-optimum claim or diagnosis of every individual element.'],
['6. stiffness/mass interpolation balance (strengthening cancellation)','EVIDENCE AGAINST','Median C rises 0.398→0.490→0.960, meaning weaker cancellation for the exact active branch. C<0.1 in only 11.14%,6.43%,7.39% of gray.','Opposing terms do exist; the exploratory 800 dual-weighted C is 0.469, so single-mode comparisons are basis/weight sensitive.','No counterfactual mass law; balance against the volume multiplier is distinct from gK+gM≈0.'],
['7. stopping / controller','EVIDENCE AGAINST','Broad patches already present by stage-1 end; stages 2/3 slightly reduce grayness; terminal global windows are flat; exact frozen policy events verified.','Flat global observables do not establish KKT stationarity.','No more-iterations counterfactual is authorized; controller is frozen, not a KKT certificate.'],
['8. FE discretization','NOT TESTED','Meshes differ and gray support changes qualitatively.','Native eigen residuals and derivative checks do not identify an FE assembly/derivative defect.','No common physical density re-evaluated on different meshes; derivative validation is not mesh convergence.'],
['9. other: telemetry/state indexing','STRONG EVIDENCE','400 hist.omega(466) is pre-update; reevaluated rho466 gives 166.452298433, consistent with later table.','This indexing difference cannot create the saved gray density field.','Explains a reported scalar discrepancy only, not grayness.']]
causetable=table(['cause','evidence level','supporting observations','contradicting observations','remaining uncertainty'],causes)
put('PROVENANCE.md',f'''# Provenance

Audit branch `{I['branch']}`, HEAD `{I['HEAD']}`. Initial dirty paths were only `diagnostics/nine_mesh_campaign_audit/` and `diagnostics/three_rung_canary_preflight/`; both pre-existed. This audit creates only `diagnostics/gray_kkt_forensic_audit/`. No production modifications, commits or optimization runs.

Implementation `{I['impl_tree']}`; all 75 manifest file hashes verified. The same hash is inside each MAT trajectory. Input containers were read-only; their exact hashes are in EVIDENCE.json. The completed audit rechecks both input and production hashes. Full input identity, decoded config and config hashes are retained.

400 source: two_branch_controller_validation (source-study starting HEAD `b6014ba8bca41f85671d79ab4c8bdee7419880bb`), interpreted through the frozen three_rung_architecture proof (HEAD `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c`). These are source-study repository observations, not invented per-MAT commit fields. The MAT itself records implTree, config hash and MATLAB build, not a per-run HEAD. 480/800 source: three_rung_canary_preflight, branch benchmark-methodology-r2, HEAD `{I['HEAD']}`. Known older container-transfer concerns are not inherited as current failures: the actual files now match the authoritative manifests.

MATLAB executable `/Applications/MATLAB_R2025b.app/bin/matlab`, native eigSolve/genGrad/applyFilter/deltaLambda kernels with one computational thread. Python environment is the repository `.venv`, h5py, numpy, scipy, matplotlib. No MATLAB optimizer call is made. Executable audit entry points are listed in scripts/README.md. Do not run a repository test suite for this task because some fixtures optimize.

Execution limitations and corrected audit-only recorder/read issues are disclosed in evaluations/EXECUTION_NOTES.md. Invalid integer FD containers are retained as scratch and excluded from all metrics. The final correct FD outputs contain 252 perturbed FE evaluations over 42 distinct state/element pairs, three delta values each, plus three base states. Earlier repeated evaluations are disclosed in logs; 252 is the accepted dataset, not the total number of FE calls made across attempts. All were analysis evaluations, zero density updates.

Preregistration SHA-256 `{I['preregistration_sha256']}`. The cluster robustness fit is a separately labelled exploratory extension after observing the small but resolved terminal 800 gap. It does not alter the preregistered primary test or sample.
''')
rows=[]
for c in I['cases']:
 rows.append([c['mesh'],c['source_study'],c['iteration'],c['status'],c['rho_sha256'],c['saved_config_hash'],c['effective_three_rung_config_hash']])
put('EVIDENCE_IDENTITY.md','# Evidence identity\n\nGRAY_FORENSICS_EVIDENCE_PASS\n\n'+table(['mesh','authority','iteration','endpoint status','rho SHA-256','saved config SHA-256','three-rung effective config SHA-256'],rows)+'''
The 400 endpoint is **RHO(:,466)** of the causal-controller four-rung parent, authenticated by the three_rung_architecture S3 density hash and exact-prefix proof. RHO(:,505) is explicitly excluded. 480 and 800 use the actual last trajectory columns, bitwise equal to the separate saved states and record hashes. No legacy beta endpoint is substituted.

All five input MAT containers match their expected SHA-256 and byte count. The three saved configs independently reproduce their recorded schema-ordered config hashes using the prior validated hash implementation. For 400 the effective three-rung hash changes only move.levels; it is labelled a counterfactual config, not the original stored config. Full configurations, multiplicity/MMA fields, trajectory paths, MATLAB metadata and controller status are in evaluations/identity.json and config_*.json.

All endpoints are stage 3 / move 0.01, persistent branch B under E=A OR B, stageExhaustion for move/stop, beta without controller authority. Shared p=3, q=1, eq4b mass, sensitivity/all filter, physical R=0.06, fixed N=2 subspace with offsets/offdiagonals, published increment MMA, min/max inner=5/500, relative tolerance=0.05. Element radii 3/3.6/6. No projected density or continuation.
''')
put('GRAYNESS_GEOMETRY.md','# Grayness geometry\n\n'+geomtable+'''
Physical area units are domain-coordinate units squared; full area=8. Gray is 0.1<rho<0.9; mid is 0.4<=rho<=0.6. Broad core means gray distance to the nearest non-gray element centre exceeds R=0.06. It is a conservative core measure, not the area of every entire component that contains a core. Domain edges are not treated as density interfaces. Distances have finite-grid centre uncertainty of order h. A separate rho=0.5 interface distance distribution is retained; there is no direct solid/void contact through a diffuse band without defining a threshold.

400: no gray core deeper than R; the gray zones are resolved interface/member bands. 480: two leading gray components have areas 0.79444 and 0.79056, each bounding box 1.91667×0.76667. 800: one leading connected gray network has area 2.8676 and spans 7.8×0.76; this connectivity includes thin bridges and does not mean the whole box is gray. Core maps and snapshots identify broad end patches. The maximum-depth diameter proxies 2d are 0.11314,0.73333,0.76; these are not exact member widths. The transition is primarily 400→480, with only a small further increase in maximum depth at 800 but a large increase in broad area.

Every component size/bounding box/depth, four/eight-neighbour component count, density quantiles, fractions below .01 and above .99, gray depth and threshold-interface distances is in evaluations/geometry.json. F01–F04 use identical physical coordinates; histograms use identical bins.
''')
put('SENSITIVITY_DECOMPOSITION.md','''# Elementwise spectral decomposition

All three native frozen-state eigensolves and the complete raw/filtered N=2 generalized tensors are retained. F=gK+gM with gM signed negative for a diagonal mode; off-diagonal contributions can have either sign. Diagonal blocks use their own lambda; off-diagonals use lambda1; fJJ is retained separately. No optimized increment was computed. The plotted active derivative is obtained from native deltaLambda at zero increment with the recorded diagonal offsets.

The lowest eigenvalue is simple at each exact saved state, including 800: its frequency gap is small but approximately 3.49e-5 and well above the measured eigen residual. The first derivative at zero is therefore the first diagonal block. This statement is local; it does not treat the 800 optimization path as single-mode. Its nearby nonsmooth optimum may involve both modes, and the full tensor robustness analysis in KKT_STATIONARITY.md is essential.

The maps F05–F07 show gLambda, stiffness and signed mass in a common mesh-comparable scale NE/lambda1. Color limits clip pooled 99.5% tails, disclosed on titles; numeric files retain all values. Raw unitful distributions are in stationarity.json. The derivative is of lambda=omega²; divide by 2omega for omega derivatives. That positive conversion does not change first-order stationarity if multipliers/scales are converted consistently.

Eigen residuals are ~2.3e-10,3.4e-10,9.3e-10 for the first mode; mass-orthogonality Frobenius errors are below 7e-14. These are measured residuals after assembly, not claims of exact arithmetic or a proof of FE discretization accuracy.
''')
balrows=[]
for n in N:
 for cls in ['gray','mid','solid','void','broad']:
  c=S[n]['classes'][cls]
  if c['n']:balrows.append([n,cls,f(c['gK_abs']['median']),f(c['gM_abs']['median']),f(c['gRaw_abs']['median']),f(c['cancellation']['median']),f(100*c['cancellation_fraction_lt_01'])])
put('MASS_STIFFNESS_BALANCE.md','# Mass versus stiffness balance\n\n'+table(['mesh','class','median |gK|','median |gM|','median |net|','median C','C<0.1 (%)'],balrows)+'''
C=|gK+gM|/(|gK|+|gM|+machine epsilon), with gM signed. Small C indicates cancellation; large C does not. Medians of individual terms cannot be subtracted to recover the median net.

There is **no measured strengthening of cancellation** as the gray area grows. Gray median C increases from .398 to .490 to .960 for the exact active branch. This rejects the proposed explanation based on a mesh-growing population of almost-zero net spectral derivatives. The 800 active mode is strongly mode-dependent: an exploratory PSD trace-one mixed dual gives raw gray median C=.469 (filtered-fit weights give .464). That also does not establish increasing cancellation relative to 400; it prevents overinterpreting .960 as an invariant property of the whole subspace.

Constrained stationarity requires balance against a **volume multiplier**, not cancellation of stiffness and mass to zero. Gray rho>.1 lies on the linear mass branch; the special low-density polynomial acts outside the gray class. This audit does not establish that changing the mass law or p would improve scientific validity. No such perturbation was made.
''')
fr=[]
for n in N:
 c=S[n]['classes']['gray'];fr.append([n,f(c['filter_RMS_ratio']),f(c['filter_std_ratio']),f(100*c['filter_sign_flip_fraction']),f(c['raw_grayfit_normalized']['RMS']),f(c['filtered_grayfit_common_scale']['RMS'])])
put('FILTER_GRADIENT_AUDIT.md','# Filter gradient audit\n\n'+table(['mesh','gray RMS filtered/raw','gray std filtered/raw','gray sign flips %','raw gray-fit KKT RMS','filtered gray-fit subproblem RMS'],fr)+'''
Residuals share the raw interior objective-gradient RMS scale. The filter mostly suppresses **spatial variation**, not the absolute magnitude of the whole gradient. At 400/480 it replaces heterogeneous signed gray sensitivities by nearly constant positive values. It is inaccurate to describe this as a universal near-zero objective gradient: the filtered derivative can be balanced by a positive volume multiplier.

The exact first-mode 800 gradient is not similarly flattened. A relaxed two-mode fit lowers its filtered gray residual to .05045 on the same scale, while its optimistic raw lower bound remains .12676. Thus single-mode attenuation comparisons alone are inadequate there. This fit is not an exact KKT certificate because lambda2 is still separated.

The formula is recovered exactly, all tensor blocks are filtered, and finite differences of the frozen filtered subspace prediction validate its increment derivative. FE finite differences validate the raw derivative instead. This explained discrepancy is a **formulation/surrogate consistency issue**, not evidence of an incorrectly transcribed filter implementation. No claim of an unknown density chain rule or of a proved non-integrable vector field is made.

No causal filter A/B was run. Quantitative first-order mismatch supports investigating the filter/optimality interface; correlation with mesh alone would not.
''')
put('KKT_STATIONARITY.md','''# Constrained KKT stationarity

GRAY_REGIONS_NOT_KKT_STATIONARY

Scope: the retained designs do not satisfy first-order stationarity of the evaluated relaxed FE eigenvalue problem. The filtered algorithm is separately audited through its exact frozen local subproblem. It is not renamed as a different undisclosed physical objective. This is not a theorem that no nearby gray optimum exists, and does not diagnose a defect in the three-rung controller or in published MMA.

Let L=-lambda1/lambda_ref + mu*(sum rho-0.5NE)/(0.5NE), with mu>=0 and lambda_ref fixed. The reduced gradient is r_e=-g_e/lambda_ref+mu/(0.5NE). At a lower bound require r>=0; at an upper bound r<=0; in the interior r=0. The projected first-order sign residual retains r on interior elements, min(r,0) on lower bounds, max(r,0) on upper bounds. Preregistered bound tolerance=1e-7. All saved values are farther than that from either bound, so there are no bound elements under that convention. Solid/void classes are **not** synonymous with active box bounds. Bound-sign arrays are empty, reported as such rather than zero-valued evidence of a pass.

No exact MMA dual is retained. With v_e=g_e/lambda_ref and fit set I, the least-squares nonnegative dual is mu=max(0,(0.5NE)*mean_I(v)). The primary fit uses all interior elements; the gray-only fit minimizes gray residual over **every possible nonnegative volume multiplier**. Its residual is therefore a decisive optimistic test for gray stationarity, not an arbitrary multiplier choice. The volume slack is small and negative; treating volume as active is generous. Strict complementarity at negative slack would require mu=0 and cannot improve on this best nonnegative fit.

Normalize by s=sqrt(mean_interior((g_raw/lambda_ref)^2)). s is 2.09159e-4,1.48909e-4,1.26285e-4. Gray is bounded well away from both box limits, so changing near-bound classification cannot make those elements active-bound variables.

'''+kkttable+'''
The best gray-only raw RMS values 0.355,0.334,1.163 are above the preregistered 0.1 diagnostic bar. This bar is an audit scale convention, not a theorem or a standard MMA tolerance. The distributions, signs and finite-difference errors provide the stronger evidence. Broad-core raw RMS is .1716 at 480 and .3115 at 800 under the same gray-fit dual. Only 20.1% and 2.46% of those core elements fall below |r|/s=.1; isolated small values do not certify a stationary region or a common dual. Accordingly the primary category is nonstationary, not mixed on the strength of isolated small residuals.

Fit details and max/median/RMS/p90/p95/p99 for gray, mid, solid, void and broad regions, primary/gray-fit multipliers, volume complementarity, and global projected sign residuals are retained in stationarity.json. Figures F10–F12 and F16 show the distribution and mesh comparison. Bound-tolerance robustness at 1e-5,1e-4,1e-3 leaves raw gray RMS materially nonzero; the best gray-only fit is independent of those choices.

## What the filtered residual does and does not establish

Use gFiltered instead of gRaw in the same algebra to test **zero increment** stationarity of the exact frozen filtered subproblem. Its all-interior dual fit is distorted by near-void filtered values: it hits mu=0 at 400/480, with gray residuals .322/.343 on the shared raw scale. Its best gray-only fit gives .0222/.0490. That is evidence of local flattening in gray regions, **not a global filtered KKT pass**. Normalizing by the filtered RMS instead would misleadingly shrink some values because near-void filtered gradients are large; both scales are retained and the shared raw scale is used for comparisons.

800 remains nonstationary under the exact simple lowest branch. However, a nearby eigenvalue crossing can be material: its exact relative lambda gap is about 6.98e-5. The exploratory robustness fit uses Q=[[a,b],[b,1-a]] and a free constant threshold to minimize gray residual of sum_sk Q_sk F_sk. This is a linear least-squares fit over an enlarged dual space, not a density optimization. It drops PSD, nonnegative volume and spectral complementarity restrictions, so the result is an **optimistic lower bound**. The raw bounds are .3393,.3306,.1268; even this enlargement does not remove the gray residual.

At 800 the fitted Q happens to be PSD with eigenvalues .01794/.98206; its volume multiplier is .4301. But trace(QD)/lambda1=3.07e-5 and ||QD||F/lambda1=4.54e-5, not zero. The analogous filtered bound is .05045. These explain why approximate near-cluster optimality can look much better than exact one-branch KKT, without silently treating separated eigenvalues as exactly multiple. At 400/480 the unrestricted fits violate PSD (and the 400 raw volume sign), so they remain lower bounds only. Exact admissible spectral dual at each saved state is Q=diag(1,0); fJJ is inactive. Native deltaLambda confirms that active derivative.

The last actual MMA subproblem was based on the **previous** density, with an inner solution and dual state that are not retained. Zero-increment testing at the final density does not prove that last subproblem was solved inaccurately. The evidence establishes physical nonstationarity and filtered-surrogate mismatch; it does not allocate all blame to MMA.
''')
fdrows=[]
for n in N:
 for row in S[n]['FD']['by_delta']:fdrows.append([n,S[n]['FD']['n_samples'],f(row['delta']),f(row['relative_error']['median']),f(row['relative_error']['max']),f(row['error_over_raw_interior_RMS']['max'])])
put('FINITE_DIFFERENCE_VALIDATION.md','# Finite-difference validation\n\nFINAL_STATE_SENSITIVITY_VALIDATED\n\n'+table(['mesh','unique elements','delta','median relative error','max relative error','max error / raw gradient RMS'],fdrows)+'''
Validation scope is the raw first-order derivative at the accuracy relevant to the KKT findings, and the increment derivative of the implemented filtered subspace prediction. It is **not** uniform relative-accuracy validation of tiny void sensitivities. The deterministic sample rule was frozen before inspecting derivatives; sample IDs depend solely on saved rho classes and column-major quantiles. 400 has no broad-core class and has 12 unique samples; 480/800 each have 15. IDs, classes and deduplication are in FD_SAMPLE_PREREGISTERED.json. Three step sizes, centered or second-order one-sided as preregistered, produce 126 accepted rows / 252 perturbed FE solves. No rho was advanced by an optimization update.

All accepted rows have absolute derivative error below 7.85e-5 of the raw interior gradient RMS, over a thousand times below the .1 stationarity scale. Median relative errors are ~1e-5–8e-5. Weak void derivatives have large relative errors (up to 8.77); shrinking delta often worsens error, consistent with eigensolver/subtraction roundoff. Those weak derivative signs are not certified. The gray/broad KKT verdict does not rely on them. This is an explained numerical limitation, not an unexplained material analytic/FD disagreement.

The frozen subspace directional checks use the smallest eigenvalue of the full 2×2 offset matrix, not an untracked individual mode. Raw and filtered model derivative errors are tiny at 400/480; at 800 their maximum normalized errors are 1.83e-6 / 2.38e-5. Physical FE FD does not match the filtered vector: normalized RMS discrepancies .365/.348/.0471 across sampled rows are expected because the filter modifies sensitivities without changing FE rho. This validates the distinction, not a supposed density chain rule.

Full numerical rows are FD_RESULTS_*.csv; delta convergence is F19. Corrected floating-point recordings are fd_*.mat; INVALID_INTEGER_SERIALIZATION files are explicitly unusable scratch. See EXECUTION_NOTES.md for all attempts. No result-dependent sample replacement or best-delta filtering was performed.
''')
put('MULTIPLICITY_RELATION.md','''# Multiplicity and next-mode relation

MULTIPLICITY_SECONDARY_OR_TRANSIENT

The classification concerns a primary explanation of grayness across the sequence. It does not mean the terminal 800 near-cluster is irrelevant to KKT.

Warnings: 400 has 2, last at 13; 480 has 4, last at 16; 800 has 81 between 26 and 113, none in 114–468 (355 iterations). The 81 flags occupy an early regime but are **not present at every iteration** in that 88-iteration span. Thus “contiguous transient” refers to the containing regime, not 88 consecutive warnings. The warning flag concerns mode J=3 being near its successor, not whether modes 1 and 2 remain close. N=2 is fixed throughout regardless.

At the 800 last warning, Mnd is 55.555%, versus 34.412% final; broad-core area fraction is 49.87%, versus 20.87% final. The final broad-core elements are already gray then (97.98% mid), but they still change: average density distance over that final core to the endpoint is .09416; whole-domain threshold flips to endpoint are 19.205%. At 480 warning end, final core is also already gray (92.10% mid) with average remaining core density change .10046. Many later changes occur without warnings. Early warnings can correlate with the initially gray trajectory without establishing causality.

Final gap12: .21065,.12980,3.4876e-5; gap23 is ordinary (see MASTER_METRICS). The 480 broad-patch transition occurs with a well-separated lowest eigenvalue and only four early warnings. Consequently next-mode warnings cannot be a sufficient common primary explanation. 800's terminal first-pair separation is **not ordinary**; its two-mode dual robustness is important and retained. Do not conflate absence of J warnings with absence of a close first pair.

Spatial correspondence to early clustered-mode sensitivity influence cannot be determined causally from densities alone: modes/gradients/duals along the transient were not retained here, and no alternate trajectory was run. Full-trajectory N=2 means a location being inside the subspace treatment is not itself discriminating. The retained evidence supports a temporal and endpoint classification, not a claim that warnings permanently changed the basin.
''')
stagerows=[]
for n in N:
 for a in G[n]['stages']:stagerows.append([n,a['stage'],f"{a['start']}–{a['end']}",f(a['Mnd_start']),f(a['Mnd_end']),f(100*a['gray_end']),f(100*a['mid_end']),f(100*a['broad_end'])])
put('GRAYNESS_TRAJECTORY.md','# Trajectory of grayness\n\n'+table(['mesh','stage','iterations','Mnd start %','Mnd end %','gray end %','mid end %','broad core end %'],stagerows)+'''
The initial field is rho=.5 everywhere: all elements start gray/mid. It is misleading to ask when gray material is first “generated” without this fact. What changes is which regions become discrete and which persist as broad gray patches. 400 resolves its broad cores away; 480/800 retain broad end patches by stage-1 end. The final-stage grayness is largely inherited from stage 1. Stages 2/3 decrease Mnd and mid/broad fraction slightly; they do not create the mesh-growing gray phase.

Mnd stays within .1 percentage point of the endpoint after iterations 419/348/441 at 400/480/800. Thus it is not literally frozen hundreds of iterations before termination, especially at 800. It is nevertheless very flat in the final 20 iterations: Mnd ranges .03196/.03554/.05919 points; pre-update omega1 ranges .00995/.00464/.01289%. Small residual evolution is not evidence of local KKT convergence.

F13/F14 plot omega1, Mnd, gray/mid/broad fractions, move, A/B/E, gap12/gap23 and warnings, with stage boundaries. F15 combines warning/stage/grayness timelines. F20 snapshots show warning-end and stage-end densities in identical coordinates. CSVs preserve pre-update frequency labeling; saved densities are post-update. Raw histories, final-patch histories and stage statistics are retained.
''')
put('CROSS_MESH_REGIME_CHANGE.md','# Cross-mesh regime change\n\n'+geomtable+'\n'+kkttable+'''
400→480 is qualitative: the domain acquires broad end-region gray patches rather than just more elements in a fixed-width diffuse boundary. Mid area triples (.3136→.94889); broad core grows from zero to 1.04111. At 800 broad core expands to 1.6696 and a connected gray network spans most of the beam. Maximum depth largely saturates after 480 even as area continues growing.

The transition is not a matching jump in cancellation strength, which weakens for the exact active branch. Raw-gray stationarity is already deficient at 400, so the nonstationarity metric does **not** uniquely identify why the topology changes at 480. Filtering sharply flattens gray gradients at both 400 and 480. At 800 the close first pair introduces a further local-optimality regime, seen in the full-tensor dual reconstruction. No claim of a proven topology bifurcation, filter-support artifact or mesh-convergence limit follows from three endpoints alone.
''')
put('CAUSAL_RANKING.md','# Ranked causal hypotheses\n\n'+causetable+'\nRanking is for directing the next diagnostic action. STRONG EVIDENCE for a measured mismatch is not a claim of an isolated causal experiment. No production, controller, p, q, filter radius, mass law, multiplicity, projection or MMA parameter changed.\n')
put('PROJECTION_DECISION.md','''# Projection decision

PROJECTION_CANARY_EXPERIMENT_PREMATURE

Broad physical gray patches are established, but stationarity of the current physical relaxed problem is not. Therefore the user-specified prerequisite for a projection experiment—genuinely stationary relaxed gray regions—is not satisfied. Projection would mix a physical representation change with a change from heuristic sensitivity filtering to a density-map chain rule, while an optimality mismatch remains unresolved.

The existing projected preset is recorded, not run: design z∈[0,1], ztilde=Hz/Hs, rho=.001+.999*P(ztilde;beta,eta), tanh projection, eta=.5, beta levels [1,2,4,8], outer-convergence trigger, full chain rule on every tensor block and exact physical-volume evaluator. It inherits its own parent preset and maxOuter=1200. It is **not** automatically a single-factor treatment of these three-rung canaries. No treatment configuration is authorized or frozen for execution here, and no sweep is proposed. The two already-valid baseline canaries remain reusable if a future evidence-based decision authorizes an experiment.
''')
put('P_CONTINUATION_DECISION.md','''# p-continuation decision

P_CONTINUATION_REOPENING_NOT_JUSTIFIED

The decomposition does not show increasingly strong stiffness–mass cancellation in growing gray patches. Gray elements use the linear high-density mass branch and the observed nonstationarity/filter mismatch is a more direct obstruction. There is no new measured mechanism specifically supporting a p schedule.

Previous frozen evidence is not overturned. `docs/olhoff_penalty_continuation_experiment.md` explicitly refuted the connectivity-preservation hypothesis for a different clamped-clamped 40×5 case and path; it does not scientifically refute every possible p schedule at these simply-supported fine meshes. Its disclosed instability is also not transferable as a quantitative diagnosis here. We do not recommend repeating that failed experiment or silently substituting the coupled/decoupled presets. p=3 stays fixed.
''')
nextaction='Resolve the physical-objective versus filtered-subproblem optimality mismatch before any new scientific optimization. The single highest-information follow-up is a separately authorized frozen-state 480 subproblem certification that retains its complete inner primal/dual state and evaluates KKT/complementarity, without applying the proposed density increment. That isolates inadequate inner solution from accurate solution of a surrogate that is inconsistent with physical KKT. It is specified only; no inner optimization was executed in this audit.'
put('PERFORMANCE_IMPLICATIONS.md','# Performance implications\n\nPERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE\n\n'+nextaction+'''

The current endpoints cannot be presented as certified stationary optima of the stated relaxed eigenvalue formulation. It remains possible to report them transparently as outcomes of the specified filtered algorithm and controller, but that is not the requested scientifically certified final performance campaign. Neither visually smoothing the table nor simply making densities more discrete addresses the first-order issue. Another nine-mesh campaign is not justified now. Existing canaries remain valid evidence of controller behavior.

The separate 160×20 filter-support anomaly (rminEl=1.2<sqrt(2)) is excluded and cannot explain this 400/480/800 finding.
''')
answers=[
'Yes. All three exact density hashes and all five source MAT containers match; 400 is the proven exact S3 counterfactual at 466, not the four-rung last column.',
'Maximize the smallest generalized FE eigenvalue with p=3 stiffness, eq4b mass, mean rho<=.5 and .001<=rho<=1, approached by filtered nonlinear spectral increment subproblems. See FORMULATION_RECOVERY.md for the crucial surrogate distinction.',
'MMA updates [Delta rho; beta/lambda1_ref] within each inner subproblem; the outer solver adds Delta rho to the unfiltered FE/design density.',
'Only generalized sensitivity vectors are filtered on this path; physical/design rho is not filtered.',
'The objective derivative is zero in density coordinates and -1 in scaled beta; spectral constraint rows use filtered subspace eigenvalue derivatives /lambda_ref, while volume uses 1/(.5NE).',
'480/800 contain broad physical end-region patches, not merely increasingly resolved thin interfaces.',
'Gray fractions are 17.960%,28.729%,37.7625%; physical areas 1.4368,2.29833,3.0210 out of area 8.',
'Mid fractions are 3.920%,11.861%,16.090%; physical areas .3136,.94889,1.2872.',
'Maximum gray depth grows .05657→.36667→.38 (R=.06). The large jump is 400→480; further refinement primarily expands area.',
'gK is nonnegative and gM nonpositive for diagonal modes; full signed tensors are retained. Gray median |gK|/|gM| is 1.586/1.423,1.792/.782,.458/.0487 in unitful eigenvalue derivatives.',
'Some do, but not the dominant mesh-growing mechanism: median gray C=.398,.490,.960, with only 11.14%,6.43%,7.39% at C<.1.',
'No. The exact-active cancellation becomes weaker. The exploratory two-mode-weighted 800 C≈.469 also does not establish systematic strengthening.',
'At 400/480 it suppresses spatial variation dramatically (std ratios .0627/.1467), with many sign reversals, while RMS magnitude ratios are about .906. The exact first-mode 800 comparison is different; near-cluster dual treatment matters.',
'No for the physical relaxed problem in its actual rho variable. Filtered gray-subproblem residuals can be much smaller, which is not a physical KKT certificate.',
'All-interior-dual normalized gray RMS=.5553,.4446,1.1631. Best possible nonnegative gray-only dual gives .3548,.3340,1.1625. Normalizer is raw interior objective-gradient RMS; full quantiles are retained.',
'Flat observables coexist with unresolved physical first-order stationarity. The data do not establish whether the last inner MMA solve itself was inaccurate; it may have solved a filtered surrogate adequately.',
'Yes at the scale needed here: every accepted raw derivative error is <7.85e-5 of raw gradient RMS. Tiny void derivative relative signs remain roundoff-limited; full tensor directional checks also pass at relevant scale.',
'It materially affects terminal 800 stationarity through the close first pair, but does not explain the 400→480 broad-patch transition as a common primary mechanism.',
'No terminal alignment: 81 flags within 26–113, then 355 warning-free iterations. Broad final-core elements are already gray then and continue evolving afterward.',
'Everything starts gray at rho=.5. The persistent broad end patches are already present by stage-1 end; later stages slightly reduce grayness.',
'Broad end patches appear: broad core 0→13.0% of the domain, mid area triples. The first eigenpair remains well separated at 480; neither raw nonstationarity nor filter flattening begins uniquely there.',
'No direct controller inconsistency was found. Its frozen convergence rule does not claim to be a KKT certificate.',
'Unresolved local optimality is implicated; a published-MMA implementation defect or inaccurate final inner solve is not established without its dual/iterate state.',
'Not as a mesh-strengthening cancellation mechanism. Opposing terms exist, but cancellation-to-zero is not the volume-constrained stationarity condition.',
'Yes, strongly in the physical-gradient versus optimizer-subproblem mismatch; direct causality for the whole topology transition remains unisolated.',
'Premature: broadness is proved but stationary relaxed grayness is not. Projection would additionally change filtering/variable representation.',
'No new evidence justifies reopening p-continuation; the prior failed connectivity experiment is respected within its own scope.',
'No. The final performance campaign remains blocked by the optimality issue.',
nextaction,
'Yes: zero optimization runs, zero optimizer calls, zero density updates. Only retained-data analysis, FE/sensitivity evaluation, derivative perturbations and numerical dual reconstruction. Earlier FD recorder errors caused repeated analysis evaluations, fully disclosed.']
questions=['Are the final artifacts authoritative?','What exact mathematical problem is solved?','What does MMA update?','What variable is filtered?','What derivative reaches MMA?','Thin interfaces or broad gray regions?','How much area is gray?','How much is strongly mid-density?','Does physical thickness grow?','What are stiffness and mass contributions?','Do they strongly cancel?','Does cancellation strengthen with mesh?','Does filtering suppress raw gradients?','Are gray elements KKT stationary?','What are normalized residuals?','Stationary or merely stuck?','Did finite differences validate sensitivities?','Does multiplicity explain grayness?','Are 800 warnings aligned with final grayness?','When is most grayness generated?','What changes between 400 and 480?','Is the controller implicated?','Is MMA implicated?','Is stiffness/mass balance implicated?','Is the filter implicated?','Is projection justified?','Should p-continuation reopen?','Is a nine-mesh campaign justified?','Single highest-information next action?','Were zero optimization runs executed?']
report='''BOTTOM LINE

The authoritative saved designs develop **broad physical gray patches**, but they are **not KKT-stationary optima of the evaluated relaxed FE problem**. Even the volume multiplier fitted to minimize gray-region residual leaves normalized RMS 0.355 / 0.334 / 1.163 at 400 / 480 / 800. Native finite differences validate the raw sensitivities at a much finer scale. The result is robust to bound classification and is not evidence for changing the validated controller.

The strongest demonstrated issue is the mismatch between the physical eigenvalue derivative and the sensitivity-filtered local spectral model supplied to MMA. At 400/480 the latter is nearly flat within gray regions while the former is not. At 800, a close two-mode pair makes approximate subspace optimality important: an optimistic relaxed dual fit reduces raw/filtered residuals to .127/.050, but is not an exact KKT certificate. No claim of a proved MMA coding defect or an isolated causal explanation of the entire mesh transition is made.

Projection is premature; p-continuation is not newly justified; no nine-mesh campaign should start. Resolve filtered-subproblem versus physical optimality first. **Zero optimization runs and zero rho updates were executed.**

'''+geomtable+'\n'+kkttable+'\n## Final verdicts\n\n```\n'+'\n'.join(verdicts)+'\n```\n\n## Required answers\n\n'+''.join(f'{i}. **{q}** {a}\n\n' for i,(q,a) in enumerate(zip(questions,answers),1))+'## Ranked causes\n\n'+causetable+'''
## Evidence and practical limits

Identity/provenance/config are verified; the exact filter/sensitivity chain is recovered; physical-problem KKT and frozen-subproblem residuals are mathematically distinguished; no unexplained material derivative disagreement remains. Therefore the requested stop conditions were not triggered. The approximate near-cluster dual fit is explicitly not used as an exact physical multiplier. The audit convention .1 is not a standard optimization tolerance, and FD validation does not certify tiny void derivative signs.

All elementwise fields, tensors, samples, trajectories, metrics, source declarations and figures are retained. Read KKT_STATIONARITY.md and FINITE_DIFFERENCE_VALIDATION.md for the key qualifications. Repeated evaluations due to an audit recorder type error are disclosed in evaluations/EXECUTION_NOTES.md. Scientific generation of the 400→480 topology regime is not uniquely identified by the nonstationarity result; a valid stationary-relaxed-gray conclusion is nevertheless ruled out for these exact endpoints at the stated diagnostic scale.
'''
put('REPORT.md',report)
metrics={'study':'gray_kkt_forensic_audit','verdicts':verdicts,'optimization_runs':0,'optimizer_calls':0,'rho_updates':0,'accepted_perturbed_FE_evaluations':252,'geometry':G,'stationarity':S,'cluster_robustness':C,'timeline':T,'causal_ranking':[dict(zip(['cause','evidence_level','supporting_observations','contradicting_observations','remaining_uncertainty'],r)) for r in causes],'next_action':nextaction}
put('METRICS.json',json.dumps(metrics,indent=2,allow_nan=False))
rows=[]
for n in N:
 g=G[n];s=S[n];c=s['classes']['gray'];row={'mesh':n+'x'+str(int(n)//8),'Mnd_percent':g['Mnd_percent'],'gray_fraction':g['gray_fraction'],'gray_area':g['gray_area'],'mid_fraction':g['mid_fraction'],'mid_area':g['mid_area'],'broad_core_fraction':g['broad_core_fraction'],'max_gray_depth':g['max_depth'],'gray_components_4':g['gray_components_4'],'largest_gray_component_area':g['largest_gray_component_area'],'raw_KKT_gray_RMS':c['raw_reduced_normalized']['RMS'],'raw_KKT_gray_best_dual_RMS':c['raw_grayfit_normalized']['RMS'],'filtered_gray_best_dual_RMS_common_scale':c['filtered_grayfit_common_scale']['RMS'],'raw_relaxed_cluster_dual_RMS_lower_bound':C[n]['Fraw']['gray_RMS_lower_bound_over_raw_interior_RMS'],'filtered_relaxed_cluster_dual_RMS_lower_bound':C[n]['Ffiltered']['gray_RMS_lower_bound_over_raw_interior_RMS'],'cancellation_gray_median':c['cancellation']['median'],'filter_gray_std_ratio':c['filter_std_ratio'],'filter_gray_RMS_ratio':c['filter_RMS_ratio'],'filter_gray_sign_flip_fraction':c['filter_sign_flip_fraction'],'omega1':s['omega'][0],'gap12':s['gap12'],'gap23':s['gap23'],'subspace_N':2,'terminal_warning':False,'raw_FD_max_normalized_error':s['FD']['all_rows_error_over_raw_interior_RMS']['max']}
 rows.append(row)
with (OUT/'MASTER_METRICS.csv').open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
put('scripts/README.md','''# Audit-only reproduction

From repository root, using the repository .venv:

1. `python scripts/identity.py` (prefix paths with this audit directory): validate source/evidence hashes and exact endpoints, recover configs.
2. `python scripts/geometry.py`: read saved trajectories, generate geometry and deterministic sample files.
3. Only after step 2 completes, MATLAB `frozen_evaluate`: native FE/gradient/FD kernels, no optimizer. Existing valid fd_*.mat cases are skipped. It requires MATLAB/license access. Do not run cp_fixedwork, any production solver or test suite.
4. `python scripts/stationarity.py`, then `python scripts/cluster_robustness.py`, `python scripts/local_core.py`, `python scripts/timeline.py`: offline array diagnostics. The cluster fit is an exploratory dual-space least-squares lower bound, not density optimization.
5. `python scripts/write_report.py`, then `python scripts/finalize.py`: write documents/manifests, verify input/source integrity and output hashes.

Set MPLCONFIGDIR to a writable temporary directory for plots. Invalid integer FD recordings are not inputs to any analysis. Source inputs and production are read-only. No runs directory is created. All density perturbations in the MATLAB evaluator are independent derivative checks around one immutable saved state; no perturbation becomes the next baseline.
''')
print('Wrote report, metrics and required phase documents')
# Additional local-core sensitivity of the multiplier reconstruction.
local=json.loads((OUT/'evaluations/local_core.json').read_text())
qualification='''

### Local-core qualification

A separate core-only multiplier fit gives raw broad-core normalized RMS **.06063 at 480** and **.18259 at 800**. At 480 the fitted raw derivative threshold 1.35315 closely matches the filtered gray threshold 1.35847; using that filtered-derived dual gives core RMS .06064. This is positive evidence of **approximate local balance inside much of the 480 broad patch**. It does not eliminate raw residuals in the surrounding gray field: all-gray RMS is .37060 with the same core-fitted multiplier. At 800 even the core-only fit remains above .1; using its filtered gray dual gives core RMS .37089.

Thus the primary nonstationarity category applies to the **complete constrained design and its substantial unresolved gray regions**, not every gray element. Some gray locations are locally balanced, and the 480 core could participate in a nearby stationary gray design. This audit cannot rule that out. The mixed-stationarity category is not issued as a certification of those patches because no common admissible physical multiplier makes the surrounding free design stationary. Local fits cannot be assigned independently to different parts of one volume-constrained problem. These results narrow the conclusion: “the whole endpoint is a genuine relaxed optimum” is unsupported; “every broad patch is itself necessarily nonstationary” is also unsupported.
'''
for name in ['KKT_STATIONARITY.md','CROSS_MESH_REGIME_CHANGE.md','REPORT.md']:
 p=OUT/name;p.write_text(p.read_text()+qualification)
metrics['local_core_robustness']=local
put('METRICS.json',json.dumps(metrics,indent=2,allow_nan=False))
