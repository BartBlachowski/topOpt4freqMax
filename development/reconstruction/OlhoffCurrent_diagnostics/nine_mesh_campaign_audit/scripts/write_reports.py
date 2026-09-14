"""Render audit Markdown from measured offline outputs. No scientific execution."""
from pathlib import Path
import json,csv
P=Path(__file__).resolve().parents[1]
def j(n):return json.loads((P/n).read_text())
def csvread(n):return list(csv.DictReader((P/n).open()))
def fmt(x):
 if x is None or x=='':return 'unavailable'
 if isinstance(x,float):return f'{x:.6g}'
 return str(x)
def table(rows,cols):
 return '| '+' | '.join(label for key,label in cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(k)) for k,label in cols)+' |' for r in rows)+'\n'
def write(n,text):(P/n).write_text(text.strip()+'\n')
s=j('summary.json');r=s['master'];v=j('verification.json');counter=s['counterfactuals'];legacy=s['legacy_comparison'];fits=s['fits'];events=csvread('HISTORICAL_CONTROLLER_EVENTS.csv')
base='The nine-mesh observations below are the **legacy beta/four-rung campaign**, not a three-rung E-controller campaign. This distinction applies to every numerical table and plot unless explicitly labelled historical E-controller evidence.'
verdicts={'CAMPAIGN':'NINE_MESH_CAMPAIGN_INTEGRITY_FAIL','CONTROLLER':'THREE_RUNG_CONTROLLER_CROSS_MESH_INCONCLUSIVE','TERMINATION':'TERMINATION_CROSS_MESH_NOT_CREDIBLE','MESH OBJECTIVE':'OBJECTIVE_MESH_CONVERGENCE_INCONCLUSIVE','TOPOLOGY':'TOPOLOGY_MESH_CONVERGENCE_INCONCLUSIVE','MULTIPLICITY':'MULTIPLICITY_CROSS_MESH_MIXED','PERFORMANCE':'PERFORMANCE_SCALING_COSTLY_BUT_INTERPRETABLE','VALUE OF THREE-RUNG WORK':'THREE_RUNG_WORK_PARTIALLY_JUSTIFIED','PROGRAMME':'IMPLEMENTATION_OR_CAMPAIGN_INVALID'}
write('VERDICTS.json',json.dumps(verdicts,indent=2))
write('PROVENANCE.md',f'''# Provenance and campaign identity

The intended promoted three-rung campaign is absent. The most recent nine-mesh campaign is `examples/Performance/conference_benchmark/campaign_9mesh_r2`, generated 2026-09-11 at 23:51:04+02:00 (results JSON 23:51:05). It contains 27 observations, nine each for three methods; this audit selects `method_key=olhoff`. Other similarly named directories contain earlier campaigns, including a different fixed-work stabilization formulation; they are not pooled.

Initial branch `benchmark-methodology-r2`; HEAD `013cc48451d33bed61c5c4eea174bbd898d548a2`; initial tracked and untracked status clean. Audit changes are confined to this directory. No AGENTS.md was found. The user-request attachment and repository evidence policy were read; no optimization-capable tests were run.

## Verified source identity

All **21/21** manifest source hashes match the current files. All **75/75** production implementation files match `SOURCE_MANIFEST.json`, with no extra source files. Independently reconstructed `+impl` tree hash:

`{v['impl_tree']}`

Both dirty campaign driver files are now the exact HEAD versions. The campaign recorded parent commit `bba45e72ea18eca7615315fcc543572d42a436bc-dirty`. The subsequently committed changes add common record fields and a warm-up record-schema check; inspection finds no controller promotion in that diff. A git subject such as “Nine test passed” is not scientific evidence.

`olhoffcurrent_config.m` SHA-256: `16b431039ec47a9131a5b9365d1e4e28628507cc88e986684bf982c28e462b50`. The effective per-mesh hashes were independently recreated using the 81-row schema and MATLAB-compatible value formatting. **9/9 match**. For C320, production is `2a5b500991ff5931eef4919e768503d41fd558fe56ffbffe40c382c74d676cad`; the validated three-rung candidate is `afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab`. Config hashes correctly differ across meshes because mesh and derived tolerance belong in them.

## Exact effective formulation

The archive stores canonical configs in `records.effective_config`; the manifest independently stores their canonical form. They agree field by field excluding provenance metadata. [effective_configs.json](effective_configs.json) and `config_<mesh>.txt` retain every value.

Domain 8×1, thickness 1, Q4 plane stress, consistent mass; mid-height simple supports with both ends axially restrained (four constrained DOFs). E=10^7, nu=0.3, solid density=1; uniform initial density 0.5, minimum 0.001, volume fraction 0.5. SIMP p=3 fixed; mass eq4b, q=1, low-density exponent 6, cutoff 0.1. Sensitivity filter on all generalized gradients, physical R=0.06; no projection or material continuation. Fixed subspace size 2, diagonal offsets retained and off-diagonal gradients active. Eigensolver `eigs`, tolerance 10^-12, cap 5000, deterministic sine-based start vector. Published MMA on increments, reset for each outer solve, tolerance 0.05, min/max inner 5/500.

**Actual ladder [0.04,0.02,0.01,0.005]; continuation `boundVariable` (legacy alias beta), window 10, tolerance 0.005; stop `designChange`, L2, tolerance 0.05 sqrt(NE/3200), settledMove guard only; cap 400.** No stageExhaustion authority. No runtime fallback occurred: legacy is the configured default.

All nine share this formulation. Only mesh dimensions, derived L2 tolerance, runtime name and provenance metadata differ; physical radius and scientific settings do not. There are no mesh-specific cap or solver overrides. The other method's Yuksel cap override is not an Olhoff change.

## Machine, time and retention

Campaign host `PMS.local`, MATLAB `25.2.0.2998904 (R2025b)`, MACA64, Apple Accelerate ILP64 BLAS, requested and reported computation threads=1. This is one sequential driver invocation with a discarded 48×6, five-outer warm-up. CPU model, RAM capacity, macOS version, competing-load record, thermal state, and peak memory were **not recorded**. Current machine values would not prove campaign conditions, so none are substituted.

Manifest configuration-resolution times are around 20:12; those are not exact job starts. Per-run canonical `resolvedAt` values are retained in effective_configs.json; they locate pre-solve configuration resolution, not a complete start/end timing log. Export at 23:51:04–05 is known; exact solver start/end times are unavailable. Sum of the nine Olhoff solver times is {sum(x['runtime_total_s'] for x in r)/3600:.4f} h. Model preparation and final modal evaluation are in solver wall time; config/path setup, export and common evaluator are outside it.

The driver has no load/resume route for these records. No resume is declared, and nothing in the stored terminal logs indicates one. This does not prove the absence of discarded earlier attempts. The run-time files were dirty but are hash-identified. No append-only execution log or original output checksum seal exists. JSON/MAT agreement is internal consistency, not independent execution authentication.

Existing topology/export ancillary files include older September 6 timestamps beside September 11 results; they cannot establish current topology identity. All audit plots were rebuilt from the MAT density vectors. Original output files may have been overwritten by the campaign; subsequent regeneration cannot be fully excluded. We do not repair or rewrite any source or historical artifact.

## Promotion history and retained historical defects

The pre-campaign `three_rung_promotion_closure/REPORT.md` explicitly says promotion was blocked and production remained legacy. The September 11 “ready” commit added reports and validation artifacts, but changed no production policy. Campaign `manifest_role` expressly says OUTPUT, not preregistration. No prospective nine-mesh three-rung preregistration or successful production-promotion seal was found.

Historical four-rung C160/C240/C320/C400 trajectories are present and match their own current EVIDENCE declarations; independent A/B replay succeeds. The separate C320 three-rung candidate MAT is missing. Its 352-row CSV and scalar record remain hash-valid, and all 52 scientific CSV columns match the retained C320 oracle prefix (timing and two declared shadow columns excluded). A historical cross-study baseline digest is stale; the candidate's optional old C320 container digest differs from the current original-oracle declaration. P400/F400 raw containers differ from their declarations. See [historical_evidence_checks.json](historical_evidence_checks.json) and [historical_seal_checks.json](historical_seal_checks.json). These defects narrow claims; they do not justify deleting the surviving positive evidence.

Reproduction of this audit: run `scripts/audit.py`, `scripts/supplement.py`, then `scripts/write_reports.py` with repository `.venv/bin/python`. These are read-only postprocessors of source data. Run `scripts/seal.py` last. No MATLAB invocation is needed.
''')
write('CAMPAIGN_INTEGRITY.md',f'''# Campaign integrity

**NINE_MESH_CAMPAIGN_INTEGRITY_FAIL** — for the requested intended three-rung campaign. Every case is INVALID for that identity, starting at 160×20. They remain usable legacy endpoint observations with caveats. “Invalid campaign” here does not mean numerical file corruption was discovered.

{table([{'mesh':x['mesh'],'label':'INVALID','legacy':'VALID_WITH_CAVEAT','status':x['status'],'n':x['outer'],'inner':x['inner_MMA'],'rmin':x['rminEl'],'stage':x['final_stage']} for x in r],[('mesh','Mesh'),('label','Intended campaign'),('legacy','Legacy endpoint'),('status','Recorded status'),('n','Outer'),('inner','Inner'),('rmin','rminEl'),('stage','Last stage')])}

All nine density vectors have exactly NE finite values within [0.001,1], finite ordered positive final frequencies, matching volume/grayness, empty error strings, and zero reported unconverged inner solves. The raw MAT agrees with JSON on all compared counts, timings, terminal metrics, logs, frequency arrays, statuses and config hashes. All effective configs and physical radii are verified. The 400-outer cap was not hit by any Olhoff record.

However, **none retains its full Olhoff trajectory**. `olhoffcurrent_run` reduces `res.hist` to aggregate accounting and final stopping fields without returning the history; `confbench_run_case` preserves those fields but leaves Olhoff telemetry empty. `benchmark_records.mat` therefore contains no Olhoff `hist`, `RHO`, `DRHO`, beta path or E path. JSON additionally removes effective_config, x and telemetry. A final vector of the right length cannot prove complete iteration history, transient finiteness or absence of truncated trajectory data.

Expected campaign export files (MAT, results JSON, manifest, tables, timing schema, notes) exist. Raw data have no contemporaneous scientific-evidence declaration or checksum seal; current audit hashes establish today's bytes only. Tracked JSON matches committed HEAD. Stale ancillary graphics are not used. [INTEGRITY_TABLE.csv](INTEGRITY_TABLE.csv) records every per-case check.

The source path guard passed, but it guards implementation identity, not promotion of the intended policy. The software contained E-controller code while the selected preset remained beta. No evidence supports a silent exception-driven fallback. This is a configuration/promotion failure and an evidence-retention failure, not a falsification of the unexecuted controller.
'''.replace("{{", "{").replace("}}", "}"))
write('CONTROLLER_GENERALIZATION.md',f'''# Controller generalization

**THREE_RUNG_CONTROLLER_CROSS_MESH_INCONCLUSIVE.** None of the nine campaign jobs executed the controller under test. Neither generalization nor its failure can be inferred from this campaign. The first problematic mesh is 160×20 by policy identity; among legacy refinements the first loss of even reaching move=0.01 is 320×40.

Actual legacy paths: 160/240 end at stage 3, move 0.01; the remaining seven end at stage 2, move 0.02. Stage-1 start is 1 for each run. The logs identify final stage starts at outer 90,103,130,138,163,189,198,222,169 respectively, followed immediately by stopping at 91,104,131,139,164,190,199,223,170. Earlier beta declarations on 160/240 cannot be reconstructed from the final archive. They must not be labelled S1/S2 E events. A/B histories, S1/S2/S3 declarations, and first terminal persistent E are **not applicable / not recorded**, not zero.

## What survives in historical evidence

Independent offline replay from stored RHO and DRHO gives:

{table([e for e in events if int(e['stage'])<=3],[('mesh','Mesh'),('stage','Stage'),('start','Start'),('declaration','E declaration'),('duration','Duration'),('branch','Branch'),('amp_over_tol','Amplitude/tol'),('medcos','Median cosine'),('mednet','Median net/path')])}

Every per-iteration A, B, nA and nB matches the historical CSV exactly. Definitions are A: median20 cosine<0 AND median20 net/path<0.5 AND ||drho||≥tol; B: ||drho||<tol AND median20 cosine>0. Each branch must persist for 20 consecutive iterations; alternating A/B is not sufficient. Windows are reset locally at a stage transition. Net/path spans 10 increments. The normalized threshold tol/sqrt(NE)=0.00088388347648 is mesh-independent in RMS units; it is not numerically relaxed under refinement, although the number/spatial concentration of moving elements can alter its effectiveness.

Coarse historical stage-1 declarations grow 102→206→274→388; branch sequence is A/B/A/B. C320 S1 lies only 3.13% above the amplitude boundary, a real near-threshold observation. At all historical S3 endpoints B fires with amplitude only 3.27–10.36% of tol, while median cosine ranges 0.191–0.786: mature in amplitude, but not a stationarity proof. A-terminal stopping would explicitly accept persistent nonzero cancellation, not conventional design convergence.

Every historical stage 2 and stage 3 lasts exactly **39 updates**, the minimum from a 20-step median plus 20-step persistence. Consequently the existing four meshes do not establish that adaptive late-stage declaration timing buys anything over two fixed 39-update dwells after the validated S1 event. They also do not establish that those dwells work at 480–800. Fine-mesh low-amplitude cancellation holes, threshold trends, accidental terminal events and asymptotic coherence cannot be tested without fine-mesh trajectories of the intended policy.

Figures [F08](figures/F08_historical_E_declarations.png) and [F09](figures/F09_historical_branch_map.png) deliberately label historical data and absent fine data. No E events were invented for the campaign.
''')
write('TERMINATION_QUALITY.md',f'''# Termination quality

**TERMINATION_CROSS_MESH_NOT_CREDIBLE** as a scientific maturity claim for the actual legacy campaign. The requested vocabulary has no inconclusive termination code; this verdict concerns its affirmative convergence claims, not the unexecuted three-rung policy.

All nine satisfy their actual programmed design-change test. None is a CAP_HIT disguised as convergence. But all stop exactly one iteration after a move halving, and their final stage contains only two updates. That is strong evidence that the one-step settledMove guard still measures schedule-induced contraction rather than demonstrating a mature trajectory. It does not alone prove what later evolution would be on each finer mesh.

{table(r,[('mesh','Mesh'),('terminal_max_abs_drho','max|drho|'),('terminal_l2_drho','L2'),('terminal_rms_drho','RMS'),('amp_over_tol','L2/tol'),('final_move','Move'),('M_nd','Mnd %'),('gap12','Relative gap')])}

Final 20/50/100 changes in omega1 and Mnd, directional coherence, cancellation, stagewise inner distributions, and multiplicity trajectories are unavailable for **all nine actual campaign runs**. Master cells are blank. No estimate substitutes for absent history. Terminal small RMS updates coexist with 13–51% grayness and large thresholded topology changes. Small step size alone does not distinguish a poor stationary point, prematurely suppressed motion, or a nearly mature design.

Historical legacy CSVs match campaign endpoint outer/inner counts, terminal density norm and Mnd at 160/320/400; they are **supporting historical analogues**, not restored campaign histories. Their last-20 pre-update omega changes are +0.105%, +0.466%, +0.471%; Mnd changes are −0.709, −2.782, −1.862 percentage points. Last-50 changes are +6.659%, +3.816%, +3.549% and −26.073, −13.274, −9.785 Mnd points. All these windows cross move transitions; they are not steady-stage estimates. See [HISTORICAL_LEGACY_WINDOWS.csv](HISTORICAL_LEGACY_WINDOWS.csv). Matched final numbers do not prove bitwise history identity across runs.

For historical E-controller S3 designs, stored density paths do support low residual evolution. Final 20-step net density L1 is 0.000458/0.000243/0.000156/0.000155 for 160/240/320/400. Their Mnd changes are −0.0203/−0.00156/+0.00808/−0.0141 points. The 50/100 windows cross stages because S3 lasts 39 updates. Full measured values and frequency conventions are in [HISTORICAL_TERMINAL_WINDOWS.csv](HISTORICAL_TERMINAL_WINDOWS.csv). The subsequent 0.005 continuation barely changes those designs, strengthening maturity evidence at these four meshes only.

An indexing issue in earlier summaries matters for honest comparison: `hist.omega(:,k)` belongs to the **pre-update** state rho_(k−1), while RHO(:,k) and Mnd are post-update. This audit obtains exact post-update S3 frequencies from the next stored modal evaluation `hist.omega(:,S3+1)`, before any new design update; fixed p, mass and filter make this valid. C320 is 166.4263044135, matching the separate validated candidate record, rather than the older S3 table's 166.4272692776. The difference is tiny but conventions are not silently mixed.

No available nine-mesh evidence proves inner KKT stationarity; “inner converged” means the configured relative-step criterion succeeded. Reported inner failure count is zero, but per-outer diagnostics were discarded.
''')
write('MESH_REFINEMENT.md',f'''# Objective and scientific refinement

**OBJECTIVE_MESH_CONVERGENCE_INCONCLUSIVE.** There is no nine-point sequence for the intended three-rung method. Even the actual legacy sequence does not justify an asymptotic convergence claim.

{base}

{table(r,[('mesh','Mesh'),('h_over_b','h/b'),('omega1','omega1 rad/s'),('omega2','omega2 rad/s'),('gap12','Relative gap'),('M_nd','Mnd %'),('volume_error','Volume error')])}

Omega1 decreases at every refinement: 169.495→167.070→165.951→162.889→161.906→161.034→159.725→159.086→153.302. The net loss is 9.55%; the last step alone loses 3.64%, larger than the preceding fine steps. This is not an observed plateau or a smooth asymptotic sequence. Monotone decrease is compatible with several possible limits; nine finite points cannot mathematically refute eventual convergence. The final discontinuity prevents a credible fitted limit or empirical order. **No y_inf+C h^p fit is forced.**

Mnd increases monotonically 13.40→50.66%; the intermediate-density fraction also rises strongly. The volume stays close to 0.5 but increasingly undershoots, with final error −1.194e−5 at 800. These are not feasibility failures on their own. They demonstrate that objective and topology quality cannot be reduced to accurate volume satisfaction.

Omega2 peaks at 240 then decreases. Gap12 is highly nonmonotonic across the full sequence, then settles below 1% at 640–800 while omega1 and topology worsen. Small spectral gap is therefore not evidence of successful overall refinement. Spectral and density metrics must be read jointly.

The five new fine legacy designs are not demonstrated scientific improvements over 160–400: their first frequencies are lower, grayness higher, and no mature-state or optimality comparison is available. FE discretization error and optimization endpoint error are confounded because each design changes with mesh. A fixed-design FE refinement study was not stored; none is run here.

[Figure F01](figures/F01_omega1_refinement.png) plots both NE and h/b=1/nely. [F02](figures/F02_spectrum.png) and [F07](figures/F07_grayness.png) show the other full-sequence measures.
''')
write('TOPOLOGY_CONVERGENCE.md',f'''# Topology convergence

**TOPOLOGY_MESH_CONVERGENCE_INCONCLUSIVE** for the intended method. Actual legacy endpoints preserve a broad beam layout but fail to demonstrate a stable refined density field.

![Actual nine legacy topologies](figures/F10_topology_atlas.png)

The top and bottom chords, central opening and end load paths persist. The coarse 160 design has multiple small end holes and narrow diagonals. From 240 onward the end layout simplifies; by 400 the end regions contain substantial gray material. The fine sequence becomes more diffuse, and at 800 the central opening shrinks markedly and diagonal members fade into gray regions. Vertical mirror symmetry holds to approximately 10^-11 in mean absolute density; left/right differences are small (0.002–0.016) but nonzero. Thus gross asymmetry is not the main failure. Persistent gray zones are.

## Physical-coordinate metrics

Cell-centre bilinear interpolation onto a common domain x/b∈[0,8], y/b∈[0,1], with edge extension to the domain boundary. No registration, reflection, density projection, threshold-volume correction or smoothing beyond interpolation. L1 is mean absolute density difference; L2 is RMS (domain-normalized). Correlation uses full sampled densities. IoU uses rho≥0.5. Boundary metric is symmetric distance between eroded-mask boundary pixels, including domain-edge boundaries; mean, 95th percentile and Hausdorff are in units of b. Thresholded metrics are diagnostics of a gray design, not manufacturability certificates.

{table(s['topology'],[('pair','Adjacent meshes'),('L1','L1'),('L2_RMS','L2 RMS'),('correlation','Correlation'),('IoU_05','IoU'),('boundary_95pct_over_b','Boundary d95/b')])}

The 640→720 pair looks close (L1=0.01914, IoU=0.9602), but 720→800 reverses that trend (L1=0.05669, IoU=0.7531, boundary d95=0.145b). The latter IoU is the worst adjacent overlap in the series. This directly blocks a conclusion based on the penultimate close pair. Common grids with 200 and 400 cells through height agree closely; the complete sensitivity results are in [TOPOLOGY_METRICS.csv](TOPOLOGY_METRICS.csv).

Native-grid four-connected component counts at rho≥0.5 are 23,1,1,1,1,3,1,1,3. These are threshold-sensitive island counts, not evidence of 23 macroscopic load paths. The atlas and boundary metrics matter more than count alone. There is evidence of changing member/void morphology and increasingly diffuse end zones, but no controlled evidence of a genuine optimization-basin bifurcation rather than stopping-induced changes.

Fixed physical radius does not deliver demonstrated mesh-independent morphology in this campaign. It also does not prove that the filter causes the changes; see FILTER_AUDIT.md.
''')
write('FILTER_AUDIT.md',f'''# Filter and physical length scale

R/b=0.06 is verified in every canonical config, actual MAT config, and solver radius-conversion source. With b=1 and square elements h=1/nely, rminEl=R/h gives the intended 1.2,1.8,2.4,3.0,3.6,4.2,4.8,5.4,6.0. A manifest legacy-view `rminEl=null` is not a missing filter: nonempty physical radius takes precedence and is converted inside the solver.

`prepFilter.m` uses cone weights max(0,rminEl−distance) with row-sum normalization and boundary truncation. Positive interior support includes:

{table(s['filter'],[('mesh','Mesh'),('rminEl','rminEl'),('positive_interior_stencil','Positive stencil entries'),('center_weight_fraction','Center fraction'),('weighted_rms_radius_over_b','Weighted RMS radius/b')])}

The continuum cone's weighted RMS radius is sqrt(3/10)R≈0.03286b. The discrete values cluster near this, including the fine 720→800 step (0.032872→0.032789), while that step loses 3.64% of omega1 and 0.207 of adjacent IoU relative to the preceding pair. There is no comparably large discrete length-scale jump there. At 160, rminEl<sqrt(2) excludes diagonal neighbours: five positive weights and a 60% center weight make it special. The 160→240 stencil expands to nine entries and center weight falls to 27.5%; the topology and gap also change. That is an association, not isolated causality.

Every mesh changes discrete support, so merely matching a kink to a stencil change is weak evidence. The largest grayness jumps also align with premature stopping regimes (320's terminal stage changes; 800 stops unusually early). The evidence ranks the proven policy/stopping defect above an alleged physical-radius inconsistency. A sensitivity filter does not itself guarantee a minimum feature size or binary morphology. No filter change or new controlled experiment is recommended from these data alone.
''')
write('LITERATURE_FIDELITY.md',f'''# Literature fidelity

The actual source was inspected: Du & Olhoff (2007), *Structural and Multidisciplinary Optimization* 34, 91–110, [DOI 10.1007/s00158-007-0101-y](https://doi.org/10.1007/s00158-007-0101-y), local `references/Du2007_Topological.pdf`. Printed p.100 was rendered and checked visually, including Figs.2a,3a,4a. The local publisher's erratum (DOI [10.1007/s00158-007-0167-6](https://doi.org/10.1007/s00158-007-0167-6)) corrects missing increment symbols in determinant equations and initial-frequency notation; it does not change the Fig.3a target.

The relevant target is **Fig.3a, simply supported beam, omega1=174.7 rad/s, bimodal**, not Fig.3b (288.7) or Fig.3c (456.4). Fig.2 gives a/b=8 and uniform initial density 0.5, with initial first frequency 68.7. Section 4.1 specifies 50% volume, isotropic E=10^7, nu=0.3, density=1, and plane stress. The broad campaign layout resembles Fig.3a (outer chords, end cells, central opening), most visibly on coarse meshes. Fine gray zones and disappearing internal members are not a faithful reproduction of its illustrated near-binary topology.

The actual 160 endpoint is 2.98% below 174.7; the 800 endpoint is 12.25% below. Historical three-rung endpoints are approximately 169.975,167.039,166.426,166.452, also below the published target. These are contextual reconstruction discrepancies, **not like-for-like discretization errors**: the paper does not disclose the mesh, R=0.06, this ladder, A/B, numerical stopping thresholds or exact support-node idealization. We do not claim a matched numerical reproduction from coincident domain/material data.

The paper specifies a sensitivity filter, the eigenvalue bound/determinant framework, SIMP-type interpolation and MMA lineage. It describes p normally increasing from 1 to 3; holding p=3 is a reconstruction choice. Mesh, Q4/consistent element realization, both-end axial restraint at mid-height, fixed two-mode subspace with diagonal offsets, precise mass-model selection, move schedule, all A/B mechanics and stopping tolerances require reconstruction choices. No projection is used. The corrected determinant uses eigenvalue **increments**; retained diagonal offsets for separated eigenvalues are a documented reconstruction extension.

Fixed N=2 is not proof of the paper's bimodality: the actual gap reaches 16.18% at 240 and historical validated C320 has a 22.32% gap. Fine legacy gaps below 1% move toward the paper's spectral shape while the objective and density field degrade. The achieved level is a documented, internally traceable reconstruction of the formulation family, with partial qualitative resemblance. It is neither an exact historical implementation nor a demonstrated mesh-independent reproduction of Fig.3a.
''')
write('THREE_RUNG_VALUE.md',f'''# What the three-rung work bought

**THREE_RUNG_WORK_PARTIALLY_JUSTIFIED.** There is strong local computational value and meaningful diagnosis of legacy stopping. Nine-mesh generalization was never tested by the completed campaign.

## Single-factor removal of 0.005

These comparisons use the same E=A OR B controller up to S3. The only architectural difference is whether S3 terminates or descends. For 160/240/400, the three-rung endpoint is a causally valid stored prefix counterfactual; C320 also has an actual validated candidate CSV/record. Its raw candidate MAT is missing now, but the original four-rung raw prefix and all 52 candidate scientific CSV columns remain independently checkable.

{table(counter,[('mesh','Mesh'),('old_status','Four-rung status'),('old_outer','Old outer'),('old_inner','Old inner'),('old_omega1','Old omega1'),('old_M_nd','Old Mnd %'),('three_outer','Three outer'),('three_inner','Three inner'),('three_omega1','Three omega1'),('three_M_nd','Three Mnd %'),('delta_outer','Δouter'),('delta_inner','Δinner'),('delta_omega1','Δomega1'),('delta_M_nd','ΔMnd points')])}

Exact post-update density endpoints are used, with frequencies from the next stored modal analysis. Relative scientific changes are tiny: removing 0.005 changes omega1 by at most about 0.022%, Mnd by at most 0.0521 percentage points, density L1 by at most 0.002121 and threshold-flip fraction by at most 0.139%. C320 saves 1248/1600=78.00% outer and 70034/76532=91.51% inner work. C240 saves 1074/1358=79.09% outer and 38675/44181=87.54% inner work. C160 and C400 save 39 outer and 791/1453 inner respectively. Those are real counterfactual savings, not cheaper computers or a mislabeled capped endpoint.

**Scientific effect:** nearly neutral relative to the four-rung E design. **Computational effect:** eliminate costly low-amplitude late-rung dynamics, dramatic at 240/320. Removal was justified on the tested meshes.

## Against original beta production

This is a broader policy comparison, not a single-factor 0.005 comparison. The physical formulation is common, but continuation/stopping differ; historical wall times are not directly comparable across MATLAB/host conditions. Work counts and density endpoints are informative.

{table(legacy,[('mesh','Mesh'),('legacy_outer','Legacy outer'),('legacy_inner','Legacy inner'),('legacy_omega1','Legacy omega1'),('legacy_M_nd','Legacy Mnd %'),('three_outer','Three outer'),('three_inner','Three inner'),('three_omega1','Three omega1'),('three_M_nd','Three Mnd %'),('delta_outer','Δouter'),('delta_inner','Δinner'),('delta_omega1','Δomega1'),('delta_M_nd','ΔMnd points')])}

Three-rung costs **more** than prematurely stopped beta production. It improves C400 frequency by about 2.19% and reduces Mnd from 32.33 to 15.37%; C320 Mnd falls from 23.36 to 12.94% with a smaller objective gain. At C240 objective is essentially unchanged but the density field is cleaner. Thus the work did not merely deliver faster versions of the original production runs. It delivered more mature designs at additional cost, then removed waste from the longer E-controller solve.

## Answers A–F

A. Relative to four-rung E, no material scientific redesign; relative to legacy, meaningful density improvements and C400 objective improvement.

B. Yes: removing 0.005 preserves the useful E-controller design while eliminating waste on all four known meshes.

C. No material sacrifice is observed against four-rung E. Fine-mesh early-stop sacrifice by three-rung remains untested. The actual legacy campaign is strongly exposed to early stopping.

D. Unknown beyond C400. Legacy cost scaling is not a measurement of three-rung savings.

E. The finer legacy meshes reveal stronger grayness and a large C800 regime change, but not a new three-rung mechanism. The campaign identity error predates every mesh.

F. Two precise simpler alternatives have evidence. First, after the same S1 E declaration, **39 updates at 0.02 then 39 at 0.01, then stop** reproduces the historical three-rung endpoints on all four meshes because both late stages reach their earliest legal E declaration. This is retrospective equivalence, not prospective validation. Second, two-rung E stopping at S2 is close but C160 remains in branch A with amplitude 2.19×tol; the historical preregistered 0.10% objective materiality gate rejected deleting both remaining rungs because the gain to the four-rung endpoint was 0.114%. Relaxing that bar after observing it would be post-hoc rescue. Plain fixed move=0.04 is not universal: C160 hits its cap, whereas C240/C400 have coherent amplitude decay. Stored fixed-move summaries and limits are in [FIXED_MOVE_EVIDENCE.csv](FIXED_MOVE_EVIDENCE.csv).

The available fixed-move reports support different terminal regimes, but several raw histories are missing or have stale container declarations. They do not support a new fine-mesh counterfactual or a claim that one scalar stagnation rule would work everywhere.
''')
write('PERFORMANCE_SCALING.md',f'''# Performance scaling and timing quality

**PERFORMANCE_SCALING_COSTLY_BUT_INTERPRETABLE** for the actual legacy campaign. Three-rung fine-mesh performance is unmeasured.

Total solver time ranges from 119.76 s to 2397.57 s, with C800 dropping to 1987.06 s because outer work falls 223→170. The nine Olhoff runs consume 2.6681 solver-hours. This is feasible as a small workstation campaign but expensive for repeated studies, and it does not establish cost to reach comparable scientific maturity.

## Fits with prefactors

N means **number of elements**, not DOFs or elements per edge. For timings T is in seconds. Fit log(T)=log(C)+p log(N) by unweighted OLS. Counts use the same descriptive form. All rows below are legacy solves; none is excluded as capped.

{table([f for f in fits if f['subset'] in ['all9','fine5'] and f['metric'] in ['runtime_total_s','eigen_s','gradient_s','inner_s','outer','inner_MMA']],[('metric','Quantity'),('subset','Subset'),('C','C'),('p','p'),('R2_log','R² log'),('p_CI95_lo','p 95% lower'),('p_CI95_hi','p 95% upper')])}

All-nine total: **T=0.03701857 N^0.9825066 s**, R²log=0.98002; C 95% interval [0.0104451,0.1311975], p interval [0.85713,1.10788]. Fine-five: **T=0.1217553 N^0.8765449 s**, R²log=0.85324; C interval [8.86245e−5,167.2715], p interval [0.20861,1.54448]. Fine-four gives C=1.483635, p=0.650307, R²log=0.66401 with p interval [−0.7571,2.0577]. The huge fine-subset uncertainty and sensitivity rule out a reliable asymptotic exponent. Prefactors depend on the chosen N units and are correlated with exponent estimates.

[SCALING_FITS.csv](SCALING_FITS.csv) includes all 33 fits, C and p confidence intervals, log RMSE, all-nine/fine-five/fine-four subsets and per-iteration quantities. [SCALING_RESIDUALS.csv](SCALING_RESIDUALS.csv) gives every observed, fitted, absolute, relative and log residual. The intervals are conditional on independent homoscedastic log residuals; these are single ordered observations, not performance-repeat confidence intervals. [F05](figures/F05_total_runtime.png) and [F12](figures/F12_scaling_residuals.png) show fit and residuals.

## Work versus per-iteration cost

Total exponent 0.98251 decomposes algebraically into outer-count exponent 0.26513 plus cost-per-outer exponent 0.71738 on the same all-nine log fit. Fine-five is 0.10908+0.76747. Outer growth is moderate through 720 (91→223); C800's decline is a stopping-regime effect, not better asymptotic complexity. The fine-five outer fit R²=0.1265 is particularly uninformative. Cumulative inner work has exponent 0.23687 all-nine; cost per inner step has exponent 0.72385. Mean inner count per outer stays roughly 20–25. Most scaling comes from cost per step, not exploding inner iterations per outer.

Assembly+eigensolve per outer has exponent 1.14233 all-nine and 1.46197 fine-five. It accelerates near 720, but remains a minority of total cost. Nested MMA per outer grows from 1.284 to 10.530 s; assembly+eigensolve per outer from 0.0282 to 1.1142 s. At 800 the decomposition is MMA 1790.086 s (90.087%), assembly+eigensolve 189.417 s (9.533%), gradients/filtering 6.070 s (0.3055%), other 1.491 s (0.0750%). At 160, MMA is 97.597% and assembly+eigensolve 2.145%.

`tEig` includes FE K/M assembly plus eigSolve, not pure ARPACK time. Pure eigensolver cost cannot be isolated. `tGrad` includes generalized gradients and filtering. Other includes outer bookkeeping and setup/final analysis. These are non-overlapping; summed components reproduce caller-side solver wall time to numerical roundoff. The “outer excluding inner” column is not the eigensolver column. Timing excludes path/configuration work and common evaluator/export, so it is not total human wait time.

## Timing quality verdicts

**Strong:** recorded nesting/accounting consistency, positive costs, same reported host/MATLAB/thread policy/source and timing definitions, overwhelming nested-MMA share, and the distinction between inner counts and per-step cost.

**Indicative:** empirical per-outer scaling, the fine eigensolve cost increase, fitted C/p over this mesh range, and rough workstation practicality. The single ordered sequence confounds mesh with thermal/load effects.

**Unreliable:** bitwise timing reproducibility; machine-independent prefactors; asymptotic complexity from nine endpoints with different maturity; fine-only exponent precision; attributing the 800 time decrease to an optimization improvement; comparing historical controller wall-time ratios without host/toolchain normalization.

No competing-load, temperature, CPU model, RAM or peak memory telemetry is available. Warm-up is recorded but cold/warm effects at larger meshes are not isolated. No resumes are declared; no full external process log authenticates uninterrupted execution. None of these limitations makes the retained scaling data useless; they limit interpretation.
''')
write('INNER_MMA_SCALING.md',f'''# Nested MMA behavior

{base}

{table(r,[('mesh','Mesh'),('outer','Outer'),('inner_MMA','Inner total'),('mean_inner_per_outer','Inner/outer'),('inner_nonconverged','Reported nonconverged'),('inner_share_pct','MMA share %')])}

Cumulative counts grow 2241→4831 through 720 then fall to 3713 at 800. Mean inner/outer falls from 24.63 to approximately 20–22, so there is no aggregate sign of growing nested iteration difficulty. Dimension-dependent per-inner cost grows from 0.05216 to 0.48211 s. The optimizer is still the dominant wall-time expense, despite FE/eigensolve growth. Cumulative fine-mesh work is driven by both outer trajectory length and more expensive inner steps, not an observed explosion in inner solves per outer.

Every actual row reports zero nonconverged inner solves; the driver independently maps any positive count to SOLVER_FAILURE. Per-outer maxima, histograms, stage distributions and isolated spikes cannot be recovered from these aggregates. The known maxInner=500 is a cap, **not an observed maximum**. No such maximum is inserted into MASTER_TABLE.csv.

For the retained historical E-controller, [HISTORICAL_STAGE_WORK.csv](HISTORICAL_STAGE_WORK.csv) supplies stage counts, means, medians, p90, maxima and failures. Stage 3 mean inner counts are 16.31/20.85/19.13/20.79; maxima 25/35/34/35; no failed inner solves. Stage 4 at C240 takes 1074 outer and 38675 inner (mean 36.01, max97); C320 takes 1248 outer and 70034 inner (mean56.12, max139). C160/C400 stage4 costs only 39 outer each. This is the directly observed late-rung pathology that removal addresses.

The historical C320 cap was reached despite every inner solve meeting its own convergence criterion. Inner success does not imply outer maturity. Conversely, no evidence permits saying the five fine three-rung runs avoid this pathology: those runs are absent.
''')
write('MULTIPLICITY_AUDIT.md',f'''# Multiplicity and eigenvalue structure

**MULTIPLICITY_CROSS_MESH_MIXED** for the available evidence; intended fine three-rung behavior is unknown.

The code's `subspace` method fixes N=2 every iteration. The constant final N is therefore imposed, not an observed spectral multiplicity classification. Diagonal offsets retain separation and make treating two distinct modes within a subspace a documented reconstruction choice; gap>5% is not by itself a coding error in this mode. It is, however, incompatible with calling those frequencies a reproduced near-double eigenvalue.

{table(r,[('mesh','Mesh'),('omega1','omega1'),('omega2','omega2'),('gap12','Relative gap12'),('multiplicity_N','Fixed N'),('multJ_warning_count','Multiple-J warnings')])}

Five actual endpoints from 240 through 480 have substantial gaps (240=16.18%,320=10.74%,400=7.74%,480=6.97%; 560 drops to 2.90%). The three finest are 0.899%,0.923%,0.601%, suggesting local approach of the first two modes, not proof of exact coalescence or global stabilization. Historical S3 C320 is even more separated, at 22.3248%, despite a mature density field and credible terminal B. Neither A/B nor fixed N enforces bimodality.

A separate material caveat appears in **all nine** solver logs: the third mode J is sometimes itself near-multiple with the next mode under the 5% numerical warning criterion. Counts are 2,3,1,2,4,4,7,43,81. The solver expressly logs `(25b) undefined` and continues with the simple-J constraint; it does not repair the spectral cluster. At 800 this occurs on 81/170=47.65% of outer iterations. At 720 it is 43/223=19.28%. These warnings are not exceptions or NaNs and are not included in the reported inner failure count.

This is a documented domain-of-validity limitation of the next-mode constraint, increasingly exercised on fine meshes. The available logs prove occurrence, not its causal effect on objective/topology. No per-iteration eigenpairs, fourth-frequency history or native residual history is retained for the campaign, so cluster identity, eigenvector rotation and mode exchange cannot be independently checked. The final common evaluator's alternative interpolation results are not substituted for native mode histories.

The multiple-J caveat warrants disclosure in any benchmark, independently of controller failure. Changing mass, multiplicity handling or MMA is outside this audit, and no such repair was made.
'''.replace('Five actual endpoints from 240 through 480','Four actual endpoints from 240 through 480'))
write('REGIME_CHANGE_AUDIT.md',f'''# Discontinuities and alternative explanations

Regimes are evaluated on all points; no smooth-family assumption is made.

| Transition | Observed discontinuity | Most supported reading |
| --- | --- | --- |
| 160→240 | gap 1.45→16.18%; stencil5→9; end holes simplify; IoU0.797 | Different endpoint morphology/spectrum; filter-support association possible, causality unisolated |
| 240→320 | final stage3→2; Mnd15.60→23.36%; gap remains10.74% | Direct change in where legacy stopping is admitted |
| 320→400 | Mnd23.36→32.33%; density L1=0.0891; omega loses1.85% | Greater premature-stop quality loss; historical E common-mesh comparison supports this |
| 400→560 | gradual objective loss/gray increase; gap falls | Fine legacy evolution, no demonstrated three-rung phenomenon |
| 560→640 | gap enters<1%; grayness keeps increasing | Spectral coalescence trend does not certify density maturity |
| 640→720 | close densities (IoU0.960); eigensolve/outer jumps; multiple-J warnings7→43 | Apparently stable pair plus separate numerical spectral/cost regime change |
| 720→800 | outer223→170; omega159.086→153.302; Mnd41.24→50.66; IoU0.753; warnings43→81 | Strong endpoint/stopping regime change; no convincing asymptotic refinement |

## Competing causes ranked by evidence

For **failure to answer the central three-rung question**, the proven cause is unpromoted configuration and discarded histories. FE, filter, MMA or mass interpolation cannot explain away which config was executed.

For **grayness growth and unstable endpoint quality**, rank: (1) stopping/continuation defect—direct logs show every stop immediately after a move halving, and historical same-mesh E paths recover substantially cleaner C320/C400 designs; (2) mesh-dependent topology evolution—direct field differences, but inseparable from stopping; (3) multiplicity/next-mode approximation—direct increasing warning count, unknown impact; (4) filter discretization and FE discretization—real discretization changes, no controlled attribution; (5) mass interpolation and MMA globalization—fixed across meshes, plausible interacting factors but no isolating comparison. The *three-rung* controller defect hypothesis has no fine-mesh test here.

For **C800 discontinuity**, evidence ranks premature admission and altered trajectory ahead of simple FE order arguments. Near-multiple J handling is a stronger specific numerical concern than an alleged radius mismatch. Stable effective cone radius and abrupt loss of iterations weaken a filter-only explanation. A distinct optimization basin is possible, but not established without histories or a controlled same-mesh comparison. Small gap12 at C800 does not exonerate the J=3 approximation.

For **computational expense**, direct cost data rank nested MMA per-step expense first, outer trajectory length second, FE/eigensolve dimension third in total share (although its exponent is higher). No evidence shows increasing mean nested iteration count per outer. Host load/thermal effects could contribute to the late eigensolve kink, but no load telemetry can rank them quantitatively.

For **historical 0.005 pathology**, same-prefix four-rung evidence isolates the extra rung and its persistent low-amplitude/cancellation dynamics as the leading explanation. Physical formulation is fixed. Changing filter, mass or FE is unnecessary to explain 1248 additional C320 iterations with negligible density change.

These rankings concern evidence strength, not asserted causal probabilities. No recommendation to change several ingredients at once follows.
''')
score=[
('Technically correct implementation?','MIXED','Controller mechanics reproduce every retained historical A/B predicate and counter. Production selection is wrong for the intended policy; the next-mode simple-eigenvalue assumption is violated without a corresponding treatment.'),
('Honest convergence criterion?','MIXED','Historical persistent E distinguishes cancellation from coherent small updates and labels caps honestly. The executed legacy criterion admits all nine endpoints one iteration after halving the move; neither criterion proves KKT optimality.'),
('Three-rung eliminated real wasted work?','YES','C320 saves 1248 outer/70034 inner; C240 saves 1074/38675 relative to four-rung E. These are same-prefix computational counterfactuals.'),
('Preserved scientific quality where counterfactuals exist?','MOSTLY YES','All four retained S3 densities differ negligibly from continued 0.005 endpoints. C320 candidate raw evidence is missing, but the surviving CSV and oracle reinforce the result.'),
('Generalized to all nine meshes?','INCONCLUSIVE','The nine jobs ran legacy beta, so no fine three-rung observations exist.'),
('Objective mesh-convergent?','INCONCLUSIVE','Legacy omega1 decreases by 9.55% with an abrupt final 3.64% loss; no credible asymptotic fit. No intended nine-point sequence exists.'),
('Topology mesh-convergent?','INCONCLUSIVE','Broad beam members persist, but adjacent overlap worsens sharply at 800 and grayness rises to50.66%. This is not demonstrated continuum topology convergence.'),
('Multiplicity treatment stable?','MIXED','Fixed N=2 is stable by construction, while actual gaps vary widely. Multiple-J warnings rise to81/170 iterations at800.'),
('Computationally practical?','MIXED','Actual legacy campaign takes2.668 solver-hours; individual solves2–40min are manageable.90–98% is nested MMA, and cost to reach intended fine-mesh maturity is unknown.'),
('More defensible than original beta/fixed-ladder production?','MOSTLY YES','Historical E design paths correct demonstrated premature stops and support removal of wasted0.005 work. This improvement was never selected in the completed campaign.'),
('Controller investigation scientifically informative?','YES','It falsified beta stagnation as universal maturity, exposed two terminal regimes, and isolated scientifically immaterial but costly late-rung dynamics.'),
('Suitable for benchmark publication?','MOSTLY NO','Not as a validated three-rung or mesh-independent method. Legacy endpoint timings can be published only with explicit actual-policy labels, stopping caveats, retained-data limits and multiple-J warnings.'),
('Suitable only as a documented reconstruction?','MOSTLY YES','A transparent reconstruction/diagnostic case study is supported. Exact historical reproduction, mature cross-mesh benchmarking and a new universal algorithm are not established.'),
('New research question worth pursuing?','INCONCLUSIVE','No new three-rung mechanism is observed. Increasing multiple-J warnings identify a concrete unresolved approximation, but current evidence does not establish that further controller tuning is the right research programme.')]
score=[(q,a,t.replace('takes2.','takes 2.').replace('solves2','solves 2').replace('.90–98','. 90–98').replace('to50','to 50').replace('to81','to 81').replace('at800','at 800').replace('wasted0.','wasted 0.')) for q,a,t in score]
write('SCORECARD.md','# Was the struggle for nothing?\n\nNo. The development produced useful coarse-mesh evidence, but the final campaign did not test the developed policy.\n\n'+table([{'n':i+1,'question':q,'answer':a,'evidence':t} for i,(q,a,t) in enumerate(score)],[('n','#'),('question','Question'),('answer','Verdict'),('evidence','Evidence')]))
write('RECOMMENDATION.md','''# Programme recommendation

**IMPLEMENTATION_OR_CAMPAIGN_INVALID** — option E, specifically the intended campaign is invalid because the three-rung policy was not promoted/selected. This is the one primary recommendation. It does not mean all retained numbers are corrupt, or that historical controller science is refuted.

Stop this audit at diagnosis. Preserve the existing campaign as legacy beta evidence, preserve the validated three-rung work separately, and do not publish or cite the former as the latter. Do not claim cross-mesh validation, mesh-independent convergence or fine-mesh savings for three-rung. No tuning, cap extension, rerun, source repair or regeneration is authorized by this conclusion.

The immediate next action is documentary: resolve the mistaken campaign identity and the owner's expectations before deciding whether any further compute has value. The audit itself makes that correction reviewable without changing production. Missing evidence and stale historical declarations should be disclosed; originals can be restored only from existing archives, not silently replaced by regenerated runs.

There is no justified new *controller-tuning* experiment from this campaign. The increasing multiple-J warnings are a specific unresolved numerical concern, but their effect is unmeasured and the evidence does not identify further A/B tuning as the remedy. No new scientific experiment is proposed. The campaign provides no reason to reopen radius, p, mass, q, projection and controller thresholds together.

Whether to stop the research programme permanently cannot be honestly decided from a campaign that never ran its candidate. Equally, “the candidate was not tested” is not an automatic argument to fund another campaign. The demonstrated coarse-mesh benefit can stand as the final outcome if the owner chooses to stop here.
''')
# Main report with the mandated direct opening and all 25 questions.
answers=[
('Are all nine runs trustworthy?','As legacy endpoint records, internally consistent with caveats: source/config hashes, raw/JSON scalars and final densities agree. As the requested three-rung campaign, no: all nine have the wrong policy and lack retained histories.'),
('Did all nine converge?','All nine report NATIVE_CONVERGED under legacy designChange. None demonstrated terminal persistent E at0.01; seven stop at0.02. A programmed stop is not proof of mature topology.'),
('Did the three-rung controller generalize?','Unknown. It was not executed in the nine-mesh campaign.'),
('First problematic mesh?','160×20 for policy identity.320×40 is the first legacy endpoint to stop atstage2;800×100 shows the strongest fine-mesh deterioration.'),
('Systematic refinement trend?','Omega1 falls and grayness rises throughout. Inner work grows through720 then drops at800; multiple-J warnings increase sharply at720/800.'),
('Does omega1 converge?','No supported asymptotic conclusion.169.495→153.302, with a3.64% loss at the last refinement; no justified limit/order fit.'),
('Does topology converge?','A broad beam family persists, but density convergence is not demonstrated. Adjacent IoU improves to0.960 at640→720 then falls to0.753 at720→800.'),
('Do gap and multiplicity stabilize?','Only the last three legacy gaps are below1%; full-sequence gaps are not stable. Fixed N=2 is imposed, and J=3 approximation warnings grow to81/170 iterations.'),
('Is stopping physically credible?','Not as an across-mesh maturity claim for these records: every stop is one iteration after a move halving. The separate historical E endpoints have much better supporting trajectory/counterfactual evidence.'),
('How much cost is eigensolving?','Recorded assembly+eigensolve share rises from2.15% to9.53%; pure eigensolver time is not separable from FE assembly.'),
('How much is nested MMA?','97.60% at160 and90.09% at800, dominant on every mesh. Mean inner work per outer remains about20–25.'),
('Fitted C and p?','Legacy total wall time in seconds with N=NE: all-nine C=0.03701857,p=0.9825066,R²log=0.98002; fine-five C=0.1217553,p=0.8765449,R²log=0.85324. Fine-four p=0.6503 is unstable. Full component fits, residuals and conditional intervals are provided.'),
('Does iteration count scale badly?','Not in the measured aggregate: all-nine outer p=0.2651; per-outer cost p=0.7174. The800 count reduction is a problematic stopping-regime change, not efficiency validation.'),
('Are fine meshes scientifically better?','Not demonstrated: lower omega1, much higher grayness and no evidence of comparable terminal maturity. Smaller gap alone is insufficient.'),
('What did three-rung improve?','Against four-rung E, remove real wasted work with nearly unchanged useful designs. Against beta production, more mature density fields and a meaningful C400 frequency gain, at increased work.'),
('What did it fail to improve?','It did not establish exact paper reproduction, spectral degeneracy, optimality, continuum topology or fine-mesh practicality. Its production adoption also failed.'),
('Could something simpler do as well?','Retrospectively, keeping S1 E then fixed39-update dwells at0.02 and0.01 reproduces all four known three-rung endpoints. Plain fixed0.04 and earlier S2 termination are not universally supported; no fine-mesh result tests the simpler dwell alternative.'),
('Did0.005 deserve removal?','Yes on all four known meshes: C320 saves78.00% outer and91.51% inner; C240 saves79.09% outer and87.54% inner, with negligible scientific differences.'),
('Was A/B investigation useful?','Yes. Independent replay confirms the different regimes and causal timing; objective stagnation and amplitude alone were inadequate universal maturity signals.'),
('Did this campaign show further controller tuning has low value?','It did not test controller tuning at all. Historical minimum-duration late stages suggest limited demonstrated value of late-stage adaptivity, but that is separate evidence.'),
('Good enough for a benchmark table?','Only as explicitly labelled legacy reconstruction endpoint/timing data with limitations. Not as a validated three-rung performance benchmark or equal-quality comparison.'),
('Good enough for a mesh-independent convergence claim?','No.'),
('Faithful reconstruction rather than new method?','Best called a documented reconstruction, with partial qualitative fidelity and explicit unsupported choices; “faithful” must not imply exact numerical or spectral reproduction.'),
('One high-value follow-up experiment?','No new controller experiment is justified now. The first need is correct campaign identity and retained evidence. The multiple-J warning concern is documented, without proposing a new experiment.'),
('Should the project stop here?','Stop the present audit without repair or rerun. Preserving the coarse-mesh result and ending the programme is defensible; condemning the three-rung controller on these nine legacy jobs is not. The primary decision is campaign invalidity, not a speculative tuning programme.')]
answers=[(a,b.replace('at0.','at 0.').replace('at160','at 160').replace('at800','at 800').replace('atstage','at stage').replace('through720','through 720').replace('at720','at 720').replace('to81','to 81').replace('below1','below 1').replace('from2','from 2').replace('to9.','to 9.').replace('about20','about 20').replace('fixed39','fixed 39').replace('fixed0.','fixed 0.').replace('saves78','saves 78').replace('and91','and 91').replace('saves79','saves 79').replace('and87','and 87').replace('the800','the 800').replace('to0.','to 0.').replace('at640','at 640').replace('a3.','a 3.').replace('.169','. 169').replace('.320','. 320').replace(';800','; 800')) for a,b in answers]
write('REPORT.md',f'''# BOTTOM LINE

**No, the struggle was not for nothing. But the supposed final test of its outcome did not happen.** The completed September11 campaign ran the old beta/four-rung policy on all nine meshes. The three-rung E-controller was validated locally but never promoted into the production preset used by this campaign. Calling these nine results a failure—or a success—of that controller is scientifically wrong.

The owner's impression is **partly correct** about delivery: substantial development did not reach the final production experiment, and the actual fine-mesh outputs are increasingly gray and poorly supported as mature designs. It is **wrong** about scientific value: removing0.005 demonstrably eliminated enormous wasted work at240/320 with negligible endpoint change, and the earlier investigation exposed real deficiencies of beta stopping.

The one primary recommendation is **IMPLEMENTATION_OR_CAMPAIGN_INVALID** (option E: wrong campaign policy, not blanket numerical corruption). No new optimization, retuning, source repair, rerun or cap extension was performed. This audit does not automatically recommend another campaign.

## Evidence that decides the case

1. All nine MAT effective configs and manifest configs say `[0.04,0.02,0.01,0.005]`, `boundVariable`, `designChange`, cap400. Independently recomputed config hashes match9/9; source hashes match21/21 and the75-file implementation tree is exact.
2. Production config still delegates to the unpromoted old preset. The earlier promotion-closure report explicitly recorded promotion blocked. There is no hidden three-rung fallback explanation.
3. All nine legacy solves stop **one iteration after a move reduction**; seven stop at0.02. Full Olhoff histories were discarded, so terminal E cannot be replayed for any campaign mesh.
4. Legacy omega1 decreases169.495→153.302 while Mnd increases13.40→50.66%. The final refinement loses3.64% frequency and yields adjacent topology IoU0.753 after the previous pair's0.960.
5. Historical retained E trajectories independently reproduce every A/B decision on160/240/320/400. Removing0.005 saves70034 inner steps at320 and38675 at240 with negligible density/objective change. The separate validated C320 MAT is missing, but its recorded352-row scientific CSV matches the retained oracle prefix.

## Final verdicts and their scope

{table([{'category':k,'verdict':x} for k,x in verdicts.items()],[('category','Category'),('verdict','Verdict')])}

Campaign/controller/objective/topology conclusions address the requested intended method and the limits of the available evidence. Termination, multiplicity and performance describe the actual legacy records where measurements exist; they must not be misread as fine-mesh three-rung observations. The termination vocabulary lacks an inconclusive code, so NOT_CREDIBLE rejects the actual campaign's scientific-maturity claim while leaving the absent intended fine-mesh test unassessed.

## Observed endpoint summary

{table(r,[('mesh','Mesh'),('outer','Outer'),('inner_MMA','Inner MMA'),('omega1','omega1'),('M_nd','Mnd %'),('final_move','Final move'),('runtime_total_s','Wall s')])}

The authoritative full table is [MASTER_TABLE.csv](MASTER_TABLE.csv), with blanks for unavailable data and definitions in the section reports. It never presents these as three-rung endpoints.

## Direct answers

'''.replace('{{','{').replace('}}','}')+'\n\n'.join(f'**{i+1}. {a}** {b}' for i,(a,b) in enumerate(answers))+'''

## What supports the recommendation

The audit distinguishes three comparisons. The actual nine-mesh legacy campaign has poor scientific-maturity support. Historical E-controller versus legacy beta shows that cleaner designs required additional work. Historical three-rung versus four-rung E shows that removing0.005 retained those cleaner designs while eliminating largely useless continuation. Mixing those comparisons would make the three-rung method look either universally faster or scientifically ineffective, neither of which the evidence says.

The strongest unresolved fine-mesh numerical concern is the increasing incidence of a near-multiple third mode while the next-mode constraint assumes simplicity. It is logged on81 of170 outer iterations at800. That is material disclosure, not proven causality and not a mandate to change multiplicity, mass, filter and MMA together.

No asymptotic objective limit was fitted. No missing terminal trajectory was synthesized. No historical wall-time ratio was passed off as a same-machine causal saving. Current hashes establish retained bytes, not an original missing campaign seal. Every plot was created from stored numerical evidence; the historical branch figures label the five fine meshes as absent.

## Artifact guide

Provenance/integrity: PROVENANCE.md, CAMPAIGN_INTEGRITY.md, verification.json, INTEGRITY_TABLE.csv, effective_configs.json.

Controller/science: CONTROLLER_GENERALIZATION.md, TERMINATION_QUALITY.md, MESH_REFINEMENT.md, TOPOLOGY_CONVERGENCE.md, FILTER_AUDIT.md, LITERATURE_FIDELITY.md, MULTIPLICITY_AUDIT.md, REGIME_CHANGE_AUDIT.md.

Value/cost/decision: THREE_RUNG_VALUE.md, PERFORMANCE_SCALING.md, INNER_MMA_SCALING.md, SCORECARD.md, RECOMMENDATION.md. Twelve figures are supplied in PNG and SVG, including the actual topology atlas and residual plots. Figures8/9 necessarily use the four historical E meshes; the requested nine-mesh E data do not exist in the campaign.

Reproducibility: scripts/, INPUT_MANIFEST.json, DATA_MANIFEST.json, EVIDENCE.json, FINAL_SHA256.txt. The original paper and its erratum were inspected; no external optimization or evaluation was run.

# WHAT WE LEARNED

The local three-rung result has measurable value. Its adoption into production failed. The legacy nine-mesh campaign shows increasingly gray endpoints, schedule-linked stopping, dominant nested-MMA cost and a growing next-mode multiplicity warning regime. Source correctness, policy identity, stopping maturity and mesh convergence are separate claims.

# WHAT WE DID NOT LEARN

We did not learn whether the intended three-rung method generalizes to480–800, whether its terminal E stays credible there, its fine-mesh computational savings, or a continuum objective/topology limit. The missing trajectories also prevent reconstructing exact late dynamics of the actual campaign. These are unavailable observations, not negative measurements.

# WHAT I WOULD DO NEXT

Preserve this audit and correct the description of the completed campaign. Keep the coarse-mesh three-rung finding as a useful, bounded result. Make no controller change or new scientific run on the strength of this audit. If the owner chooses to stop the programme, stop without pretending the legacy campaign refuted the untested controller. If the owner later chooses more work, campaign identity and durable evidence must be resolved first; this report does not authorize that work.
''')
# Clean accidental compact prose in authored Markdown only.
for p in P.glob('*.md'):
 if p.name=='AUDIT_PREREGISTRATION.md':continue
 t=p.read_text()
 for a,b in [('Removing0.005','Removing 0.005'),('stage2','stage 2'),('max97','max 97'),('mean56','mean 56'),('max139','max 139'),('stage4','stage 4'),('and90','and 90'),('The800','The 800'),('and0.01','and 0.01'),('Did0.005','Did 0.005'),('September11','September 11'),('removing0.005','removing 0.005'),('cap400','cap 400'),('match9/9','match 9/9'),('match21/21','match 21/21'),('the75-file','the 75-file'),('at0.02','at 0.02'),('decreases169','decreases 169'),('increases13','increases 13'),('loses3','loses 3'),('IoU0','IoU 0'),("pair's0","pair's 0"),('on160','on 160'),('saves70034','saves 70034'),('at320','at 320'),('and38675','and 38675'),('at240','at 240'),('recorded352','recorded 352'),('on81','on 81'),('of170','of 170'),('at800','at 800'),('Figures8/9','Figures 8/9'),('to480','to 480'),('be720','be 720'),('and800','and 800')]:t=t.replace(a,b)
 p.write_text(t)
print('Wrote audit reports')
