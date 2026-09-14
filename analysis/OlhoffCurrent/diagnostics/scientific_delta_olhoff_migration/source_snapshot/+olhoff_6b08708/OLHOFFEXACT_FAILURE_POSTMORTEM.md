# OlhoffExact failure post-mortem

**Forensic comparison of the previous `OlhoffApproachExact` effort, the clean-room Du–Olhoff reproduction, and Du & Olhoff (2007)**  
**Evidence cut-off:** 25 August 2026  
**Scope:** Fig. 3a / Fig. 4, simply supported beam, target mode `n = 1`

This is an audit, not a solver-development document. No production source was changed and no optimization campaign was launched. The only execution performed for this report was read-only loading of existing MAT files to expose saved configurations and histories.

## 1. Executive verdict

The earlier effort did not fail because its finite-element model, generalized-gradient derivation, eigenvector normalization, or simply-supported boundary condition was fundamentally wrong: at equal mesh its uniform-design spectrum is numerically identical to the clean-room result, and its derivative tests were strong local validations.  
It failed because a correct local model was embedded in a materially different optimization path.  
The original nested-MMA implementation normally used `rmin=2.5`, returned increments that had not met its declared inner stopping test, ignored that status, and consequently took destructive full-box steps or entered move-saturated oscillations.  
The later rebuilt solver used the favorable `rmin=1.2` but combined a full-coupling LMI cutting-plane subproblem with adaptive trust acceptance; it reached `N=2` at iteration 16 and then rejected every further SS trial until `trust_region_exhausted` at iteration 23, while a terminal audit showed the filtered predicted-ascent direction was a true descent direction.  
The clean-room success comes from the interaction of a much smaller filter (`1.1–1.5` elements), a genuine Eq. (22) equality-constrained LP, and an unrejected small fixed update that continues through coalescence.  
Controlled clean-room sweeps make filter scale the strongest isolated basin selector, and the old `rmin=2.5` fixed-step endpoint closely matches the clean large-radius failure family.  
Support interpretation is **not** causal for the old SS case: both codes restrain both translations at the two mid-height end nodes and give `68.3986/253.3851/420.7672` on `160×20`, although the old investigation spent disproportionate effort on the mechanically different CC case.  
The clean best endpoint, `170.4709/170.8659/285.1939`, is a convincing low-grayness, near-bimodal benchmark reproduction, but it is not a convergence certificate because all 1600 iterations use the full `0.005` move.  
The Fig. 4 trace likewise remains move-saturated, and its terminal pair is called multiple only under a permissive 5% frequency rule.  
The overarching research-process failure was over-trusting local verification while testing the highest-information global discriminants—filter scale, early three-frequency trajectory, and post-coalescence behavior—too late.

## 2. Old vs new implementation matrix

### 2.1 How to read the old columns

“Old—legacy” means the June/July nested-MMA production/reconstruction path now archived under `Matlab/legacy/`. “Old—rebuilt” means the 30–31 July `topopt_freq_exact.m` / `olhoff2014_case.m` implementation. They must not be collapsed into one algorithm: the second was a substantial replacement, not merely a parameter change.

Difference classifications are:

- **material**: capable of selecting a different topology, basin, or multiplicity path;
- **secondary**: credible effect on pace/noise but weak evidence for changing the final basin here;
- **irrelevant**: controlled or cross-code evidence excludes it as the explanation;
- **unknown**: the paper or recorded artifacts are insufficient.

### 2.2 Forward model and spectral analysis

| Item | Du & Olhoff (2007) | Old—legacy | Old—rebuilt | Clean-room reproduction | Forensic classification |
|---|---|---|---|---|---|
| Element | 4-node plane-stress element | Q4, `2×2` Gauss | Same | Same | **irrelevant**; same initial spectrum at equal mesh |
| Stiffness interpolation | SIMP, normally `p: 1→3` | Fixed `ρ^3` in principal campaigns; continuations tested later | Fixed `ρ^3` | Fixed `ρ^3` | **secondary/unknown**; continuation not needed for clean result and old tests did not rescue reproduction |
| Mass matrix | Consistent mass discussed; three density laws Eq. (4), (4a), (4b) reported as giving negligible final differences | Consistent Q4; principal reconstruction used smooth Eq. (4b), `c1=6e5`, `c2=-5e6` | Same default | Consistent Q4; successful runs use discontinuous Eq. (4): `ρ` above 0.1, `ρ^6` below | **minor contributor**; different but paper says alternatives are close and initial uniform designs are unaffected |
| Constants | `a=8`, `b=1`, `t=1`, `E=10^7`, `ν=0.3`, material density 1 | Same | Same | Same | **irrelevant** |
| Mesh/aspect | Element count not stated; beam `8:1` | Historical diagnostics included `40×5`; faithful campaign `160×20`, `240×30` | Paper cases `160×20` | Primary clean result `240×30`; also `160×20` | **secondary/unknown**; initial `ω1` is converged, but `ω3` and the selected filter's physical scale change |
| SS support location | Figure is visually ambiguous | Mid-height node at each end from first exact commit | Same | `support='mid'` | **irrelevant for old-vs-new SS** |
| SS restrained DOFs | Not stated unambiguously by the drawing | Both `ux,uy` at both mid-height end nodes | Same | `axial='both'`: both `ux,uy` at both nodes | **irrelevant**; identical spectrum is decisive |
| Initial design | Uniform `ρ=0.5` | Same | Same | Same | **irrelevant** |
| Volume | `V*/V=0.5` | `0.5` | `0.5` | `0.5` | **irrelevant** |
| Target | Maximize first eigenfrequency, `n=1` for Fig. 3a/4 | Many historical campaigns targeted CC; SS path still `n=1` | SS `n=1` | SS `n=1` | **material as research scope**, not a code error |
| Eigenpairs | Enough for cluster and `J=n+N`; exact count unstated | Usually 4 | 7 in paper-case record | `Jcalc=n+Nmax=5` | **secondary** |
| Eigensolver | Unstated | MATLAB `eigs` | Deterministic `eigs`, tight tolerance | Deterministic `eigs`, `tol=1e-12`, `maxit=5000`, fixed `v0` | **irrelevant** to failure; deterministic checks and identical initial low modes |
| Eigenvector normalization | Required by generalized eigenproblem; implementation unstated | Mass normalized | Mass normalized | Explicit `ΦᵀMΦ=I` | **irrelevant**; verified in old work |

### 2.3 Multiplicity and sensitivities

| Item | Old—legacy | Old—rebuilt | Clean-room reproduction | Classification |
|---|---|---|---|---|
| Multiplicity rule | Frequency-relative `abs(ωj-ωn)/ωn <= 10^-3`; no hysteresis | Eigenvalue-relative hysteresis: join `2%`, leave `5%` | Frequency-relative `5%`; no hysteresis in successful artifacts | **material contributor**, not sufficient alone; see H6 |
| Cluster value | Cluster average used in generalized-gradient model | Lowest cluster eigenvalue for objective; average eigenvalue in tensor construction as configured | `λ~=mean(λcluster)` in `f_sk` model | **unknown/material**; locally legitimate variants with path consequences |
| `J=n+N` | Yes | Yes | Yes | **irrelevant** |
| Nominal `J` also multiple | No explicit robust expansion in legacy path | Guarded/logged by rebuilt solver | Detected/logged; 9 occurrences in the best `1600`-iteration run and 0 in the `400`-iteration Fig. 4 run | **secondary/unknown**; present but not correlated with the old decisive failures |
| Ordinary sensitivity | `φᵀ(K'-λM')φ` | Same | Same | **irrelevant**; finite differences passed |
| Generalized tensor `f_sk` | Full `N×N` tensor | Full tensor | Full tensor computed | **irrelevant as a formula**; exact degenerate directional test error `1.4e-5` and FD worst about `1.7e-4` in clean work |
| Off-diagonal terms | Included in full-coupling subproblem | Included in full-coupling LMI cuts | Computed, then Eq. (22) enforces their linearized increments to zero and the remaining diagonal model is used | **material operational difference**; dropping them without Eq. (22) gives 11–250% directional error |
| Filtering formula | Sigmund/top88 sensitivity filter | Same formula | Same formula | **irrelevant as implementation** |
| Filtered components | Every `f_sk` in legacy campaigns | Every `f_sk` and `J` sensitivity | Successful route filters diagonal `f_ss` and `J`; off-diagonal Eq. (22) equalities use the unfiltered tensor in the implemented route | **material/unknown**; paper does not specify generalized-tensor filtering semantics |
| Radius, elements | Normally `2.5`; diagnostics also swept larger values. July rebuilt SS final used `1.2` | Final rebuilt SS `1.2`; fixed-step calibration unintentionally/defaulted to `2.5` | Bimodal band `1.1–1.5`; best `1.3` at `240×30` | **material; strongest isolated basin selector** |
| Radius, physical | On `160×20`, `2.5×0.05=0.125`; on `240×30`, `2.5×0.0333=0.0833` | `1.2×0.05=0.060` | Best `1.3×0.0333=0.0433`; `1.2` at `160×20` is `0.060` | **material/unknown**; successful band is closer to fixed element count than fixed physical radius |

### 2.4 Optimization and stopping

| Item | Old—legacy | Old—rebuilt | Clean-room reproduction | Classification |
|---|---|---|---|---|
| Increment problem | Full Eq. (25) nested multiple-eigenvalue problem | Full-coupling semidefinite/LMI model solved by iterative LP cuts | Eq. (22) equality-constrained linear program | **material** |
| MMA versus LP | MMA inner loop; first simple-mode full-box problem independently shown to be an LP | Called `lp`, but not the paper's Eq. (22) LP: it is a cutting-plane realization of the full coupled LMI | One `linprog` solve per outer iteration | **material** |
| Eq. (22) | Not used in production | Not used; full coupling retained | Off-diagonal `f_skᵀΔρ=0` imposed as equalities | **material** |
| Feasible interior at multiplicity | Full coupled MMA model is nonsmooth/ill-conditioned at repeated subeigenvalues, but the archive does not prove an empty feasible set | LMI cutting-plane problem remained feasible and reported satisfied subproblems | Eq. (22) LP has linear equalities and does not need a strict interior; attempting paired inequalities inside MMA produced near-singular systems (`RCOND≈9e-18`) and freezes | **material realization issue**, not proof that full-coupling mathematics is wrong |
| Per-step move | Literal path none; stabilized path `0.2` plus `α=0.5` | Initial `0.05`, adaptive up to `0.2`, down to `10^-4` | Fixed `0.02` for trajectory figure; fixed `0.005` for best endpoint | **material** |
| Accumulated bounds | `ρmin-ρ <= Δρ <= 1-ρ`, plus optional move | Same | Same plus fixed move | **irrelevant as formula** |
| MMA asymptotes | Restarted each outer iteration; persistence tested and made key runs worse; `β` upper bound `10^6` enlarged scale | MMA alternative exists with fresh state per subproblem; primary result uses LP/LMI | Not applicable to successful Eq. (22) LP | **material for legacy**, irrelevant to successful route |
| Inner convergence | `‖Δρ_k-Δρ_{k-1}‖₂ < tolInner√nEl`; default 30 iterations | LP/LMI cut satisfaction/KKT-style checks; no nested MMA in primary SS result | LP is one solve, marked converged | **material for legacy** |
| Inner status consumption | Logged but ignored; production accepted cap-hit steps | Used by subproblem/acceptance logic | LP solver status checked | **root-causal legacy defect** |
| H5 absolute-tolerance no-op | Not the legacy test and not observed as a false convergence | Not applicable to primary LP | Found in an abandoned clean-room MMA stopping rule; fixed with relative-to-accumulated increment test | **not causal as H5 is worded** |
| Outer convergence | RMS design change after a full step; thresholds varied | Small accepted step, KKT/trust exhaustion; SS stopped on trust exhaustion | `max abs(Δρ)<10^-3` | **material reporting difference** |
| Damping | Stabilized legacy path `α=0.5` | No separate damping in primary rebuild | None | **material in legacy trajectory** |
| Acceptance/globalization | Literal path unconditional; later experimental line search/trust variants | Predicted/actual ratio, trial eigensolve, accept/reject, adaptive trust radius | None: feasible clipped LP step always applied | **root-causal in rebuilt SS failure** |
| Clipping/projection | Clamp to `[ρmin,1]`; no production density projection | Same | Same | **irrelevant** |
| Connectivity | Legacy CC campaigns often formed disconnected islands/local modes | Rebuilt SS terminal has one thresholded component but wrong topology/high grayness | Best clean topology is visually paper-like; no formal connectivity certificate saved | **outcome/diagnostic**, not an input cause |
| Terminal selection | Final iterate; “best” snapshots separately diagnostic | Terminal iterate; stop reason explicit | Terminal post-update design returned; comparisons sometimes highlight a selected/best saved run | **material to claims**, not to trajectory |
| Best vs converged | Old reports eventually separated near-paper snapshots from terminal results | Final SS is not called converged; `trust_region_exhausted` | Best endpoint is not converged (`1600/1600`, move saturated) | **material qualification** |

## 3. Root-cause ranking

### 3.1 Ranked causes

| Rank | Finding | Category | Confidence | Estimated impact | Sufficient by itself? |
|---:|---|---|---|---|---|
| 1 | Legacy filter radius `rmin=2.5` selected the non-bimodal basin | **ROOT CAUSE** for the original/fixed-step SS path | High | Changes topology and whether the first pair coalesces | Sufficient to explain that path's failure under the clean LP controls; **not** sufficient to explain the rebuilt `rmin=1.2` failure |
| 2 | Legacy nested MMA returned cap-hit increments and the outer solver ignored the failed stopping status | **ROOT CAUSE** for original solver nonviability | High | Full-box collapse or persistent move-saturated oscillation before a reliable multiple-mode path can form | Sufficient to invalidate the recorded legacy trajectory as a solution of its declared nested procedure; not proof that all full-coupling MMA realizations fail |
| 3 | Rebuilt trust acceptance/globalization froze the SS design at first coalescence | **ROOT CAUSE** for the July rebuilt SS endpoint | High | Prevented any post-`N=2` topology evolution | Sufficient for the observed early termination; whether an unrejected full-LMI path would reach the clean basin is unresolved |
| 4 | Eq. (22) one-shot LP versus full-coupling MMA/LMI consumption of `f_sk` | **MAJOR CONTRIBUTOR** | Medium-high | Removes nested conditioning and changes the admissible local direction at multiplicity | Not isolated as the sole cause in a controlled retrofit of the old solver |
| 5 | Old multiplicity tolerances were much tighter/different | **MAJOR CONTRIBUTOR** for the legacy path; minor for rebuilt endpoint | Medium | Delays entry to the multiple-mode model and changes the branch around a near-pair | No: rebuilt old solver reaches `N=2` and still freezes; clean filter sweep holds tolerance fixed and changes outcome |
| 6 | Mass Eq. (4b) versus Eq. (4) | **MINOR CONTRIBUTOR** | Medium | Can shift low-density modes and late topology | No; paper itself reports negligible final differences and initial spectra agree |
| 7 | Fixed `p=3` versus continuation | **MINOR CONTRIBUTOR / UNRESOLVED historically** | Medium | May affect path regularity | No; clean reproduction succeeds fixed at 3 and old tested continuations did not recover the benchmark |
| 8 | SS support location/DOFs | **NOT CAUSAL** | Very high | None for the compared SS runs | No; the spectra are identical to numerical precision |
| 9 | Q4 FE, constants, mass normalization, ordinary and generalized derivative formulas | **NOT CAUSAL** | Very high | None detectable in this comparison | No; cross-code spectrum and directional tests agree |
| 10 | Exact undocumented 2007 optimizer, filter radius, tensor filtering, and stopping rule | **UNRESOLVED** | High that they are under-specified | Limits historical-fidelity claims | Not answerable from the paper/artifacts |

### 3.2 H1 — Filter radius put the old code in the wrong basin

**Verdict: supported as a root cause of the principal old fixed-step/legacy SS failure, but not a universal explanation of every old version.**

The production and faithful-reconstruction defaults were `rmin=2.5` elements. On `160×20`, where the square element edge is `0.05`, that is a physical radius of `0.125`. The clean-room controlled `160×20` LP sweep gives:

| `rmin` (elements) | Final `ω1/ω2/ω3` | Pair gap | Persistent bimodal result? |
|---:|---:|---:|---|
| 1.1 | `170.13 / 170.76 / 312.3` | 0.37% | Yes |
| 1.2 | `168.24 / 168.60 / 286.01` | 0.22% | Yes |
| 1.5 | `166.49 / 167.36 / 260` | 0.52% | Yes |
| 2.0 | `163.77 / 172.02 / 297.1` | 5.04% | No |
| 2.5 | `159.49 / 167.52 / 318` | 5.04% | No |
| 3.0 | about `156 / 164 / 325` | about 5.3% | No |

At `240×30`, `rmin=1.1` and `1.3` remain bimodal, while `1.5` is already outside a strict near-pair and `2.2` is not bimodal. A separate clean-room move sweep at `rmin=3` with moves `0.02, 0.01, 0.005, 0.002` stays near `ω1≈156`; that rules out “the large-radius failure is merely a step-size artifact.”

The most probative old comparison is its fixed `move=0.02` Fig. 4 calibration, which silently retained the old default `rmin=2.5`. It reaches `154.93/161.81/328.16` at iteration 80, remains `N=1`, and displays the same non-coalesced large-radius family as the clean sweep. This is strong cross-run causal evidence, not merely a difference between two final configurations.

However, the rebuilt old SS run used `rmin=1.2` and still stopped at `148.78/150.26`. Filter radius therefore cannot explain that later failure; its trust/globalization history can.

### 3.3 H2 — We validated the wrong support idealization

**Verdict: not causal for OlhoffExact SS; partially true as a research-focus criticism.**

The exact old support builder has restrained both translations at both mid-height end nodes since commit `5ffeaa3`. The clean-room support sweep identifies exactly that case as the one matching Fig. 4:

| `160×20` support interpretation | `ω1/ω2/ω3` at uniform `ρ=0.5` | Interpretation |
|---|---:|---|
| Mid-height, both translations at both ends | `68.40 / 253.39 / 420.77` | Old exact and clean reproduction; paper-compatible |
| Mid-height, only one axial restraint | `68.40 / 248.0 / 253.4` | Wrong ordering; extensional mode intrudes |
| Corner nodes, one axial restraint | `64.53 / 168.4 / 275` | Wrong problem |
| Corner nodes, both translations | `95.50 / 195.9 / 363.1` | Wrong problem |

The paper's support sketch is ambiguous enough that a visual reading alone can mislead, but the spectrum is not ambiguous. Old rebuilt and clean code both obtain `68.398592/253.385067/420.767152` on `160×20`. H2 is therefore excluded as an old SS code cause.

What did go wrong methodologically is that many June/July audits concentrated on the clamped–clamped target `456.4` and on disconnected CC topologies. Those experiments answered real questions about that case but did not validate the Fig. 4 SS trajectory that would later expose the filter and post-coalescence optimizer differences quickly.

### 3.4 H3 — The nested MMA reconstruction was the wrong operational realization

**Verdict: major contributor; the legacy implementation was operationally unreliable, while full-coupling mathematics itself is not disproved.**

The legacy solver used MMA on variables `[β_hat; Δρ]`, restarted asymptotes at every outer iteration, bounded `β_hat` by `10^6`, and allowed only 30 inner iterations in the recorded production configuration. Its best-instrumented faithful trace is for CC rather than SS, so it is direct evidence about the optimizer failure mechanism and not a direct SS endpoint comparison; that trace says:

- declared stopping threshold: `0.005657`;
- last successive-increment change: `0.40043`;
- inner iterations: `30`;
- `converged=0`;
- returned `||Δρ||∞=0.499994`, with 81.5% of variables near a box bound;
- outer loop accepted the increment unconditionally;
- `ω1: 145.569 → 0.114`.

Across the stabilization-basin campaign, the baseline had `0/120` inner solves meet the declared test and hit the cap `120/120` times. Raising the budget allowed many solves to meet the chosen test but did not rescue the unrestricted first update: an independent exact LP at the initial simple eigenvalue also selected a destructive box vertex. Thus “30 MMA iterations” was not the only problem. The printed full-box incremental model, evaluated with that reconstruction and no additional step restriction, is itself a catastrophic local approximation over the allowed box.

The successful clean route uses the alternative explicitly permitted by Eq. (22): force the off-diagonal linearized couplings to zero, then solve the diagonal linear inequalities plus volume/bounds in one LP. It is about 75 times faster in the recorded comparison and has no inner fixed-point convergence question. When Eq. (22)'s equalities were instead represented as paired inequalities inside MMA, the system became almost singular (`RCOND≈9×10^-18`) and emitted about 97,000 warnings in one diagnostic. That is an optimizer/interface failure, not a sensitivity-formula failure.

Two cautions prevent overclaiming. First, §3.5.3 of the paper says MMA was used; the Eq. (22) LP is a paper-sanctioned route, but its clean-room success does not prove that it reconstructs the authors' undocumented code more literally. Second, the rebuilt old LMI cutting-plane solver did find feasible subproblems and passed frozen-design tests. It is a legitimate alternative full-coupling realization that followed a different path, not simply “bad mathematics.” Its observed failure resulted from the combination of its model, filtering, and globalization.

| Required distinction | Finding |
|---|---|
| Wrong mathematics | **Not established.** Eq. (19)/(25), `f_sk`, and the full-coupling eigenvalue-increment model pass local checks |
| Correct mathematics, wrong numerical realization | **Demonstrated for legacy use.** Cap-hit MMA increments were consumed; unrestricted local models were applied far beyond their predictive range |
| Legitimate alternative, different local optimum/path | **Best description of the rebuilt LMI solver.** Its subproblems are coherent, but the filter/model/trust combination freezes in a different basin |

### 3.5 H4 — Globalization prevented the correct trajectory

**Verdict: directly demonstrated for the rebuilt old SS result.**

The rebuilt solver begins at move `0.05`, expands to `0.1` and `0.2`, and reaches `127.09/229.32/555.53` by iteration 5. It rejects the trial at iteration 6, accepts a reduced step at 7, rejects 8–9, and continues by trust contractions. At iteration 16 it reaches `148.784/150.258/425.371` and declares `N=2`. Iterations 16–23 accept no density change at all; the spectrum and grayness remain identical until `trust_region_exhausted`.

The clean `move=0.02` trace reaches its first `N=2` classification at iteration 26 (the compact table below shows the near-pair at iteration 27) and then continues hundreds of post-coalescence updates. Its low pair rises from roughly `148.0/150.8` around coalescence to roughly `170/175`, while the topology sharpens. This is exactly the phase the rebuilt solver suppresses.

The old terminal-direction audit makes the mechanism more specific. At its minimum trust radius `10^-4`, the filtered local model predicted `Δλ1=+1.68`, while the raw generalized gradient predicted `-6.27`; finite differences gave slope `-0.492` and the actual trial `-0.518`. The acceptance layer was therefore responding correctly to the true decrease, but the combined filtered-model/trust algorithm had no accepted escape direction. “Stabilization” did not merely slow the right path; it made the current local model and acceptance objective mutually incompatible and terminated at the first cluster.

The converse caution is equally important: removing acceptance does not produce a convergence certificate. It lets the clean solver traverse the desired benchmark basin, but both highlighted clean runs remain move-saturated at their iteration limits.

### 3.6 H5 — Inner convergence logic caused silent ineffective steps

**Verdict: the stated hypothesis is not historical; a related and more serious failure is demonstrated.**

The clean-room work found that a naïve absolute successive-step tolerance can call a small-move MMA update converged before the accumulated increment is useful. That defect was repaired by normalizing the last step by the accumulated increment, and its MMA diagnostics then needed 89–120 inner iterations.

The legacy OlhoffExact criterion was instead an absolute Euclidean test scaled by `√nEl`. The evidence does **not** show it declaring convergence early at a materially unresolved accumulated increment. It shows the opposite: the flag remained false, the 30-iteration cap was hit, and the outer loop ignored the warning. H5 is therefore **NOT CAUSAL as worded**. “Uncertified cap-hit increments were knowingly consumed” is a root cause of the legacy trajectory.

The rebuilt primary SS solver and the successful clean Eq. (22) solver are LP-based and have no repeated MMA stopping logic, so this hypothesis cannot explain their difference.

### 3.7 H6 — Near-multiplicity interpretation was wrong

**Verdict: contributory and claim-sensitive, but not the dominant isolated cause.**

The three thresholds are not comparable without converting variables:

- legacy: `0.1%` relative frequency, no hysteresis;
- rebuilt: join at `2%` relative eigenvalue (about `1%` frequency for a small gap), leave at `5%` eigenvalue (about `2.5%` frequency);
- clean saved runs: `5%` relative frequency, no hysteresis.

The legacy rule can engage the generalized multiple-mode model much later than the clean rule. This plausibly compounds its inability to retain a near-pair. It does not explain the clean filter sweep, because that sweep holds `tolMult=0.05` fixed while the filter changes the outcome. Nor does it explain the rebuilt freeze, because that solver actually reaches `N=2`.

The threshold also affects the strength of the clean claim. The `FIG4_definitive` terminal pair is `170.7449/175.0656`, a 2.53% frequency gap: multiple under 5%, but not under either the paper's unspecified “very small tolerance” or the legacy 0.1% rule. The best `rmin=1.3`, `move=0.005` endpoint is `170.4709/170.8659`, a 0.232% gap and therefore a much stronger bimodality result, though still not legacy-`N=2` under the strict 0.1% threshold. The report consequently uses “near-bimodal” for numerical proximity and reserves `N=2` for a stated algorithmic tolerance.

### 3.8 H7 — We over-trusted local verification

**Verdict: strongly supported as the central research-process failure.**

The old tests established all of the following, legitimately:

- Q4 assembly and initial eigenvalues were internally consistent;
- mass normalization was correct;
- ordinary derivatives agreed with finite differences at simple modes;
- `f_sk` was symmetric and basis-invariant at a repeated eigenspace;
- the multiple-eigenvalue directional prediction was locally correct;
- the rebuilt LMI/LP subproblem satisfied its frozen-design constraints;
- eigensolver results were deterministic after pinning `v0`.

None of those propositions implies that:

- `rmin=2.5` represents the numerical benchmark used for Fig. 3a;
- a full-box local linearization remains predictive at its optimizer's returned vertex;
- an inner MMA cap-hit iterate is an adequate solution of the intended subproblem;
- a full-coupling LMI route and an Eq. (22) diagonal LP enter the same basin;
- filtering the model tensor preserves ascent for the unfiltered trial eigenproblem;
- a trust acceptance rule will allow the post-coalescence topology transition;
- a frequency tolerance has the same semantics as an eigenvalue tolerance;
- a near-paper snapshot is a retained or converged solution.

Local derivative verification answers “is this tangent correct here?” Benchmark reproduction asks “did the complete, under-specified discrete dynamical system follow the published trajectory and reach the published structure?” The latter includes parameter identification and globalization. The old program repeatedly treated success on the first question as increasing confidence in the second, then searched for ever more sophisticated defects downstream instead of reopening the filter scale and operational update at the primary Fig. 4 gate.

### 3.9 Numerical comparison and direct histories

#### 3.9.1 Initial spectrum: the strongest support/FE control

| Source/configuration | Mesh | `ω1` | `ω2` | `ω3` | Finding |
|---|---:|---:|---:|---:|---|
| Paper Fig. 4 first marker (read from plot) | Unstated | about 71 | about 245 | about 428 | Reference is graphical, not exact tabular data |
| Old rebuilt SS | `160×20` | 68.3986 | 253.3851 | 420.7672 | Exact match to clean at same mesh |
| Clean SS | `160×20` | 68.3986 | 253.3851 | 420.7672 | Same FE/support problem |
| Clean SS | `240×30` | 68.3209 | 252.5916 | 406.6631 | `ω1` stable; higher modes retain mesh sensitivity |

Clean mesh checks give `ω1=68.75` at `64×8`, `68.62` at `80×10`, `68.40` at `160×20`, and `68.32` at `240×30`. This evidence excludes a gross FE scaling or support error and cautions against demanding exact agreement with a rasterized Fig. 4 `ω3` marker on an unknown mesh.

#### 3.9.2 Comparable first 80 iterations

The table compares two fixed-`0.02` routes because this is the least confounded old/new trajectory comparison. The old route uses its default `rmin=2.5`; the clean route uses `rmin=1.3` on `240×30`. Values are pre-update spectra saved at each iteration.

| Iter. | Old fixed `0.02`, `160×20`, `rmin=2.5`: `ω1/ω2/ω3`, `N` | Clean fixed `0.02`, `240×30`, `rmin=1.3`: `ω1/ω2/ω3`, `N` |
|---:|---:|---:|
| 1 | `68.40 / 253.39 / 420.77`, 1 | `68.32 / 252.59 / 406.66`, 1 |
| 5 | `80.63 / 286.32 / 432.52`, 1 | `80.61 / 286.37 / 415.80`, 1 |
| 10 | `95.34 / 311.39 / 482.24`, 1 | `95.45 / 311.55 / 461.74`, 1 |
| 15 | `109.28 / 282.59 / 494.21`, 1 | `109.47 / 282.10 / 494.11`, 1 |
| 20 | `123.61 / 191.44 / 435.23`, 1 | `123.99 / 193.16 / 443.16`, 1 |
| 25 | `142.41 / 153.40 / 314.71`, 1 | `143.07 / 150.45 / 324.42`, 1 |
| 27 | `146.63 / 154.95 / 317.84`, 1 | `148.01 / 150.81 / 321.63`, 2 |
| 30 | `148.79 / 154.53 / 320.72`, 1 | `151.56 / 152.16 / 326.78`, 2 |
| 40 | `152.16 / 156.80 / 326.55`, 1 | `160.43 / 160.73 / 318.08`, 2 |
| 60 | `155.11 / 161.13 / 328.60`, 1 | `169.24 / 170.87 / 307.47`, 2 |
| 80 | `154.93 / 161.81 / 328.16`, 1 | `170.28 / 175.32 / 305.44`, 2 |

The near identity through roughly iteration 20 is important. FE, support, and the early simple-mode LP direction are not where the paths separate. The separation occurs at the approach to coalescence, exactly where filter scale changes the spatial direction and the multiple-mode optimizer becomes active.

Paper Fig. 4 appears to coalesce near iteration 20, with higher-mode peaks around iterations 7–9. Clean move studies show the same three-phase structure but not an exact timing match: `move=0.05, 0.03, 0.02, 0.01` coalesce at about iterations `13, 18, 27, 54`; their early `ω2` peaks occur about `5, 7, 10, 20`. Move `0.03` best matches the paper's timing but later disconnects/collapses; `0.02` gives the smoother retained trajectory. No single clean run perfectly matches both the paper timing and endpoint.

#### 3.9.3 Adaptive old run: the globalization freeze

| Iter. | `ω1/ω2/ω3` | `N` | Accepted `‖Δρ‖∞` | What happened |
|---:|---:|---:|---:|---|
| 1 | `68.40 / 253.39 / 420.77` | 1 | 0.05 | Accepted |
| 2 | `76.07 / 274.74 / 424.05` | 1 | 0.10 | Trust expanded |
| 3 | `90.18 / 301.96 / 427.32` | 1 | 0.20 | Trust expanded |
| 5 | `127.09 / 229.32 / 555.53` | 1 | 0.20 | Large excursion |
| 6 | `133.95 / 139.00 / 349.33` | 1 | 0 | Trial rejected |
| 8–9 | `143.23 / 160.63 / 439.51` | 1 | 0 | Trials rejected |
| 10 | same | 1 | 0.025 | Accepted after contraction |
| 15 | `147.75 / 153.12 / 431.07` | 1 | 0.025 | Approaches cluster |
| 16 | `148.78 / 150.26 / 425.37` | 2 | 0 | First `N=2`, trial rejected |
| 17–23 | unchanged | 2 | 0 | All trials rejected; trust exhausted |

Volume remains `0.5`. Gray fraction (`0.1<ρ<0.9`) is `0.5891` at termination, versus `0.3863` for the old fixed-step run at iteration 80 and `0.1201` for the clean best endpoint. The old rebuilt topology has one thresholded component, so its failure is not the disconnected-island pathology that dominated the earlier CC campaigns; it is a premature, gray, pre-paper topology.

#### 3.9.4 Final artifacts and adverse qualifications

| Artifact | Config | Final/terminal spectrum | Volume | Grayness | Multiplicity/stop |
|---|---|---:|---:|---:|---|
| Paper Fig. 3a | Unstated mesh/filter/update | Reported maximum `ω1=174.7`; bimodal topology | 0.5 | Not reported | Bimodal |
| Old rebuilt SS | `160×20`, `rmin=1.2`, full LMI LP, adaptive trust | `148.7843 / 150.2577 / 425.3715` | 0.5 | 0.5891 | `N=2`; `trust_region_exhausted`, 23 iters |
| Old fixed-step calibration | `160×20`, `rmin=2.5`, move 0.02 | At iter. 80: `154.9319 / 161.8077 / 328.1596` | 0.5 | 0.3863 | `N=1`; move still saturated |
| Clean Fig. 4 trace | `240×30`, `rmin=1.3`, move 0.02, 400 iters | Post-terminal update `170.7449 / 175.0656 / 301.8717` | 0.5 | 0.1250 | `N=2` only under 5% rule; move still 0.02 |
| Clean best endpoint | `240×30`, `rmin=1.3`, move 0.005, 1600 iters | `170.4709 / 170.8659 / 285.1939` | 0.5 | 0.1201 | Persistent near-pair; move still 0.005 |

The clean best spectrum is 2.42% below the paper's reported `174.7`; the first-pair gap is 0.232%. Its history first enters the 5% `N=2` class at iteration 95 and remains there for 1506 recorded iterations. It logs 9 instances in which the nominal `J` mode is itself multiple. The pre-update spectrum at iteration 1600 is `170.2599/170.7591/284.5264`; the returned post-update terminal design is the row reported above. The difference is expected from the save convention, not evidence of data corruption.

Topology evidence:

- Paper reference: `docs/figs/paper_fig3a.png`.
- Old rebuilt terminal: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact/experiments/paper_examples/ss_n1/topology.png`.
- Clean side-by-side comparison: `results/BEST_240x30_rmin1.3_vs_paper.png`.
- Clean Fig. 4 history comparison: `results/FIG4_definitive_vs_paper_80.png`.

The clean topology is visibly far closer to Fig. 3a and much less gray. It should be called a successful reproduction in the benchmark sense, not a proven converged optimum.

#### 3.9.5 Filter output and acceptance telemetry limits

No common scalar “filter output” was archived by both implementations, so a numerical overlay would be invented evidence. What is available is stronger but narrower:

- the formulas are code-equivalent top88 sensitivity filters;
- the controlled radius sweep isolates the parameter's outcome;
- the old terminal directional audit compares filtered-model prediction, raw-gradient prediction, finite difference, and actual trial and shows a sign conflict;
- old adaptive history records zero accepted steps after coalescence, whereas clean history has no rejection mechanism and records a saturated move every iteration.

The directly comparable control telemetry is:

| Route | Inner work/status | Applied step factor | Move behavior | Terminal implication |
|---|---|---:|---|---|
| Legacy literal MMA | 30 MMA iterations; first trace `converged=0`; stabilization baseline `0/120` met declared test | 1.0 | Full density box | Destructive first update |
| Legacy stabilized MMA | Usually 30 cap-hit iterations | 0.5 fixed damping; no rejection on default path | `0.2` inner/outer cap | Period-2-like/move-saturated histories |
| Rebuilt old LMI/trust | One LP when `N=1`; 4 LP rounds/3 added cuts at terminal `N=2` | 1 for accepted trial, 0 for rejected trial | Expanded `0.05→0.1→0.2`, then contracted to `10^-4` | No accepted change after first `N=2` state |
| Clean Eq. (22) LP | One LP solve; status checked | 1.0 always | Fixed `0.02` or `0.005`, saturated throughout highlighted runs | Traverses the paper-like basin but does not meet outer convergence |

## 4. Why the previous audits did not catch the decisive causes

### 4.1 Historical failure timeline

Dates are repository dates. Some July production source was later deleted or replaced; the timeline distinguishes source provenance from surviving run artifacts.

| Phase | Symptom | Working hypothesis and experiment | What it established | What it did not establish / overreach |
|---|---|---|---|---|
| 10–12 Jun: initial exact solver (`5ffeaa3`–`25d7901`) | Correct local tests, wrong/oscillatory endpoints | Implement full tensor and nested MMA; verify initial modes, multiplicity, filter, inner/outer paths; test MMA persistence | Core FE/sensitivity mechanics were plausible; MMA state handling materially changes steps | Did not identify paper's filter radius or validate Fig. 4 trajectory. “More faithful MMA state” was treated as progress without a global benchmark gate |
| 18–30 Jun: broad audit/stabilization (`b98cc96`, `47737e7`) | CC design enters near-paper frequency then collapses or disconnects | Sweep filters, mode tracking, damping, move limits, acceptance ideas, continuation, symmetry, connectivity diagnostics | Large local steps can destroy a coalesced basin; disconnected modes explain spectacular high CC frequencies | Mostly studied CC at coarse or different settings. Near-paper transient frequency was sometimes given more significance than retained topology/convergence |
| 1–9 Jul: basin retention/globalization (`6f2e04c`–`6275215`) | Coalescence lasts only a few iterations | Persistent MMA, 30 vs 300 inner iterations, line search/trust variants, low-mode guards | Baseline met inner test `0/120`; persistence did not help; accepted large steps caused basin exits; no variant met strict retained paper guard | Did not show the paper was unreproducible. It showed this full-coupling, `rmin=2.5`, mainly CC realization was unstable. Globalization was layered onto a possibly wrong basin/model |
| 8–9 Jul: disconnected/local-mode and “missing regularization” audits | High coalesced frequencies but nonstructural topologies | Test density filters, projection, larger radii, p-continuation, symmetry, passive support paths, lumped mass, alternate supports | High CC optima were disconnected/localized; the tested extras did not produce a valid CC reproduction | Negative results for these variants did not rank the untested small-radius SS Eq. (22) route. Larger filters reinforced rather than challenged the key bad scale |
| 13–29 Jul: faithful full-box reconstruction (`8938332`–`310043e`) | Literal Eq. (25) collapses; bounded variants oscillate | Independent exact LP and converged MMA at initial state; 19 variants, two meshes, gates and independent reviews | Unrestricted initial LP vertex collapses; 30-iteration inner status was not met; no tested full-box trajectory converged or retained the cluster | The corrected report appropriately limits its verdict to the tested reconstruction. It could not infer undocumented 2007 controls or exclude the Eq. (22) LP route |
| 30 Jul: rebuilt Olhoff2014 solver (`79c533c`, `6f86f7e`) | Mathematical tests pass; SS ends around 149 rather than 174.7 | Replace legacy path with full-coupling LMI cutting-plane LP, hysteretic clustering, deterministic eigensolve, adaptive trust; calibrate cases | Frozen-design subproblem and derivatives are strong; SS uses correct supports and later `rmin=1.2`; reaches an `N=2` connected state | “LP” could be mistaken for the Eq. (22) LP, but it is a different full-coupling algorithm. Final stop is trust exhaustion, not convergence |
| 31 Jul: terminal-direction audit (`cf290fc`) | Trust radius reaches floor with no progress | Compare filtered and raw directions to finite difference and true trial | At the SS terminal state, filtered predicted ascent is true descent; acceptance has no viable step under that model | Does not prove all sensitivity filtering is invalid. It proves this filter/model/globalization combination is self-blocking there |
| Clean-room effort | Need a positive Fig. 3a/Fig. 4 reproduction | Start from paper-figure spectrum; sweep supports, then small filter radii, optimizer routes, moves; use Eq. (22) LP | Correct support identified by spectrum; small filter is a sharp basin selector; Eq. (22) LP and unrejected small steps traverse coalescence to a paper-like topology | Does not reveal the original authors' exact hidden parameters or prove asymptotic convergence |

The recurring process error was repair momentum: once the full-coupling `rmin=2.5` implementation existed, each new failure invited another safeguard or diagnostic inside that frame. The clean-room effort instead treated the paper's plotted trajectory as a system-identification target and varied early modeling/algorithm choices before adding safeguards.

### 4.2 They optimized evidence strength locally, not information gain globally

Finite-difference audits, basis invariance, KKT checks, deterministic eigensolves, and MAC diagnostics are high-quality tests. After they passed, repeating or elaborating them had diminishing power to distinguish the remaining hypotheses. A single early SS table of `(ω1,ω2,ω3)` at iterations `1,5,10,20,30,60` across `rmin={1.2,2.5}` and `{full coupling, Eq.22 LP}` would have been more discriminating.

### 4.3 The investigation conflated three objects

At different moments, “OlhoffExact” meant:

1. the printed Eq. (25) mathematical program;
2. a particular nested-MMA reconstruction with 18 or more undocumented choices;
3. the historical 2007 code.

The later faithful report corrected this explicitly after independent review. Earlier causal language sometimes allowed a negative result for object 2 to sound like a verdict on objects 1 or 3. The clean result confirms why that distinction matters: Eq. (22), also in the paper, produces a materially different operational program.

### 4.4 The wrong benchmark case dominated the expensive diagnostics

The CC case was attractive because its target `456.4` and disconnected high modes made failures dramatic. But Fig. 4 for SS contains far more trajectory information: three frequency curves, coalescence timing, and the mode interaction. Optimizing the SS case first would have excluded support and FE errors immediately and exposed the correct early trajectory through iteration 20 before the old and clean filters diverged.

### 4.5 The audits separated correctness from parameter identification too late

The sensitivity filter was repeatedly verified as a formula. That does not validate `rmin=2.5`. The paper states that a sensitivity filter was used but does not disclose the radius. The old program often treated “filter verified” as if the filtering choice were validated; the clean sweep shows that radius is a bifurcation parameter.

### 4.6 Stabilization obscured rather than resolved model disagreement

Acceptance and trust logic prevented catastrophic steps, but a rejected step can mean either “the local model needs a smaller radius” or “the modeled direction is incompatible with the true objective.” The terminal audit found the latter. Continuing to contract the radius could only reach the stop floor, so apparent safety became a silent trajectory veto.

### 4.7 Near-paper snapshots were not global validation

Several CC runs briefly entered a frequency/coalescence guard and then collapsed or migrated to disconnected structures. Those snapshots proved reachability of a numerical neighborhood, not retention, connectivity, convergence, or correctness of the basin. The later reports became appropriately fail-closed, but substantial effort had already accumulated around preserving an off-target reconstruction.

## 5. Minimum counterfactual fix

### 5.1 Smallest likely changes to the original legacy solver

Ranked by expected importance:

1. **Set the sensitivity-filter radius near `1.2–1.3` elements, not `2.5`.** This is the only candidate with a controlled, monotone basin sweep and a direct old large-radius trajectory analogue.
2. **Replace the nested MMA consumption of the multiple-mode model with the Eq. (22) equality-constrained LP.** This removes the ignored inner-convergence failure, asymptote/`β` scaling, and ill-conditioned multiplicity interface in one change while remaining within the paper's formulation.
3. **Use a small unrejected fixed move through the Fig. 4 transition:** approximately `0.02` to match the trajectory shape, or `0.005` to refine the endpoint. Do not use the old `0.2`/damped step or ratio-based trust rejection for this reproduction gate.
4. **Use a declared frequency-relative clustering tolerance large enough to activate the multiple-mode branch during approach**, with `5%` as the empirically successful clean value, then report the actual eigengap separately so the label cannot substitute for numerical bimodality.

Items 1–3 are the probable minimum interacting set. The evidence does not justify claiming that only the filter change would have rescued the legacy nested-MMA implementation, because its cap-hit/full-box behavior is independently destructive.

### 5.2 Smallest likely changes to the rebuilt July solver

The rebuilt solver already has the correct SS FE model and `rmin=1.2`. Its minimum likely retrofit is therefore narrower:

1. switch its full-coupling LMI cutting-plane subproblem to the Eq. (22) equality LP;
2. replace ratio-based adaptive trust acceptance with the clean fixed small move for the reproduction experiment;
3. continue past the first `N=2` state rather than treating trust exhaustion as a terminal design.

This is a counterfactual supported by cross-run evidence, not a completed controlled retrofit. It must not be described as proven until the exact old model is replayed with only those changes.

### 5.3 Fixes probably unnecessary for reproducing Fig. 3a/Fig. 4

The following may be useful in general topology optimization, but the evidence says they were not prerequisites for this benchmark reproduction:

- persistent MMA asymptote state;
- complex line searches, basin guards, or trust-ratio heuristics;
- density filtering, Heaviside projection, or projection continuation;
- forced symmetry;
- passive support paths or connectivity constraints;
- mass lumping;
- corner-support alternatives;
- elaborate MAC tracking for the first few modes;
- `p=1→2→3` continuation;
- “best-seen” snapshot selection;
- repeated ordinary-sensitivity checks after the cross-code derivative tests already passed.

Connectivity and mode-localization audits were scientifically useful for rejecting false CC successes. They were unnecessary as modifications to produce the clean SS result.

## 6. What the clean-room reproduction proves

### 6.1 FE-model correctness

It strongly corroborates the old SS forward model. Equal mesh, constants, Q4 integration, consistent mass, density, and supports give the same first three frequencies. Support variants give dramatically different spectra, making the agreement discriminating rather than accidental.

### 6.2 Benchmark reproduction

It reproduces the defining qualitative facts of Fig. 3a/Fig. 4:

- correct initial frequency scale and modal ordering;
- smooth early rise and intermediate higher-mode peak;
- approach and retention of a low-frequency pair;
- a final first frequency within about 2.4% of `174.7`;
- a visibly paper-like, low-grayness final topology.

It also shows that no single saved run is perfect: move settings that best match coalescence timing can later fail, and the run with the best endpoint does not match every Fig. 4 transient.

### 6.3 Optimizer reconstruction

It proves that a paper-sanctioned Eq. (22) LP realization is sufficient to reproduce the benchmark family under the identified filter/move settings. It proves that repeated MMA inner iterations are unnecessary for that route. It does **not** prove that the historical authors used Eq. (22), because the paper says MMA was used and omits procedural detail.

### 6.4 Convergence and optimality

It does not prove either. The best run never satisfies `tolOuter=10^-3`; its applied move remains `0.005`. The Fig. 4 trace also remains at its `0.02` cap. Their endpoints are samples from bounded small-step dynamics, albeit stable-looking and benchmark-consistent ones. Claims should be “successful reproduction of the published benchmark behavior/topology,” not “recovered the exact converged 2007 optimum.”

## 7. Remaining uncertainties

1. **Original filter scale.** The paper specifies a sensitivity filter but no radius or whether radius was fixed in element or physical units. Clean results favor `1.1–1.5` elements, and a fixed physical radius of `0.06` gives `rmin=1.8` at `240×30` and a worse endpoint, but this does not identify the authors' value.
2. **Original mesh.** The paper does not state the Fig. 3a mesh. `ω1` is insensitive by `160×20`, while `ω3`, topology thickness, and physical filter scale are not.
3. **Original step/globalization control.** Fig. 4's smooth history is incompatible with the tested unrestricted full-box update, but the paper gives neither a move limit nor an acceptance rule. The successful fixed moves are inferred numerical reconstructions.
4. **LP versus full-coupling MMA.** Eq. (22) permits LP, but §3.5.3 reports MMA. The successful route may be equivalent to an undocumented authors' simplification, or merely a different route to the same published basin.
5. **Generalized-gradient filtering.** The paper does not say whether every off-diagonal `f_sk`, only diagonal terms, the physical sensitivities before tensor assembly, or some other representation was filtered. Old and new differ materially here.
6. **Multiplicity threshold.** “Very small tolerance” is not quantified. The clean 5% frequency threshold is useful operationally but too permissive to treat its `N=2` flag as self-proving. The actual reported gap is the defensible observable.
7. **Cluster representative.** Average versus lowest cluster eigenvalue and hysteresis can change the local model near coalescence; no complete controlled factorial experiment isolates these choices.
8. **Mass interpolation late in the run.** Eq. (4) versus Eq. (4b) is probably secondary, not rigorously isolated at otherwise identical successful settings.
9. **Asymptotic behavior.** The clean fixed-move trajectories may approach a small cycle rather than a fixed point. A diminishing-step or convergent globalization scheme that preserves the reproduced basin remains to be demonstrated.
10. **Terminal topology connectivity metric.** The clean topology is visually structural and paper-like, but the saved report does not include the formal support-connected modal-energy audit applied to old CC results.

## 8. Lessons for future reproduction work

1. **Turn published trajectories into acceptance gates before optimizing code.** Record the initial spectrum, mode ordering, key peak times, coalescence iteration, and terminal topology. A final scalar objective is too weak.
2. **Separate model verification, parameter identification, and optimizer identification.** Passing one layer must not increase the evidential grade of an untested layer.
3. **Treat every omitted numerical parameter as a hypothesis.** Filter radius, tolerance units, move limits, mesh, and stopping rules are part of the reproduced numerical problem, not implementation trivia.
4. **Use controlled one-factor tests at the earliest discriminating state.** The clean radius sweep is more causal than comparing two large codebases. At multiplicity, compare directions at the same `ρ`, not only final runs.
5. **Preserve distinct solver generations in reports.** “Old OlhoffExact” hid a legacy MMA solver and a later LMI/trust solver with different failure mechanisms.
6. **Never ignore solver status.** If an inner stopping test is declared, record and consume it. If the test is not an optimality certificate, say exactly what it certifies and no more.
7. **Distinguish stabilization from fidelity.** Acceptance logic can prevent collapse while also blocking the published transition. Audit both predicted and true directional signs before contracting indefinitely.
8. **Report actual eigengaps alongside multiplicity labels.** A Boolean `N=2` depends on tolerance and cannot substitute for the spectrum.
9. **Do not equate a near-paper snapshot with reproduction.** Require retained topology, volume, modal character, and a transparent stop condition.
10. **Use clean-room implementations when repair momentum dominates.** A second implementation is valuable not because it is automatically correct, but because it breaks correlated assumptions and makes parameter/algorithm differences visible.
11. **State adverse qualifications in the headline result.** Here the clean endpoint is excellent evidence of the right basin but remains a move-saturated finite trajectory.

## 9. Evidence ledger

### 9.1 Paper and erratum

- `docs/Du and Olhoff - 2007 - Topological design of freely vibrating continuum s.pdf`
- `docs/Du and Olhoff - Topological design of freely vibrating continuum s.pdf` (erratum inserting the missing `Δ` in Eqs. 25d, 26f, 26g)
- `docs/Olhoff and Du - 2014 - Structural Topology Optimization with Respect to E.pdf`
- `docs/figs/paper_fig3a.png`, `docs/figs/paper_fig4_hist.png`

### 9.2 Clean-room executable evidence

- Configuration and outer algorithm: `algo/defaultCfg.m`, `algo/olhoffOpt.m`
- Generalized gradients and increment model: `algo/genGrad.m`, `algo/deltaLambda.m`
- Full-coupling MMA and Eq. (22) LP: `algo/innerLoop.m`, `algo/innerLoopLP.m`
- FE and eigenproblem: `fem/model2D.m`, `fem/elemMats2D.m`, `fem/assemble2D.m`, `fem/eigSolve.m`, `fem/massScale.m`
- Filter: `filter/prepFilter.m`, `filter/applyFilter.m`, `filter/top88_reference.m`
- Investigation narrative and recorded sweep values: `NOTES.md`
- Primary saved results: `results/lp240_rmin1.3.mat`, `results/FIG4_definitive.mat`
- Visual comparisons: `results/BEST_240x30_rmin1.3_vs_paper.png`, `results/FIG4_definitive_vs_paper_80.png`
- Sweep artifacts: `results/lp240_rmin*.mat`, `results/lprmin*.mat`, `results/fig4_mv*.mat`, and corresponding logs/PNGs

### 9.3 Old executable and recorded evidence

Old root: `/Users/piotrek/Programming/topOpt4freqMax/analysis/OlhoffApproachExact`

- Legacy implementation: `Matlab/legacy/topopt_freq_exact_persist_mma_experiment.m`, `Matlab/legacy/inner_loop_mma.m`, `Matlab/legacy/audit_optimizer_nochange.m`
- Rebuilt implementation: `Matlab/topopt_freq_exact.m`, `Matlab/olhoff2014_case.m`, `Matlab/subproblem_lp.m`, `Matlab/subproblem_mma.m`
- FE/support/filter: `Matlab/assemble_KM_exact.m`, `Matlab/fe_q4_exact.m`, `Matlab/build_supports_exact.m`, `Matlab/generalized_gradients.m`, `Matlab/apply_sensitivity_filter.m`
- Rebuilt SS result: `experiments/paper_examples/ss_n1/{summary.txt,history.csv,result.mat,topology.png}`
- Old fixed-move trace: `experiments/step_calibration/results/hist_lp_m0p02.csv`
- Terminal direction: `experiments/terminal_direction_audit/results/REPORT.md`
- Current synthesis: `REPORT.md`, `PLAN_Olhoff2014_exact.md`
- Historical reports recoverable at commits `6275215` and `310043e`: basin retention, globalization, persistent MMA, disconnected/local-mode, missing-regularization, faithful-reconstruction, and independent-review reports
- Git chronology: `5ffeaa3`, `25d7901`, `b98cc96`, `47737e7`, `6f2e04c`, `ea99dd9`, `6275215`, `8938332`, `310043e`, `79c533c`, `6f86f7e`, `cf290fc`

## 10. Bottom line

The old project mostly solved the equations it chose, but it chose—and later stabilized—a different numerical dynamical system from the one that reproduces the published SS benchmark. The original system combined the wrong filter scale with an uncertified nested-MMA update; the rebuilt system corrected the filter and local mathematics but stopped itself at coalescence. The clean-room system succeeds because it changes the basin selector, the operational multiple-mode subproblem, and the permission to continue through the transition. That is not a single coding bug, and it is not evidence that the old mathematical work was wasted. It is evidence that local correctness and historical numerical fidelity are separate scientific claims, and only the second one can be validated by the paper's global trajectory and topology.
