# Constrained KKT stationarity

GRAY_REGIONS_NOT_KKT_STATIONARY

Scope: the retained designs do not satisfy first-order stationarity of the evaluated relaxed FE eigenvalue problem. The filtered algorithm is separately audited through its exact frozen local subproblem. It is not renamed as a different undisclosed physical objective. This is not a theorem that no nearby gray optimum exists, and does not diagnose a defect in the three-rung controller or in published MMA.

Let L=-lambda1/lambda_ref + mu*(sum rho-0.5NE)/(0.5NE), with mu>=0 and lambda_ref fixed. The reduced gradient is r_e=-g_e/lambda_ref+mu/(0.5NE). At a lower bound require r>=0; at an upper bound r<=0; in the interior r=0. The projected first-order sign residual retains r on interior elements, min(r,0) on lower bounds, max(r,0) on upper bounds. Preregistered bound tolerance=1e-7. All saved values are farther than that from either bound, so there are no bound elements under that convention. Solid/void classes are **not** synonymous with active box bounds. Bound-sign arrays are empty, reported as such rather than zero-valued evidence of a pass.

No exact MMA dual is retained. With v_e=g_e/lambda_ref and fit set I, the least-squares nonnegative dual is mu=max(0,(0.5NE)*mean_I(v)). The primary fit uses all interior elements; the gray-only fit minimizes gray residual over **every possible nonnegative volume multiplier**. Its residual is therefore a decisive optimistic test for gray stationarity, not an arbitrary multiplier choice. The volume slack is small and negative; treating volume as active is generous. Strict complementarity at negative slack would require mu=0 and cannot improve on this best nonnegative fit.

Normalize by s=sqrt(mean_interior((g_raw/lambda_ref)^2)). s is 2.09159e-4,1.48909e-4,1.26285e-4. Gray is bounded well away from both box limits, so changing near-bound classification cannot make those elements active-bound variables.

| mesh | gray RMS, all-interior dual | gray RMS, best gray dual | gray p95, best gray dual | broad RMS, best gray dual | filtered gray RMS, best gray dual |
| --- | --- | --- | --- | --- | --- |
| 400 | 0.555345 | 0.354769 | 0.781865 | empty | 0.0222383 |
| 480 | 0.444567 | 0.334039 | 0.754716 | 0.171565 | 0.0489931 |
| 800 | 1.16312 | 1.16254 | 1.69269 | 0.311465 | 1.15835 |

The best gray-only raw RMS values 0.355,0.334,1.163 are above the preregistered 0.1 diagnostic bar. This bar is an audit scale convention, not a theorem or a standard MMA tolerance. The distributions, signs and finite-difference errors provide the stronger evidence. Broad-core raw RMS is .1716 at 480 and .3115 at 800 under the same gray-fit dual. Only 20.1% and 2.46% of those core elements fall below |r|/s=.1; isolated small values do not certify a stationary region or a common dual. Accordingly the primary category is nonstationary, not mixed on the strength of isolated small residuals.

Fit details and max/median/RMS/p90/p95/p99 for gray, mid, solid, void and broad regions, primary/gray-fit multipliers, volume complementarity, and global projected sign residuals are retained in stationarity.json. Figures F10–F12 and F16 show the distribution and mesh comparison. Bound-tolerance robustness at 1e-5,1e-4,1e-3 leaves raw gray RMS materially nonzero; the best gray-only fit is independent of those choices.

## What the filtered residual does and does not establish

Use gFiltered instead of gRaw in the same algebra to test **zero increment** stationarity of the exact frozen filtered subproblem. Its all-interior dual fit is distorted by near-void filtered values: it hits mu=0 at 400/480, with gray residuals .322/.343 on the shared raw scale. Its best gray-only fit gives .0222/.0490. That is evidence of local flattening in gray regions, **not a global filtered KKT pass**. Normalizing by the filtered RMS instead would misleadingly shrink some values because near-void filtered gradients are large; both scales are retained and the shared raw scale is used for comparisons.

800 remains nonstationary under the exact simple lowest branch. However, a nearby eigenvalue crossing can be material: its exact relative lambda gap is about 6.98e-5. The exploratory robustness fit uses Q=[[a,b],[b,1-a]] and a free constant threshold to minimize gray residual of sum_sk Q_sk F_sk. This is a linear least-squares fit over an enlarged dual space, not a density optimization. It drops PSD, nonnegative volume and spectral complementarity restrictions, so the result is an **optimistic lower bound**. The raw bounds are .3393,.3306,.1268; even this enlargement does not remove the gray residual.

At 800 the fitted Q happens to be PSD with eigenvalues .01794/.98206; its volume multiplier is .4301. But trace(QD)/lambda1=3.07e-5 and ||QD||F/lambda1=4.54e-5, not zero. The analogous filtered bound is .05045. These explain why approximate near-cluster optimality can look much better than exact one-branch KKT, without silently treating separated eigenvalues as exactly multiple. At 400/480 the unrestricted fits violate PSD (and the 400 raw volume sign), so they remain lower bounds only. Exact admissible spectral dual at each saved state is Q=diag(1,0); fJJ is inactive. Native deltaLambda confirms that active derivative.

The last actual MMA subproblem was based on the **previous** density, with an inner solution and dual state that are not retained. Zero-increment testing at the final density does not prove that last subproblem was solved inaccurately. The evidence establishes physical nonstationarity and filtered-surrogate mismatch; it does not allocate all blame to MMA.


### Local-core qualification

A separate core-only multiplier fit gives raw broad-core normalized RMS **.06063 at 480** and **.18259 at 800**. At 480 the fitted raw derivative threshold 1.35315 closely matches the filtered gray threshold 1.35847; using that filtered-derived dual gives core RMS .06064. This is positive evidence of **approximate local balance inside much of the 480 broad patch**. It does not eliminate raw residuals in the surrounding gray field: all-gray RMS is .37060 with the same core-fitted multiplier. At 800 even the core-only fit remains above .1; using its filtered gray dual gives core RMS .37089.

Thus the primary nonstationarity category applies to the **complete constrained design and its substantial unresolved gray regions**, not every gray element. Some gray locations are locally balanced, and the 480 core could participate in a nearby stationary gray design. This audit cannot rule that out. The mixed-stationarity category is not issued as a certification of those patches because no common admissible physical multiplier makes the surrounding free design stationary. Local fits cannot be assigned independently to different parts of one volume-constrained problem. These results narrow the conclusion: “the whole endpoint is a genuine relaxed optimum” is unsupported; “every broad patch is itself necessarily nonstationary” is also unsupported.
