# Olhoff fixed-budget quality audit

## Executive finding

The faithful Du–Olhoff trajectory is already spectrally mature in the **raw**
common evaluators at 200 outer iterations: relative to each mesh's own k=1600
late-run state, raw-E1 loss is 0.794%, 1.391%, 1.543%, and 2.098% from 160×20
through 400×50. All four states are finite, volume-feasible and connected from
support to support. Thus k=200 is a defensible **coarse** fixed budget under the
2.5% raw-E1 band.

It is not the preregistered practical operating point. None of the four meshes
satisfies the complete 1% practical rule: 160×20 meets the loss band but its 1%
bimodal gap later reopens; 240×30 has established bimodality but loses 1.391%;
320×40 and 400×50 miss the 1% band and later reopen the gap. At 400×50 the
volume-preserving binary state is particularly immature: its E1 loss is 28.1%
and its E2/E3 losses are 97.7%, even though one component still spans the two
supports. This representation dependence rules out an evaluator-robust claim.

The result is therefore **INSUFFICIENT_BUDGET** at the declared 1% practical
criterion, not resolution-sensitive: zero of four meshes is practically
adequate, while all four are only coarsely adequate. The numerical evidence
supports treating Yuksel's wording as a termination convention, but does not
support transferring 200 as a general practical profile for this faithful
Du–Olhoff reproduction.

## Scientific frame and preregistered rules

The primary question was how much useful spectral/topological quality has been
obtained after the predeclared budget k=200. The optimization was not rerun.
All results use saved states from the same continued k=0…1600 trajectories at
160×20, 240×30, 320×40 and 400×50. No move, filter, tolerance or multiplicity
parameter was swept or changed.

The rules were frozen in [`study_preregistration.json`](study_preregistration.json)
before checkpoint outcomes were inspected:

- primary quality: signed raw common-E1 ω1 loss relative to the same mesh at
  k=1600; negative losses are retained;
- loss bands: 0.5%, 1.0%, and 2.5%;
- persistent crossing: the condition remains true at every later saved state
  through k=1600;
- `BIMODAL_ESTABLISHED` at k=200: N=2 and gap12≤1% at k=200 and at every later
  saved state; otherwise a currently valid state that later fails is
  `BIMODAL_TRANSIENT`;
- healthy/connected: finite values, successful LP flags, |volume residual|≤10⁻⁶,
  and left-to-right connectivity in both raw-0.5 and exact-count binary fields;
- declared cross-resolution practical criterion: the 1% class, including the
  health, connectivity and established-modal requirements.

k=1600 is called the **late-run reference**, not a converged solution. There is
no mathematical-stationarity requirement in the adequacy rule.

## Table A — Frozen configuration and provenance

| Item | Frozen value / result |
|---|---|
| Branch / HEAD | `benchmark-methodology-r2` / `cb6353feb941f12b2aaa927e622649e1ccc926f7` |
| Working tree | already dirty at audit start; all pre-existing changes preserved |
| MATLAB | 25.2.0.3042426 (R2025b) Update 1, MACA64 |
| Threads | 1 in every saved optimization (`res.cfg.threads`); 1 in offline evaluators |
| Profile | `repro2007_config('fig3a_best')` |
| Numerical settings | move=0.005; rmin=1.3 element widths; tolMult=0.05; rhomin=0.001; LP; diagonal filtering; offDiag=false |
| Source-manifest verification | 61 matched, 0 mismatched, 0 missing |
| Frozen reproduction tree hash | `d0fcc873310aeea504a84bc6b93f484b073ceecc06685cf304e1df73f82a8747` |
| Trajectories | four saved 1600-update trajectories, snapshot stride 1 |
| 240×30 identity | bit-identical to the frozen `lp240_rmin1.3.mat` baseline |
| Solver telemetry | zero LP failures and zero nonfinite iterations on all trajectories |
| Common evaluator | unchanged R3 `study_evaluate_design.m`; E1 extraction verified at all 44 declared checkpoints, max absolute difference 0 |
| Authoritative numerical code changed | **No** |

Full paths, SHA256 values, the initial dirty-tree inventory and migration mapping
are in [`provenance.json`](provenance.json).

Native values below are aligned to the post-update design x_k: for k<1600 the
modal solve at the start of update k+1 is used; x_1600 uses the optimizer's
post-loop modal solve. Common metrics are evaluated directly on x_k.

## Table B — Central k=200 result

| Mesh | native ω1 | native ω2 | gap12 | N | raw E1 ω1(200) | raw E1 ω1(1600) | E1 loss | E2 loss | E3 loss | bin E1 loss | ρ RMS | bin turnover | conn. raw/bin | time to 200 | class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---|
| 160×20 | 166.117 | 166.181 | 0.038% | 2 | 165.429 | 166.753 | 0.794% | 0.771% | 0.771% | 0.646% | 0.1656 | 6.69% | yes/yes | 10.76 s | COARSELY_ADEQUATE |
| 240×30 | 168.081 | 168.367 | 0.170% | 2 | 167.395 | 169.755 | 1.391% | 1.402% | 1.402% | 0.684% | 0.1591 | 5.17% | yes/yes | 22.69 s | COARSELY_ADEQUATE |
| 320×40 | 167.976 | 168.224 | 0.147% | 2 | 167.460 | 170.084 | 1.543% | 1.504% | 1.504% | 0.851% | 0.2069 | 7.72% | yes/yes | 41.52 s | COARSELY_ADEQUATE |
| 400×50 | 168.146 | 168.321 | 0.104% | 2 | 167.680 | 171.272 | 2.098% | 2.067% | 2.067% | **28.054%** | 0.2058 | 7.18% | yes/yes | 70.23 s | COARSELY_ADEQUATE |

The class is the preregistered primary-rule class. It must be read with the
secondary evaluator disclosure: 400×50 is not binary-evaluator robust. Binary
E2 and E3 losses there are 97.723% each. The binary field has 20 components at
k=200 (one spans both supports), versus one component at k=1600; the small
floating components introduce very low modes. On the other three meshes all
binary evaluator losses at k=200 are within 2.5%.

At k=200 all native spectra satisfy N=2 and gap12≤1%, but only 240×30 retains
that condition without interruption through the late-run reference.

## Table C — All checkpoint quality and cost results

Loss is signed relative to k=1600. Native progress is the fraction of the
initial-to-terminal native-ω1 improvement; values above 100% and negative loss
are deliberately not clipped. Loop time is the measured cumulative sum of
eigensolve, gradient and LP-inner telemetry—not a prorate of wall time.

| Mesh | k | E1 raw loss | E2 raw loss | E3 raw loss | E1 bin loss | native progress | gap12 | N | ρ RMS | bin turnover | loop time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 160×20 | 50 | 36.632% | 36.816% | 36.816% | 42.480% | 37.69% | 185.500% | 1 | 0.3229 | 13.06% | 3.50 s |
| 160×20 | 100 | 9.961% | 10.086% | 10.086% | 73.107% | 82.94% | 8.631% | 1 | 0.2051 | 10.25% | 6.36 s |
| 160×20 | 150 | 2.633% | 2.788% | 2.788% | 47.838% | 95.47% | 0.399% | 2 | 0.1780 | 8.88% | 8.73 s |
| 160×20 | 200 | 0.794% | 0.771% | 0.771% | 0.646% | 98.83% | 0.038% | 2 | 0.1656 | 6.69% | 10.76 s |
| 160×20 | 250 | 0.578% | 0.519% | 0.519% | 0.610% | 99.18% | 0.272% | 2 | 0.1550 | 6.12% | 12.86 s |
| 160×20 | 300 | 0.589% | 0.570% | 0.570% | 56.646% | 99.15% | 0.121% | 2 | 0.1424 | 5.06% | 14.89 s |
| 160×20 | 400 | 0.860% | 0.848% | 0.848% | 69.651% | 98.64% | 0.197% | 2 | 0.1199 | 3.62% | 18.89 s |
| 160×20 | 600 | 0.124% | 0.150% | 0.150% | 0.452% | 99.74% | 0.214% | 2 | 0.0710 | 1.50% | 26.97 s |
| 160×20 | 800 | 0.206% | 0.038% | 0.038% | 0.291% | 99.96% | 0.239% | 2 | 0.0402 | 1.00% | 34.99 s |
| 160×20 | 1200 | −0.125% | −0.156% | −0.156% | −0.014% | 100.25% | 0.184% | 2 | 0.0168 | 0.31% | 50.96 s |
| 160×20 | 1600 | 0.000% | 0.000% | 0.000% | 0.000% | 100.00% | 0.222% | 2 | 0.0000 | 0.00% | 66.63 s |
| 240×30 | 50 | 37.665% | 37.813% | 37.813% | 60.521% | 36.71% | 183.560% | 1 | 0.3402 | 16.92% | 7.06 s |
| 240×30 | 100 | 10.952% | 11.083% | 11.083% | 57.838% | 81.33% | 2.126% | 2 | 0.2386 | 11.00% | 13.18 s |
| 240×30 | 150 | 4.493% | 4.362% | 4.362% | 2.199% | 92.53% | 0.082% | 2 | 0.1953 | 7.69% | 17.92 s |
| 240×30 | 200 | 1.391% | 1.402% | 1.402% | 0.684% | 97.66% | 0.170% | 2 | 0.1591 | 5.17% | 22.69 s |
| 240×30 | 250 | 0.692% | 0.682% | 0.682% | 57.961% | 98.89% | 0.241% | 2 | 0.1281 | 3.97% | 27.25 s |
| 240×30 | 300 | 0.469% | 0.454% | 0.454% | 64.008% | 99.25% | 0.169% | 2 | 0.1008 | 2.58% | 31.87 s |
| 240×30 | 400 | 0.064% | 0.058% | 0.058% | 0.028% | 99.91% | 0.220% | 2 | 0.0628 | 1.22% | 41.44 s |
| 240×30 | 600 | 0.015% | 0.020% | 0.020% | 0.009% | 99.97% | 0.235% | 2 | 0.0150 | 0.22% | 60.04 s |
| 240×30 | 800 | 0.001% | 0.002% | 0.002% | 0.000% | 100.00% | 0.242% | 2 | 0.0011 | 0.00% | 78.30 s |
| 240×30 | 1200 | −0.000% | 0.000% | 0.000% | 0.000% | 100.00% | 0.241% | 2 | 0.0005 | 0.00% | 114.77 s |
| 240×30 | 1600 | 0.000% | 0.000% | 0.000% | 0.000% | 100.00% | 0.232% | 2 | 0.0000 | 0.00% | 150.35 s |
| 320×40 | 50 | 37.776% | 37.910% | 37.910% | 55.883% | 36.76% | 183.387% | 1 | 0.3435 | 16.78% | 12.24 s |
| 320×40 | 100 | 11.004% | 11.100% | 11.100% | 55.243% | 81.62% | 1.648% | 2 | 0.2473 | 12.31% | 23.83 s |
| 320×40 | 150 | 4.514% | 4.475% | 4.475% | 2.412% | 92.75% | 0.025% | 2 | 0.2189 | 9.86% | 32.86 s |
| 320×40 | 200 | 1.543% | 1.504% | 1.504% | 0.851% | 97.57% | 0.147% | 2 | 0.2069 | 7.72% | 41.52 s |
| 320×40 | 250 | 0.500% | 0.500% | 0.500% | 0.467% | 99.24% | 0.133% | 2 | 0.1970 | 6.95% | 50.08 s |
| 320×40 | 300 | 0.245% | 0.275% | 0.275% | 0.588% | 99.62% | 0.262% | 2 | 0.1909 | 6.38% | 58.77 s |
| 320×40 | 400 | 0.223% | 0.187% | 0.187% | 0.507% | 99.73% | 0.220% | 2 | 0.1820 | 5.62% | 75.78 s |
| 320×40 | 600 | 0.587% | 0.597% | 0.597% | 0.890% | 99.03% | 0.049% | 2 | 0.1701 | 4.84% | 110.53 s |
| 320×40 | 800 | 0.172% | 0.174% | 0.174% | 0.291% | 99.73% | 0.135% | 2 | 0.1521 | 4.58% | 145.31 s |
| 320×40 | 1200 | −0.368% | −0.356% | −0.356% | −0.087% | 100.60% | 0.148% | 2 | 0.0871 | 2.09% | 216.59 s |
| 320×40 | 1600 | 0.000% | 0.000% | 0.000% | 0.000% | 100.00% | 0.146% | 2 | 0.0000 | 0.00% | 284.92 s |
| 400×50 | 50 | 38.220% | 38.322% | 38.322% | 60.782% | 36.32% | 183.373% | 1 | 0.3537 | 17.58% | 20.24 s |
| 400×50 | 100 | 11.833% | 11.865% | 11.865% | 66.384% | 80.36% | 2.135% | 2 | 0.2594 | 11.76% | 40.57 s |
| 400×50 | 150 | 5.489% | 5.428% | 5.428% | 27.633% | 90.93% | 0.337% | 2 | 0.2300 | 9.79% | 56.02 s |
| 400×50 | 200 | 2.098% | 2.067% | 2.067% | 28.054% | 96.59% | 0.104% | 2 | 0.2058 | 7.18% | 70.23 s |
| 400×50 | 250 | 0.940% | 0.902% | 0.902% | 0.429% | 98.57% | 0.116% | 2 | 0.1864 | 5.84% | 85.15 s |
| 400×50 | 300 | 0.423% | 0.429% | 0.429% | 14.882% | 99.34% | 0.179% | 2 | 0.1717 | 5.22% | 100.23 s |
| 400×50 | 400 | 0.478% | 0.473% | 0.473% | 0.748% | 99.16% | 0.526% | 2 | 0.1498 | 4.10% | 128.61 s |
| 400×50 | 600 | 0.299% | 0.246% | 0.246% | 22.101% | 99.58% | 0.155% | 2 | 0.1204 | 2.96% | 186.90 s |
| 400×50 | 800 | 0.245% | 0.252% | 0.252% | 0.293% | 99.60% | 0.084% | 2 | 0.0991 | 2.24% | 243.99 s |
| 400×50 | 1200 | 0.225% | 0.203% | 0.203% | 0.359% | 99.64% | 0.115% | 2 | 0.0734 | 1.55% | 358.71 s |
| 400×50 | 1600 | 0.000% | 0.000% | 0.000% | 0.000% | 100.00% | 0.214% | 2 | 0.0000 | 0.00% | 473.05 s |

The full CSV also contains volume, grayness, component counts, all binary
E2/E3 losses, timing per iteration/window, and eigensolve time/share. At k=200
the cumulative eigensolve shares are 58.5%, 61.7%, 61.1%, and 59.0% with mesh
refinement; mean loop costs are 53.8, 113.4, 207.6, and 351.2 ms/iteration.

## Table D — First and persistent quality-threshold crossings

These are exact outer-iteration budgets from the raw-E1 evaluation at every
saved x_k, not checkpoint interpolation.

| Mesh | E1 band | first crossing | persistent crossing through k=1600 |
|---|---:|---:|---:|
| 160×20 | ≤2.5% | 153 | 155 |
| 160×20 | ≤1.0% | 186 | 459 |
| 160×20 | ≤0.5% | 226 | 843 |
| 240×30 | ≤2.5% | 177 | 177 |
| 240×30 | ≤1.0% | 219 | 219 |
| 240×30 | ≤0.5% | 294 | 294 |
| 320×40 | ≤2.5% | 178 | 1150 |
| 320×40 | ≤1.0% | 218 | 1150 |
| 320×40 | ≤0.5% | 250 | 1150 |
| 400×50 | ≤2.5% | 192 | 507 |
| 400×50 | ≤1.0% | 246 | 1104 |
| 400×50 | ≤0.5% | 290 | 1510 |

The large gaps between first and persistent crossings on 320×40 and 400×50
are caused by later quality spikes, not solver failure. A fixed budget of 1200
is observationally inside and remains inside the 1% raw-E1 band for all four
meshes over the remaining 400 observed iterations. This does not replace 200,
does not establish convergence, and is not a stopping detector.

## Table E — Bimodality establishment

| Mesh | condition | first entry | persistent entry | later closes/opens | state at k=200 |
|---|---|---:|---:|---:|---|
| 160×20 | N=2 | 115 | 115 | 0 / 1 | true |
| 160×20 | gap≤5% | 115 | 115 | 0 / 1 | true |
| 160×20 | gap≤2% | 133 | 133 | 0 / 1 | true |
| 160×20 | gap≤1% | 145 | 1504 | 3 / 4 | **BIMODAL_TRANSIENT** |
| 240×30 | N=2 | 94 | 94 | 0 / 1 | true |
| 240×30 | gap≤5% | 94 | 94 | 0 / 1 | true |
| 240×30 | gap≤2% | 101 | 101 | 0 / 1 | true |
| 240×30 | gap≤1% | 106 | 106 | 0 / 1 | **BIMODAL_ESTABLISHED** |
| 320×40 | N=2 | 94 | 94 | 0 / 1 | true |
| 320×40 | gap≤5% | 94 | 94 | 0 / 1 | true |
| 320×40 | gap≤2% | 99 | 99 | 0 / 1 | true |
| 320×40 | gap≤1% | 103 | 1548 | 7 / 8 | **BIMODAL_TRANSIENT** |
| 400×50 | N=2 | 94 | 157 | 4 / 5 | true |
| 400×50 | gap≤5% | 94 | 157 | 4 / 5 | true |
| 400×50 | gap≤2% | 101 | 157 | 4 / 5 | true |
| 400×50 | gap≤1% | 104 | 1511 | 7 / 8 | **BIMODAL_TRANSIENT** |

“Closes/opens” counts all state transitions over k=0…1600; the initial entry is
included among opens. N=2 itself is persistent on three meshes, but the tighter
1% gap is not. Hence k=200 has the intended pair on every mesh as a snapshot,
yet that pair is only *established* on 240×30 under the declared rule.

## Quality-vs-budget curves and late-phase behavior

![All-mesh common-E1 loss](figures/all_meshes_common_E1_quality_loss.png)

Per-mesh eight-panel figures contain normalized E1, signed loss, native ω1/ω2,
gap12, N, density RMS, binary turnover and measured cumulative runtime:

- [`quality_vs_budget_160x20.png`](figures/quality_vs_budget_160x20.png)
- [`quality_vs_budget_240x30.png`](figures/quality_vs_budget_240x30.png)
- [`quality_vs_budget_320x40.png`](figures/quality_vs_budget_320x40.png)
- [`quality_vs_budget_400x50.png`](figures/quality_vs_budget_400x50.png)

Every k=200 line is labelled “fixed budget”; none is labelled convergence.

The trajectories are not uniformly stationary late in the run. Over the last
100 states, raw-E1 ranges are 166.111–166.914 (160×20),
169.7550–169.7614 (240×30), 169.625–170.951 (320×40), and
170.188–171.306 (400×50). The mean lag-two density RMS over the last 200
updates is respectively 0.00338, 0.000104, 0.00309 and 0.00217. Therefore the
single terminal reference is disclosed alongside late-block statistics in
[`late_phase_diagnostics.csv`](late_phase_diagnostics.csv); it is not silently
promoted to a converged state. All predeclared checkpoints are even, so their
same-phase late reference is x_1600; full-curve same-phase distances are stored
separately from raw-terminal distances.

## Table F — Cross-resolution budget classification

| Test at k=200 | Passing meshes | Cross-resolution conclusion |
|---|---:|---|
| Strict: ≤0.5% + healthy + connected + established modal state | 0/4 | not supported |
| Practical: ≤1.0% + healthy + connected + established modal state | **0/4** | **INSUFFICIENT_BUDGET** |
| Coarse: ≤2.5% + healthy + connected | 4/4 | robust only at the coarse raw-E1 level |
| Binary evaluator robustness within 2.5% | 3/4 | fails at 400×50 |

Loss-only, 160×20 is within 1%; its `BIMODAL_TRANSIENT` label prevents promotion
to the full practical class. The result is not `RESOLUTION_SENSITIVE_BUDGET`
under the declared criterion because no mesh passes the complete criterion.

## Yuksel Table 1 interpretation

The three hypotheses rank as follows:

1. **H3 — non-transferable convention (strongest).** Yuksel states that its
   Dynamic Code optimization is “terminated after 200 iterations.” The present
   faithful Du–Olhoff trajectory reaches different maturity at different mesh
   resolutions, fails the complete 1% rule everywhere, and is not binary-robust
   at 400×50. The number cannot be transferred as a validated general profile.
2. **H2 — fixed-budget interpretation (partially supported, contextual).** The
   wording and Table 1 arithmetic are consistent with a prescribed work budget,
   and all four raw-E1 states are within the coarse 2.5% band while healthy and
   connected. Thus fixed budget is more defensible than a convergence reading,
   but **200 specifically** is only a coarse/contextual budget here.
3. **H1 — convergence interpretation (weakest).** Nothing in this audit
   establishes native convergence at 200. The native move statistic remains
   saturated in the frozen evidence; three meshes later reopen the 1% gap; the
   design is still 0.159–0.207 RMS away from x_1600; and later E1 spikes occur.

“Terminated after 200 iterations” is not paraphrased as “converged in 200
iterations.” The audit tests the former and rejects the latter.

## Table G — Implications for `performance_comparison.m`

| Use | Recommendation | Reason |
|---|---|---|
| General representative Olhoff operating point | **Do not use k=200** | it is insufficient under the declared practical criterion and lacks cross-resolution modal/evaluator robustness |
| Separate Yuksel-Table-1 interpretation | **Context only** | retain only as an explicitly labelled `FIXED BUDGET / NOT NATIVE CONVERGENCE` diagnostic or reproduction row |
| R3 native convergence/stationarity table | Keep separate and unchanged | a fixed budget is not a native stopping rule |
| Claim of a validated Olhoff practical profile | Prohibited | the prior detector failure remains frozen evidence |

At 240×30, changing only the illustrative Olhoff observation point from the
previous detector fire k=812 to k=200 reduces measured Olhoff loop time from
78.21 s to 22.69 s and raw-E1 ω1 from 169.754 to 167.395. Against the frozen
selected Yuksel/Proposed results, the Olhoff cost multipliers become about 2.7×
and 3.3× rather than about 10×, while its raw-E1 advantages become about 5.0%
and 6.2%. This changes a **contextual fixed-budget picture**, not the frozen R3
native-practical hierarchy. Olhoff remains more expensive per useful run and
spectrally stronger; Yuksel versus Proposed remains evaluator-dependent.

## Evidence map and reproducibility

- [`checkpoint_metrics.csv`](checkpoint_metrics.csv): all checkpoint native,
  common-evaluator, topology and measured-cost fields;
- [`budget_adequacy.csv`](budget_adequacy.csv): central k=200 rows and classes;
- [`modal_establishment.csv`](modal_establishment.csv): all modal first/persistent entries;
- [`minimum_quality_budget.csv`](minimum_quality_budget.csv): exact E1 and 1%-gap budgets;
- [`quality_budget_curves.csv`](quality_budget_curves.csv): full k=0…1600 curve evidence;
- [`raw/`](raw/): per-iteration E1 evaluations and checkpoint MAT evidence;
- [`run_checkpoint_audit.m`](run_checkpoint_audit.m),
  [`run_e1_trajectory.m`](run_e1_trajectory.m), and
  [`build_outputs.m`](build_outputs.m): offline reproduction scripts.

No 480×60, 560×70, 640×80, 720×90 or 800×100 solve was run. No authoritative
numerical implementation was changed, no parameter was retuned, and no new
convergence detector was created.

## Final verdict

**FIXED-200 PRACTICAL INTERPRETATION NOT SUPPORTED**

- k=200 represents convergence: **NO**
- k=200 may be used in a Yuksel-Table-1-style comparison: **CONTEXT ONLY**
- Olhoff now has a cross-resolution validated native practical stopping profile: **NO**
