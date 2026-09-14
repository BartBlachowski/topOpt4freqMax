# Phase 2I Candidate-C Olhoff trajectory precision qualification

## Outcome

The principal Olhoff-LP float32 trajectory-storage path is **not qualified**. Candidate C
removes the old Eq. (4) spectral discontinuity: modal selection, spectral errors, reference,
quality, and persistence endpoints all agree. But the complete frozen hard gate differs at
four same-state updates because float32 collapses density orderings into cutoff ties. Q7 is
binding and requires identity, so the verdict is FAIL without evaluator or methodology
retuning.

## Binding evidence

- Same-state 96×12 reference trajectory: 3,200 post-update pairs, captured in double and
  independently repeated through the untouched single-snapshot runner.
- Density maximum absolute error: `2.98019426914e-08`; rho=.1 branch crossings: `19517`.
- Selected ordinal/classifier/adaptive-search mismatches: `0 / 0 / 0`.
- Maximum relative omega errors: E1 `5.59511131921e-08`,
  E2 `5.59553986329e-08`, E3
  `5.59482294612e-08`.
- Formal evidence bound: relative omega `1.12e-7`, absolute omega `8.18e-6`; raw maximum is
  `1.11911e-05` of the q=.995 0.5% band.
- The 2x safety factor is a transparent conservative reporting envelope over the
  no-exclusion raw maximum; it was not tuned to rescue qualification, which independently
  fails Q7.
- Binary differences: `95` states / `232` entries, all explained
  by float32 cutoff ties; hard-gate mismatches: `4` at `[41, 45, 48, 99]`.
- `b_ref`: 2100/2100; `B_meas`: 3200/3200.
- P=100 endpoints are identical for all q; see `PERSISTENCE_EQUIVALENCE.csv` for the
  machine-derived values.
- Explicit difficult-case coverage reaches selected ordinal 18,
  including the >12-mode and maximum-ordinal-18 Phase-2G cases.
- Production-scale offline evidence: eight available meshes, up to `12477` at-risk
  elements in one state; no final-pair modal/classifier/hard-gate mismatch. This does not
  override the binding 96×12 hard-gate failure.

## Required final summary

1. Branch and HEAD: `benchmark-methodology-r2` / `632e9b01811845709de33f93051fd853373ed5e1`.
2. Repository state before work: dirty with three tracked modifications and pre-existing untracked audit/campaign trees; preserved.
3. Candidate-C contract hash: `cc900b4ad4cae18b0bcd9b7a559f51e04e5167db587f64180b371d3c399bf95b`.
4. Candidate-C evaluator hash: `e14a21efe0bb2d9b9d7f3187b4c3f671ec089f6ff96773074b8f3b56cacd79e9`.
5. Native optimizer modified? **NO**.
6. Frozen methodology modified? **NO**.
7. Principal route tested: **Olhoff-LP**.
8. Same-state pairing mechanism: exact double optimizer state `x_d` and `double(single(x_d))`; protected runner cast checked across all columns.
9. Prefix determinism result: **PASS** (full repeat plus 8 strategic lossless capped checks; historical 45 float32-prefix checks remain supporting evidence).
10. Number of paired states: **3,200 binding**, plus 12 new supporting difficult/production final pairs and 236 historical paired states.
11. Meshes represented: 24×4, 96×12, and production 160×20 through 720×90 (800×100 unavailable).
12. Maximum density absolute error: `2.98019426914e-08`.
13. rho≈0.1 crossing count: `19517` on the binding trajectory.
14. Maximum at-risk elements per state: `12477` (720×90 historical trajectory); binding maximum `192`.
15. Selected-mode mismatch count E1: `0`.
16. Selected-mode mismatch count E2: `0`.
17. Selected-mode mismatch count E3: `0`.
18. Classifier mismatch count: `0` relevant; `0` across all examined aligned modes.
19. Minimum observed classifier margin: `0.00134902433628`.
20. Maximum voidKE perturbation: `0.0449434051516`.
21. Maximum voidSE perturbation: `0.334731187521`.
22. Maximum densityParticipation perturbation: `1.42371802014e-07`.
23. Maximum E1 relative omega error: `5.59511131921e-08`.
24. Maximum E2 relative omega error: `5.59553986329e-08`.
25. Maximum E3 relative omega error: `5.59482294612e-08`.
26. Formal documented precision bound: relative `1.12e-7`, absolute omega `8.18e-6` (2× raw maximum, rounded upward).
27. Error/band ratio for q=.995: `1.11910797266e-05` raw; `2.24e-05` formal-bound ratio.
28. Adaptive escalation mismatch count: `0`.
29. Hard-gate mismatch count: `4`.
30. Binary-field difference count: `95` states / `232` entries.
31. Unexplained binary difference count: `0`.
32. Maximum |Delta Q|: `2.50559931136e-08`.
33. Binding-evaluator change count: `2`.
34. q=.98 crossing differences: `0`.
35. q=.99 crossing differences: `0`.
36. q=.995 crossing differences: `0`.
37. b_ref double: `2100`.
38. b_ref single: `2100`.
39. B_meas double: `3200`.
40. B_meas single: `3200`.
41. k_enter .98 double/single: `229 / 229`.
42. k_cert .98 double/single: `328 / 328`.
43. k_enter .99 double/single: `309 / 309`.
44. k_cert .99 double/single: `408 / 408`.
45. k_enter .995 double/single: `453 / 453`.
46. k_cert .995 double/single: `552 / 552`.
47. Status identity: **PASS / PASS for all q**.
48. P=50 sensitivity result: identical endpoints for all q.
49. P=200 sensitivity result: identical endpoints for all q.
50. Production-scale offline check result: no contradictory paired final-state failure across 8 available meshes.
51. Worst production-scale mesh/state: 720×90 by exposure (`12477` at-risk); 640×80 final E3 by relative error (`1.6593e-9`).
52. Independent replay result: **PASS**.
53. Phase-2B old maximum E2/E3 error: `0.0226523752185` / `0.0226523756441`.
54. Candidate-C new maximum E2/E3 error: `5.59553986329e-08` / `5.59482294612e-08`.
55. Phase-2B endpoint mismatch reproduced/explained? **YES historically explained; Candidate-C endpoints now identical**.
56. MMA secondary evidence available? **NOT TESTED — NONBLOCKING FOR LP**; no usable saved BASE-MMA density artifact.
57. Q1–Q16: `PASS,PASS,PASS,PASS,PASS,PASS,FAIL,PASS,PASS,PASS,PASS,PASS,PASS,PASS,PASS,PASS`.
58. Qualification artifact written? **Negative artifact only; no pass artifact installed**.
59. Precision preflight blocker cleared? **NO**.
60. Remaining preflight blockers: precision, cross-method, reference-length.
61. Residual rho=.1 precision pathology? **No spectral Eq. (4) pathology; yes, exact-count cutoff-tie topology sensitivity**.
62. New scientific issue discovered? **YES — four binding hard-gate flips from float32-created cutoff ties**.
63. Production campaign run? **NO**.
64. Exact next action: retain lossless double Olhoff trajectory storage and keep precision preflight blocked; any change to hard-gate equivalence requires a separately authorized methodology phase.

PHASE 2I FAILED —
OLHOFF SINGLE-PRECISION TRAJECTORY NOT QUALIFIED UNDER CANDIDATE C
