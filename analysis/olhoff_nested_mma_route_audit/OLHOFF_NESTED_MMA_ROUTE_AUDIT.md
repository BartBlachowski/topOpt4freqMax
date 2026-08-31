# Du–Olhoff reproduction — nested-MMA route audit

READ-ONLY FORENSIC AUDIT — NO PRODUCTION CAMPAIGN — NO METHOD SWITCH — NOTHING RE-RUN

Date 2026-08-30 · branch `benchmark-methodology-r2` · HEAD `632e9b01811845709de33f93051fd853373ed5e1`
No optimizer was run. No algorithm, methodology, contract or campaign file was modified.
Everything new is confined to `analysis/olhoff_nested_mma_route_audit/`.

---

## Verdict in one paragraph

`BASE_mma_160x20` is a **genuine, deterministic, well-behaved nested-MMA reproduction that
reaches the correct benchmark basin** — and it is decisively *not* ready to become the
principal paper-facing Olhoff route. It is legitimate: the route is real full-coupling MMA
with no LP fallback, it is bit-identically reproducible, its topology is paper-like, and it
yields a genuinely publishable result about the cost of multiplicity that the LP route
structurally cannot produce. But its only artifact is a log from a MATLAB session that died
at outer 752 of a declared 800, with no density field and no saved result; its eigengap is
an order of magnitude worse than the matched LP run on the property that defines the Fig. 3a
benchmark; and it costs roughly **400× the LP route per outer iteration**, which makes a
nine-mesh campaign infeasible independently of any quality argument. The correct disposition
is to keep LP as the principal reproduction route and report nested MMA as a secondary
paper-native variant — which, notably, captures essentially all of the MMA evidence's
scientific value at almost none of the switching cost.

---

## 1. What was audited, and how

The run's source tree (`/Volumes/HP911Pro/Combobulating/Olhoff/`) is **byte-identical** to
the repository's `Matlab/reproduction2007/` — 46 of 46 `.m` files plus `setpaths.m` and
`top88.m` hash-match — so the repository code is the exact code that produced the run.

The complete log was parsed, not sampled: **752 rows, no missing iterations, no duplicates,
`cumInner` arithmetic internally consistent at every row**.

A second artifact turned out to be decisive. `fm_mma_diag.mat` is the **same configuration**
(every field matches except `maxOuter`, 400 vs 800) and it *was* saved. Its trajectory is
**bit-identical to BASE_mma over all 400 shared outer iterations** — `nInner`, `cumInner` and
`N` match exactly, `omega1` and `max|drho|` to the log's printed precision, `cumInner@400 =
55278` in both. It therefore supplies the density field, the timing and the topology that
BASE_mma itself lacks, up to outer 400.

## 2. The filter-radius question, resolved (WP2)

The log header prints `rminEl: 3` and `rminPhys: 0.0600`. Taken literally the first would
place this run in the *failing* large-radius family.

It does not. `olhoffOpt.m` overrides `rminEl` from `rminPhys` before building the filter:

    cfg.rminEl = cfg.rminPhys/(cfg.b/cfg.nely) = 0.06/(1/20) = 1.2

`run_case.m` prints `disp(cfg)` **before** calling `olhoffOpt`, so the echo is the
pre-override configuration. `rminEl` is the only quantity that reaches `prepFilter`, which
documents its argument as element units. **The effective radius is 1.2 elements.**

Confirmed independently rather than only by reading code: `fm_mma_diag.mat` stores the
*post-override* representation (`rminEl = 1.200`, `rminPhys` emptied) and is bit-identical to
BASE_mma. Two runs, two representations, one trajectory.

This places BASE_mma squarely inside the successful bimodal band (1.1–1.5) and is the
principal reason it behaves where the legacy `rmin = 2.5` nested MMA did not. Details in
`FILTER_RADIUS_SEMANTICS.md`. **The log must never be quoted as configuration evidence for
this run.**

## 3. The route is genuine full-coupling MMA (WP1)

`innerSolver='mma'` selects `innerLoop`, and `offDiag=1` routes the increment model through
`deltaLambda`, which solves the `N × N` subeigenvalue problem
`det|f_sk' drho − delta_sk Δ(omega²)| = 0` in its erratum form and returns
`ddlam(e,j) = Σ_sk v_js v_jk (f_sk)_e`. That is real Eq. (25d) coupling, not a relabelled
diagonal approximation.

**No LP fallback exists on this path** at either `N = 1` or `N = 2`: the selector is a single
`if strcmpi(cfg.innerSolver,'lp')`, with no failure branch and no `N`-dependence. All 752
recorded iterations are pure nested MMA. Full call graph in `EXECUTABLE_ROUTE.md`.

## 4. The result worth publishing (WP6, WP16)

| regime | outers | mean | median | min | p90 | p95 | max | total | cap-hit | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| `N = 1` simple | 69 | **93.4** | 93 | 83 | 100 | 102 | 108 | 6 448 | **0.00%** | **100.00%** |
| `N = 2` multiple | 683 | **147.8** | 111 | 26 | **300** | **300** | 300 | 100 925 | **16.69%** | 83.31% |
| all | 752 | 142.8 | 107 | 26 | 300 | 300 | 300 | **107 373** | 15.16% | 84.84% |

Mann–Whitney U, one-sided: **U = 12 677, p = 1.1e-10**; ratio of means **1.581**.

**Multiplicity increases inner subproblem effort by ~58% in the mean and introduces a heavy
upper tail the simple-mode phase does not have at all** — p90 and p95 both saturate the cap
under multiplicity, while 69 simple-mode solves never exceed 108 and never fail to converge.
The prompt's expectation of "about 80–110" for the simple phase is confirmed and is in fact
tighter than that: min 83, max 108, zero outliers.

The LP route cannot produce this result: its `nInner` is exactly 1 at every one of 1600
outers, verified. This is the strongest argument for retaining MMA as a reported variant.

## 5. Do cap-hit inner solves harm the outer trajectory? (WP7)

**No detectable systematic harm — but this does not certify the subproblem.**

Conditioning each `N = 2` outer step on the previous inner status:

| previous inner status | n | mean Δω₁ | median Δω₁ | mean abs Δω₁ | P(Δω₁ < 0) |
|---|---|---|---|---|---|
| CONVERGED | 572 | +0.0325 | +0.0100 | 0.0873 | 0.411 |
| CAP-HIT | 111 | +0.0414 | +0.0500 | **0.1255** | 0.414 |

Mann–Whitney two-sided on Δω₁: **p = 0.515**. Cap-hit steps drift *slightly more*, not less,
and regress no more often. What they do show is **44% larger mean absolute step** — more
volatility at equal drift.

So the naive assumption "conv = NO ⇒ bad step" is refuted. But 16.7% of multiple-mode solves
still terminate at 300 sub-iterates without meeting the declared inner test, and the inner
histories (`st.dxHist`, `st.relHist`) are computed and then **discarded**, so WP8 cannot be
answered from this evidence: it is not possible to tell whether those solves were near the
tolerance or far from it. **This remains an outstanding qualification, not an accepted
practice** (finding M8).

## 6. Why the route is nevertheless not principal

**(a) The artifact does not exist.** No `.mat`, no density field, no topology, no stop
status. `run_case.m` saves only after `olhoffOpt` returns, and it never returned. The
session died at outer **752 of a declared 800** (log lines 791–792: WindowServer event port
death). The run is truncated and self-incomplete (M1, M2).

**(b) It is not converged.** Terminal `max|drho| = 0.00990` against `tolOuter = 1e-3` — ten
times the threshold, sitting on the move limit. ω₁ peaked at **169.19 at outer 482** and ends
at **168.75**; the last-200 trend is **−2.6e-04 per iteration**. Classification:
**MOVE-SATURATED_FINITE_TRAJECTORY** with a mildly regressive tail (M4).

**(c) It does not reproduce the benchmark's defining feature.** At the same mesh and the same
effective filter radius:

| | matched LP (`lprmin1.2`) | MMA (BASE_mma) |
|---|---|---|
| final ω₁ | 168.240 | **168.750** |
| final relative eigengap | **0.217%** | **2.376%** |
| outers with gap < 1% | ~1440 of 1600 | **2 of 752** |
| outers with gap < 0.5% | most of the tail | **0** |
| final ω₃ | 286.01 | 341.54 |

The LP route settles into a stable near-bimodal state by outer ~160 and holds it. The MMA
route chatters at the 5% tolerance to outer ~231, then wanders between 1% and 4% and ends at
2.4%. Cumulative move budgets are comparable (MMA 7.52 vs LP 8.00), so this is not simply a
matter of LP having travelled further — though the runs are **not move-matched** (LP 0.005 ×
1600 vs MMA 0.010 × 752) and that caveat is stated rather than hidden (M6).

**(d) Cost.** Measured, not estimated:

| | LP (1600 outer) | MMA (400 outer) |
|---|---|---|
| wallclock | **70.7 s** | **7 121.7 s** |
| per outer | 0.044 s | **17.8 s** |
| mean `tInner` | 0.012 s | 17.775 s |
| inner share of optimizer time | 27.4% | **99.8%** |
| mean eigensolve | 0.031 s | 0.029 s |

The eigensolve cost is *identical*; the whole difference is the inner solver. **~400× per
outer iteration.** Since `mmasub` is O(design variables) per sub-iterate, 720x90 (20× the
variables) projects to order 150 hours for one mesh and one method. The nine-mesh campaign is
infeasible on this route (M7).

**(e) Switching cost.** `olhoffOptStabilized.m` — the study's protected Olhoff comparator —
**hard-codes `innerLoopLP`** with no `innerSolver` switch. It is a `protected_numerical_sources`
entry whose SHA-256 every audit phase from 2B through 2F has re-verified unchanged. Selecting
MMA changes what "an iteration" means for one of three methods, re-opens the Phase-2B/2E
precision-qualification chain, and requires re-running all eight stored Olhoff trajectories
(M14, `CAMPAIGN_IMPACT.md`).

## 7. The asymmetry that decides it

Adopting MMA as the **principal** route costs items (a)–(e) above. Reporting it as a
**secondary paper-native variant** costs a single completed 160x20 run and delivers the
multiplicity-cost result, the paper-literal MMA claim, and the inner/outer work
decomposition. **Almost all of the scientific value, almost none of the cost.**

---

## Required final summary

1. **Exact executable nested-MMA route.** `run_case` → `olhoffOpt` → `innerLoop` →
   `mmasub`/`subsolv`, with `deltaLambda` for the Eq. (25d) increments, `genGrad` for the
   tensor, `prepFilter`/`applyFilter` for the filter, `assemble2D`+`eigSolve` per outer.
   **No LP is called at any point.**
2. **Is `offDiag=1` genuinely full coupling?** **Yes.** `deltaLambda` solves the `N × N`
   subeigenvalue problem and returns `Σ_sk v_js v_jk (f_sk)_e`. Real Eq. (25d).
3. **Exact meaning of `rminEl=3`.** A **stale pre-override display value**, printed by
   `run_case.m`'s `disp(cfg)` before `olhoffOpt` overrides it. Never used.
4. **Exact meaning of `rminPhys=0.06`.** Filter radius in physical units. Not used directly;
   it *overrides* `rminEl` by dividing by the element height `b/nely = 0.05`.
5. **Which one controls the filter?** `rminEl` **after** the override = **1.2 elements**.
6. **Available outer iterations.** **752** (declared `maxOuter = 800`).
7. **First `N=2` iteration.** **56.**
8. **Persistent `N=2` onset.** **231** (and it holds to 752 — 522 iterations).
9. **`N=2 → N=1` returns.** **14** (29 transitions total, all between outer 56 and 231).
10. **Mean `N=1` inner iterations.** **93.4** (median 93, range 83–108).
11. **Mean `N=2` inner iterations.** **147.8.**
12. **Median `N=2` inner iterations.** **111.**
13. **p95 `N=2` inner iterations.** **300** (the cap).
14. **Total cumulative MMA inner iterations.** **107 373.**
15. **Inner cap hits.** **114 of 752 = 15.16%** overall; **16.69%** within the `N=2` regime;
    **0.00%** within `N=1`.
16. **Inner solves meeting the declared test.** **638 of 752 = 84.84%**; 100% at `N=1`,
    83.31% at `N=2`.
17. **Do cap-hit solves systematically harm outer progress?** **No** (Mann–Whitney
    p = 0.515; mean Δω₁ +0.041 vs +0.032). They are associated with **44% larger mean
    absolute steps** — more volatility at equal drift. This does not certify the subproblem.
18. **Maximum ω₁.** **169.19**, at outer **482**.
19. **Terminal ω₁.** **168.75.**
20. **Terminal ω₂.** **172.76.**
21. **Terminal eigengap.** **4.01 absolute, 2.376% relative.**
22. **Best persistent near-bimodal interval.** **None.** Only **2 of 752** outers reach a
    relative gap below 1%, and **none** below 0.5%.
23. **Final volume.** **0.500** (exactly, throughout).
24. **Final grayness.** **0.1442** — from the verified `fm_mma_diag` proxy at outer 400;
    BASE_mma saved no density field.
25. **Final topology hard-gate result.** **FAIL** (16 components, 15 detached, worst island
    14 elements against `a_sig = 4`). The matched LP run also FAILS at this mesh.
26. **Is the final topology paper-like?** **Yes** — a symmetric X-braced truss in the same
    family as the accepted LP reproduction and Fig. 3a, and clearly not the blurred
    `rmin = 2.5` family. See `figures/topology_comparison.png`.
27. **Is the endpoint natively converged?** **No.**
28. **Is the endpoint still move-saturated?** **Yes** — `max|drho| = 0.00990` = the move
    limit, against `tolOuter = 1e-3`.
29. **Does the trajectory show a limit cycle?** Not a clean one. Last-100 ω₁ std 0.085,
    peak-to-peak 0.38, with a mildly **regressive** trend (−2.6e-04/iteration over the last
    200). Classified **MOVE-SATURATED_FINITE_TRAJECTORY**.
30. **LP final ω₁ at matched mesh/config.** **168.240** (`lprmin1.2`, 160x20, `rminEl = 1.2`).
31. **MMA final ω₁.** **168.750** at outer 752 (168.513 at outer 400).
32. **LP vs MMA topology.** Both fail the study gate at 160x20; MMA is worse on every metric
    (16 vs 14 components, 15 vs 13 detached, worst island 14 vs 4 elements, grayness 0.1442
    vs 0.1321). Not maturity-matched (outer 400 vs 1600).
33. **LP outer iterations.** **1600.**
34. **MMA outer iterations.** **752.**
35. **LP calls.** **1600** — exactly one `linprog` solve per outer, verified for all 1600.
    HiGHS/simplex iterations are *not* counted as inner iterations anywhere in this audit.
36. **MMA cumulative inner iterations.** **107 373.**
37. **Principal cause of legacy MMA failure.** Two, jointly: the `rmin = 2.5` filter selected
    the non-bimodal basin, and unrestricted/cap-hit full-box increments were consumed without
    regard to their failed inner status.
38. **Principal reason BASE_mma behaves better.** The filter radius (1.2, not 2.5) and the
    bounded 0.01 move with no rejection layer, supported by a workable 300-iteration inner
    budget, a scale-invariant relative inner test, and `lamref` normalisation of the MMA
    subproblem. **Not "MMA was fixed"** — the inner solver is the same published `mmasub`.
39. **Is nested MMA more paper-literal than the Eq. (22) LP?** **Yes, on exactly two counts**
    — the use of MMA (sec. 3.5.3) and retention of the full Eq. (25d) coupling. On nine other
    controls both routes are equally inferred.
40. **Is LP still paper-sanctioned?** **Yes, unambiguously.** Eq. (22) is printed in the
    paper and attributed to Krog & Olhoff (1999).
41. **Does BASE_mma demonstrate that nested MMA can work?** **Yes** — decisively, and it
    refutes any claim that full-coupling nested MMA is inherently unviable.
42. **Does it demonstrate historical 2007 code fidelity?** **No.** The paper omits the filter
    radius, mesh, move limit, all stopping rules, the multiplicity tolerance, the
    tensor-filtering semantics and the asymptote handling.
43. **Are the inner stopping semantics sufficiently justified?** **Partially.** The relative
    test is well-motivated and declared a reconstruction in the source. But 16.7% of `N=2`
    solves hit the cap, and the inner histories are discarded, so whether those subproblems
    were materially unresolved **cannot be determined from this evidence**. Outstanding
    qualification.
44. **Is a confirmation run required?** **Yes — but only to make the MMA evidence citable,
    not to decide the route.** Reproducibility is already proven bit-identically (M15).
45. **Is a second mesh required before route selection?** **No.**
46. **What campaign files/semantics would change if MMA is selected?** Fifteen items,
    enumerated in `CAMPAIGN_IMPACT.md` — of which four are methodological: iteration
    accounting, status taxonomy, trajectory storage (re-opening the Phase-2B/2E precision
    chain), and the paper table layout.
47. **Recommended Olhoff route.** **Eq. (22) LP as principal; nested MMA as a reported
    secondary paper-native variant.**
48. **Is the route decision ready for independent review?** **Yes.**
49. **Is the nine-mesh campaign authorized?** **No.**
50. **Exact next action.** Obtain an independent route-selection review of this audit. Do not
    edit the methodology, the contract, `olhoffOptStabilized.m`, or any campaign file. If the
    secondary-variant recommendation is accepted, commission the single completed 160x20
    confirmation run specified in `CONFIRMATION_RUN_REQUIREMENTS.md` — configuration
    unchanged, logging only.

---

# FINAL VERDICT

    NESTED MMA SHOULD REMAIN SECONDARY; LP REMAINS PRINCIPAL ROUTE

    ITERATION-EFFICIENCY METHODOLOGY: NOT YET UPDATED

    PRODUCTION STATUS: BLOCKED
