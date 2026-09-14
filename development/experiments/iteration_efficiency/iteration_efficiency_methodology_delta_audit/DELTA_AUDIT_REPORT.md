# Independent delta audit — iteration-efficiency study, Phase 1D

**Reviewer:** the same independent methodologist who issued the original hostile audit
(2 CRITICAL, 8 MAJOR, 9 MODERATE, 5 MINOR; verdict *NOT READY — MAJOR METHODOLOGICAL REVISION
REQUIRED*).

**Scope:** closure verification of the Phase-1C repair, not a second full review.

**Constraints honoured:** read-only. Nothing in
`analysis/iteration_efficiency_methodology_audit/` or
`analysis/iteration_efficiency_study_design/` was modified; no MATLAB, no optimizer, no new
trajectory, no frozen artifact touched, no threshold tuned, and no defect silently repaired —
every defect found is reported, not fixed.

---

## Verdict

**DO NOT FREEZE — SPECIFIC BLOCKERS REMAIN** (blockers **N1**, **N2**).

This is a narrow verdict. All 24 original findings are **CLOSED**, both CRITICAL repairs
verify independently against frozen evidence and source, and both deliberate departures from
my original minimum corrections are **ACCEPTED**. No accepted repair is reopened, and no
redesign is requested. Two blockers remain, each with a forced, parameter-free correction; two
further corrections are non-blocking.

---

## What I verified, rather than read

I did not accept the closure ledger as proof. The load-bearing claims were re-derived:

| Claim | How verified | Result |
|---|---|---|
| Repaired topology gate is satisfiable | **Independent reimplementation** (union-find, not the author's `scipy.ndimage` path) rerun over frozen Olhoff trajectories | 160x20, 240x30, 640x80, 720x90 reproduce the author's table **exactly** |
| Aggregate clause was the binding constraint | Same recomputation, scoring T1 both ways at 640x80 | T1-with-aggregate **0.56%** vs per-component-only **45.74%** |
| Original C1 diagnosis | Final-state metrics at 640x80 | `det_max = 4`, `det_tot = 20` — my own original anchor, reproduced |
| Gate still rejects pathology | Real failure breakdown + 5 synthetic probes | 53/1067 rejected; first at iter 55 on a 64-element body; severed/isolated/144-element-blob all fail; 49-element speck passes |
| `A_sig` is mesh-invariant | Recomputed as share of solid volume | **0.2500% at every mesh**; retired 5-element rule swung 0.3125%→0.0125% |
| Yuksel Stage-1 handoff | `top99neo_inertial_freq.m:237` | `x = xPhys;` confirmed — the repair's corrected statement is accurate |
| `nInner` is not a solver iteration | `innerLoopLP.m:65,67` | `[x, ~, flag] = linprog(...)`; `st.nInner=1` hard-coded — 4th output genuinely unavailable |
| Proposed has no in-loop eigensolve | `topopt_freq.m` `dueFlags` logic | With `ua==0` + `'solid'`, `dueFlags` is false every iteration — confirmed |
| Even-`nely` precondition | `model2D.m` case `'mid'` | Explicit `error(...)` on odd `nely` — real precondition |
| `volume_residual` is absolute | `study_evaluate_design.m:10` | `mean(x) - volfrac` — confirmed; gate correctly names `H.rV` |
| Evaluator identities | `study_evaluate_design.m` `solve_modes` | E1 linear mass (Proposed), E2 piecewise `x^6` (Yuksel), E3 `rho_min=1e-3` clipped (Olhoff) |
| Evaluator agreement | Recomputed over `common_evaluators.csv` | **0.429%** max spread, ordering preserved at every mesh — matches my original ~0.43% |
| Olhoff quality lead | Recomputed over `common_evaluators.csv` | **6.2%–8.5%** over Proposed on **8** meshes — **not** the 6.1–7.2%/nine-mesh figure in the package (see N4) |

---

## Closure summary

| Severity | Count | CLOSED | PARTIAL | OPEN |
|---|---:|---:|---:|---:|
| CRITICAL | 2 | 2 | 0 | 0 |
| MAJOR | 8 | 8 | 0 | 0 |
| MODERATE | 9 | 9 | 0 | 0 |
| MINOR | 5 | 5 | 0 | 0 |

Per-finding evidence is in `FINDING_CLOSURE_VERIFICATION.csv`; the two CRITICAL and eight
MAJOR rechecks are in `CRITICAL_FINDINGS_RECHECK.md` and `MAJOR_FINDINGS_RECHECK.md`.

### Deliberate departures

- **Mo7 — retire `a_res` entirely rather than rename it: ACCEPTED.** Strictly stronger than my
  minimum correction. I verified the replacement actually fixes the objection (constant 0.25%
  of solid volume across the family) rather than relabelling it.
- **Mo6 — decline the filter-footprint sensitivity: ACCEPTED.** My original suggestion would
  have reintroduced filter-derived topology scales, which the C1/Mo7 closure prohibits. The
  substituted 1x1/3x3 FE patch scales satisfy the finding's real requirement — a sensitivity
  able to probe the **permissive** direction. The author also did the other half of my
  either/or by stating T0's known outcome up front.

I record explicitly that neither departure is penalised for differing from my original wording.

---

## New findings

Four, against a deliberately high bar. Two are blocking.

| ID | Sev | Blocking | Summary |
|---|---|---|---|
| **N1** | MAJOR | **yes** | Measurement horizon not tied to `b_ref`; extension rule is the logical negation of the stabilisation rule, so a stabilised cell is denied its tranche. Method-correlated censoring created by the C2 repair. |
| **N2** | MAJOR | **yes** | Three live contradictions in the master protocol document's narrative: evaluators "remain supplementary" (vs M4), `k_enter`/`k_cert` as co-equal fits (vs Mo1), "T1/T0 sensitivity" as a live control (vs C1/Mo6). |
| **N3** | MODERATE | no | Aggregate detached area — the deleted CRITICAL clause's own quantity — appears in no paper-facing table, yet reaches 2.633% of solid volume among accepted states at 640x80. |
| **N4** | MODERATE | no | The mandatory "6.1–7.2% over nine meshes" disclosure is wrong: actual 6.2%–8.5% over Proposed and 5.9%–7.7% over Yuksel across eight meshes. **The error originates in my own original M3 wording** and understates the gap the disclosure exists to expose. |

Details and minimum corrections in `NEW_FINDINGS.csv`.

Nothing was raised for stylistic preference, alternative-but-reasonable methodology, or
optional robustness. Two items that could have been findings are recorded instead as
non-blocking qualifications: the E2/E3 mass-law near-duplication (M4), and the 800x100
labelling (D14).

---

## D15 — expected-result firewall, method-blind

Re-read with the methods as A, B, C. The repaired protocol survives every enumerated outcome:

| Outcome | Survives? | Why |
|---|---|---|
| C has the fewest outer iterations | yes | counts are per-method units, never equated; F3 sits adjacent to F2 with the per-update cost note |
| A has the most method-level iterations | yes | no ordering is assumed anywhere; `k_native` and `k_gate` are printed beside every count |
| B has the lowest wall time | yes | timing is secondary, platform-scoped, and never combined into a score |
| C has substantially better quality | yes | F4, Main Table 2 and the mandatory best-observed benchmark exist for exactly this; frozen evidence already says so |
| rankings cross with requested quality | yes | the q-family is co-primary precisely so a crossing is visible; explicitly declared a reportable result |
| exponents change with quality fraction | yes | `p(q)` reported per level; material dependence declared a scientific result, not a nuisance |
| a method never reaches some q levels | yes | `QUALITY_NOT_REACHED` / `REFERENCE_NOT_ESTABLISHED` are publishable outcomes with visible censoring |

One rule is **not** indefensible but is *unfair* under a specific outcome, and that is N1: a
method whose reference stabilises late is censored for having been assigned a smaller
measurement budget. Under method-blind reading this is the only asymmetry I could not justify,
and it is why the freeze is withheld.

Residual asymmetries that are defensible and disclosed: R's structural preference for early
plateau (M2), the count-raising method-specific gates (M8, exposed via `k_gate`), the unequal
proportional burden of `P-1` (Mo3), and the exact-count projection's harsher treatment of gray
fields (F06).

---

## Assessment of the repair as a whole

The Phase-1C package does what a targeted repair should. The two CRITICAL repairs are not
documentation gestures: the topology gate was re-derived from FE geometry rather than
relabelled, and the reference construction was moved onto a separate trajectory with the
decisive property — **no cap fallback** — that severs the horizon-to-quality-bar causal chain
at its first link. The author twice chose a stricter repair than I requested and documented
both departures rather than quietly diverging.

The failures are of harmonisation, not of design. N1 is the C2 repair's own consequence left
unpropagated into the budget contract; N2 is three narrative sections left behind when the
rule-bearing sections were rewritten. N4 is an error I introduced and the author faithfully
inherited.

**The methodology is close.** Once N1 and N2 are corrected — neither requires a new scientific
choice — the experiment will be specified fairly enough that implementation can proceed without
a further decision after seeing results.

---

## Deliverables in this directory

- `DELTA_AUDIT_REPORT.md` — this report
- `FINDING_CLOSURE_VERIFICATION.csv` — all 24 findings with independent delta disposition
- `CRITICAL_FINDINGS_RECHECK.md` — C1, C2 in full, including both departures
- `MAJOR_FINDINGS_RECHECK.md` — M1–M8
- `INTERNAL_CONSISTENCY_CHECK.md` — obsolete-concept sweep, historical vs live
- `PHASE2_READINESS.md` — evidence matrix, 800x100 artifact, D16 classification
- `METHODOLOGY_FREEZE_GATE.md` — freeze decision, blockers, and what the freeze will cover
- `NEW_FINDINGS.csv` — N1–N4
- `independent_gate_recheck.py` — independent union-find recheck (read-only, reproducible)
