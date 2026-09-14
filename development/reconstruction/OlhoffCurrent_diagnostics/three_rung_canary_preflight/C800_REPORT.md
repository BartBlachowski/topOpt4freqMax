# C800_REPORT — canary 2, 800×100

## Verdicts

```
C800_THREE_RUNG_CONTROLLER_PASS
C800_ENDPOINT_SCIENTIFICALLY_SUSPICIOUS
C800_RUNTIME_BEHAVIOR_EXPLAINED
```

Three separate questions, deliberately not collapsed. The controller did
everything it was designed to do. The design it produced is not, by itself,
scientifically convincing. The runtime is fully accounted for.

## The run

| | |
|---|---|
| mesh / NE / free DOF | 800×100 / 80 000 / 161 798 |
| config hash | `7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe` (= frozen) |
| `+impl` tree | `edbfe47e…52cb` |
| status | **CONVERGED** |
| outer iterations | 468 (cap 1600, headroom 1132 — **no cap hit**) |
| inner MMA total | 10 404; max 37 per outer; **0 non-converged** |
| final move / stage | 0.01 / 3 |
| ω₁ / ω₂ / ω₃ | 161.94583808332942 / 161.95148607903909 / 382.47940150449614 |
| gap12 / gap23 | **3.4875831182270845e−05** / 1.3616912123784417 |
| M_nd / grayness | **34.41232077776818 %** / 0.377625 |
| volume | 0.49999367812652545 |
| terminal max\|dρ\| / ‖dρ‖₂ | 9.556946e−04 / 2.113134e−02 (ε = 0.25) |
| multiplicity N final | 2 |
| wall | 7554.096479416667 s (2 h 05 m 54 s) |
| ρ SHA-256 | `50e4e625c330e1aadddf75ea121966213173b5817af7337d2ef3603e287bb9c7` |
| trajectory SHA-256 | `aef7af1c3ec669b41d6ff6799aaaea4bc35feac0d20ff739f91a532bece012eb` (536.1 MB) |

Trajectory rebuilt from `res.diag.drho` and **proved exact**: bitwise equal to
`res.rho` at the end, clamp-displacement residual 5.551e−17.

## 1. Controller — PASS

### Events

| event | iteration | window | branch | move | amp | amp/ε | med₂₀cos | med₂₀net | persistence |
|---|---|---|---|---|---|---|---|---|---|
| S1 declaration | 390 | 371–390 | **B** | 0.04 | 0.218466 | 0.87387 | 0.9434 | 0.9577 | nB = 20 |
| S2 declaration | 429 | 410–429 | **B** | 0.02 | 0.059329 | 0.23731 | 0.9172 | 0.8856 | nB = 20 |
| S3 declaration (terminal) | 468 | 449–468 | **B** | 0.01 | 0.021131 | 0.08453 | 0.8208 | 0.9032 | nB = 20 |

Stage starts 1 / 391 / 430. Two descents, both consumed by the exhaustion rule
at the iteration after declaration. Every declaration reached full persistence
P = 20 on branch B; counter A never fired at this mesh either.

### The structure is identical to every validated mesh

| mesh | S1 declaration | S2 | S3 | total | pattern |
|---|---|---|---|---|---|
| 160×20 | 102 | 39 | 39 | 180 | S1 + 78 |
| 240×30 | 206 | 39 | 39 | 284 | S1 + 78 |
| 320×40 | 274 | 39 | 39 | 352 | S1 + 78 |
| 400×50 | 388 | 39 | 39 | 466 | S1 + 78 |
| 480×60 | 308 | 39 | 39 | 386 | S1 + 78 |
| **800×100** | **390** | **39** | **39** | **468** | **S1 + 78** |

Six meshes spanning NE = 3 200 → 80 000, a factor of 25. Every one: a single S1
declaration, then both lower rungs at the minimum possible dwell of
39 = (W−1) + P. **No new controller behaviour appeared at either canary.**

S1 length remains non-monotone in mesh (102, 206, 274, 388, 308, 390) — the
480×60 canary already showed this and 800×100 confirms it is not an accident.
The pre-run budget projection of ≈ 1130 outer (`PREREGISTRATION.md` §4) was
wrong by 2.4×; the run needed 468. That projection was explicitly not an
acceptance criterion and no gate read it.

### The ten gates

All ten hold, evaluated mechanically (`evidence/analysis_800x100.json`).

| # | gate | result |
|---|---|---|
| 1 | three-rung policy actually used | **PASS** — runtime hash = frozen hash, levels `[0.04 0.02 0.01]`, both switches on |
| 2 | valid stage sequence | **PASS** — 1 → 2 → 3, monotone, no skips |
| 3 | terminal persistent E at move = 0.01 | **PASS** — iteration 468, stage 3, CONVERGED |
| 4 | no hidden legacy/beta transition | **PASS** — beta holds neither authority; exactly 2 descents, both exhaustion events |
| 5 | no numerical failure | **PASS** — no NaN, eigensolver never failed |
| 6 | inner MMA solves acceptable | **PASS** — 0 of 468 non-converged; max 37 against a cap of 500 |
| 7 | no substantial unresolved terminal evolution | **PASS** — ω₁ +0.014 %, M_nd −0.064 pts over the last 20 iterations; amp/ε = 0.0845, inside the validated range [0.0327, 0.1036] |
| 8 | no config drift | **PASS** |
| 9 | telemetry / evidence complete | **PASS** — 468 rows, no missing field, rebuild exact |
| 10 | no outcome-driven intervention | **PASS** — one run, nothing touched |

### Did the strange legacy low-iteration behaviour repeat?

**No.** The legacy 800×100 run stopped at 170 outer iterations — *fewer* than
its own 720×90 run (223) — because the `designChange` test admitted convergence
one iteration after a move halving. The three-rung canary ran 468 iterations and
stopped only on a 20-iteration persistent exhaustion window at the terminal rung.

On this same realized trajectory the legacy `designChange` rule would first have
admitted convergence at **iteration 279** — 111 iterations and one full ladder
descent before the frozen rule considered the design exhausted. (A predicate
replay on the realized path, not a counterfactual trajectory.)

### Was there a low-amplitude cancellation hole?

No. Branch B requires amp < ε *and* med₂₀cos > 0 — coherent motion, not merely
small motion. All three declarations carry high positive med₂₀cos (0.9434,
0.9172, 0.8208) and med₂₀net (0.9577, 0.8856, 0.9032), i.e. the design was
moving in a consistent direction with little cancellation, and simply had little
left to move. Branch A, the cancellation branch, never fired at this mesh.

## 2. Scientific endpoint — SUSPICIOUS

This verdict is **not** about the controller, which behaved correctly, and not
about the comparison with legacy, which the three-rung run wins decisively.

### Against legacy 800×100 it is a large improvement

| | legacy 800×100 | three-rung 800×100 | change |
|---|---|---|---|
| outer / inner | 170 / 3 713 | 468 / 10 404 | +298 / +6 691 |
| final move / stage | 0.02 / 2 | **0.01 / 3** | reaches the last rung |
| ω₁ | 153.3020068531485 | **161.94583808332942** | **+8.64 (+5.64 %)** |
| M_nd | 50.65606664059261 % | **34.41232077776818 %** | **−16.24 points** |
| topology IoU @ 0.5 | — | 0.7021 vs legacy | 15.45 % of elements flip side |
| density L1 distance | — | 0.0989 | — |
| wall | 1987.06 s | 7554.10 s | ×3.80 (**not like-for-like**) |

`figures/FIG_12_topology_legacy_vs_three_rung_800.*` shows the difference
plainly: the legacy design is a gray blur; the three-rung design has resolved
black flanges, a defined central void and distinct diagonal members.

### But as a design, in absolute terms, it is poor — and getting worse with refinement

| mesh | three-rung M_nd % | three-rung ω₁ | legacy M_nd % | advantage |
|---|---|---|---|---|
| 160×20 | 12.756 | 169.975 | 13.402 | −0.65 pts |
| 240×30 | 12.917 | 167.039 | 15.604 | −2.69 pts |
| 320×40 | 12.940 | 166.426 | 23.360 | −10.42 pts |
| 400×50 | 15.373 | 166.452 | 32.328 | −16.96 pts |
| 480×60 | 26.342 | 163.932 | 34.672 | −8.33 pts |
| **800×100** | **34.412** | **161.946** | 50.656 | −16.24 pts |

Under the **correct, validated controller**, held fixed across all six meshes:

* **M_nd rises monotonically and accelerates** — 12.8, 12.9, 12.9, 15.4, 26.3,
  34.4 %. A third of the 800×100 domain is intermediate density.
* **ω₁ falls monotonically** — 169.98 → 161.95, with no sign of an asymptote.
* Most telling: the three-rung 800×100 design (M_nd 34.41 %) is **as gray as the
  legacy 480×60 design** (34.67 %). Refining the mesh 2.8× while running the
  correct controller bought a design no more discrete than a coarser mesh
  produced under the wrong one.

The terminal window is genuinely flat (ω₁ +0.014 %, M_nd −0.064 pts over 20
iterations), so this is **not** a case of stopping too early. The controller
stopped where the design had stopped moving. The design simply stopped moving
while still a third gray.

`figures/FIG_11_topology_480_vs_800.*` shows both canaries converge to the
**same topology family** — flanges, end blobs, X-bracing, central void. The
800×100 version is that same design with larger and grayer end regions. So
topology is mesh-consistent; discreteness is not.

### Why "suspicious" and not "credible"

`C800_ENDPOINT_CREDIBLE` would assert the endpoint is scientifically
believable as a converged design. It is not: neither the objective nor the
discreteness shows convergence under refinement, and the density field is a
third intermediate. `_INCONCLUSIVE` would understate it — the measurements are
complete and unambiguous; it is the *quality* that fails, not the evidence.

**No causal claim is made** about why. Candidates are spectral (see §3),
discretization, the fixed p = 3 without continuation, or the sensitivity-filter
formulation at a fixed physical radius. This study changed none of them and
tested none of them.

## 3. Next-mode warnings

| | 480×60 | 800×100 | legacy 800×100 |
|---|---|---|---|
| warnings | 4 | **81** | 81 |
| of iterations | 386 | 468 | 170 |
| fraction | 1.04 % | **17.31 %** | 47.65 % |
| first warning | 12 | 26 | — |
| **last warning** | — | **113** | — |
| stage 1 / 2 / 3 | 4 / 0 / 0 | **81 / 0 / 0** | — |
| overlap terminal window (449–468) | none | **none** | — |

The decisive detail: at 800×100 the 81 warnings are a **contiguous early
transient, iterations 26–113**. Nothing in the remaining **355 iterations**
(76 % of the run), nothing in stage 2 or 3, nothing near the terminal
declaration. The endpoint spectrum is clean — gap23 = 1.3617, ω₃ well separated.

So on all five of Part H's diagnostic questions: the regime **appears early**,
does **not** increase late, does **not** overlap terminal E, does **not**
correlate with branch A/B behaviour (all declarations are branch B, 335+
iterations after the last warning), and its resolution *precedes* the phase in
which the endpoint quality is determined.

Classification and its bounds are in `MULTIPLICITY_WARNING_AUDIT.md`.

## 4. gap12 collapses to near-degeneracy

ω₁ = 161.94583808 and ω₂ = 161.95148608 — **gap12 = 3.49e−05**, degenerate to
five significant figures. Trajectory: 2.88e−03 (it 117) → 9.60e−03 (it 234) →
7.06e−03 (it 351) → 4.16e−05 (it 468).

This is the **expected physics**: maximizing ω₁ drives the first two modes
together, and a bimodal optimum is the characteristic outcome of this problem
class. The fixed N = 2 subspace treatment with diagonal offsets and off-diagonal
terms exists precisely for this, and `multN` is 2 at the endpoint.

Worth noting as a mesh trend rather than a defect: gap12 at the three-rung
endpoint is 0.2232 (320×40), 0.1298 (480×60), 3.49e−05 (800×100). The optimum
becomes bimodal only at fine mesh. Legacy shows the same direction
(0.0697 → 0.0060 from 480 to 800).

## 5. Runtime — EXPLAINED

| | value |
|---|---|
| total wall | 7554.096 s |
| Σ tOuter | 7552.475 s |
| mean wall / outer | 16.1378 s |
| assembly + eigensolve / outer | 1.18725 s (**7.36 %**) |
| gradients / outer | 0.04893 s (0.30 %) |
| inner MMA / outer | 14.89901 s (**92.32 %**) |
| other (total) | 1.208 s (0.02 %) |
| mean inner steps / outer | 22.23 |
| mean s per MMA step | 0.67020 s |

`T_total = N_outer × T_outer_avg` = 468 × 16.1378 = 7552.5 s ✓

### Per stage

| stage | move | iterations | tOuter/it | tEig/it | tGrad/it | tInner/it | inner/it | s / MMA step |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.04 | 390 | 13.9432 | 1.1636 | 0.0487 | 12.7284 | 21.71 | 0.58628 |
| 2 | 0.02 | 39 | 20.9952 | 1.2994 | 0.0499 | 19.6436 | 22.59 | 0.86958 |
| 3 | 0.01 | 39 | 33.2257 | 1.3120 | 0.0502 | 31.8607 | 27.08 | 1.17667 |

The move-limit effect first seen at 480×60 **repeats**: seconds per MMA step
doubles from stage 1 to stage 3 (0.586 → 1.177, ×2.01) while assembly+eigensolve
and gradient times stay flat. Stages 2 and 3 are 16.7 % of iterations but
28.0 % of wall time.

### 480 → 800 scaling (stage 1, NE ratio 2.778)

| component | 480×60 | 800×100 | ratio | exponent in NE |
|---|---|---|---|---|
| wall / outer | 5.9597 | 13.9432 | 2.340 | 0.832 |
| assembly + eigensolve | 0.2721 | 1.1636 | 4.276 | **1.422** |
| gradients | 0.0167 | 0.0487 | 2.915 | 1.047 |
| inner MMA / outer | 5.6693 | 12.7284 | 2.245 | 0.792 |
| s per MMA step | 0.2915 | 0.5863 | 2.011 | 0.684 |

Consistent with the legacy nine-mesh fits (assembly+eigensolve 1.142, gradients
0.895, s per MMA step 0.724). Nothing anomalous; every component grows, the
eigensolve grows fastest and its share rises from 3.75 % at 480×60 to 7.36 % at
800×100, and the nested MMA still dominates everywhere.

**Total wall is 3.80× the legacy 800×100 run.** That is not a controller cost
measurement: the canary runs 2.75× more iterations, spends 16.7 % of them in
stages that cost 1.5–2.4× a stage-1 iteration, and retains a full trajectory
(`diagnostics = true`) which the legacy campaign did not. See
`PERFORMANCE_DECOMPOSITION.md` §5.

### Host

| probe | load avg | dgemm 1200³ | sparse solve 200k |
|---|---|---|---|
| C800 pre | 3.19 | 0.013 s | 0.003 s |
| C800 post | 2.85 | 0.012 s | 0.003 s |

Identical to the C480 probes (0.013/0.012 s, 0.003 s). Both canaries ran on a
host in materially the same state; no thermal or cache control was attempted and
none is claimed.

## 6. Files

`runs/C800x100_three_rung_iterations.csv` (468 × 55, frozen `cv_export` schema) ·
`runs/C800x100_three_rung_supplement.csv` (468 × 21) ·
`runs/C800x100_three_rung_record.json` ·
`evidence/analysis_800x100.json` · `evidence/stage_timing_both.json` ·
`evidence/fixedwork_800x100.json` · `evidence/preflight_800x100.json` ·
`evidence/hostprobe_C800x100_three_rung_{pre,post,fixedwork}.json` ·
`evidence/topology_800_comparison.json` ·
raw trajectory `evidence/three_rung_canary_preflight/C800x100_three_rung_trajectory.mat` ·
figures 2, 3, 4, 5, 6, 7, 8, 9, 10 for this mesh, plus 11 and 12.
