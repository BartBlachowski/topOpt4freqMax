# PREREGISTRATION — is the four-rung move ladder still justified?

A **zero-scientific-run** offline audit. No optimization of any kind is executed.
Every number comes from trajectories that already exist.

| | |
|---|---|
| Repository | `/Users/piotrek/Programming/topOpt4freqMax` |
| Branch | `benchmark-methodology-r2` |
| **HEAD at task start** | `1438aa3f4bd934f5b588587ef65f4f2ca35bac1c` |
| Tree at task start | dirty, 17 paths (the previous study's uncommitted deliverables) |
| `+impl/` tree SHA-256 | `edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb` (75 files) |
| MATLAB | not required; audit is Python/offline |
| Prior status | `TWO_BRANCH_CONTROLLER_PARTIALLY_VALIDATED` · `PRODUCTION_CONTROLLER_NOT_PROMOTED` · `NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` |

---

## 1. The question

> After `move = 0.04` has already reached the frozen `A OR B` exhaustion event,
> do the lower rungs `0.02 → 0.01 → 0.005` provide scientifically material
> additional benefit that justifies their computational cost and failure risk?

## 2. The counterfactual

```
SINGLE-STAGE POLICY  (S)
    move = 0.04, held
    terminate at the first persistently satisfied frozen E = A OR B
    no descent to 0.02, 0.01 or 0.005
```

compared against

```
FOUR-RUNG CANDIDATE  (F)   the controller already validated in
                           two_branch_controller_validation
PRODUCTION           (P)   beta-stall ladder, the frozen baselines
```

**No simulation is required to know S.** Under the single-stage policy the run
is bitwise the four-rung candidate up to the first declaration, because the two
policies differ only in what happens *after* that event, and the exhaustion
predicate is a function of the trajectory alone. S is therefore the recorded
state of the existing candidate trajectory at its first declaration. The prefix
argument is stated in full and checked in Phase 7 of the brief; it is not
assumed.

## 3. Meshes and sources

| mesh | role | source | what it can support |
|---|---|---|---|
| 160×20 | primary | `C160x20_trajectory.mat` + `runs/C160x20_iterations.csv` | full S vs F vs P |
| 320×40 | primary | `C320x40_trajectory.mat` + CSV | full S vs F vs P |
| 400×50 | primary | `C400x50_trajectory.mat` + CSV | full S vs F vs P |
| 240×30 | supporting | `two_branch_maturity_240/METRICS.json` **scalars only** | the value of continuing at `move = 0.04` past the event; **no** S vs F, because no four-rung 240×30 run exists and none will be made |

Production baselines are the frozen ones already recorded in
`two_branch_controller_validation/evidence/baselines.json`.

**Known missing evidence, declared now:** `runD_240x30.mat`, `runB_320x40.mat`,
`runC_400x50.mat`, `fm_analysis.mat`, `tb_analysis.mat`, `phaseA_stats.mat`,
`dr_analysis.mat`, `runA_400x50.mat`, and the four `move_stop` `.mat` arms are
absent from this machine. They are **not** regenerated. Where they would have
been needed the audit reports a gap.

## 4. Event extraction (frozen)

The exhaustion event is recovered by **independently recomputing** the frozen
rule from the raw trajectory (`RHO`, `hist.dxNorm2`), not by reading the
controller's log:

```
tol(NE) = 0.05*sqrt(NE/3200)          W = 20, P = 20, W_np = 10
d_k     = rho_k - rho_{k-1}           amp(k) = ||drho_k||_2 = hist.dxNorm2(k)
cos(k)  = <d_k, d_{k-1}> / (||d_k|| ||d_{k-1}||)
net(k)  = n2(rho_k - rho_{k-10}) / sum_{j=k-9..k} n2(d_j),   n2 = ||.||/sqrt(NE)
med20   = trailing 20-median, 'omitnan', defined only for k >= 20
A(k) = med20 cos < 0 AND med20 net < 0.5 AND amp >= tol
B(k) = amp < tol AND med20 cos > 0
declaration = first k at which either branch has held for 20 consecutive iterations
```

The recomputation must agree with the controller's recorded `exA`/`exB` trace
**element-wise over the whole `move = 0.04` prefix**, and the declaration must
equal the recorded one. Disagreement ⇒ `MOVE_LADDER_NECESSITY_INCONCLUSIVE`.

## 5. Metrics

At S and at F, per mesh: outer iterations, cumulative inner MMA iterations, wall
time, `ω₁`, `ω₂`, relative gap, volume, `M_nd`, gray fraction, mid-density
fraction, `max|Δρ|`, `max|Δρ|/move`, `‖Δρ‖₂`, RMS `Δρ`, `cosθ`, net/path, bound
fraction, β-stall state, native-stop state, subspace size, and the SHA-256 of the
density vector.

Incremental lower-rung value is `F − S`; incremental cost is the outer
iterations, inner MMA iterations and wall seconds spent after the first descent.
Density-field distance is reported as mean `|Δρ_e|` and `‖Δρ‖₂/√NE` between the
S and F designs.

## 6. Materiality thresholds — fixed now

Anchored to scales this project has already committed to, **not** to any number
computed in this audit. Each is deliberately an order of magnitude below the
project's own "material" bar, so the test is generous to the ladder.

| quantity | material if | anchor |
|---|---|---|
| `M_nd` | lower rungs improve `M_nd` by **≥ 2 % relative** | the controller study preregistered **20 %** relative as "materially better" at the fine meshes; this is one tenth of that |
| `ω₁` | lower rungs improve `ω₁` by **≥ 0.10 % relative** | the controller study preregistered **1 %** relative as the acceptable-degradation bound; one tenth of that, and ≈ 0.17 in absolute ω₁ here, far above numerical noise |
| topology | gray or mid-density fraction changes by **≥ 0.01 absolute**, or mean \|Δρ_e\| between S and F **≥ 0.01** | the project reports these to 4 decimals and has called 0.11 absolute a large change; 0.01 is one tenth of the smallest change it has treated as meaningful |
| volume feasibility | \|volume − 0.5\| worsens by **≥ 1e-5** | production and candidate both achieve ≈ 1e-6–1e-7 |
| multiplicity / physics | subspace size leaves 2, mode order changes, ω₂ ≤ ω₁, or a NaN/Inf appears | qualitative; gap magnitude alone is **not** material, because the objective is ω₁ |
| cost domination | a rung block costs **≥ 2×** the outer iterations used to reach S while delivering sub-material benefit on every metric above | the ladder must earn its cost, not merely change ρ |
| failure risk | any rung sequence that produces `CAP_HIT` or a non-terminating stage | a controller that cannot stop is a defect regardless of quality |

A lower rung is **not** retained merely because it changes ρ numerically.

## 7. Decision rule — fixed now

Count the primary meshes (160×20, 320×40, 400×50) on which the lower rungs
deliver material benefit by **at least one** of the `M_nd`, `ω₁`, topology,
volume or multiplicity criteria:

* **`FOUR_RUNG_LADDER_NECESSARY`** — material on **≥ 2 of 3**.
* **`FOUR_RUNG_LADDER_PARTIALLY_USEFUL`** — material on **exactly 1 of 3**.
* **`FOUR_RUNG_LADDER_NOT_JUSTIFIED`** — material on **0 of 3**, *and* the cost
  or failure-risk criterion is met on at least one mesh.
* **`MOVE_LADDER_NECESSITY_INCONCLUSIVE`** — event verification fails, or
  missing evidence prevents evaluating a primary mesh.

Next-step verdict `SINGLE_STAGE_POLICY_PREREGISTRATION_JUSTIFIED` requires all
seven of the brief's Phase-19 conditions; otherwise
`MORE_MOVE_LADDER_EVIDENCE_REQUIRED` or `RETAIN_MOVE_LADDER_PENDING_REDESIGN`.

## 8. Disclosure — what was already visible before this file was frozen

This audit is **not blind to its own headline**, and pretending otherwise would
be dishonest. Before freezing, the following were already published in
`two_branch_controller_validation`:

* the F-state finals (`M_nd` 12.7041 / 12.9233 / 15.3311; `ω₁` 170.0113 /
  166.4267 / 166.4563) and the P baselines;
* the S-state `ω₁` and `M_nd` at the declaration iteration, in that study's
  transition-audit table (160×20: 168.9804 / 13.0364; 320×40: 166.4216 /
  13.0121; 400×50: 166.4176 / 15.6649);
* the prose claim that ≥ 98 % of fine-mesh gain is banked before first descent;
* that 320×40 ended `CAP_HIT`.

So the *direction* of the answer is foreseeable. What is **not** yet computed,
and what this preregistration governs, is: the full S-state extraction, the
rung-by-rung decomposition, the cost fractions, the density-field distances, the
multiplicity comparison, the exact recomputed percentages, and — decisively —
the **thresholds** above, which are fixed here and anchored to prior commitments
rather than chosen once the table is on screen.

The 160×20 mesh is explicitly **not** assumed to follow the fine meshes
(brief Phase 12); its numbers are computed and reported on their own.

## 9. What this task will not do

No scientific optimization run of any mesh, fixed-move arm, production arm or
controller arm. No modification of `A`, `B`, `W`, `P`, `tol`, the union, the
persistence, or the move values. No Branch C, no terminal exception, no
mesh-dependent condition, no hybrid ladder, no alternative initial move, no new
continuation law. No single-stage production implementation, no promotion, no
nine-mesh campaign. No projection, no change to `R`, `p`, mass interpolation or
`q`. No regeneration of lost evidence and no rewriting of prior history.

Production remains unchanged; the campaign remains
`NINE_MESH_PERFORMANCE_CAMPAIGN_BLOCKED` regardless of this audit's outcome.
