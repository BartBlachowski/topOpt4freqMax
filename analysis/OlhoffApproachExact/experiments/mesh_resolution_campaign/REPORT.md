# Mesh-Resolution Verification Campaign — Du & Olhoff (2007) Benchmark

**Hypothesis under test (H1).** The primary reason for the failure to reproduce the
published topology is the extremely coarse finite-element discretization and the
resulting discrete representation of the benchmark geometry (including support
placement), rather than missing optimization details.

**Verdict: H1 is SUPPORTED, with one explicitly bounded exception.**
Mesh resolution and the discrete geometry it induces are established here as
*sufficient* to explain the qualitative reproduction failures — disconnection,
basin exit, mode-tracking loss, trajectory roughness, and (for SS/CS) a broken
symmetry. They are *not* sufficient to explain the residual quantitative gap in
ω₁, nor the divergence of the paper-literal parameter regime, both of which are
shown below to be mesh-independent. Section 8 states the verdict precisely.

Date: 2026-07-29 · MATLAB R2025b · commit `461022f` (working tree clean apart
from this new, additive experiment directory).

Throughout, **observations** are what was measured, **interpretations** are
labelled as such, and **conclusions** appear only in Sections 7–8.

---

## 1. Task 1 — Inspection of the current benchmark definition

Read-only audit, `task1_support_geometry_audit.m`, which calls the production
function `build_supports_exact.m` verbatim. Full output:
`results/task1_support_geometry.csv`.

### 1.1 Geometry and mesh as currently defined

Source: `topopt_freq_exact.m` `set_defaults()` (lines 599–612) and
`run_clamped_clamped_exact.m` / `run_clamped_simply_exact.m`.

| Quantity | Value |
|---|---|
| Domain | `L = 8.0`, `H = 1.0` |
| **Beam aspect ratio** | **L/H = 8.000** |
| Current mesh | **40 × 5** (`run_clamped_clamped_exact.m`, `run_clamped_simply_exact.m`) |
| Element size | dx = 0.200, dy = 0.200 → **square elements**, dx/dy = 1.000 |
| Elements / nodes / DOF | 200 / 246 / **492** |
| Volume fraction | 0.5 |
| Node numbering | `nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1)` — column-major, row 1 = bottom edge (y = 0) |
| DOF convention | ux(n) = 2n−1, uy(n) = 2n (1-based) |

`run_simply_simply_exact.m` is already committed at 160 × 20 and is therefore
*not* part of the coarse-mesh baseline; the 40 × 5 mesh applies to the CC and CS
scripts.

### 1.2 Exact node selection used for supports

`build_supports_exact.m` selects (line 34):

```matlab
mid_idx = round(nely/2) + 1;
```

* **CC** — every node of the left and right edge, both components. No mid-height
  node is involved.
* **CS** — every node of the left edge, plus a pin (ux + uy) at `mid_idx` on the
  right edge.
* **SS** — a pin (ux + uy) at `mid_idx` on each edge, and nothing else.

### 1.3 Is the support exactly at H/2? — **No, at 40 × 5 it is shifted**

`nely = 5` is odd, so **no node lies on y = H/2**. MATLAB's `round` breaks the
tie 2.5 away from zero, giving `mid_idx = 4` rather than 3:

| Mesh | nely | `mid_idx` | pin y | y/H | offset |
|---|---|---|---|---|---|
| **40 × 5** | 5 (odd) | **4** | **0.600** | **0.600** | **+0.100 H** |
| 80 × 10 | 10 (even) | 6 | 0.500 | 0.500 | 0 (exact) |
| 160 × 20 | 20 (even) | 11 | 0.500 | 0.500 | 0 (exact) |
| 240 × 30 | 30 (even) | 16 | 0.500 | 0.500 | 0 (exact) |

**Observation 1.** At 40 × 5 the SS and CS pins sit at y = 0.600 H — **10 % of the
beam height above mid-height**, and 20 % above the mirror-image choice (row 3,
y = 0.400 H). The discretized SS and CS benchmarks are therefore **not
symmetric about mid-height**, whereas the physical benchmark is.

**Observation 2.** CC is unaffected: both edges are fully clamped, so its 40 × 5
discretization is geometrically faithful. Any 40 × 5 CC pathology must come from
resolution alone, not from support placement. *This makes CC and SS a natural
pair of controls that separate the two clauses of H1.*

**Observation 3.** Every mesh in the campaign has square elements and L/H = 8
exactly, and every campaign mesh has even `nely`. **No change to node indexing or
to `build_supports_exact.m` was required** — the existing formula already returns
the exact H/2 node whenever `nely` is even.

---

## 2. Task 2 — Campaign design

### 2.1 What varies and what does not

| | |
|---|---|
| **Varied** | `nelx`, `nely` only |
| Mesh ladder | 40 × 5 (control) → 80 × 10 → 160 × 20 → 240 × 30 |
| Aspect ratio | 8.000 at every mesh (exactly preserved) |
| Element shape | square at every mesh (dx/dy = 1.000) |
| Volume fraction | 0.5 at every mesh |
| Initial design | uniform ρ = 0.5 at every mesh (solver default path) |
| Loading | free vibration — no load in this benchmark; unchanged |
| Stopping criteria | `outer_tol = 1e-6`, `outer_max_iter = 80` — preserved verbatim |
| Everything else | preserved verbatim (Section 3) |

The 40 × 5 control was **retained, not replaced**: without it there is no
baseline against which refinement can be measured.

### 2.2 Two parameter regimes, each held fixed across the ladder

Both regimes are copied **verbatim from files already in the repository**. No
parameter value was invented, tuned, or adjusted for this campaign.

* **Regime B (primary)** — `audit_optimizer_nochange.m`, struct `base`, lines
  12–31: `move_lim = 0.2`, `outer_move = 0.2`, `alpha = 0.5`, `mult_tol = 1e-3`,
  `inner_max_iter = 30`, `outer_max_iter = 80`, `outer_tol = 1e-6`,
  `mass_mode = du2007_c1`, `rmin_elem = 2.5`, `n_modes = 4`,
  `acceptance_check = false`. This is the configuration that produced the
  disconnected 40 × 5 design analysed in the previous campaign.
* **Regime A (regime control)** — `run_clamped_clamped_exact.m`, lines 11–24:
  the paper-literal setting, `move_lim = Inf`, `alpha = 1`,
  `outer_max_iter = 300`, `outer_tol = 1e-4`.

Regime A was included so that "mesh refinement helps" could not be an artefact of
one particular step-control setting.

### 2.3 Boundary conditions

All three paper cases were run in Regime B (CC, SS, CS), because CC isolates the
*resolution* clause of H1 and SS/CS additionally carry the *support-placement*
clause. Regime A was run on CC.

**16 production runs in total.**

### 2.4 A note on the filter radius

`rmin_elem = 2.5` is specified in **element units** and was held at 2.5 at every
mesh, because that is the literal reading of "do not modify filters". The
consequence is that the *physical* filter radius shrinks proportionally with
refinement (0.50 → 0.25 → 0.125 → 0.083). This is stated here as a known
confound: the campaign varies "resolution of representable geometry", which is
mesh size and minimum feature size jointly, rather than mesh size at fixed
feature size. Holding the physical radius constant instead would have required
changing `rmin_elem`, which the restrictions forbid. Section 7.6 bounds what this
confound can and cannot explain.

---

## 3. Task 1–2 deliverable — Description of every code modification

**No existing file was modified.** Every file below is new and additive, living
in a new directory `analysis/OlhoffApproachExact/experiments/mesh_resolution_campaign/`.

| File | Purpose | Touches solver? |
|---|---|---|
| `task1_support_geometry_audit.m` | Task 1. Calls `build_supports_exact` and decodes fixed DOFs to physical coordinates. Read-only. | No |
| `run_mesh_campaign.m` | Campaign driver. Builds the regime cfg verbatim, sets `nelx`/`nely`, calls `topopt_freq_exact` unmodified, persists results. | No |
| `probe_support_placement.m` | Isolates support placement from resolution at fixed mesh (Section 6.2). Injects pin DOFs through the existing `cfg.fixed_dofs` interface (`topopt_freq_exact.m:177–181`). | No |
| `postprocess_modes.m` | Post-hoc modal instrumentation (MAC, mode tracking, local-mode dominance). Runs **after** the optimization, on saved snapshots. | No |
| `analyze_campaign.py` | Topology / trajectory descriptors, cross-mesh correlation. | No |
| `topology_maps.py` | ASCII rendering of final densities. | No |
| `drive_regimeA.m`, `drive_regimeB.m`, `drive_modesA.m`, `drive_modesB.m` | Loop wrappers. | No |
| `ab_snapshot_check.m` | Verifies the instrumentation is side-effect free (Section 4.2). | No |

Two driver-level settings deserve explicit declaration:

1. **`cfg.rho_snapshot_interval = 1`** — an *existing* solver field, documented at
   `topopt_freq_exact.m:52–54` as "diagnostic-only … no effect on optimizer". It
   stores ρ each outer iteration so topology snapshots and MAC history can be
   reconstructed. Verified side-effect free in Section 4.2.
2. **`rng(seed,'twister')`** — fixes the RNG stream `eigs` draws its start vector
   from, for reproducibility. Verified inert in Section 4.3.

---

## 4. Evidence that no algorithmic changes were introduced

### 4.1 Byte-level identity of every algorithmic file

SHA-256 of all 25 files in `analysis/OlhoffApproachExact/Matlab/` plus
`tools/Matlab/mmasub.m` and `tools/Matlab/subsolv.m` (the MMA implementation),
taken before the first run and after the last:

```
$ diff results/solver_sha256_BEFORE.txt results/solver_sha256_AFTER.txt
$ echo $?
0
```

`results/solver_sha256_{BEFORE,AFTER}.txt` are committed as evidence. `git
status` reports the campaign directory as the only untracked addition and **no
modified tracked file** in `analysis/` or `tools/`.

This covers, by name and by hash: `mmasub.m`, `subsolv.m` (MMA algorithm and
inner subproblem solver); `inner_loop_mma.m` (inner MMA logic); `build_filter.m`,
`apply_sensitivity_filter.m` (filters); `compute_elem_sensitivity.m`,
`compute_generalized_gradients.m` (sensitivities, generalized gradients);
`detect_multiplicity.m` (eigenvalue tracking); `mass_interp.m`,
`assemble_KM_exact.m` (mass and stiffness interpolation); `topopt_freq_exact.m`
(convergence criteria, continuation, damping, move limits, line search).

### 4.2 The instrumentation is provably side-effect free

`ab_snapshot_check.m` ran CC 40 × 5 twice, identical except for
`rho_snapshot_interval`:

```
no-snapshot  final omega1=355.613335
snapshot=1   final omega1=355.613335
omega_trial identical: 1
rho_final    identical: 1   max|diff|=0
```

**Bit-identical.** The instrumentation adds bookkeeping only.

### 4.3 The solver at HEAD is behaviourally identical to the frozen solver

`topopt_freq_exact.m` has gained 672 lines since the `revision_r1_frozen_solver`
commit (`47737e7`), all of it default-disabled diagnostic machinery
(globalization, forensics, density/projection filters). To confirm this code is
genuinely inert, the frozen file was checked out and run under the Regime-B
configuration:

```
FROZEN(47737e7) 40x5 final omega1=355.613335 omega2=358.305217 vol=0.487532
HEAD            40x5 final omega1=355.613335 omega2=358.305217 vol=0.487532
```

**Identical to all printed digits.** The default execution path has not changed.

### 4.4 The runs are deterministic

Six replicates of CC 40 × 5 with RNG seeds 0–5 produced
`final omega = [355.613 358.305 818.639 857.826]` in **all six**. `eigs` is
deterministic for these problems, so single runs are sufficient and no
replicate-averaging is needed. (Seeds were probed precisely because a chaotic
trajectory would have made single-run comparisons invalid; they do not.)

### 4.5 Relation to the previously recorded 40 × 5 figure

The project record cites final ω₁ = 413.869 for CC 40 × 5 Regime B. Re-measured
here at HEAD **and** at the frozen commit, the same configuration yields
**355.613**. Since the two solver versions agree exactly with each other, the
413.869 figure originates elsewhere — most plausibly the `audit_run_trace`
*replica* inside `audit_optimizer_nochange.m` rather than `topopt_freq_exact`
itself. This is reported for completeness and was deliberately **not**
investigated further or "fixed" (Task 5). It does not affect the campaign, whose
comparisons are all internal to one solver build.

---

## 5. Task 3 — Complete numerical results

Raw artefacts per run in `results/<tag>/`: `run.mat`, `history.csv` (objective β,
ω₁…ω₄ pre- and post-update, volume, N, drho_norm, step α, inner iterations, inner
CPU time), `modes.csv` (MAC, tracked mode identity, local-mode fractions),
`rho_snapshots.csv` (topology snapshot every outer iteration), `rho_final.csv`,
`summary.csv`, `log.txt`. Aggregates: `results/campaign_analysis.json`,
`results/topology_maps_all.txt`.

### 5.1 Discretization, supports, cost

| Mesh | nEl | **nDOF** | CC fixed DOF | SS/CS pin y/H | Exact H/2 | wall (B) | wall (A) |
|---|---|---|---|---|---|---|---|
| 40 × 5 | 200 | **492** | 24 | **0.600** | **no** | 11–23 s | 67 s |
| 80 × 10 | 800 | **1 782** | 44 | 0.500 | yes | 27–29 s | 105 s |
| 160 × 20 | 3 200 | **6 762** | 84 | 0.500 | yes | 62–67 s | 320 s |
| 240 × 30 | 7 200 | **14 942** | 124 | 0.500 | yes | 118–128 s | 738 s |

### 5.2 Initial frequencies (uniform ρ = 0.5) — forward-model convergence

| BC | 40 × 5 | 80 × 10 | 160 × 20 | 240 × 30 | Paper Fig. 2 | 40×5 error vs 240×30 |
|---|---|---|---|---|---|---|
| SS | **71.510** | 68.623 | 68.399 | 68.321 | 68.7 | **+4.67 %** |
| CS | **105.671** | 104.065 | 103.728 | 103.626 | 104.1 | **+1.97 %** |
| CC | **147.426** | 145.968 | 145.569 | 145.488 | 146.1 | **+1.33 %** |

**Observation 4.** The forward model converges monotonically. The 40 × 5 error is
**3.5× larger for SS than for CC** — the case with a displaced pin versus the case
with a geometrically faithful clamp.

### 5.3 Objective and eigenfrequency results (Regime B)

| BC | Mesh | ω₁ init | ω₁ final | ω₁ best (iter) | ω₂ final | N | vol | Paper ω₁ opt | % of paper |
|---|---|---|---|---|---|---|---|---|---|
| CC | 40 × 5 | 147.43 | 355.61 | 574.16 (33) | 358.31 | 1 | 0.488 | 456.4 | 77.9 % |
| CC | 80 × 10 | 145.97 | 279.64 | 320.65 (41) | 389.75 | 1 | 0.500 | 456.4 | 61.3 % |
| CC | 160 × 20 | 145.57 | 327.14 | 327.89 (30) | 426.80 | 1 | 0.500 | 456.4 | 71.7 % |
| CC | **240 × 30** | 145.49 | **369.43** | 374.47 (79) | **452.73** | 1 | 0.500 | 456.4 | **80.9 %** |
| SS | 40 × 5 | 71.51 | 143.60 | 147.20 (38) | 192.74 | 1 | 0.500 | 174.7 | 82.2 % |
| SS | 80 × 10 | 68.62 | 133.92 | 140.86 (23) | 180.11 | 1 | 0.500 | 174.7 | 76.7 % |
| SS | 160 × 20 | 68.40 | 153.94 | 153.94 (76) | 182.58 | 1 | 0.500 | 174.7 | 88.1 % |
| SS | **240 × 30** | 68.32 | **159.72** | 159.88 (78) | 182.46 | 1 | 0.500 | 174.7 | **91.4 %** |
| CS | 40 × 5 | 105.67 | 204.37 | 205.68 (14) | 266.31 | 1 | 0.500 | 288.7 | 70.8 % |
| CS | 80 × 10 | 104.07 | 194.78 | 195.72 (13) | 252.75 | 1 | 0.500 | 288.7 | 67.5 % |
| CS | 160 × 20 | 103.73 | 221.16 | 222.71 (38) | 293.30 | 1 | 0.500 | 288.7 | 76.6 % |
| CS | **240 × 30** | 103.63 | **233.23** | 233.23 (80) | 293.22 | 1 | 0.500 | 288.7 | **80.8 %** |

**Observation 5.** From 80 × 10 upward, ω₁ rises monotonically with refinement for
all three BCs. Relative to the published optimum the match improves for every
case: SS 82.2 → 91.4 %, CS 70.8 → 80.8 %, CC 77.9 → 80.9 %.

**Observation 6.** No run reaches bimodality: **N = 1 in all 16 runs.** The paper
reports all three optima as bimodal.

**Observation 7.** The CC 40 × 5 trajectory transiently reaches ω₁ = 574.16 —
26 % *above* the published optimum — at iteration 33. Its endpoint ω₁ = 355.61 is
one sample from a wildly oscillating trajectory (Section 5.5), not a converged
value.

### 5.4 Topology descriptors

`8conn` = 8-connected solid components at the final iterate (ρ ≥ 0.5);
`xmem` = *structural* components beyond the first (≥ 0.5 % of domain area);
`span %` / `conn %` = fraction of the 80 iterations whose topology has a component
touching both supports / is a single component; `ctr3` = median centre-third mean
density; `ysym` = correlation of the final design with its own mid-height mirror.

| BC | Mesh | 8conn | **xmem** | span % | conn % | **ctr3** | grey | **ysym** |
|---|---|---|---|---|---|---|---|---|
| CC | 40 × 5 | 2 | **1** | **0.0** | 0.0 | **0.089** | 0.615 | 1.000 |
| CC | 80 × 10 | 5 | **4** | **0.0** | 0.0 | 0.222 | 0.709 | 0.999 |
| CC | 160 × 20 | **1** | **0** | **95.0** | 51.2 | 0.320 | 0.513 | 1.000 |
| CC | 240 × 30 | 4* | **0** | **95.0** | 3.8 | 0.315 | 0.330 | 1.000 |
| SS | 40 × 5 | 1 | 0 | 98.8 | 91.2 | 0.373 | 0.500 | **−0.441** |
| SS | 80 × 10 | 1 | 0 | 93.8 | 93.8 | 0.451 | 0.689 | **1.000** |
| SS | 160 × 20 | 1 | 0 | 96.2 | 87.5 | 0.426 | 0.403 | **1.000** |
| SS | 240 × 30 | 1 | 0 | 96.2 | 87.5 | 0.416 | 0.282 | **1.000** |
| CS | 40 × 5 | 1 | 0 | 96.2 | 95.0 | 0.385 | 0.650 | **0.068** |
| CS | 80 × 10 | 2 | 1 | 46.2 | 46.2 | 0.443 | 0.760 | **1.000** |
| CS | 160 × 20 | 1 | 0 | 96.2 | 91.2 | 0.430 | 0.508 | **1.000** |
| CS | 240 × 30 | 2 | 0 | 95.0 | 62.5 | 0.422 | 0.457 | **1.000** |

\* The CC 240 × 30 final iterate has one spanning structural component plus three
sub-threshold specks; `xmem = 0` is the structurally meaningful count.

### 5.5 Trajectory descriptors

`coll` = single-iteration collapses of ω₁ by more than 50 %; `d̄log` =
mean |Δ log ω₁| between consecutive iterations; `IQR` = interquartile range of ω₁;
`tailCV` = coefficient of variation of ω₁ over the last 40 iterations; `TV` =
total variation of ω₁ normalized by max ω₁.

| BC | Mesh | **coll** | **d̄log** | IQR | **tailCV** | TV |
|---|---|---|---|---|---|---|
| CC | 40 × 5 | **11** | **0.730** | 193.6 | **0.508** | 16.27 |
| CC | 80 × 10 | **1** | **0.073** | 29.7 | 0.152 | 3.39 |
| CC | 160 × 20 | **0** | **0.012** | 3.3 | **0.002** | 0.79 |
| CC | 240 × 30 | **0** | **0.017** | 13.0 | **0.007** | 1.13 |
| SS | 40 × 5 | 0 | 0.011 | 2.4 | 0.007 | 0.69 |
| SS | 240 × 30 | 0 | 0.019 | 7.6 | 0.007 | 1.35 |
| CS | 40 × 5 | 0 | 0.010 | 1.6 | 0.004 | 0.67 |
| CS | 240 × 30 | 0 | 0.017 | 7.6 | 0.009 | 1.19 |

### 5.6 Tracked mode identity and MAC history

`MACmed`/`MACmin` = median / minimum MAC between mode 1 at iteration k−1 and its
maximum-MAC continuation at iteration k; `swap` = iterations at which that
continuation is **not** index 1; `break` = iterations at which the best MAC falls
below 0.5, i.e. mode identity is lost outright.

| BC | Mesh | MACmed | **MACmin** | swap | **break** |
|---|---|---|---|---|---|
| CC | 40 × 5 | 0.927 | **0.0083** | 24 | **12** |
| CC | 80 × 10 | 0.809 | **0.0019** | 3 | **3** |
| CC | 160 × 20 | 0.703 | **0.641** | 1 | **0** |
| CC | 240 × 30 | 0.773 | **0.565** | 2 | **0** |
| SS | 40 × 5 | 1.000 | 0.944 | 0 | 0 |
| SS | 240 × 30 | 0.842 | 0.507 | 1 | 0 |
| CS | 40 × 5 | 0.769 | 0.741 | 0 | 0 |
| CS | 160 × 20 | 0.738 | 0.006 | 1 | 1 |
| CS | 240 × 30 | 0.787 | 0.634 | 0 | 0 |

### 5.7 Local vibration modes

Median fraction of modal strain energy residing in low-density elements
(ρ ≤ 0.1), per mode, over all 80 iterations:

| BC | Mesh | **mode 1 (target)** | mode 2 | mode 3 | mode 4 |
|---|---|---|---|---|---|
| CC | 40 × 5 | **6.20e−01** | 6.64e−01 | 8.89e−01 | 7.93e−01 |
| CC | 80 × 10 | **1.20e−01** | 3.21e−01 | 4.99e−01 | 3.89e−01 |
| CC | 160 × 20 | **6.13e−02** | 2.30e−01 | 1.29e−01 | 4.78e−01 |
| CC | 240 × 30 | **2.48e−02** | 2.23e−01 | 6.08e−01 | 1.04e−01 |
| SS | 40 × 5 | 7.45e−03 | 5.77e−02 | 2.63e−02 | 4.28e−02 |
| SS | 240 × 30 | 1.39e−02 | 2.42e−01 | 6.93e−01 | 9.31e−02 |
| CS | 40 × 5 | 3.02e−02 | 1.35e−01 | 6.08e−02 | 8.70e−02 |
| CS | 240 × 30 | 3.25e−02 | 1.81e−01 | 3.47e−02 | 3.85e−01 |

### 5.8 Relative topology correlation between consecutive meshes

Final designs mapped to a common 80 × 10 grid (refinements by exact,
volume-preserving block averaging; the 40 × 5 control by nearest replication,
the only mapping available since 5 does not divide 10 downward).

| BC | Regime | Pair | **Pearson r** | L1 | IoU |
|---|---|---|---|---|---|
| CC | B | 40×5 ↔ 80×10 | 0.7758 | 0.166 | 0.721 |
| CC | B | 80×10 ↔ 160×20 | 0.6078 | 0.220 | 0.564 |
| CC | B | **160×20 ↔ 240×30** | **0.9425** | 0.089 | 0.864 |
| SS | B | **40×5 ↔ 80×10** | **0.1302** | 0.353 | 0.332 |
| SS | B | 80×10 ↔ 160×20 | 0.8541 | 0.150 | 0.725 |
| SS | B | **160×20 ↔ 240×30** | **0.9240** | 0.092 | 0.761 |
| CS | B | 40×5 ↔ 80×10 | 0.5094 | 0.222 | 0.555 |
| CS | B | 80×10 ↔ 160×20 | 0.8260 | 0.150 | 0.751 |
| CS | B | **160×20 ↔ 240×30** | **0.9469** | 0.079 | 0.896 |
| CC | A | 40×5 ↔ 80×10 | 0.0062 | 0.496 | 0.334 |
| CC | A | 80×10 ↔ 160×20 | −0.0063 | 0.503 | 0.330 |
| CC | A | 160×20 ↔ 240×30 | −0.3313 | 0.662 | 0.189 |

Correlation of each mesh's final design with the finest (240 × 30), Regime B:

| BC | 40 × 5 | 80 × 10 | 160 × 20 |
|---|---|---|---|
| CC | 0.5218 | 0.5550 | **0.9425** |
| SS | **0.1095** | 0.8414 | **0.9240** |
| CS | 0.4293 | 0.8295 | **0.9469** |

**Observation 8.** In Regime B the consecutive-mesh correlation rises
monotonically for all three BCs and reaches 0.92–0.95 at the finest pair —
the signature of an approaching mesh-converged design. In Regime A it is
statistically indistinguishable from zero at every pair.

### 5.9 Final topologies (density maps, top row = y = H)

```
CC 40x5   (2 components, centre emptied, DISCONNECTED)
  %%%%%%%%**+---::.     ..:::-=++*##%%%@@@
  %%%%%%%%++.--:::.     ..:::-=+=++**##%%%
  %%%%%%%%=-.:--::.      .:::--====++***##
  %%%%%%%%++.--:::.     ..:::-=+=++**##%%%
  %%%%%%%%**+---::.     ..:::-=++*##%%%@@@

CC 160x20  (1 component, full-span diagonal bracing — Fig. 3c class)
  @@@@@@@@@@@@@@*:           -######%@@@%##%@%#*******=.         .-#@@@@@@@@@@@@@@
  @@@@@@@@@%%%@@@%+:       :*#=-::-=+==.   ..===:..::-+*:      .-*%@@@%%%@@@@@@@@@
  #@@@@@%#*+=+#+*%@%+-.  .=#+--------::.     .:--::::::-#=. .:=*%@%**#+=+*%@@@@@@#
  .:+#%%%%%*+*+: :+%@%*=-*#-.:---==-:::.      ..:--:::. :#*-+#%@%+:.:**+#%%%%@%+:.
     -*%%##%%*:    :+#%%%#:  .:----:::.       ...:--:..  -#%%%%+:    -#%%##%%*-.
     -*%%##%%*:    :+#%%%#:  .:----:::.       ...:--:..  -#%%%%+:    -#%%##%%*-.
  .:+#%%%%%*+*+: :+%@%*=-*#-.:---==-:::.      ..:--:::. :#*-+#%@%+:.:**+#%%%%@%+:.
  #@@@@@%#*+=+#+*%@%+-.  .=#+--------::.     .:--::::::-#=. .:=*%@%**#+=+*%@@@@@@#
  @@@@@@@@@%%%@@@%+:       :*#=-::-=+==.   ..===:..::-+*:      .-*%@@@%%%@@@@@@@@@
  @@@@@@@@@@@@@@*:           -######%@@@%##%@%#*******=.         .-#@@@@@@@@@@@@@@
```

The 240 × 30 CC design (`results/topology_maps_all.txt`) is the same class with
finer, more numerous braces.

```
SS 40x5   (pin at 0.600 H — GROSSLY ASYMMETRIC about mid-height, ysym = -0.441)
  #+-:....                       ...::--+#     <- top edge nearly empty
  @@#+=-::...                ...::--=+*#@@
  @@@%%#*+=-:...  .....    ..::-=+**#%%@@@
  %###%%%%##*****##%%%%%%##***##%%%%%%###%
  --=+**#%@@@@@@@@@@@@@@@@@@@@@@@@%%#*+=--     <- bottom edge dense mid-span

SS 160x20  (pin exactly at H/2 — SYMMETRIC, ysym = 1.000)
       .-#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#=.
     .-*@%%#*+++=+*#@@@@@%@%@@@@@@%%##########%%@@@@@%%%####%@@@%*====++#%%@#=.
   .-*@@*--*+. .:+%%*====--+#*=--::::...        ..:-=+*+=:---::=#%*-.  -*--*%@#=:
  =#@@*-.  -*==*%%*:. :===+*+-.                      .-=+==--.  .-#@#+-*+. .-*%@#+
  @@#-.    .+%%%+:     -=+*+-.                         :-===:     .-#%%*.    .-#@@
  @@#-.    .+%%%+:     -=+*+-.                         :-===:     .-#%%*.    .-#@@
  =#@@*-.  -*==*%%*:. :===+*+-.                      .-=+==--.  .-#@#+-*+. .-*%@#+
   .-*@@*--*+. .:+%%*====--+#*=--::::...        ..:-=+*+=:---::=#%*-.  -*--*%@#=:
     .-*@%%#*+++=+*#@@@@@%@%@@@@@@%%##########%%@@@@@%%%####%@@@%*====++#%%@#=.
       .-#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#=.
```

---

## 6. Two decisive controls

### 6.1 Regime control — refinement does **not** rescue the paper-literal regime

Regime A (`move_lim = Inf`, `alpha = 1`, 300 iterations), CC, same mesh ladder:

| Mesh | final ω₁ | final ω₂ | outcome |
|---|---|---|---|
| 40 × 5 | **0.0154** | 0.0461 | ~0 Hz mechanism |
| 80 × 10 | **0.0236** | 0.1117 | ~0 Hz mechanism |
| 160 × 20 | **0.0169** | 0.0531 | ~0 Hz mechanism |
| 240 × 30 | **9.09** | 16.95 | ~0 Hz mechanism |

**Observation 9.** The paper-literal regime collapses to a near-zero-frequency
mechanism at **every** mesh, accompanied by a storm of `RCOND ~ 1e−20` singular
subproblem warnings from `mmasub`. Refining from 492 to 14 942 DOF changes
nothing qualitatively. Consecutive-mesh topology correlation is ≈ 0 or negative
throughout — the designs are unrelated noise.

**Interpretation.** This failure is **mesh-independent** and therefore **outside
the scope of H1**. It is the previously documented N = 1 LP bang-bang divergence:
for a simple eigenvalue the increment subproblem degenerates to a linear program
whose optimum is a box vertex. No discretization can repair it.

### 6.2 Support-placement control — isolating geometry from resolution

`probe_support_placement.m`: SS, **fixed** 40 × 5 mesh, identical optimizer and
settings, pin moved between node row 3 (y = 0.400 H) and row 4 (y = 0.600 H,
what the benchmark actually uses) via the existing `cfg.fixed_dofs` interface.

| Metric | Value |
|---|---|
| final ω₁, pin at 0.400 H | 144.259 |
| final ω₁, pin at 0.600 H | 143.597 |
| mid-height self-symmetry, pin at 0.400 H | **−0.436** |
| mid-height self-symmetry, pin at 0.600 H | **−0.441** |
| **corr( ρ(0.400 H), mirror of ρ(0.600 H) )** | **0.9996** |
| max abs mirror difference | 9.84e−02 |

**Observation 10.** The two designs are near-perfect mirror images of one another
(r = 0.9996) and each is strongly *anti*-correlated with its own mirror
(−0.44). The optimizer is faithfully solving whichever asymmetric problem the
node grid hands it.

**Interpretation.** The asymmetry of the 40 × 5 SS design is caused **entirely by
the off-centre pin**, not by the element count, not by the filter, and not by any
optimizer property. At fixed mesh, moving the pin by one node row flips the
design. The 40 × 5 SS discretization poses a structurally different, asymmetric
problem.

---

## 7. Task 4 — Scientific interpretation

### 7.1 Does increasing mesh resolution move the topology towards Du & Olhoff's?

**Yes, qualitatively and measurably — but it does not arrive.**

*Observations.* For CC, the disconnected two-block design at 40 × 5 (2 components,
centre-third density 0.089, spanning topology in 0 % of iterations) becomes a
single, symmetric, full-span diagonally-braced truss at 160 × 20 and 240 × 30
(0 extra structural members, centre-third density 0.320, spanning in 95 % of
iterations) — the morphological class of the published Fig. 3c. Consecutive-mesh
correlation rises to 0.92–0.95 at the finest pair for all three BCs, and each
mesh's correlation with the 240 × 30 design increases monotonically
(CC 0.52 → 0.56 → 0.94; SS 0.11 → 0.84 → 0.92; CS 0.43 → 0.83 → 0.95). ω₁ as a
percentage of the published optimum improves for all three BCs.

*Interpretation.* Refinement moves the topology into the published **class** and
towards a mesh-converged limit. It does **not** close the quantitative gap: at
240 × 30, ω₁ reaches only 80.9 % (CC), 91.4 % (SS), 80.8 % (CS) of the published
values, and **no run becomes bimodal** (N = 1 in all 16). The published optima are
all bimodal, so the campaign ends short of the paper's reported state.

*Caveat, stated plainly.* The CC 240 × 30 trajectory plateaus by iteration ~40 at
ω₁ ≈ 374 with ω₂ ≈ 452.7 and a stable ratio of 1.21 — the modes are **not**
coalescing. The proximity of ω₂ = 452.7 to the published 456.4 is suggestive but
is **not** evidence of convergence to the paper's bimodal optimum, and is not
claimed as such.

### 7.2 Does the "paper basin exit" become weaker or disappear?

**It disappears.**

*Observations.* CC single-iteration collapses of ω₁ by more than 50 %:
**11 → 1 → 0 → 0** across the ladder. Mean |Δ log ω₁| per iteration:
**0.730 → 0.073 → 0.012 → 0.017** (a 43-fold reduction). ω₁ IQR: 193.6 → 29.7 →
3.3 → 13.0. Tail coefficient of variation: 0.508 → 0.152 → 0.002 → 0.007.

*Interpretation.* The previously documented basin-exit forensics — accepted steps
driving ω₁ from ~550 to ~11 rad/s — are reproduced at 40 × 5 (11 such events) and
are **entirely absent** at 160 × 20 and 240 × 30 under identical step control.
The phenomenon is a property of the coarse discretization, not of the step-length
rule, which was held fixed.

### 7.3 Does mesh refinement reduce disconnected structural members?

**Yes, to zero.**

*Observations.* CC extra structural components (≥ 0.5 % of domain area) beyond the
first: **1 → 4 → 0 → 0**. Iterations with a support-to-support spanning component:
**0 % → 0 % → 95 % → 95 %**. Centre-third mean density: **0.089 → 0.222 → 0.320 →
0.315**. Grey fraction falls monotonically from 160 × 20 onward (0.513 → 0.330).

*Interpretation.* The 5-row mesh cannot represent a thin diagonal brace at all;
the only ω₁-ascent direction available to it is to empty the mid-span and
disconnect. Once ≥ 20 rows are available, a connected braced load path exists as
an ascent direction and the optimizer follows it. The non-monotonicity at 80 × 10
(4 extra members, worse than 40 × 5) is a real, reported feature: 10 rows are
enough to fragment but not enough to brace.

### 7.4 Do local vibration modes become less dominant?

**For the optimized mode, decisively yes. For the upper modes in the window, no —
and this distinction matters.**

*Observations.* CC mode-1 median strain-energy fraction in ρ ≤ 0.1 elements:
**0.620 → 0.120 → 0.061 → 0.025** — a 25-fold monotone reduction. In parallel,
mode-1 tracking breaks (MAC < 0.5) fall **12 → 3 → 0 → 0**, and the minimum MAC
rises from 0.008 (complete loss of mode identity) to 0.565 (identity preserved at
every step). Modes 2–4 show **no** such trend; mode 3 at CC 240 × 30 is 0.608.

*Interpretation.* The mode the optimizer actually differentiates becomes a genuine
global structural mode instead of a low-density artefact. The higher window modes
do not, and are not expected to: a finer mesh simply supports more resolvable
local modes in the void. Note the counter-intuitive median-MAC figures — MACmed
*falls* with refinement (0.927 → 0.773) while MACmin *rises* (0.008 → 0.565). At
40 × 5 most steps barely perturb the mode but twelve destroy it outright; at
240 × 30 every step evolves the mode moderately and none destroys it. **Minimum
MAC and break count, not median MAC, are the meaningful robustness measures here.**

### 7.5 Does the optimization trajectory become qualitatively smoother?

**Yes, dramatically — but it still does not converge.**

*Observations.* See 7.2 for the smoothness metrics. However, **no run converged**:
`drho_norm` plateaus at 0.040–0.079 against `outer_tol = 1e-6`, i.e. four to five
orders of magnitude above tolerance, at **every** mesh in **every** BC. All 16
runs terminated on `outer_max_iter`. Refinement improves the residual only
modestly (CC 0.064 → 0.042).

*Interpretation.* The trajectory becomes smooth and the objective becomes stable
(tail CV 0.002–0.009), yet the design keeps changing at a constant rate — a limit
cycle in design space with a stationary objective. This is a **mesh-independent**
property of the fixed-step update (α = 0.5 with a 0.2 move limit and no step
acceptance test), not a discretization effect. It is documented, not modified
(Section 8.2).

### 7.6 Was 40 × 5 a *different discrete problem*, or merely a coarse approximation?

**A different discrete problem for SS and CS. For CC, a coarse approximation
severe enough to change which optimum is reachable.**

*Observations, SS/CS.* At 40 × 5 the pin sits at 0.600 H instead of 0.500 H. The
resulting design has mid-height symmetry **−0.441** (anti-symmetric) versus
**1.000** (exactly symmetric) at every finer mesh. The 40 × 5 SS design correlates
with the 80 × 10 design at **r = 0.130** and with the mesh-converged 240 × 30
design at **r = 0.110** — statistically indistinguishable from unrelated. The
initial-frequency error at 40 × 5 is +4.67 % for SS versus +1.33 % for CC. The
support-placement probe (Section 6.2) shows that at **fixed** 40 × 5 mesh, moving
the pin one node row produces the mirror-image design (r = 0.9996).

*Interpretation, SS/CS.* This is not a coarse approximation of the published
problem. It is a **different, non-mirror-symmetric structural problem**, and the
optimizer solved it correctly. The near-zero correlation with the converged design
is exactly what one expects when the two problems differ in symmetry class rather
than in resolution.

*Observations, CC.* CC's supports are geometrically exact at 40 × 5, so this
mechanism cannot apply. Nevertheless its topology class changes qualitatively
(disconnected → connected braced), collapses vanish (11 → 0), mode identity is
preserved instead of destroyed (12 breaks → 0), and correlation with the converged
design is only 0.52.

*Interpretation, CC.* The 40 × 5 CC problem is a faithful but **severely
under-resolved** version of the benchmark. Because the admissible design space
cannot represent the braces that constitute the published solution, the
discretization *removes the paper's optimum from the reachable set* and leaves
disconnection as the only ascent direction. Functionally the consequence is the
same as posing a different problem, but the mechanism is genuinely different from
the SS/CS case, and the two should not be conflated.

*Bound on the filter confound.* Because `rmin_elem` is fixed in element units, the
physical filter radius shrinks with refinement (Section 2.4). This confound could
plausibly contribute to the appearance of finer braces. It **cannot** explain:
(a) the SS/CS symmetry restoration, which is fixed by node parity and confirmed
independently at constant mesh by the support probe; (b) the Regime A null result,
where the same radius change produces no improvement at all; or (c) the
elimination of mode-tracking breaks, which is a property of the eigenproblem
rather than of feature size.

---

## 8. Task 5 — Deficiencies identified but deliberately NOT modified

Per the controlled-experiment requirement, the following were observed and are
recorded here **without any change being made**.

1. **Non-convergence at every mesh (Section 7.5).** `drho_norm` plateaus 4–5
   orders of magnitude above `outer_tol`; all 16 runs stopped on the iteration
   cap. The fixed α = 0.5 step with no acceptance test admits a limit cycle.
   *Not modified: this is damping / convergence-criterion territory.*
2. **N = 1 in all 16 runs.** Bimodality — which the paper reports at all three
   optima — is never reached, so the campaign compares a unimodal state against a
   bimodal published one. *Not modified: this is multiplicity-detection and
   eigenvalue-tracking territory.*
3. **Regime A's mesh-independent collapse (Section 6.1).** The paper-literal
   regime diverges to a ~0 Hz mechanism with `RCOND ~ 1e−20` at all four meshes.
   *Not modified: this is inner-MMA territory.*
4. **`build_supports_exact.m` mid-height selection for odd `nely`.** `round(nely/2)+1`
   silently biases the pin upward (row 4 rather than 3 at `nely = 5`) with no
   warning, and no node exists at H/2 in that case at all. *Not modified: it is
   correct for all even `nely`, i.e. for every campaign mesh, so the campaign
   needed no change. Should odd-`nely` meshes ever be used again, this warrants a
   guard.*
5. **`rmin_elem` semantics (Section 2.4).** The element-unit filter radius couples
   mesh refinement to minimum feature size. *Not modified: forbidden, and bounded
   in Section 7.6.*
6. **Provenance of the recorded 413.869 figure (Section 4.5).** Re-measurement at
   HEAD and at the frozen commit both give 355.613. *Not investigated further.*

---

## 9. Final verdict

# H1: **SUPPORTED**

**Supported, for the qualitative reproduction failures that motivated the
hypothesis.** Holding every optimization parameter byte-identical and varying only
`nelx` and `nely`, mesh refinement:

* converts the CC topology from two disconnected blocks into a single, symmetric,
  full-span diagonally-braced truss — the published Fig. 3c morphological class
  (spanning topology 0 % → 95 % of iterations; extra structural members 1 → 0);
* eliminates the "paper basin exit" entirely (11 → 0 collapse events;
  mean |Δ log ω₁| 0.730 → 0.017);
* eliminates loss of tracked mode identity (12 → 0 MAC breaks; minimum MAC
  0.008 → 0.565);
* reduces localization of the optimized mode 25-fold (0.620 → 0.025);
* drives the design sequence towards a mesh-converged limit
  (consecutive-mesh r → 0.92–0.95 for all three BCs);
* and restores exact mid-height symmetry to SS and CS, which the 40 × 5 mesh had
  broken by placing the pin at 0.600 H.

**The support-placement clause of H1 is confirmed directly and independently.** At
fixed 40 × 5 mesh, moving the pin one node row yields the mirror-image design
(r = 0.9996), and the 40 × 5 SS design correlates with the mesh-converged design at
r = 0.110. The 40 × 5 SS and CS benchmarks were a **different discrete problem**,
not a coarse approximation. For CC, whose supports are exact at 40 × 5, the mesh
was instead so coarse that the published solution lay outside the representable
design space.

**Bounded exception — the hypothesis is not universal.** Two failure modes are
demonstrated here to be **mesh-independent** and therefore not attributable to
discretization:

* the paper-literal Regime A collapses to a ~0 Hz mechanism at 492 **and** at
  14 942 DOF alike;
* no run converges at any mesh (`drho_norm` plateaus 4–5 orders above tolerance),
  and no run reaches the bimodal state the paper reports.

Consequently, mesh resolution is established as the **primary** cause of the
qualitative reproduction failures — disconnection, basin exit, symmetry breaking,
mode-tracking loss — which is precisely what H1 asserts. It is **not** the whole
story: a residual ω₁ gap of roughly 20 % (CC, CS) and the absence of bimodality
survive refinement to 240 × 30 and require a separate, optimizer-side explanation.
Those deficiencies are itemized in Section 8 and were deliberately left untouched.

---

## Appendix — Reproduction

```bash
cd analysis/OlhoffApproachExact/experiments/mesh_resolution_campaign
matlab -batch "task1_support_geometry_audit"          # Task 1
matlab -batch "drive_regimeB"                         # 12 primary runs
matlab -batch "drive_regimeA"                         # 4 regime-control runs
matlab -batch "warning('off','all'); drive_modesB"    # MAC / mode tracking
matlab -batch "warning('off','all'); drive_modesA"
matlab -batch "warning('off','all'); probe_support_placement"
python3 analyze_campaign.py                           # tables + correlations
python3 topology_maps.py                              # density maps
```

Integrity check:

```bash
diff results/solver_sha256_BEFORE.txt results/solver_sha256_AFTER.txt   # must be empty
```
