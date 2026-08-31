# Topology gate audit — T1, T0, T2

Audit-only. `TOPOLOGY_SANITY_SPEC.md` was not modified.

Stated purpose of the gate (which I accept and audit against): **reject grossly or
pathologically wrong results, not rank topology aesthetics.**

## 1. Provenance of `a_res = 5`

**Arithmetic — correct.** Integer cells with `p² + q² < r²` for `r = 1.3` (`r² = 1.69`):
`(0,0)`, `(±1,0)`, `(0,±1)` = **5**. `(±1,±1)` has `p²+q² = 2 > 1.69` and is excluded.

**Radius — verified but not common.** `r = 1.3` is the **Olhoff** filter radius:
`res.cfg.rminEl = 1.3` in every frozen `s1_*.mat`, and `rmin_element: 1.3` in
`profile_freeze_manifest.json` for both Olhoff profiles. The three frozen profiles use
**three different radii**:

| method | `rmin_element` | integer footprint `p²+q² < r²` |
|---|---:|---:|
| Olhoff | 1.3 | **5** cells |
| Proposed | 2.0 | **9** cells |
| Yuksel | 2.5 | **21** cells |

The manifest's `"common"` block contains `filter_radius_units:
"element_widths_at_every_resolution"` — it declares that radii are expressed in element
widths, **not** that the radius is shared. Naming 1.3 `r_common` misstates this.

Choosing the smallest of three yields the strictest allowance, which is the conservative
direction and is a defensible choice. The defect is the name, not the direction. It should
read "the strictest of the three frozen filter footprints".

## 2. Mesh independence

The filter radius is fixed in element widths at every resolution, so the footprint is
mesh-invariant **in resolution units** — which is the sense the spec claims, and in that
sense the claim is true.

It is not mesh-independent in any other sense:

| mesh | element size | physical area of 5 elements | 4 elements as a share of `nSolid` |
|---|---|---:|---:|
| 160x20 | 0.05 × 0.05 | 1.25e-2 | 4/1600 = **0.25 %** |
| 400x50 | 0.02 × 0.02 | 2.00e-3 | 4/10000 = 0.04 % |
| 800x100 | 0.01 × 0.01 | 5.00e-4 | 4/40000 = **0.010 %** |

The physical area shrinks 25×, and the relative allowance shrinks 25×, across the mesh
range. **Five elements does not represent the same physical feature across the nine
meshes.** For a study whose primary output is a scaling exponent over exactly that range,
a gate whose relative severity varies 25× with `N_e` is a structural hazard.

## 3. The two clauses are not equally derived

T1 as specified is a conjunction of three conditions:

```
(a) C_required == 1                                   support-footprint connectivity
(b) max( detached component area ) <  a_res           per-component
(c) sum( detached component area ) <  a_res           aggregate
```

- **(a)** is geometry-derived and correct. See §5.
- **(b)** is a genuine resolution argument: a detached feature smaller than the method's own
  filter footprint is below the resolvable scale and should not veto an otherwise sound
  structure. Sound.
- **(c)** does **not** follow from (b)'s argument. Nothing about filter resolution implies
  that the *total* of many independent sub-resolution specks must itself be below one
  footprint. As the mesh refines, the number of sub-resolution specks grows with element
  count while the aggregate cap stays at 4 elements. Clause (c) is a global purity
  requirement wearing clause (b)'s constant.

The spec presents (b) and (c) as one derived rule. They are one derived rule and one
undefended rule sharing a number.

## 4. Empirical feasibility — the decisive test

I recomputed the full gate on every recorded Olhoff state: exact-count volume-preserving
binary projection with stable index tie-break (per `study_evaluate_design.m`), four-connected
components, and support-footprint connectivity per `TOPOLOGY_SANITY_SPEC.md`. Read-only,
Python, no MATLAB, no optimization. 14 409 states across nine meshes.

**Verification status (see `DATA_PROVENANCE_AND_LIMITATIONS.md`).** The eight meshes
160x20 through 720x90 were reproduced **exactly** by two independent component-labelling
implementations (`scipy.ndimage.label` and a pure-Python BFS). The 800x100 trajectory row
rests on one implementation only: the second scan failed on that file because the artifact
was truncated mid-write by a concurrently running regeneration job. The 800x100 **final
state** in the second table below *was* cross-verified across both implementations and both
precisions. §7 records why Finding C1 does not depend on the unverified cell.

Definitions used:
`T1` = (a) ∧ (b) ∧ (c) — the specified baseline.
`T1a` = (a) ∧ (b) — per-component clause only.
`T0` = (a) ∧ `n_components == 1`.

| mesh | states | `C_req` % | T1 % | longest T1 run | T1 ≥ P=100? | T1a % | longest T1a run | T1a ≥100? | longest T0 run |
|---|---:|---:|---:|---:|:--:|---:|---:|:--:|---:|
| 160x20 | 1601 | 98.94 | 66.31 | 967 | yes | 74.62 | 1065 | yes | 944 |
| 240x30 | 1601 | 98.94 | 77.75 | 1132 | yes | 88.19 | 1265 | yes | 1122 |
| 320x40 | 1601 | 98.94 | 56.50 | 861 | yes | 60.12 | 890 | yes | 181 |
| 400x50 | 1601 | 99.06 | 37.44 | 347 | yes | 62.00 | 526 | yes | 25 |
| 480x60 | 358 | 95.80 | 4.48 | **11** | **no** | 26.61 | **50** | **no** | 8 |
| 560x70 | 400 | 96.24 | 1.25 | **5** | **no** | 6.02 | **14** | **no** | 5 |
| 640x80 | 1067 | 98.59 | 0.56 | **5** | **no** | 45.78 | **226** | **yes** | 1 |
| 720x90 | 1601 | 99.06 | 16.50 | 261 | yes | 55.94 | 415 | yes | 122 |
| 800x100 | 1601 | 99.06 | **0.00** | **0** | **no** | 27.38 | **298** | **yes** | 0 |

Final states, double precision (`res.rho`), for cross-checking:

| mesh | `nSolid` | `C_req` | components | `f_LCC` | largest detached | total detached | T1 | T0 |
|---|---:|:--:|---:|---:|---:|---:|:--:|:--:|
| 160x20 | 1600 | 1 | 1 | 1.00000 | 0 | 0 | pass | pass |
| 240x30 | 3600 | 1 | 1 | 1.00000 | 0 | 0 | pass | pass |
| 320x40 | 6400 | 1 | 8 | 0.99344 | 12 | 42 | fail | fail |
| 400x50 | 10000 | 1 | 2 | 0.99980 | 2 | 2 | pass | fail |
| 480x60 | 14400 | 1 | 48 | 0.99507 | 13 | 71 | fail | fail |
| 560x70 | 19600 | 1 | 64 | 0.99469 | 10 | 104 | fail | fail |
| 640x80 | 25600 | 1 | 9 | 0.99922 | **4** | **20** | **fail (c) only** | fail |
| 720x90 | 32400 | 1 | 19 | 0.99873 | 8 | 41 | fail | fail |
| 800x100 | 40000 | 1 | 29 | 0.99920 | **4** | **32** | **fail (c) only** | fail |

### Findings

1. **T1 is unsatisfiable for Du–Olhoff at 800x100.** Zero of 1601 states pass. Not "rarely
   passes" — never. Under the specified baseline, Olhoff's result at the largest mesh does
   not exist.
2. **Clause (c) is the sole cause at 640x80 and 800x100.** At both, the largest detached
   component is exactly 4 elements — clause (b) passes — while the detached total is 20 and
   32. Removing (c) restores 100-runs of 226 and 298.
3. **480x60 and 560x70 fail under both T1 and T1a**, but those trajectories terminate at
   358 and 400 states because of the LP failures. They are already
   `GENUINE_SOLVER_FAILURE` rows under status precedence; topology is not the operative
   cause.
4. **Connectivity is never the binding constraint.** `C_required` passes at 95.8–99.1 % of
   states everywhere. This is exactly what a "reject grossly wrong results" gate should look
   like: it fires on early, unformed states and almost never afterwards. Clause (a) is
   doing its job; clauses (b) and (c) are doing something else.
5. **T0 is not a usable sensitivity.** Longest T0 runs are 25/8/5/1/0 at 400x50 through
   800x100 — it would censor Olhoff at five of nine meshes. The `TOPOLOGY_RESOLUTION_
   SENSITIVE` verdict is therefore already determined by frozen data. T0 is also *stricter*
   than T1, so it cannot probe the direction that matters, which is whether T1 is too
   strict.

### Precision check

The trajectory scan uses `res.rho_snapshots`, which is `float32`. The final-state table
above uses `res.rho`, which is `float64`. At all nine final states the two give **identical**
component counts, largest detached areas and detached totals. The exact-count tie-break is
therefore not precision-sensitive at these fields, and the trajectory conclusions are not a
float32 artifact. (Individual borderline states elsewhere in a trajectory could still differ;
the aggregate picture cannot.)

## 5. Required connectivity — verified correct

The benchmark is an 8×1 beam with both translations fixed at the mid-height node of each end
face. Verified in four independent sources:

| source | mid-node index |
|---|---|
| `study_evaluate_design.m` | `jMid = round(nely/2)`; `nL = jMid`, `nR = nelx*(nely+1)+jMid` |
| `topopt_freq.m` `localBuildFixedDofs` | `j_mid = floor(nely/2)` |
| `top99neo_inertial_freq.m` case `"simply"` | `midRow = round(nely/2)+1` |
| `Matlab/reproduction2007/fem/model2D.m` | `support='mid'`, errors unless `nely` is even |

`floor` and `round` agree because all nine meshes have even `nely`. That should be recorded
as a precondition (Finding Mi1).

Requiring one four-connected binary component to intersect both support element footprints
is correct, and it is a real improvement on the evaluator's existing
`left_right_connected`, which only tests whether *any* labelled cell in the first element
column shares a label with *any* cell in the last — a condition a structure could satisfy
without a load path through either support node.

**Should load-path connectivity also be required?** No. There is no common external load
point across the three formulations — Proposed uses a frozen-reference semi-harmonic load
distributed by `M(x)`, Yuksel Stage 1 a single point load, Yuksel Stage 2 a design-dependent
inertial load, Olhoff none at all. Any load-path rule would be method-specific and would
violate the symmetry principle the spec applies elsewhere. The spec's reasoning here is
correct and needs no change.

## 6. Is exact-count binarization method-neutral?

**Procedurally, yes.** Fixing `nSolid = round(Vf·N_e)` and taking the top-`nSolid`
densities with a stable index tie-break means one method's grayness cannot admit more
material to the structural graph than another's. That is the right design and the
implementation is deterministic.

**In consequence, no**, and the frozen evidence is stark:

| endpoint | E1-raw ω₁ | E1-binary ω₁ | change |
|---|---:|---:|---:|
| Yuksel 400x50 | 159.968 | **109.440** | **−31.6 %** |
| Olhoff 480x60 | 170.244 | 138.138 | −18.9 % |
| Olhoff 800x100 | 172.945 | 168.557 | −2.5 % |
| Proposed 160x20 | 153.675 | 162.760 | +5.9 % |

Binarization can destroy a load path (Yuksel 400x50) or improve a gray design (Proposed
160x20). These are real properties of those designs, and exposing them is the point. The
spec's division of roles — **raw** for the spectral gate, **binary** for the hard topology
gate, both reported — is correct and must be preserved: it is what stops the 400x50-style
collapse from contaminating `Q(k)`.

## 7. Verdict and minimum correction

**T1 as specified is not suitable as the baseline.** Not because it is strict, but because
its strictest clause is its least-derived clause, and that clause is empirically
unsatisfiable for one method at the top of the mesh range.

**Minimum correction (conceptual; not applied):**

1. **Delete clause (c).** Baseline becomes `C_required == 1 ∧ n_islands_resolved == 0` —
   two conditions, both derived, both already named and computed in the spec's own metrics
   list.
2. **Keep as mandatory reported diagnostics:** aggregate detached area, `f_LCC`, component
   count, sorted component-area distribution, `n_islands_all`, and the implied mesh-specific
   `f_LCC` bound. Nothing is lost analytically; only the veto is removed.
3. **Rename `r_common`** and state the derivation as "the strictest of the three frozen
   filter footprints (5 / 9 / 21 cells)".
4. **Replace or re-scope T0.** Either declare its known outcome up front, or add a
   sensitivity that can discriminate in the permissive direction — the natural choice is
   clause (b) evaluated at the Proposed and Yuksel footprints (`a_res` = 9 and 21), which
   directly tests how much of the result depends on whose filter defines the allowance.
5. **Add a post-freeze feasibility rescan as a Phase-A gate.** After thresholds are frozen
   and before production, rerun the §4 computation on the existing Olhoff trajectories. It
   may only declare infeasibility and trigger a documented amendment; it may never retune a
   threshold. This audit demonstrates the check costs minutes and changes the study.

T2 (fixed fractional LCC) remains correctly rejected: no neutral fraction exists and its
absolute allowance grows with mesh size. Its retention as a labelled exploratory diagnostic
only, unable to rescue a failed row, is the right treatment.
