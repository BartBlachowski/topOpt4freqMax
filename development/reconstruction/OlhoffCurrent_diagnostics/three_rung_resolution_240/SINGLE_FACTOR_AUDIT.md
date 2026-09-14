# SINGLE-FACTOR AUDIT — the mesh is the only difference

Phase 3. Evidence: `evidence/single_factor.json`, produced **before** the run.

---

## 1. Method

The 240×30 candidate configuration is produced by
`cv_config('C', 240, 30)` — **the same function** that produced the three prior
causal arms. It is then compared **field for field** (recursively flattened)
against `cv_config('C',160,20)`, `cv_config('C',320,40)` and
`cv_config('C',400,50)`. Any difference outside a declared, justified set aborts
the study.

## 2. Result — exactly five fields differ, against each of the three arms

| field | 160×20 → 240×30 | 320×40 → 240×30 | 400×50 → 240×30 | status |
|---|---|---|---|---|
| `domain.mesh.nelx` | 160 → 240 | 320 → 240 | 400 → 240 | **the authorized factor** |
| `domain.mesh.nely` | 20 → 30 | 40 → 30 | 50 → 30 | **the authorized factor** |
| `stop.tolerance` | 0.05 → 0.075 | 0.1 → 0.075 | 0.125 → 0.075 | deterministic function of `NE` |
| `runtime.name` | `CV_C_160x20` → `CV_C_240x30` | ⋯ | ⋯ | a label; enters no computation |
| `provenance.overrides` | cell[18] → cell[18] | ⋯ | ⋯ | recorded override list — **verified below** |

**Nothing else differs.** No solver flag, tolerance rule, guard, filter,
material, multiplicity, optimizer, eigensolver, objective or constraint field.

### `stop.tolerance` is not a free choice

`0.075 = 0.05·√(7200/3200)`, the inherited `meshScaled` law. It is a
deterministic function of the element count, identical in form to the value used
at every other mesh, and was verified equal to `0.05·√(NE/3200)` at
160×20, 240×30, 320×40 and 400×50 in the Phase-0 gate.

### `provenance.overrides` — verified element-wise, not assumed

Declaring a differing field "allowed" without checking would be an assumption.
Every element of the 18-entry override cell was compared:

```
vs 160x20:  [ 2] 160          -> 240
            [ 4] 20           -> 30
            [14] CV_C_160x20  -> CV_C_240x30
vs 320x40:  [ 2] 320 -> 240   [ 4] 40 -> 30   [14] CV_C_320x40 -> CV_C_240x30
vs 400x50:  [ 2] 400 -> 240   [ 4] 50 -> 30   [14] CV_C_400x50 -> CV_C_240x30
```

Exactly three entries differ in each case, and each is either a mesh dimension or
the run label. `overridesMeshOnly = true`.

**`singleFactorOk = 1`.**

## 3. The frozen scientific lock, asserted on the 240×30 config

Checked in `evidence/single_factor.json` and again **in code inside the run
driver, before `olhoffSolve` is called**, so a violation aborts rather than
producing a contaminated trajectory:

| field | required | resolved |
|---|---|---|
| `material.stiffness.p` | 3, fixed | 3 |
| `material.stiffness.continuation.enabled` | false | false |
| `material.mass.model` | `eq4b` | `eq4b` |
| `material.mass.q` | 1 | 1 |
| `filter.type` / `applyTo` | `sensitivity` / `all` | `sensitivity` / `all` |
| `filter.radiusPhysical` | 0.06 (`R = 0.06·b`) | 0.06 |
| `projection.enabled` | **false** | false |
| `multiplicity.method` / `subspaceSize` | `subspace` / 2 | `subspace` / 2 |
| `multiplicity.diagonalOffsets` / `offDiagonal` | true / **true** | true / true |
| `optimizer.inner.variant` | `published` MMA | `published` |
| `move.policy` | `ladder` | `ladder` |
| `move.levels` | `[0.04 0.02 0.01 0.005]` | `[0.04 0.02 0.01 0.005]` |
| `move.continuation.signal` | `stageExhaustion` | `stageExhaustion` |
| `stop.rule` | `stageExhaustion` | `stageExhaustion` |
| `stop.tolerance` | `0.05·√(NE/3200)` | 0.075 |
| `runtime.maxOuter` | 1600 | 1600 |
| threads | 1 | `maxNumCompThreads() == 1` |
| resolved config hash | preregistered | `33833323efa08fac…` **asserted** |

`lockOk = 1`. FE formulation, eigensolver policy, objective and volume
constraint are inherited unchanged through the same preset resolution and are
not touched by any override.

## 4. The safety cap is inherited, not chosen

`CAP = 1600` is a constant inside `cv_config.m` and is the cap **every** prior
candidate arm used. It was not selected for this run, and it was not changed
after the trajectory was seen. The run converged at 1358, so the cap was never
reached.

## 5. What this audit does not claim

It shows the *configuration* differs only in mesh. It does not claim bitwise
comparability of trajectories across meshes — different meshes are different
problems. Its purpose is narrower and sufficient: the 240×30 result can be read
as evidence about the **controller** rather than about some other changed
setting, because no other setting changed.
