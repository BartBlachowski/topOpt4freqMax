# SINGLE-FACTOR AUDIT (Phase 6)

**Verdict: `CONTROLLER_SINGLE_FACTOR_PASS`.**

Machine-readable: `evidence/single_factor.json`.

---

## 1. What was compared

At each of the three scientific meshes, every field of the configuration schema
was compared between

* the **production entry point** `olhoffcurrent_config(nelx, nely, 'MaxOuter',
  1600, 'Diagnostics', true)` — production, at the candidate's cap and recorder
  setting, so the cap and the recorder cannot masquerade as the intervention;
* the **candidate** `cv_config('C', nelx, nely)`.

Comparison is `isequaln` over the whole schema, not a spot check of fields
someone thought to look at.

## 2. Result

| mesh | fields that differ | unexpected | scientific lock |
|---|---|---|---|
| 160×20 | `move.continuation.signal`, `stop.rule`, `runtime.name` | **none** | OK |
| 320×40 | `move.continuation.signal`, `stop.rule`, `runtime.name` | **none** | OK |
| 400×50 | `move.continuation.signal`, `stop.rule`, `runtime.name` | **none** | OK |

`runtime.name` is a free-text run label; `olhoffcurrent_config_hash` excludes it
by construction because it is never read by the mathematics.

The two remaining differences **are** the declared intervention. Both were also
required to differ — a run in which they did not would mean the candidate had
silently reverted to production, and the gate fails in that direction too.

## 3. Effective configuration hashes

| mesh | production | candidate |
|---|---|---|
| 160×20 | `cffa417eff67def9fb8ab62ea679c6087cfd44d422e141204144524a0087f962` | `31d2ef382746a942a4036d07dd6a1012742432cba0497d2de5ec24b51d2b5904` |
| 320×40 | `a1b31c6f4259b5ee4d51165247d33b6936501fa1b910a179535b1cca47bd6677` | `2359a1112fcec9edd0971aae9a89508cd703ea3899770e7e85c850138c2550f4` |
| 400×50 | `ab7d70d8b801ab06daaf11a461ee6cf37c5051dbdf09ae7ef4f8ca33e14d4888` | `0afb0d4daa11b52aac8f184c856f71626bfd11b0a581e66d5aaacfa7f78f3a78` |

The candidate hash recorded in each run's `runs/*_record.json` equals the
candidate hash above, so the configuration audited here is the configuration that
ran.

## 4. The scientific lock, field by field

Asserted identical between production and candidate at every mesh, and
additionally re-asserted **in code inside `cv_run` immediately before each
solve**, so a run cannot start under a violated lock:

```
material.stiffness.p = 3            material.stiffness.continuation.enabled = false
material.mass.model  = eq4b         material.mass.q = 1
filter.type = sensitivity           filter.applyTo = all
filter.radiusPhysical = 0.06        filter.radiusElements = []
projection.enabled = false
multiplicity.method = subspace      multiplicity.subspaceSize = 2
multiplicity.diagonalOffsets = true multiplicity.offDiagonal = true
optimizer.inner.type = mma          optimizer.inner.variant = published
optimizer.inner.variable = increment
optimizer.inner.tolerance = 0.05    minIterations = 5   maxIterations = 500
eigen.solver = eigs                 eigen.targetMode = 1   eigen.maxCluster = 4
design.initial · design.minimum · design.volumeFraction = 0.5
domain.a · domain.b · domain.mesh.nelx · domain.mesh.nely
move.policy = ladder                move.initial = 0.04
move.levels = [0.04 0.02 0.01 0.005]                     move.minimum
stop.norm = l2                      stop.toleranceRule = meshScaled
stop.tolerance = 0.05*sqrt(NE/3200) stop.field = designVariable
stop.guards.settledMove = true      ladderExhausted = false   maxDesignChange = false
runtime.maxOuter = 1600             runtime.singleThread = true   runtime.diagnostics = true
```

Note `stop.tolerance` is **identical** between the arms and is the *same number*
the frozen rule uses as `tol(NE)`. The controller therefore introduces **no new
numerical constant** — it reads the inherited mesh-scaled tolerance.

Note also `stop.guards.settledMove` is left at its production value `true`. Under
`stop.rule = 'stageExhaustion'` it is inert (the exhaustion rule replaces the
admission test wholesale), but changing it would have added a second differing
field for no reason.

## 5. Independent corroboration that the switch is the only effect

Two further checks, neither of which is a configuration comparison:

* **Production bitwise reproduction.** With both switches at their defaults, the
  production preset re-solved 160×20 and matched the frozen conference record
  bitwise — ρ bitwise, ω₁ bitwise, 91 outer, 2241 inner, `CONVERGED`
  (`test_preset_equivalence`, 0 failures). The six edited files and one new file
  are therefore inert for production.

* **Telemetry inertness.** Diagnostics on and off give identical ρ, ω, `dxNorm2`
  and controller trace (`cv_tests` test 15). The recorder that produced all of
  this study's evidence does not perturb the trajectory it records.

## 6. Conclusion

The only intended algorithmic intervention — β-driven stage transition and
terminal admission replaced by the frozen `A OR B` stage exhaustion — is the only
difference between the arms. `CONTROLLER_SINGLE_FACTOR_PASS`; no reason to stop
before interpreting results.
