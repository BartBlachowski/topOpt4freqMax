# VALIDATED_POLICY_RECOVERY — Phase 7

The exact validated candidate configuration, recovered **by re-resolving the
retry's own frozen builder**, not from memory or prose.

## 1. Recovery method and its proof

`scripts/tr_policy_recovery.m` calls `tr_config(320,40)` — the retry's own
configuration function, on the retry's own `scripts/` path — and re-resolves it
through `olh.config.resolve`. The result is checked against the hash the
validated run actually recorded:

```
re-resolved cfgHash        afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab
recorded in retry1 METRICS afad9ea4b27da576553f128d66a8329edee963066c1ef477b1df70f9232daaab
MATCH                      yes
```

The recovered configuration is therefore provably the one that produced
`CONVERGED @352`, not a reconstruction of it.

## 2. The validated policy

| Field | Validated candidate | Production today |
|---|---|---|
| `move.policy` | `ladder` | `ladder` |
| **`move.levels`** | **`[0.04 0.02 0.01]`** | `[0.04 0.02 0.01 0.005]` |
| **`move.continuation.signal`** | **`stageExhaustion`** | `boundVariable` |
| `move.continuation.window` | `10` | `10` |
| `move.continuation.tolerance` | `0.005` | `0.005` |
| **`stop.rule`** | **`stageExhaustion`** | `designChange` |
| `stop.norm` | `l2` | `l2` |
| `stop.tolerance` (320×40) | `0.1` | `0.1` |
| `stop.toleranceRule` | `meshScaled` | `meshScaled` |
| `stop.guards.settledMove` | `true` | `true` |
| `stop.guards.ladderExhausted` | `false` | `false` |
| `stop.guards.maxDesignChange` | `false` | `false` |

Three fields differ. Everything else in the policy group is already identical.

## 3. A and B — source, constants and semantics

**Source of truth (the executed detector):**
`+impl/architecture/+olh/+move/exhaustion.m`, member of `+impl` tree
`edbfe47eb32109a2fb017f6f13d5327f2c240357caa96630064ffcf37ee152cb`.

| File | SHA-256 |
|---|---|
| `+impl/architecture/+olh/+move/exhaustion.m` | `17b37a384b1aa5d987d9c861…` |
| `+impl/architecture/+olh/+move/limit.m` | `61fa923d430121ead764a229…` |
| `+impl/architecture/olhoffSolve.m` | `1e5a114cbf91717e01e5592e…` |
| `+impl/architecture/+olh/+presets/duOlhoffFrozenM4.m` | `6ed3624cd19b3569f55b23ac…` |

**Preregistration it implements:**
`diagnostics/two_branch_maturity_240/PREREGISTRATION.md`, SHA-256
`62748225253f85f6a2fbc1bad35489a2c201cd45ee64ff279601003354b73abd` — exactly the
digest `exhaustion.m` cites in its own PROVENANCE comment.

```
W = 20    P = 20    Wnp = 10    tol = 0.05*sqrt(NE/3200)   (meshScaled; 0.1 at 320x40)

A(k) = med20 cos(k) < 0  AND  med20 net(k) < 0.5  AND  amp(k) >= tol
B(k) = med20 cos(k) > 0  AND                           amp(k) <  tol
E(k) = A(k) OR B(k)
```

**Persistence:** `cntA`/`cntB` increment on a true predicate and reset to 0 on
any false; `E` is declared at the first iteration where either reaches `P = 20`;
`declIter = t`, `declBegin = t − P + 1`.

**Reset semantics:** on descent, atomically — push
`[outer, stageFrom, declIter, declBegin]`; `stage += 1`; `stageStart = outer`;
`cntA = cntB = 0`; `declared = false`; clear `declIter`/`declBegin`/`declBranch`.
The reset can only delay a descent, never advance one.

**Stage locality:** `amp(j)` for `j ≥ s`; `cos(j)` for `j ≥ s+1`; `net(j)` for
`j ≥ s+9`; medians only once the trailing 20-window lies wholly inside the
stage. Anchor exception: `net` at `j = s+9` uses `rho_{s−1}`.

**Terminal semantics:**
`atLastLevel = stage >= numel(moveLevels)`;
`convOuter = ex.declared && atLastLevel`. With three rungs, persistent `E` at
`move = 0.01` is a genuine `CONVERGED`. `beta` appears in neither branch.

## 4. Scientific formulation — already identical

All 18 audited formulation fields are **identical** between the validated
candidate and production today:

`material.stiffness.p = 3`; p-continuation disabled; `material.mass.model =
eq4b`; `material.mass.q = 1`; `filter.type = sensitivity` applied to `all`;
`filter.radiusPhysical = 0.06`; `projection.enabled = false`;
`multiplicity.method = subspace` with `subspaceSize = 2`, diagonal offsets and
off-diagonal terms; `optimizer.inner` type/variant/variable (published MMA);
`design.initial`; `design.minimum`; `volume.fraction`.

**Promotion changes no physics.** It changes only the step controller's ladder
length and which rule governs continuation and terminal admission.
