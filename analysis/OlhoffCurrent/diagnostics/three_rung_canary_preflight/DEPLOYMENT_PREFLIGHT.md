# DEPLOYMENT_PREFLIGHT — Part A

## Verdict

```
THREE_RUNG_DEPLOYMENT_PREFLIGHT_PASS
```

Issued for both authorized meshes, from a **runtime** resolution, before any
optimization. 47 of 47 field checks pass at each mesh, 0 blockers.

| | 480×60 | 800×100 |
|---|---|---|
| verdict | PASS | PASS |
| field checks | 47/47 | 47/47 |
| blockers | none | none |
| runtime config hash | `03097a28b0ad7fdb0d977985d3b5fd279dd74553c9dd5dfbd3cc035ac2a1782e` | `7724af5ed26786e16132c30fbae840f5d8679d5ea798fc75d5c48eefd0e50ffe` |
| frozen expected hash | **identical** | **identical** |
| `+impl` tree | `edbfe47e…52cb`, verified | same |
| dispatch | `olhoffcurrent_assert_dispatch` OK | OK |
| beta continuation authority | false | false |
| beta stop authority | false | false |
| telemetry fields missing | none | none |
| retention budget | 0.74 GB worst case, fits | 2.05 GB worst case, fits |

Records: `evidence/preflight_480x60.json`, `evidence/preflight_800x100.json`.

## The hash was frozen before it was measured

`PREFLIGHT_MANIFEST.json` was written **before** MATLAB could be started on
this host, with the expected config hash for each mesh computed offline by
`scripts/cp_config.py`. That prediction was validated against ten independently
recorded hashes (the nine legacy campaign configs, 9/9, plus the validated
three-rung C320 config `afad9ea4…daaab`).

When MATLAB later became available, `olh.config.resolve` produced exactly those
two hashes. The frozen expectation and the runtime measurement agree bit for
bit, so the assertion is a genuine pre-run commitment that was met, not a
value recorded after the fact and declared to match.

## What was verified, at run time

Every value below was read back out of the resolved `cfg` with
`olh.config.getPath`, after defaults → preset → overrides → derived rules →
validation had all run. No preset name, comment, variable name or intention was
trusted. Full table in `CONFIG_ASSERTIONS.md`.

**The controller under test**

```
move.levels               = [0.04 0.02 0.01]        three rungs, 0.005 absent
move.continuation.signal  = stageExhaustion
stop.rule                 = stageExhaustion
move.policy               = ladder
```

`olh.config.validate` additionally refuses to resolve a half-applied policy:
`stop.rule = stageExhaustion` without the matching continuation signal is a
named error, so a partially promoted controller cannot reach the solver at all.

**Beta has no authority** — asserted structurally rather than by name:
`move.continuation.signal ≠ boundVariable` (so `olh.move.limit` returns from
its `stageExhaustion` branch before the bound-variable window is ever formed)
and `stop.rule ≠ designChange` (so `olhoffSolve` replaces the sec. 3.5.1
admission wholesale). Both flags read `false` at both meshes.

**The scientific locks** — p = 3 with no continuation, eq. (4b) mass with q = 1
and no continuation, sensitivity filter applied to every `f_sk` at physical
R = 0.06, projection off, fixed subspace N = 2 with diagonal offsets and
off-diagonal terms on, published MMA on the increment, `eigs` with the
deterministic start vector, uniform initial design 0.5, volume fraction 0.5,
single thread. All 37 lock fields verified.

**Mesh-derived quantities** — `stop.tolerance` = 0.15 (480) and 0.25 (800) from
`0.05·√(NE/3200)`; element-space filter radius 3.6 and 6.0. Both sets match the
values independently recorded for the same meshes in the legacy campaign.

**Instrumentation** — `runtime.diagnostics = true` (so `res.diag.drho` is
retained for every outer iteration and the raw trajectory can be rebuilt and
proved), and every required `hist` field present on the live solver source:
A/B/E telemetry, persistence counters, declaration state, stage and move,
inner-MMA counts and convergence, ω₁…ω₅, multiplicity and J diagnostics, and
the four timing channels.

**Retention** — decided before any run. Worst case under the frozen cap of 1600
is 0.74 GB at 480×60 and 2.05 GB at 800×100 for `RHO`+`DRHO`, against 68.7 GB
of host RAM. The preflight refuses to start if that exceeds a quarter of RAM,
so the checkpoint fallback is a pre-run decision, never a post-hoc discard.

## Fail-closed, demonstrated

`cp_run.m` calls `cp_preflight(..., 'Throw', true)` as its first statement and
cannot reach `olhoffSolve` past a throw. This is not a claim: earlier in this
same study the gate **did** refuse. MATLAB's network licence manager
(`zm8pc.ippt.pan.pl:27000`) was unreachable, the preflight could not resolve a
configuration, and it aborted with zero optimization executed
(`evidence/preflight_execution_attempt.log`, 2026-09-12T09:17:43Z, MathWorks
Licensing Error 15, code −15.2). Once network access was restored the same gate
passed unchanged. The refusal cost nothing; that is the point of putting it in
front of a multi-hour pair of solves.

## Material disclosures

**1. Production is still unpromoted.** `olhoffcurrent_preset()` names
`duOlhoffFrozenM4`, which carries `move.levels = [0.04 0.02 0.01 0.005]`,
`move.continuation.signal = boundVariable` and `stop.rule = designChange` — the
legacy policy. `three_rung_promotion_closure` records
`PRODUCTION_THREE_RUNG_CONTROLLER_NOT_PROMOTED` and `PRODUCTION_FREEZE_FAIL`.

The canary driver therefore does **not** resolve through
`olhoffcurrent_config`. It applies the validated overrides explicitly through
`tr_config` — the same builder that produced the validated C320 result — and
says so in `cp_config.m`'s header. This is a deliberate, disclosed deviation
from "production preset = what runs". It is also the reason a future full
campaign must settle promotion first: a campaign resolving through production
today would run the legacy policy again, exactly as the September 11 one did.

**2. MATLAB build differs from the validated run's.** This host reports
`25.2.0.2998904 (R2025b)`. The validated three-rung C320 run recorded
`25.2.0.3042426 (R2025b) Update 1` — a *later* build. The preflight does not
assert a toolchain version (none was ever observed here when the manifest was
frozen, and freezing an unobserved version would have been fabrication), so
this is not a blocker. It is recorded because a floating-point-sensitive
comparison against the validated C320 endpoint would have to account for it.
No such comparison is made in this study: the canaries are at meshes the
validated runs never visited.

**3. Host load at the preflight.** The machine had booted five minutes earlier
and the load average was still decaying from post-boot indexing (≈ 67 at the
480 preflight, ≈ 53 at the 800 preflight). Actual CPU consumption measured
immediately afterwards was ≈ 0.56 of 10 cores, i.e. the load figure was a
lagging artefact and the host was effectively idle. The canaries were launched
only after confirming that. Config resolution is not timing-sensitive in any
case; this is recorded for the Part G timing record, not as a caveat on the
preflight itself.

## Host and toolchain, as observed by MATLAB

| | |
|---|---|
| MATLAB | 25.2.0.2998904 (R2025b) |
| `computer` | MACA64 |
| CPU | Apple M1 Max, 10 cores |
| RAM | 68 719 476 736 B (64 GiB) |
| `maxNumCompThreads` at probe | 10 (the solver sets it to 1) |
| OS | macOS 26.6.2, Darwin 25.6.0, arm64 |
| hostname | PMS.local |
| swap | 0 B in use |
| BLAS thread env | unset (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` all empty) |
