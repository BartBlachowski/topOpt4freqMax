# PROVENANCE — analysis/OlhoffCurrent

Where this implementation came from, exactly, and what was changed on the way in.

The machine-readable form of everything below is [`PROVENANCE.json`](PROVENANCE.json);
it is what `olhoffcurrent_provenance()` returns and what every production result
embeds. The previous record (the 2026-09-07 promotion) is preserved verbatim in
`PROVENANCE.json` under `history[0]`.

---

## 1. Source

| | |
|---|---|
| Source repository | `/Users/piotrek/Programming/Matlab/Olhoff` |
| Its role | **DEVELOPMENT / RESEARCH UPSTREAM** |
| Source branch (recorded) | `migration/upstream-olhoffcurrent-capabilities` |
| **Source commit** | `253069262407885a8b759a9e721c4f0a7d3a397d` (tree `4571029f39b62181499ab3d4ffb6da89f1d30021`) |
| Commit subject | `Promote optional stage-exhaustion controller and outer timing` |
| **Parent successful-science commit** | `6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7` (`repro/natural-convergence`, "Nine resolution test - ultimate results") |
| Promotion route | byte copy from `git archive 253069…` (tar SHA-256 `f911240340cab69b9806f98f9413d9038649fde5e14adf3bbe4d57f3ddb0dbeb`), every blob verified against the commit; never from the upstream working tree |
| Upstream working tree at promotion | checked out `repro/natural-convergence @ 6b08708`, one uncommitted file (`repro/PLAN_OLHOFFCURRENT_UPDATE.md`, the uncommitted §7). **Not read** by the promotion. |
| Acceptance evidence (committed) | upstream `Olhoff@253069262407885a8b759a9e721c4f0a7d3a397d:repro/audits/upstream_olhoffcurrent_capabilities` (`UPSTREAM_CAPABILITY_PROMOTION_READY`); upstream `Olhoff@6b0870850d7407a0f2fc31dbf0d0f318c2e5f2f7:repro/results/S*x*` and `SWEEP_R06.md` (the Pedersen/adaptive-box nine-mesh sweep); `diagnostics/upstream_253069_migration` (this promotion's 160×20 gates and production decision, HARNESS_UPDATE.md); `diagnostics/provenance_gate_hardening` (finalization-gate provenance repair) |
| Supplementary local evidence (NOT committed, NOT required) | `diagnostics/scientific_delta_olhoff_migration` (`OLHOFFCURRENT_MIGRATION_READY_WITH_NAMED_FORMULATION_SPLIT`), `diagnostics/nine_mesh_campaign_audit`, `diagnostics/three_rung_canary_preflight`: analyses that informed the decision, untracked in the primary checkout. Nothing needed to reproduce or validate the migration depends on them; a clean clone lacks them. |
| Promotion date | **2026-09-13** |

## 2. Main repository at promotion

| | |
|---|---|
| Path | `/Users/piotrek/Programming/topOpt4freqMax` |
| Commit before promotion | `013cc48451d33bed61c5c4eea174bbd898d548a2` (`benchmark-methodology-r2`) |
| Where the promotion was made | dedicated worktree, branch `migration/olhoffcurrent-upstream-253069` |
| State | no tracked changes; eight untracked diagnostic directories in the primary checkout, left untouched |

## 3. What is promoted

```
analysis/OlhoffCurrent/+impl/
    algo/            fem/            filter/
    mma/             mma_published/  architecture/{+olh/, olhoffSolve.m, legacy/, docs/}
```

**79 files. All 79 were byte-identical to upstream `253069` at promotion. Since
2026-10-08 five carry a local, default-off adaptation (§7).**

This event replaced 16 files and added 4; the other 59 were already identical.
The per-file record is `diagnostics/upstream_253069_migration/PROMOTION_MAP.md`
and `BYTE_IDENTITY.md`.

The sibling layout is preserved exactly as upstream, because it is load-bearing:
`algo/useMMA.m` locates the MMA variants as
`fileparts(fileparts(mfilename('fullpath')))/mma*`.

### Not promoted, and why

| Excluded | Why |
|---|---|
| `audit_*/`, `results/`, `runs/`, `repro/`, `NOTES.md`, `CLAUDE.md`, `EVIDENCE_MANIFEST.sha256` | upstream scientific evidence and audits; not executable production code |
| `architecture/anchors/`, `architecture/tests/` | upstream's own regression harness; several files hard-code the upstream absolute root |
| `architecture/README.md` | excluded at 695f03b as well; not part of the executable layout |
| `setpaths.m` | replaced by `olhoffcurrent_paths.m`, which additionally *proves* the resolution |
| `top88.m` (repo root) | a byte-identical duplicate of `filter/top88_reference.m` |

## 4. Integrity manifest

| | |
|---|---|
| File | [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json) |
| Files covered | **79** |
| **Source tree SHA-256** | `4ba9a3ae10881344a0e60f2b8a8976c5ec9ceccf966bc2a3da4f37fb5aebffbf` |

The tree hash is the SHA-256 of the sorted `<relative path>  <sha256>` lines, so
it changes if any file changes, is added or is removed, and does not depend on
filesystem ordering. Verify with `olhoffcurrent_source_manifest()`. It was
computed independently (Python, before MATLAB saw the tree) and by
`olhoffcurrent_source_manifest('Write', true)`; the two agree.

## 5. Presets: shared implementation, named formulations

The solver is shared; science is selected by an **explicitly named preset**
(`olhoffcurrent_presets.m`). Every configuration call names its preset —
`olhoffcurrent_config(nelx, nely, 'Preset', name)` — and an unnamed call is
refused, so no call can change formulation silently when production changes.

| canonical preset | delegates to | role |
|---|---|---|
| `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` | `duOlhoffAdaptivePedersen` | **production** since 2026-09-13; a distinct formulation |
| `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` | `duOlhoffFrozenM4` | historical formulation (production 2026-09-07 … 2026-09-13) |
| `duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered` | `duOlhoffFrozenM4` + 3 policy overrides, cap 1600 | historical diagnostic; not production-eligible |

The only **compatibility alias** is `duOlhoffFixedPenaltySensitivityFiltered`,
the pre-2026-09-13 production name. It resolves to the historical β-stall
preset and to nothing else.

**Historical audit IDs are provenance aliases, not API.** M4, TMA, B0, REG160,
`duOlhoffFrozenM4`, TR3_C, CAN3_*, EX3_160, `duOlhoffAdaptivePedersen` and
S160x20 … S800x100 match old evidence to a preset. They are refused as preset
names.

### Production selection is a recorded event

`PROVENANCE.json → production_preset_events` is append-only:

1. **2026-09-07** — initial promotion from `695f03b`: `duOlhoffFixedPenaltySensitivityFiltered`
   (now canonically `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered`),
   config hashes as recorded by the 2026-09-11 campaign (81-row schema).
2. **2026-09-13** — production changes to `duOlhoffPedersenAdaptiveBoxSensitivityFiltered`
   (upstream `duOlhoffAdaptivePedersen` @ `253069`, parent `6b08708`). The
   historical preset is retained unchanged and stays resolvable. New config
   hashes are recorded for both presets under the 87-row schema.

`olhoffcurrent_production_preset()` reads the latest event.

## 6. Configuration hashes

The schema grew from 81 to 87 rows. `olhoffcurrent_config_hash` iterates schema
rows, so **every configuration hash changed** without any historical scientific
value changing. For all 14 recorded historical configurations, the 81 old leaves
are equal, and the 81-row hash recomputed from the migrated configuration
reproduces the recorded hash exactly. Old hashes remain valid identifiers of
the evidence that recorded them. Details:
`diagnostics/upstream_253069_migration/CONFIG_HASH_TRANSITION.md`.

## 7. Adaptations

**2026-10-14 — lower-triangle assembly (local; same matrices).**
A fairness audit of the conference benchmark aligned the per-iteration global
assembly of the three methods. `fem/assemble2D.m` now assembles the 36
lower-triangle entries of each element matrix (indices `mdl.iL`/`mdl.jL` and
values `mdl.K0L`/`mdl.M0L` from `fem/model2D.m`, taken from the symmetric parts
of K0/M0) and mirrors the result with `K + tril(K,-1)'`. The former code passed
all 64 entries to `sparse()` and then formed `(K+K')/2`, which costs twice the
triplets and becomes several times slower once ndof ≥ 2^17 (720×90 and finer).
Below 2^17 the assembled K and M are bitwise those of the former code (checked
at 160×20), so the reproduction tests are unaffected; at 720×90 and finer, where
K0 is symmetric only to round-off, K differs by at most 2.4e-16 relative.

| file | change |
|---|---|
| `fem/assemble2D.m` | lower-triangle assembly + mirror |
| `fem/model2D.m` | `mdl.iL`, `mdl.jL`, `mdl.K0L`, `mdl.M0L` |

**2026-10-14 — density-field stagnation stop rule (local, default-off).**
`stop.rule = 'stagnation_x'` is `stagnation` with the ω_n test replaced by a
test on the density field ρ that `assemble2D` receives, which is the field that
enters the stiffness and mass interpolation and the eigenanalysis. Under the
sensitivity filter this is the design variable itself. Projection, where it
would be the projected field, is refused with this rule, as it is for
`stagnation`. Over the same window of W+1 analysed designs (never the initial
design), each of the W relative changes ‖ρ_j − ρ_{j−1}‖₂/‖ρ_{j−1}‖₂ must be below
`stop.stagnation.relativeChangeTolerance` (default 1e-3, no calibrated value),
and the grey-measure test is unchanged. The largest change in the window is
tested, not the last one. It is selected from outside through
`olhoffcurrent_config(..., 'StopStagnationX', true | struct)`.

| file | change |
|---|---|
| `architecture/olhoffSolve.m` | the change of ρ between consecutive analysed designs and the grey measure recorded per outer iteration when selected; `local_stagnation_x`; `aux.stagXRel` and `aux.stagMndRange`, created only when selected |
| `architecture/+olh/+config/schema.m` | `stagnation_x` added to the `stop.rule` domain; one row `stop.stagnation.relativeChangeTolerance` |
| `architecture/+olh/+config/validate.m` | the guard and projection refusals of `stagnation` cover `stagnation_x` |
| `architecture/+olh/+config/describe.m` | the STOPPING section names the rule and its thresholds |
| `architecture/docs/CONFIG_REFERENCE.md` | the `stop.rule` row and the stagnation rows |

**Configuration hashes change** (90 → 91 rows). No scientific value changes.
The 90-row list is frozen in `tests/fixtures/schema_rows_pre_stagnation_x.json`
together with the production 160×20 hash `a08a14f4…`, which
`tests/test_preset_identity` recomputes from a current configuration.

Unselected, the option leaves the trajectory unchanged: see
`trajectory_effect` in `PROVENANCE.json`. The change is not in upstream.

**2026-10-10 — stagnation stop rule (local, default-off).**
`stop.rule = 'stagnation'` replaces the sec. 3.5.1 design-increment test. The
run stops when, over the W+1 most recent designs whose eigenvalues are known,
ω_n varied by less than `stop.stagnation.objectiveTolerance` relative to its
latest value, and 4·mean(ρ(1−ρ)) varied by less than
`stop.stagnation.graynessTolerance`. W is `stop.stagnation.window`; the
defaults are 10, 1e-3 and 5e-3. Both quantities are taken at the top of each
outer iteration, for the design whose ω is evaluated there anyway. So the
window trails the update by one and never contains the initial design. This is
rule R1 of `examples/Performance/stop_criterion_study`, and it is the common rule
of all three methods. It is selected from outside through
`olhoffcurrent_config(..., 'StopStagnation', true | struct)`.

| file | change |
|---|---|
| `architecture/olhoffSolve.m` | ω_n and the grey measure recorded per outer iteration when selected; the admission branch; `aux.stagObjRange` and `aux.stagMndRange`, created only when selected |
| `architecture/+olh/+config/schema.m` | `stagnation` added to the `stop.rule` domain; three rows `stop.stagnation.*` |
| `architecture/+olh/+config/validate.m` | refuses `stagnation` together with any `stop.guards.*` (they qualify a test it does not use) and with projection (continuation would consume a stale window) |
| `architecture/+olh/+config/describe.m` | the STOPPING section names the rule and its thresholds |
| `architecture/docs/CONFIG_REFERENCE.md` | the `stop.rule` row and the three new rows |

**Configuration hashes change.** `olhoffcurrent_config_hash` iterates schema
rows, so all three rows enter every hash, including runs that never select the
rule (87 → 90 rows). No scientific value changes. The 87-row list is frozen in
`tests/fixtures/schema_rows_pre_stagnation.json`, so a hash recorded before this
change can be recomputed from a current configuration, as was done for the
81 → 87 transition.

Unselected, the option leaves the trajectory unchanged: see
`trajectory_effect` in `PROVENANCE.json`. The change is not in upstream.

**2026-10-08 — relative-increment stop metric (local, default-off).**
`stop.norm = 'relativeL2'` stops on ||Δρ||₂/||ρ||₂ < `stop.tolerance`, where ρ
is the design variable before the update (z under projection). It is selected
from outside through `olhoffcurrent_config(..., 'StopRelativeChangeTolerance', tol)`.

| file | change |
|---|---|
| `architecture/olhoffSolve.m` | the metric branch, plus `aux.dxRel` (kept outside `hist`) |
| `architecture/+olh/+config/schema.m` | `relativeL2` added to the `stop.norm` domain; no row added, so every config hash is unchanged |
| `architecture/+olh/+config/validate.m` | refuses `relativeL2` together with `meshScaled`, the RMS guards, or stage exhaustion |
| `architecture/+olh/+config/describe.m` | the STOPPING line names the relative metric |
| `architecture/docs/CONFIG_REFERENCE.md` | the `stop.norm` row |

The upstream and promoted SHA-256 of each file are listed under `adaptations` in
`PROVENANCE.json`. `SOURCE_MANIFEST.json` was re-recorded for them.

Unselected, the option leaves the trajectory unchanged. This was checked at
160x20 on the production preset under the default rule, `StopToleranceFactor`
0.2, and `StopMaxChangeTolerance` 0.03. In each case ρ, ω, the log and `hist`
(compared with `isequaln`) are bitwise equal to HEAD `be43b90`.

The change is not in upstream. It should be offered there, then come back
through a normal promotion.

### Before 2026-10-08: none

Until 2026-09-13 this file claimed "74 files, one adaptation" (`hist.tOuter`).
That had been stale since commit `1438aa3` (2026-09-09), which added the
stage-exhaustion controller locally. From then on the tree held 75 files and
differed from its promoted base in seven. `SOURCE_MANIFEST.json` was correct
throughout.

Upstream `253069` now carries both capabilities as default-off options:

- `hist.tOuter` — tic first in the outer loop body, toc after every stopping decision;
- `move.continuation.signal = stageExhaustion` together with `stop.rule = stageExhaustion`.

Both reproduce the former local code bitwise (upstream audit T5/T8; this
migration's EX3/EX4 runs). OlhoffCurrent therefore needs no private
solver/controller edits.

## 8. Known pre-existing defect carried by the promotion

`pContinuationDecoupled` and `pMassCompatible` record `hist.move` before the
p-controller reset at p-event iterations. The design trajectory is unaffected.
The defect was introduced upstream at `6b08708` and is deliberately **not
repaired** here; no production or historical preset uses p continuation. See
`diagnostics/upstream_253069_migration/KNOWN_PREEXISTING_DEFECTS.md`.

## 9. Relationships

| Tree | Relationship |
|---|---|
| `analysis/OlhoffM4Reconstruction` | **FROZEN_EVIDENCE.** The frozen conference reconstruction. Reproduced bitwise at 160×20 by the historical β-stall preset. Not a production dependency; hard-blocked from the production MATLAB path. |
| `/Users/piotrek/Programming/Matlab/Olhoff` | **DEVELOPMENT_UPSTREAM.** Never an executable dependency of a production run. Blocked by absolute path. |
| `analysis/OlhoffExperiments` | **EXPERIMENTAL.** Blocked pre-emptively. |
| all other Olhoff trees | **HISTORICAL / AUDIT_ONLY**, classified in `analysis/OLHOFF_IMPLEMENTATION_MAP.md` and blocked. |

## 10. Currentness — and why "upstream is ahead" is not "obsolete"

`olhoffcurrent_currentness()` reports exactly one of:

| State | Meaning |
|---|---|
| `CURRENT` | promoted source intact, and the promoted commit is the head of the recorded upstream branch |
| `LOCAL_MODIFIED` | `+impl/` no longer hashes to `SOURCE_MANIFEST.json`. **The most serious state** |
| `UPSTREAM_AHEAD` | the recorded upstream branch has commits after the promoted one. **Informational** |
| `PROVENANCE_MISMATCH` | the recorded branch is missing, or the promoted commit is not in its history |
| `UPSTREAM_UNREACHABLE` | the development repository is absent; local integrity was still checked |

Since 2026-09-13 the upstream comparison is made against the **recorded branch**
(`source.branch`), not against whatever the development checkout has checked
out. The promoted commit lives on a branch that is not the checkout's current
branch, and currentness must not depend on that.

**Production currentness changes on exactly one event: a human explicitly
accepts an upstream state and promotes it.** `olhoffcurrent_currentness()`
never updates anything.

## 11. The promotion workflow

```
external development
    -> experiment / audit
    -> accepted committed state
    -> explicit promotion into OlhoffCurrent (byte copy from the committed object)
    -> regression verification  (path isolation, currentness, named-preset reproduction)
    -> production (a recorded production_preset_events entry)
```

Production scripts must **never** execute directly from the external repository.
Production source is never edited in place.
