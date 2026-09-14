# Olhoff — the supported Du–Olhoff implementation

> **Renamed 2026-09-14.** This directory was `analysis/OlhoffCurrent` until the
> repository cleanup. Function names keep the `olhoffcurrent_` prefix, and
> `PROVENANCE.json`, `PROVENANCE.md` and `SOURCE_MANIFEST.json` record the
> directory under its name at promotion time; they were moved byte-for-byte.
> Source integrity is verified relative to `+impl/`, so the rename does not
> affect `olhoffcurrent_currentness()`.

> ## `analysis/Olhoff` is the ONLY production Olhoff implementation in `topOpt4freqMax`.
>
> Every other Du–Olhoff tree on this machine — inside this repository or outside
> it — is historical evidence, experimental code, audit material or development
> upstream. **No production script may execute any of them.** A production run
> that can see a second Olhoff implementation **fails closed before it solves
> anything.**

---

## 1. The roles, explicitly

| Tree | Role | Production executable? |
|---|---|---|
| **`analysis/Olhoff`** | **PRODUCTION — the source of truth** | **YES, and only this** |
| `/Users/piotrek/Programming/Matlab/Olhoff` | DEVELOPMENT / RESEARCH UPSTREAM | no |
| everything under `development/` — `OlhoffM4Reconstruction`, `OlhoffApproach*`, `OlhoffRegularized`, `OlhoffReproduced2007`, `Matlab/reproduction2007`, the `olhoff_*` audits | FROZEN_EVIDENCE / HISTORICAL / AUDIT_ONLY | no — `development/` is forbidden as a whole |

The classification made at promotion (2026-09-07) is archived in
`development/migration_history/olhoff_current_promotion/OLHOFF_IMPLEMENTATION_MAP.md`;
the current source-of-truth analysis is
`development/repository_cleanup/SOURCE_OF_TRUTH.md`.

The diagnostics and raw evidence produced on this tree while it was named
`OlhoffCurrent` (`diagnostics/`, `evidence/`), and the study-finalization gates
that govern them (`olhoffcurrent_finalization_gate`, `olhoffcurrent_evidence_gate`,
`olhoffcurrent_evidence_declare`, `EVIDENCE_POLICY.md` and their tests), are
archived under `development/reconstruction/`. They are not needed to run or
verify production.

## 2. How to run it

```matlab
addpath('<repo>/analysis/Olhoff');

prod = olhoffcurrent_production_preset();          % the recorded production choice
out  = olhoffcurrent_run(160, 20, 'Preset', prod.name);
                                       % installs the gate, resolves the NAMED
                                       % preset, solves, and returns the nested
                                       % cost accounting (total AND per outer iteration)
```

A preset name is **required**. `olhoffcurrent_run(160, 20)` without one is
refused, so a call can never change formulation silently when production changes.

`olhoffcurrent_run` installs the fail-closed path guard **before** it solves,
so the implementation that produced a number is proved rather than assumed.

To inspect the formulation without running anything:

```matlab
guard = olhoffcurrent_paths();          %#ok<NASGU>  keep the guard alive
cfg   = olhoffcurrent_config(320, 40, 'Preset', ...
            'duOlhoffPedersenAdaptiveBoxSensitivityFiltered');
olh.config.describe(cfg);               % the mathematics, in scientific terms,
                                        % with the provenance class of every choice
```

## 3. Presets — one shared solver, named formulations

The solver in `+impl/` is byte-identical to upstream Olhoff `253069`. Scientific
behaviour is selected by an **explicitly named preset**
([`olhoffcurrent_presets.m`](olhoffcurrent_presets.m)); nothing is selected by
editing the solver.

| canonical preset | material law | controller / stop | role |
|---|---|---|---|
| `duOlhoffPedersenAdaptiveBoxSensitivityFiltered` | Pedersen (2000) low-density stiffness, linear mass eq. (2) | per-element adaptive move box; natural ‖Δρ‖₂ < ε stop, no guards | **production** (since 2026-09-13) |
| `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` | SIMP ρ³, eq. (4b) mass | four-rung ladder on the β stall; settled-move design-change stop | historical formulation (production until 2026-09-13; the 2026-09-11 campaign) |
| `duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered` | SIMP ρ³, eq. (4b) mass | three-rung ladder and terminal stop by stage exhaustion | historical diagnostic, not production-eligible |

All three share p = 3 fixed, the Sigmund sensitivity filter on every `f_sk` at a
fixed physical R = 0.06, fixed subspace N = 2 with offsets and off-diagonals,
and published MMA on the increment.

**The Pedersen preset is a distinct formulation, not a bug fix of the eq. (4b)
presets.** See `olhoffcurrent_caveat(name)` for each preset's caveat.

**Production** is the latest entry of `PROVENANCE.json → production_preset_events`,
read by `olhoffcurrent_production_preset()`. Changing it means appending an
event; earlier events are never edited, and every preset stays resolvable by
name.

### Aliases

* `duOlhoffFixedPenaltySensitivityFiltered` — the pre-2026-09-13 production name —
  is the **only compatibility alias**. It resolves to
  `duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered` and to nothing else.
* **M4**, **TMA**, **B0**, **REG160**, `duOlhoffFrozenM4`, **TR3_C**, **CAN3_***,
  **EX3_160**, `duOlhoffAdaptivePedersen` and **S160x20 … S800x100** are
  **provenance aliases**. They are refused as preset names. The same holds for
  `S2`, `R1`, `R2`, `P1`, `PD1`, `PM1` and `T800`.

## 4. Why the core lives under `+impl/`

`genpath` skips folders whose name begins with `+`, **and every folder beneath
them**. Archived scripts (e.g. the paper-revision experiments) call
`addpath(genpath(<repo>/analysis))`, and those entries persist for the rest of
the MATLAB session.

The upstream tree and `Matlab/reproduction2007` share **49 bare function
names** — `olhoffOpt`, `innerLoop`, `massScale`, `eigSolve`, `mmasub`,
`subsolv`, `genGrad`, `prepFilter` and more. A plain subfolder here would make
this a *third* competitor in that resolution race.

Under `+impl/` the core is invisible to `genpath` and reachable **only** through
`olhoffcurrent_paths()`, which asserts its identity before returning.

The sibling layout of `algo/ fem/ filter/ mma/ mma_published/ architecture/` is
**load-bearing**: `algo/useMMA.m` locates the MMA variants relative to its own
file, so `algo/` must stay a sibling of `mma_published/`.

## 5. The path invariant, and how it fails closed

> **Exactly one executable Olhoff implementation is visible to MATLAB:
> `analysis/Olhoff`.**

`olhoffcurrent_assert_dispatch` checks **every** symbol this tree owns — 32 of
them, derived from the directory rather than restated as a list — using
`which(name, '-all')`. For each it requires:

1. the name resolves at all;
2. the **winning** resolution is inside `+impl/`;
3. there is **no other candidate** anywhere outside this tree.

Point (3) is the one that matters. The gate this replaces used `which(name)` —
the *first* hit only — which proves *"the winner is right"* but not *"there is
only one candidate"*. A second copy of `innerLoop` further down the path is
invisible to that test right up until an `addpath(..., '-begin')` or a `genpath`
sweep reorders the path and it silently becomes the executed code.

The blacklist in `olhoffcurrent_forbidden_paths` names what we already know
about. It is **not** the whole gate: a bare `.m` file with one of our names,
outside this tree and outside the MATLAB installation, is a blocker **even if
nobody has declared it** — so a new Olhoff implementation added tomorrow still
fails closed.

Class methods (`@cls/f.m`) and package members (`+pkg/f.m`) are skipped: they
cannot take part in bare-name path resolution, so they are not candidates for
the resolution being tested.

## 6. Is it still current?

```matlab
olhoffcurrent_currentness()
```

reports exactly one of `CURRENT`, `UPSTREAM_AHEAD`, `LOCAL_MODIFIED`,
`PROVENANCE_MISMATCH`, `UPSTREAM_UNREACHABLE`. It never updates anything.

**`UPSTREAM_AHEAD` does not mean obsolete.** The upstream repository is a
*development* tree; experimental commits land there constantly and most will
never be promoted. Production currentness changes on one event only: a human
explicitly **accepts** an upstream state and promotes it. See
[`PROVENANCE.md`](PROVENANCE.md) §10.

## 7. Tests

```matlab
test_path_isolation()          % TEST A–E, including helper shadowing
test_currentness()             % provenance, integrity, state model, preset registry
test_source_integrity()        % artifacts ignored, source changes block
test_preset_identity()         % named presets, aliases, formulations, historical hashes (no solve)
test_preset_equivalence()      % historical beta-stall preset, 160x20 vs the frozen conference record
test_named_preset_reproduction('pedersen')         % 160x20 vs committed upstream S160x20
test_named_preset_reproduction('stageExhaustion')  % 160x20 vs pre-migration OlhoffCurrent
test_cost_reporting()          % total and per-outer-iteration cost fields
test_pedersen_adaptive_units() % adaptive move-box rule and Pedersen stiffness law (no solve)
```

## 8. What is here

| Path | What |
|---|---|
| `+impl/` | the promoted implementation, 79 files, byte-identical to upstream `253069` (no adaptations) |
| `+impl/architecture/+olh/` | the canonical configuration package: schema, validation, presets |
| `+impl/architecture/olhoffSolve.m` | the canonical solver — branches on no experiment identifier |
| `+impl/architecture/docs/` | upstream's configuration reference, presets, terminology, field-level provenance |
| `olhoffcurrent_run.m` | **the production entry point** |
| `olhoffcurrent_presets.m` / `olhoffcurrent_preset.m` | the named-preset registry and lookup |
| `olhoffcurrent_config.m` | a named preset, resolved per mesh |
| `olhoffcurrent_production_preset.m` | the recorded production choice |
| `olhoffcurrent_caveat.m` | the caveat of each preset |
| `olhoffcurrent_paths.m` / `_assert_dispatch.m` / `_forbidden_paths.m` | the fail-closed path gate |
| `olhoffcurrent_currentness.m` / `_provenance.m` / `_source_manifest.m` | provenance and integrity |
| `PROVENANCE.md` / `PROVENANCE.json` | where this came from, exactly |
| `SOURCE_MANIFEST.json` | integrity manifest over `+impl/` |
| `tests/` | path isolation, currentness, integrity, preset identity and reproduction, cost reporting |

## 9. Changing it

Production source is **not** edited in place. The workflow is:

```
external development  ->  experiment / audit  ->  accepted committed state
   ->  explicit promotion into analysis/Olhoff  ->  regression verification  ->  production
```

Editing `+impl/` directly makes `olhoffcurrent_currentness()` report
`LOCAL_MODIFIED`, which is the most serious of its states: it means the recorded
provenance no longer describes the code on disk.
