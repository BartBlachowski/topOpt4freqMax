# OlhoffCurrent

> ## `analysis/OlhoffCurrent` is the ONLY production Olhoff implementation in `topOpt4freqMax`.
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
| **`analysis/OlhoffCurrent`** | **PRODUCTION — the source of truth** | **YES, and only this** |
| `analysis/OlhoffM4Reconstruction` | FROZEN_EVIDENCE — the frozen historical reconstruction the conference numbers came from | no |
| `analysis/OlhoffExperiments` | EXPERIMENTAL — for future experimental implementations and scripts, when created | no |
| `/Users/piotrek/Programming/Matlab/Olhoff` | DEVELOPMENT / RESEARCH UPSTREAM | no |
| `analysis/OlhoffApproach*`, `analysis/OlhoffRegularized`, `analysis/OlhoffReproduced2007`, `Matlab/reproduction2007`, `analysis/olhoff_*_audit`, … | HISTORICAL / AUDIT_ONLY unless explicitly reclassified later | no |

The full classification of every discovered tree is in
[`analysis/OLHOFF_IMPLEMENTATION_MAP.md`](../OLHOFF_IMPLEMENTATION_MAP.md).
There is exactly one `PRODUCTION` entry in it.

## 2. How to run it

```matlab
addpath('<repo>/analysis/OlhoffCurrent');

out = olhoffcurrent_run(160, 20);      % installs the gate, resolves the
                                       % production preset, solves, and
                                       % returns the nested cost accounting
```

`olhoffcurrent_run` installs the fail-closed path guard **before** it solves,
so the implementation that produced a number is proved rather than assumed.

To inspect the formulation without running anything:

```matlab
guard = olhoffcurrent_paths();          %#ok<NASGU>  keep the guard alive
cfg   = olhoffcurrent_config(320, 40);
olh.config.describe(cfg);               % the mathematics, in scientific terms,
                                        % with the provenance class of every choice
```

## 3. The production preset

There is exactly one, and it has a name:

```
duOlhoffFixedPenaltySensitivityFiltered
```

* **FixedPenalty** — SIMP `p = 3` held **constant**, no `p` continuation.
  (Du & Olhoff §2.1 says `p` is "normally assigned values increasing from 1 to
  3"; fixing it is a **reconstruction ruling**, made because the reported
  initial eigenfrequencies fit `p = 3` and not `p = 1`.)
* **SensitivityFiltered** — Sigmund (1997) **sensitivity** filter applied to
  every `f_sk`, at a fixed **physical** radius `R = 0.06·b`. No density filter,
  no Heaviside projection.

The preset does **not** restate the fields it needs. It delegates to the
promoted upstream preset `olh.presets.duOlhoffFrozenM4`, so the production
realization cannot silently drift from the accepted canonical one.

Production scripts **name the preset**. They never rebuild a historical
realization out of a handful of switches.

### Historical codes are provenance, never API

The same realization appears in the historical record as **M4**, **TMA**,
**B0**, **REG160**, and upstream as the preset name `duOlhoffFrozenM4`. Those
are **provenance aliases and experiment identifiers**. They are recorded so old
evidence can be matched to new runs. They are **not** canonical user-facing
terminology, and no production script should use them. The same applies to
`S2`, `R1`, `R2`, `P1`, `PD1`, `PM1` and `T800`.

## 4. Why the core lives under `+impl/`

`genpath` skips folders whose name begins with `+`, **and every folder beneath
them**. Six scripts under `examples/Revision_v1/` call
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
> `analysis/OlhoffCurrent`.**

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
[`PROVENANCE.md`](PROVENANCE.md) §8.

## 7. Tests

```matlab
test_path_isolation()          % TEST A–E, including helper shadowing
test_currentness()             % provenance, integrity, state model
test_preset_equivalence()      % 160x20 vs the frozen conference realization
test_preset_equivalence([160 20; 320 40])
```

## 8. What is here

| Path | What |
|---|---|
| `+impl/` | the promoted implementation, byte-identical to upstream `695f03b` except one documented adaptation |
| `+impl/architecture/+olh/` | the canonical configuration package: schema, validation, presets |
| `+impl/architecture/olhoffSolve.m` | the canonical solver — branches on no experiment identifier |
| `+impl/architecture/docs/` | upstream's configuration reference, presets, terminology, field-level provenance |
| `olhoffcurrent_run.m` | **the production entry point** |
| `olhoffcurrent_config.m` / `olhoffcurrent_preset.m` | the production preset, resolved per mesh |
| `olhoffcurrent_paths.m` / `_assert_dispatch.m` / `_forbidden_paths.m` | the fail-closed path gate |
| `olhoffcurrent_currentness.m` / `_provenance.m` / `_source_manifest.m` | provenance and integrity |
| `PROVENANCE.md` / `PROVENANCE.json` | where this came from, exactly |
| `SOURCE_MANIFEST.json` | integrity manifest over `+impl/` |
| `tests/` | path isolation, currentness, preset equivalence |

## 9. Changing it

Production source is **not** edited in place. The workflow is:

```
external development  ->  experiment / audit  ->  accepted committed state
   ->  explicit promotion into OlhoffCurrent  ->  regression verification  ->  production
```

Editing `+impl/` directly makes `olhoffcurrent_currentness()` report
`LOCAL_MODIFIED`, which is the most serious of its states: it means the recorded
provenance no longer describes the code on disk.
