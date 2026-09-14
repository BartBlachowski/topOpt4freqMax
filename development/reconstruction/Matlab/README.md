# Three implementations of the Du & Olhoff eigenfrequency algorithm

This repository carries **three independent MATLAB realizations** of the
Du & Olhoff eigenfrequency-maximization algorithm family. They are kept
separate on purpose. They are *not* three versions of one code, and they must
not be refactored onto shared numerical code while the paper revision is open:
when two of them disagree, that disagreement is evidence, and it is only
evidence if neither was allowed to contaminate the other.

This organization is **temporary, for the paper-revision / reproducibility
phase**. Long-term consolidation into pyMorphoGen is explicitly out of scope.

---

## The map

| generation | what it is | where it lives |
|---|---|---|
| **1. Historical nested MMA** | The original June/July production and reconstruction path. Retained for provenance and forensic analysis only. | [`analysis/OlhoffApproachExact/Matlab/legacy/`](../analysis/OlhoffApproachExact/Matlab/legacy/) |
| **2. Full-coupling OlhoffExact** | The rebuilt full generalized-gradient / LMI realization used in our previous investigations and planned parametric study. | [`analysis/OlhoffApproachExact/Matlab/`](../analysis/OlhoffApproachExact/Matlab/) (top level) |
| **3. Du–Olhoff 2007 reproduction** | Independent clean-room Eq. (22) LP implementation that successfully reproduces the Fig. 3a / Fig. 4 benchmark family. | [`Matlab/reproduction2007/`](reproduction2007/) |

`legacy/` and `full_coupling/` in *this* directory are signposts, not code —
see the note on layout below.

---

## 1. Historical nested MMA

**Status: frozen. Provenance and forensic analysis only. Do not develop.**

The nested-MMA implementation as it stood before the July rebuild. It normally
ran at `rmin = 2.5`, returned increments that had not met its own declared
inner stopping test, ignored that status, and consequently took destructive
full-box steps or entered move-saturated oscillations.

It is kept because several recorded findings — and the reviewer-facing account
of how the earlier effort went wrong — are statements *about this code*. Delete
it and those findings become unverifiable.

Entry points: `legacy/run_simply_simply_exact.m`,
`legacy/run_clamped_clamped_exact.m`, `legacy/run_clamped_simply_exact.m`,
plus the `legacy/verify_*.m` checks.

## 2. Full-coupling OlhoffExact

**Status: active. The planned parametric study runs against this.**

The rebuilt solver: `topopt_freq_exact.m` with a full-coupling LMI
cutting-plane subproblem and adaptive trust-region acceptance. On the Fig. 3a
case it reaches `N = 2` at iteration 16 and then rejects every further trial
step until `trust_region_exhausted` at iteration 23.

Entry points: `run_olhoff_case.m`, `run_all_olhoff_2014.m`, the `run_{cc,cs,ss}_n*.m`
scripts, and the `verify/v_*.m` checks. Reachable through the benchmark as
`optimization.approach = "OlhoffExact"`.

## 3. Du–Olhoff 2007 clean-room reproduction (Eq. 22 LP route)

**Status: imported 2026-08-26. Independent. Do not merge into 1 or 2.**

Reproduces the published benchmark:

| quantity | paper | reproduction |
|---|---|---|
| ω₁ at the Fig. 3a optimum | 174.7 | **170.4709** (−2.4 %) |
| ω₂ at the optimum | 174.7 (bimodal) | **170.8659** (gap 0.23 %) |
| ω₃ at the optimum | 284.9 | **285.1939** (+0.1 %) |
| multiplicity | bimodal | bimodal, `N = 2` |

Its success comes from the interaction of a much smaller filter (1.1–1.5
elements), a genuine Eq. (22) equality-constrained LP, and a small unrejected
fixed update that continues through coalescence.

### What this does and does not establish

- It **validates that the published benchmark is reproducible.**
- It **does not prove** that its undocumented numerical choices — filter
  radius, move limit, multiplicity tolerance, mesh, inner-loop convergence
  test, support idealization — are the ones Du and Olhoff actually used. The
  paper states none of them.
- It is **intentionally kept independent** of implementations 1 and 2 until the
  paper revision is complete.

Do not describe it as "the correct implementation". The accurate phrase is
**Du–Olhoff 2007 clean-room benchmark reproduction (Eq. 22 LP route)**.

See [`reproduction2007/PROVENANCE.md`](reproduction2007/PROVENANCE.md) for the
source manifest, and [`reproduction2007/NOTES.md`](reproduction2007/NOTES.md)
for the evidence log behind every reconstructed choice.

---

## Running each one

```matlab
% 3. clean-room reproduction — standardized runner
addpath('Matlab/reproduction2007/runner');
[x, omega, tIter, nIter, info] = run_repro2007(struct('config','fig3a_best'));

% 3. clean-room reproduction — through the benchmark, like Yuksel/Proposed
data.optimization.approach = 'OlhoffDu2007Repro';
[x, omega, tIter, nIter, mem, nIterStage, telemetry] = run_topopt_from_json(data);
%    telemetry.iterations gives .outer / .inner / .inner_solver

% 3. original reproduction scripts, unchanged, for provenance
cd Matlab/reproduction2007; setpaths(); res = run_case('mycase', struct('move',0.005));

% 2. full-coupling OlhoffExact
data.optimization.approach = 'OlhoffExact';

% 1. historical nested MMA
run('analysis/OlhoffApproachExact/Matlab/legacy/run_simply_simply_exact.m');
```

### In the performance benchmark

Since 2026-08-26 the **Olhoff column of
`examples/Performance/performance_comparison.m` is produced by this
implementation**, not by `analysis/OlhoffApproach/`. The reported name is
unchanged (`Olhoff` / `OlhoffApproach` / `Olhoff--Du`); only the solver moved.
`benchmark_results.json` records the mapping under `metadata.method_dispatch`.

Two settings are scoped to this method there, because the shared benchmark
values are not transferable: `move = 0.005` (the shared 0.2 is an MMA/OC move
limit and collapses this SLP solver to ω₁ ≈ 2.9) and `max_outer = 1600` (the
method is move-saturated, so the shared `max_iters = 10000` would always run in
full). Read `s/iter` and the scaling exponent for this column — its
`iter_total` is a fixed budget, not a convergence result.

Note that `validate_determinism.m`, `validate_history_logging.m`,
`validate_extension_invariance.m`, `exp1_perf_table.m` and
`verify_comparator_counts.m` still dispatch `'Olhoff'` to the **old** solver.

Migration regression gate:

```matlab
addpath('Matlab/reproduction2007/runner');
repro2007_regression('prefix');   % ~15 s, all three frozen baselines
repro2007_regression('full');     % ~7 min, includes final topologies
```

---

## Why the layout is not literally `Matlab/{legacy,full_coupling,reproduction2007}`

Two constraints pulled against the tidy sketch, and both won.

**The historical implementations were not moved.** `OlhoffApproachExact` is
referenced from 18 places, including `tools/Matlab/run_topopt_from_json.m`, the
frozen `examples/Performance/ledger/protocol_ledger.json`, and three experiment
trees (`ablations`, `step_calibration`, `terminal_direction_audit`). Relocating
it mid-revision would churn the protocol ledger and every recorded experiment
path for no scientific gain. `legacy/` and `full_coupling/` here are therefore
**pointer READMEs** to their real locations. WP1 explicitly permits this.

**The reproduction was placed outside `analysis/` deliberately.** This is a
safety property, not a preference. Several repository scripts —
`examples/Revision_v1/exp*.m` and `run_all_revision_experiments.m` — do:

```matlab
addpath(toolsDir);
addpath(genpath(fullfile(repoRoot,'analysis')));   % prepends every subfolder
```

`genpath` prepends, so anything placed under `analysis/` **shadows
`tools/Matlab/`**. The reproduction ships its own `mmasub.m`, `subsolv.m` and
`top88.m`, all three of which also exist elsewhere in this repository. Had it
been imported under `analysis/`, those revision experiments would silently have
started executing the reproduction's copies. Keeping it at `Matlab/` puts it
out of reach of every `genpath(analysis)` sweep at zero cost and with no edits
to existing scripts.

At migration time all three colliding pairs were byte-identical (SHA256
verified), so nothing was actually mis-executed — but "currently identical" is
not a property that survives an edit, which is exactly why the arrangement was
avoided rather than tolerated.

## Path isolation

Each implementation must be able to state which root it executed. For the
reproduction this is enforced, not assumed:

- `repro2007_paths()` adds only the six directories the implementation owns,
  never with `genpath`, and returns an `onCleanup` guard that restores the
  previous path on return — including on error.
- `repro2007_assert_identity()` then checks that **every** owned function
  resolves inside `repro2007_root()`, and **errors** otherwise. A run that
  starts is a run whose implementation identity has been proved.
- Name collisions that resolve correctly are still reported, with a
  byte-identity verdict, in `info.path_identity.shadowed`.

Never add `addpath(genpath('Matlab/reproduction2007'))` anywhere.
