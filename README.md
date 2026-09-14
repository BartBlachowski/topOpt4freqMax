# Topology optimization for frequency maximization

This repository contains code for three different approaches to topology optimization for fundamental-frequency maximization. All of them are based on the SIMP method, but differ in how the fundamental frequency is maximized.

1. **Du–Olhoff** — the classical double-loop approach of Du and Olhoff (2007), as a reconstruction.
2. **Yuksel** — the two-level static approximation of Yuksel and Yilmaz (2025).
3. **Proposed** — the newly proposed one-level quasi-static approximation.

## Where things are

| path | what |
|---|---|
| [`analysis/Olhoff/`](analysis/Olhoff/) | Du–Olhoff implementation — the only supported one. Entry point `olhoffcurrent_run`; see its [README](analysis/Olhoff/README.md) |
| [`analysis/Yuksel/`](analysis/Yuksel/) | Yuksel implementation (`Matlab/top99neo_inertial_freq.m`, Python port) |
| [`analysis/Proposed/`](analysis/Proposed/) | Proposed implementation (`Matlab/topopt_freq.m`, `Python/topopt_freq.py`) |
| [`analysis/elastic2D/`](analysis/elastic2D/) | auxiliary compliance-minimization solver |
| [`tools/`](tools/) | shared code: the JSON dispatchers `run_topopt_from_json` (MATLAB, Python), plotting and helpers |
| [`examples/`](examples/) | runnable examples: `ClampedBeam`, `HingedBeam`, `ClampedHingedBeam`, `Building`, `elastic2D` |
| [`examples/Performance/`](examples/Performance/) | **the three-method performance comparison**: `performance_comparison.m` |
| [`tests/`](tests/) | repository checks (`test_development_firewall.py`); method tests live beside each method, e.g. `analysis/Olhoff/tests/` |
| [`docs/`](docs/) | the JSON task schema and technical notes |
| [`paper/`](paper/) | manuscript review material and reference literature |
| [`development/`](development/) | **archive** — see below |

## Running

**Yuksel, Proposed, elastic2D — from a JSON task file:**

```matlab
addpath('tools/Matlab');
[x, omega, tIter, nIter] = run_topopt_from_json('examples/ClampedBeam/BeamTopOptFreq.json');
```

`optimization.approach` selects the method: `"ourApproach"` (Proposed), `"Yuksel"`, `"elastic2D"`. The example folders also contain MATLAB runners (which need `examples/` on the path) and Python runners (`python3 examples/ClampedBeam/run_clamped_beam.py`). The schema is `docs/topopt_config.schema.json`.

**Du–Olhoff — with a named preset:**

```matlab
addpath('analysis/Olhoff');
prod = olhoffcurrent_production_preset();
out  = olhoffcurrent_run(160, 20, 'Preset', prod.name);
```

**Performance comparison of all three methods:** open `examples/Performance/performance_comparison.m`, edit the configuration block at the top (meshes, methods, output label), and run it.

## `development/` is not part of the supported code

`development/` holds the project's history: reconstruction attempts, superseded Olhoff implementations, experiments, audits, earlier benchmark versions and their evidence. It is kept for scientific provenance, **not** for normal runs. Current code, examples and tests never depend on it (`tests/test_development_firewall.py` checks this). Read [`development/README.md`](development/README.md) before using anything there.
