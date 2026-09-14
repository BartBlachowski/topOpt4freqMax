# Yuksel benchmark (two-stage inertial load + dynamic)

Standalone Python port of MATLAB files in `source_of_truth/fromPaper/Yuksel/`:
- `top99neo_inertial_freq.m`
- `top99neo_dynamic_freq.m`
- `run_fig4_simply_supported.m`
- `run_fig8_fixed_pinned.m`
- `run_fig9_cantilever.m`

## Run

Normal inertial runs:
- `python -m examples.fromPaper.Yuksel.run_case --case fig4`
- `python -m examples.fromPaper.Yuksel.run_case --case fig8`
- `python -m examples.fromPaper.Yuksel.run_case --case fig9`

Quick smoke run:
- `python -m examples.fromPaper.Yuksel.run_case --case fig4 --quick`

Run with dynamic solver history:
- `python -m examples.fromPaper.Yuksel.run_case --case fig4 --with-dynamic`

Custom output directory:
- `python -m examples.fromPaper.Yuksel.run_case --case fig4 --out /tmp/yuksel_out`

## Outputs

Each case writes under `output/<figure>/` (or under `--out`):
- `stage1_iteration_log.csv`
- `stage2_iteration_log.csv`
- `final_design_field.npz`
- `final_plot.png`
- `results_inertial.npz`
- `snapshots/` (if snapshots enabled)

If `--with-dynamic` is used:
- `results_dynamic.npz`
- `dynamic_iteration_log.csv`
- `dynamic_convergence.png`
