# Olhoff benchmark (SIMP frequency optimization with MMA)

Standalone Python port of MATLAB files in `source_of_truth/fromPaper/Olhoff/`:
- `run_freq_benchmark.m`
- `topFreqOptimization_MMA.m`
- `freqOpt_CC.m`, `freqOpt_CS.m`, `freqOpt_SS.m`

## Run

Normal runs:
- `python -m examples.fromPaper.Olhoff.run_case --case all`
- `python -m examples.fromPaper.Olhoff.run_case --case CC`

Quick smoke run:
- `python -m examples.fromPaper.Olhoff.run_case --case CC --quick`

Custom output directory:
- `python -m examples.fromPaper.Olhoff.run_case --case CC --out /tmp/olhoff_out`

## Outputs

Each case writes under `output/<BC>/` (or under `--out`):
- `iteration_log.csv`
- `final_design_field.npz`
- `final_plot.png`
- `results.npz`
- `snapshots/` (if snapshots enabled)

## Eigen solve detail

Smallest generalized eigenvalues are solved with shift-invert:
`eigsh(Kf, M=Mf, sigma=0.0, which="LM")`
(used instead of `which="SM"` in shift-invert mode).
