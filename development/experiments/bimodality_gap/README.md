# docs/bimodality_gap — mesh refinement and the bimodality gap of the Olhoff density field

Evidence package for `bimodality_gap.tex` / `bimodality_gap.pdf`.

## Layout

| path | content |
|---|---|
| `bimodality_gap.tex`, `.pdf` | the document (compile: `pdflatex bimodality_gap.tex` twice) |
| `scripts/bg_metrics.m` | the metric definitions (M_nd, intermediate fractions, histogram, 0.5-contour length, gray-band width, band/bulk decomposition, distances to phases) |
| `scripts/bg_extract_campaigns.m` | read-only extraction from the two recorded nine-mesh campaigns (`examples/Performance/conference_benchmark/{nine_mesh_pedersen_b21483b,campaign_9mesh_r2}`) |
| `scripts/bg_extract_prior_interventions.m` | read-only extraction of the recorded fixed-move interventions on the SIMP formulation (`analysis/OlhoffCurrent/evidence/move_activity_400`, `analysis/OlhoffCurrent/diagnostics/move_stop/runs`) |
| `scripts/bg_arms.m` | registry of the single-factor experiment arms (preset + overrides) |
| `scripts/bg_run_arm.m` | runs one arm through the production route (`olhoffcurrent_paths` gate -> `olh.config.resolve` -> `olhoffSolve`), records cfg hash, `+impl` tree hash, repo HEAD |
| `scripts/bg_launch_one.sh`, `bg_launch_queue.sh`, `bg_queue*.txt` | the exact launch commands (MATLAB R2025b `-batch`, one computational thread per run) |
| `scripts/bg_extract_new_runs.m` | metrics + per-iteration CSVs + bitwise identity checks for `runs/*.mat` |
| `scripts/bg_bulk_gray.m` | split of the bulk-gray term into coreless members and gray plateaus (`data/bulk_gray_classification.csv`) |
| `scripts/bg_analytic.py` | material-law table and discrete filter-kernel table |
| `scripts/bg_plots.py`, `bg_provenance.py` | figures, LaTeX tables, run index |
| `runs/` | every new run: `BG_<arm>_<mesh>.mat` (rho, omega, hist, aux, cfg, log) and `.json` summary |
| `data/` | CSV/JSON/MAT evidence: `campaign_metrics.csv`, `new_run_metrics.csv`, `prior_intervention_metrics.csv`, `iter_*.csv` (per outer iteration), `rho_*.mat` (final designs), `filter_kernel.csv`, `material_laws.csv`, `identity_checks.json`, `RUN_INDEX.json` |
| `figures/` | PDF/PNG figures and generated `tab_*.tex` tables |
| `logs/` | MATLAB stdout of every run, queue timestamps, pdflatex log |

## Reproduce

```
# 1. existing evidence (read-only sources)
matlab -batch "addpath('docs/bimodality_gap/scripts'); bg_extract_campaigns(); bg_extract_prior_interventions();"
python3 docs/bimodality_gap/scripts/bg_analytic.py
# 2. new arms (about 20 single-thread CPU-hours in total; the queue files list every (arm, mesh))
docs/bimodality_gap/scripts/bg_launch_queue.sh ; docs/bimodality_gap/scripts/bg_launch_queue2.sh
docs/bimodality_gap/scripts/bg_launch_one.sh budget400 640 80 ; docs/bimodality_gap/scripts/bg_launch_one.sh budget400 800 100
# 3. metrics, checks, figures, tables, document
matlab -batch "addpath('docs/bimodality_gap/scripts'); bg_extract_new_runs(); bg_bulk_gray();"
python3 docs/bimodality_gap/scripts/bg_plots.py ; python3 docs/bimodality_gap/scripts/bg_provenance.py
cd docs/bimodality_gap && pdflatex bimodality_gap.tex && pdflatex bimodality_gap.tex
```

Nothing under `examples/Performance/conference_benchmark` or `analysis/OlhoffCurrent` is modified by any script here; the production tree `analysis/OlhoffCurrent/+impl` is used unchanged (its live tree hash is recorded in every run).
