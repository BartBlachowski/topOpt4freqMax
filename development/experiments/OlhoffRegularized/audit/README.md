# audit/ — audit-only material

Nothing here is production code. It exists to produce and check the evidence in
`../AUDIT_REPORT.md`, and it deliberately does **not** reuse the production
model builder or the production stopping logic.

## scripts/

| script | role |
|---|---|
| `audit_run.m` | one instrumented production run → `results/<tag>/{run.mat,trajectory.csv}` and one `AUDITRESULT` summary line. Changes no numerical policy: every `runCfg` entry is the repository default. |
| `audit_run_variant.m` | the same, with explicitly listed deviations recorded in the saved metadata, so a diagnostic can never be mistaken for a result. |
| `audit_write_csv.m` | flat trajectory export (36 columns). |
| `audit_report_run.m` | re-emit an `AUDITRESULT` line from a saved `run.mat`. |
| `audit_model.m`, `audit_assemble.m` | **independent** rebuild of the FE model from the problem definition, not by calling the production private `localModel`. Cross-checked against the production ρ and ω. |
| `audit_ascent.m` | independent maximum feasible first-order ascent of `λ_min` by Kelley cutting planes on the sub-eigenvector, LP master. Uses only `linprog` and `eig` on an N×N — no `deltaLambda`, no MMA. |
| `audit_ascent_lp.m` | control: the same problem under the Eq. (22) equality restriction, so the effect of the off-diagonal information is measured rather than assumed. |
| `audit_stationarity.m` | WP2 driver: rebuild → cross-checked eigensolve → independent multiplicity → ascent at three radii for N = 1,2,3 → self-consistent cluster → **physical fixed-t eigensolves** on the ordered spectrum. |
| `audit_all_stationarity.m` | WP2 over a list of tags. |
| `audit_wp5_table.m` | per-run detail block required by WP5. |
| `audit_main_table.m` | the report's headline table rows. |
| `audit_compare.m` | terminal-design topology PNGs and pairwise density distances (same basin? — answered numerically). |
| `audit_volume_mma.m` | does the *penalised* volume row of the nested MMA let the filtered volume drift? |
| `probe_timing.m`, `probe_components.m`, `probe_grow.m` | cost and behaviour probes. |
| `summarize.py` | collect every `AUDITRESULT` line into one table. |

## results/

One directory per run tag (`run.mat`, `trajectory.csv`, `stationarity.mat` when
WP2 has been run on it), plus:

* `*.BEFORE` / `*.ASSHIPPED` — pre-correction copies of every file the audit
  changed;
* `Rpre1b_*` — corrected runs made **before** defect CV-1b was found. Retained
  as the before/after evidence for that correction; **not results**;
* `C_*` — an intermediate correction state retained because it exhibits defect
  CV-4 in its pure form; **not results**;
* `DIAG_*` — deliberate diagnostic deviations; **not results**.

Tag prefixes: `ss…` as shipped, `R_…` corrected, `C_`/`Rpre1b_`/`DIAG_` as
above. Problem codes: `ss` simply supported, `cs` fixed–pinned, `cf` cantilever
with concentrated mass.

## logs/

Raw stdout of every run, including the per-iteration trace.
