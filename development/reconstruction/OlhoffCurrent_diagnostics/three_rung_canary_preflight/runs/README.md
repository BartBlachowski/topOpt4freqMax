# runs/

Per-iteration data and scalar records for the two authorized canaries.

| file | rows × cols | content |
|---|---|---|
| `C480x60_three_rung_iterations.csv` | 386 × 55 | frozen `cv_export` schema — column-for-column identical to the validated C320 CSV |
| `C480x60_three_rung_supplement.csv` | 386 × 21 | Part-B columns the frozen schema omits: ω₁…ω₅, gap23, tEig/tGrad/tInner/tOther |
| `C480x60_three_rung_record.json` | — | scalar record: status, counts, ω, gaps, M_nd, controller events, timing decomposition, embedded preflight record, host probes |
| `C800x100_three_rung_iterations.csv` | 468 × 55 | same frozen schema |
| `C800x100_three_rung_supplement.csv` | 468 × 21 | same supplement schema |
| `C800x100_three_rung_record.json` | — | same schema |

Raw trajectories (`RHO`, `DRHO`, `hist`, `cfg`, `meta`, `exh`, `log`) and the
terminal states used by the fixed-work benchmark live in
`analysis/OlhoffCurrent/evidence/three_rung_canary_preflight/`, which is
git-ignored by the implementation's evidence policy and declared with SHA-256
digests in `EVIDENCE.json`.

No 560×70, 640×80 or 720×90 run exists. No nine-mesh campaign was executed.
