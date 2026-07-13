# Revision_v1 governed archive

Everything under `archive/` is **preserved provenance, not reviewer evidence**.

No file in this tree may be used to produce a manuscript table, figure,
frequency comparison, convergence claim, speedup, scaling exponent, or response-letter
statement. Archived artifacts are numerically unaltered; only their classification and
location changed. Historical wording inside archived reports (for example "exact",
"faithful", "reference", "benchmark", "production", "migrate", "freeze", "recommended")
records an intermediate assessment and is **superseded by this archive classification**.

Nothing here is scheduled for deletion. Archived material may be cited only as a record
of a closed investigation or a superseded run.

## `olhoff_exact_reconstruction/`

**Policy:** `analysis/OlhoffApproachExact` is a completed diagnostic reconstruction
campaign. It is not part of the reviewer evidence chain and must not be used by any
active Revision_v1 experiment. `analysis/OlhoffApproach` is the only active local
comparison implementation.

Contains the six standalone reconstruction scripts (`scripts/`) and their seven output
directories (`output/`), plus the original `OLHOFF_EXACT_ARCHIVE.md` index. These are the
only files in Revision_v1 that add `analysis/OlhoffApproachExact` to the MATLAB path; by
being archived they are excluded from the production path allowlist in
`run_all_revision_experiments.m`.

Supports only the negative conclusion: the attempted reconstruction did not establish
paper fidelity after extensive diagnostics.

## `obsolete_evidence/`

Experiments **retired from the reviewer evidence chain** because the claims they supported
no longer exist, or because they are construct-invalid as evidence. See
`obsolete_evidence/README.md`.

| Directory | Retired by | Reason |
|---|---|---|
| `exp1_exp5/` | [SCIENTIFIC_DECISION_EXP1_EXP5.md](../SCIENTIFIC_DECISION_EXP1_EXP5.md) | EXP1 supports zero surviving manuscript claims and is construct-invalid as a benchmark (the local comparators are not faithful reference implementations). EXP5 depended only on EXP1 and its scaling claim is withdrawn. |

## `diagnostics/`

Closed investigations with residual scientific value but no reviewer-evidence role.

| Directory | Question asked | Outcome |
|---|---|---|
| `eq4b_hypothesis_test/` | Does the Eq. 4b mass interpolation rescue Exp3 and suppress localized modes? | **Refuted** — all four sub-questions answered "no"; run capped 2000/2000 and failed the A5 check. |
| `s1_mode_diagnostic/` | Elementwise energy/localization classification of the Exp3 400x50 spectrum. | Diagnostic; established the localized low-density mode population. |
| `localized_mode_onset/` | At which mesh do localized low-density modes take over? | Diagnostic; explicitly **not** a mesh-convergence study. 3/6 meshes mode-invalid. |
| `exp2_alpha1_discrepancy_diagnosis.md` | Source of the alpha=1 discrepancy. | Closed diagnostic note. |

## `superseded_runs/`

### `pre_authoritative/`

Results produced **before** the authoritative load `F(x) = omega0^2 * M(x) * Phi0` was
adopted. They describe a different mathematical method and cannot be compared with, or
mixed into, authoritative results. Includes the legacy master result
(`all_revision_results.mat`, which records Exp2 and Exp3 as NaN/empty), the legacy Exp1
/ Exp2b / Exp4 / Exp5 result artifacts, the Exp4 variant diaries, and the per-alpha
building correlation exports.

`building_mode_plots_ambiguous/` holds 200 PNGs written under the generic
`topopt_config_*` basename. They were overwritten across alpha values, so only the last
run survives and **which alpha each plot belongs to is unrecoverable**. Archived rather
than deleted because the pixels are real output; they must not be published because their
provenance is ambiguous.

### `campaign_r1_full_20260701/`

The failed orchestrated campaign `r1_full_20260701T141604990` (status `failed`): EXP1
completed, EXP2 threw `eigs:AminusBSingular` after 2.24 s, and EXP2b/EXP3/EXP4/EXP5 never
ran. Quarantined so that the next campaign starts from a clean registry and cannot
silently resume onto, or consume, a stale stage artifact — in particular so that EXP5
cannot consume the superseded EXP1 timing `.mat`.

`exp1/` inside this directory was marked "accepted" by the runner on **artifact presence
only**. It is not scientifically accepted: its Olhoff comparator terminated at the 2000
iteration cap and its setup/loop/total timing decomposition is unsound. It must be
regenerated.
