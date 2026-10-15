# stagnation_x relativeChangeTol study: εx = 0.005 / 0.010 / 0.015 / 0.020 / 0.030 (W = 10, gray 0.02)

2026-10-15. Calibration screening for εx with the window frozen at W = 10 and graynessTol at 0.02,
Proposed and Yuksel only. Yuksel runs stagnation_x in both stages, and the window history resets at the handoff.
Du-Olhoff is the later validation case.

## Provenance

- Host BIO-2_HELI, Windows 11 (NT 10.0.26200), MATLAB 24.1.0.2603908 (R2024a) Update 3, single thread,
  campaigns run sequentially 10:52–11:36 with no concurrent MATLAB; all exited 0.
- Source: the baseline tree of the window study (W = 10, git write-tree c9e0440, Mac HEAD 2d57f52 + uncommitted
  driver edits). Each campaign tree differs from it only in `R1.relativeChangeTol` and `cfg.runLabel`. This was
  verified on the remote by SHA-256 over all 3486 files, and only `performance_comparison.m` differs. Preflight
  PASSED (18/18) for each tree.

  | εx    | label                                                     | git tree | driver SHA-256 |
  |-------|-----------------------------------------------------------|----------|----------------|
  | 0.005 | campaign_remote_stagnation_x_w10_rel5e-3_gray2e-2          | 476dbfa  | 96745724…3e  |
  | 0.010 | campaign_remote_stagnation_x_w10_rel1e-2_gray2e-2          | f87903a  | b57b1075…f5  |
  | 0.015 | campaign_remote_stagnation_x_w10_rel1p5e-2_gray2e-2        | 4e12a76  | 431e43b0…e8  |
  | 0.020 | campaign_remote_stagnation_x_w10_rel2e-2_gray2e-2 (baseline, window study) | c9e0440 | ffdbb927…3b |
  | 0.020 | campaign_remote_stagnation_x_w10_rel2e-2_gray2e-2_repeat   | 3eebf66  | 7e935e9f…10  |

  The repeat is bit-identical to the baseline (counts, ω₁, final designs). Wall-time noise between the two
  identical runs is ≤ 3.3 % (median 1.7 %).

## Files

`analyze_reltol_study.py` (uses `exports/` written by `../window_study_stagnation_x_rel2e-2_gray2e-2/wstudy_export.m`),
`reltol_study_summary.txt` (all tables), `reltol_study_rows.csv`, `reltol_study_engineering_vs_0p02.csv`,
`fig_iterations_vs_ne.png`, `fig_engineering_vs_ne.png`, `fig_walltime_vs_ne.png`, `fig_stop_relchange_vs_ne.png`,
`fig_topologies_{proposed,yuksel}.png`, `fig_yuksel_feature_closeups.png`.

## Findings

- No anomalies: all 90 rows NATIVE_CONVERGED on stagnation_x (Yuksel in both stages), with no cap hit and no
  timing-accounting failure.
- **Proposed is insensitive across the whole range.** The trajectory is identical for every εx (bit-identical
  objective-history prefixes), and the stop is simply where the relative change first crosses εx, plus the window.
  Against 0.020: ω₁(E1) changes by ≤ 0.43 % (0.005), ≤ 0.25 % (0.010), ≤ 0.15 % (0.015) and ≤ 0.17 % (0.030). |ΔM_nd| ≤ 0.002, and the topology is the same.
  Tighter εx makes the counts *less* regular (CV 0.043 → 0.098) and adds a mesh-dependent delay
  (+10 to +21 iterations at 0.005, largest on fine meshes).
- **Yuksel, ω₁:** within 0.06 % of 0.020 for εx ∈ [0.015, 0.030] at every mesh except 800×100 (−0.55 % at 0.015,
  where the end-bay X-braces are removed). At 0.010 the change is ≤ 0.47 %.
- **Yuksel, design field:** the design keeps sharpening below 0.010 (relative L2 change vs 0.020 up to 0.27 at 0.005,
  M_nd −0.007 to −0.033).
- **Yuksel 800×100 end-bay X-braces:** present at 0.020 and 0.030, removed at ≤ 0.015. Removing them does not raise ω₁
  (−0.55 %, −0.47 %, −0.03 % at 0.015 / 0.010 / 0.005), so they are topology polish, not an engineering gain.
- **Yuksel 320×40 grey central X-braces:** present for every εx ≥ 0.010, resolved only at 0.005. Resolving them gives
  ω₁(E1) +1.53 %, M_nd 6.5 % → 3.2 % and 7 → 1 binary components, at k₁ + k₂ = 164 instead of 62.
  This is the only change of more than 0.6 % in the study.
- **Which test stops the run:** at 0.030 the grayness-range test governs most stops (grayness range at the stop is
  0.013–0.020; several rows are identical to 0.020). The Yuksel 160×20 stage 1 stops at k₁ = 64 for every εx ≥ 0.010,
  so it is not governed by εx (inferred to be the grayness test).
  At 0.015–0.020 the relative-change test governs: the window maximum at the stop sits just below εx.
- **Cost (campaign totals vs 0.020):** Proposed +33 / +15 / +3 / −8 %, Yuksel +151 / +35 / +9 / −8 % at
  0.005 / 0.010 / 0.015 / 0.030. Yuksel counts at 0.005 are very irregular (k₂ 37–90, maximum jump 48).

## Conclusion

- **Plateau:** εx ∈ [0.015, 0.020] is a plateau for both methods. ω₁ differs by ≤ 0.15 % (Proposed) and ≤ 0.06 % (Yuksel, except the 800×100
  end-bay polish at −0.55 %), with small delays.
- **Recommended value: 0.020.** It is the loosest value at which εx, and not the grayness test, governs the stop.
  0.030 gives the same engineering result but is mostly a grayness-range rule.
- **Caveat:** Yuksel 320×40 holds a latent topology event (+1.5 % ω₁) that only εx = 0.005 reaches.

No production change was made. Du-Olhoff validation is next.
