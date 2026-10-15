# stagnation_x graynessTol study: εM = 0.005 / 0.010 / 0.020 (W = 10, εx = 0.02)

2026-10-15. Calibration screening for the grayness-range tolerance εM with W = 10 and εx = 0.02 fixed.
Proposed and Yuksel only; Yuksel runs stagnation_x in both stages, and the window history resets at the handoff.

## Provenance

- Host BIO-2_HELI, Windows 11 (NT 10.0.26200), MATLAB 24.1.0.2603908 (R2024a) Update 3, single thread.
  Campaigns ran sequentially (0.010 from 11:44 to 11:52, 0.005 from 11:52 to 12:00), with no concurrent MATLAB. Both exited 0.
- Source: the window-study baseline tree c9e0440 (Mac HEAD 2d57f52 + uncommitted driver edits). Each new tree differs
  from it only in `R1.graynessTol` and `cfg.runLabel`. This was verified on the remote by SHA-256 over all 3486 files.
  Preflight PASSED (18/18) for each tree.

  | εM    | label                                              | git tree | driver SHA-256 |
  |-------|----------------------------------------------------|----------|----------------|
  | 0.005 | campaign_remote_stagnation_x_w10_rel2e-2_gray5e-3  | 14285b2  | 1f029600…336f |
  | 0.010 | campaign_remote_stagnation_x_w10_rel2e-2_gray1e-2  | cd59850  | 5b45bd35…9511 |
  | 0.020 | campaign_remote_stagnation_x_w10_rel2e-2_gray2e-2 (reused audited baseline) | c9e0440 | ffdbb927…ae3b |

  The 0.020 baseline is reused. Its bit-identical repeat (`…_gray2e-2_repeat`, tree 3eebf66) removes any provenance
  ambiguity.

## Files

`analyze_graytol_study.py`, `graytol_study_summary.txt` (all tables), `graytol_study_rows.csv` (includes the
governing-test classification), `graytol_study_engineering_vs_0p02.csv`, `fig_iterations_vs_ne.png`,
`fig_engineering_vs_ne.png`, `fig_walltime_vs_ne.png`, `fig_topologies_{proposed,yuksel}.png`,
`fig_yuksel_feature_closeups.png`. The close-ups include the εx = 0.005 reference state from the relativeChangeTol study.

**Governing test.** This is decided counterfactually:
- at εM < 0.02, a stop equal to the baseline stop means the relative-change test governs, and a later stop means the
  grayness test governs;
- at the baseline, the stop is compared with the εx = 0.03 run.

## Findings

- No anomalies: all 54 rows NATIVE_CONVERGED on stagnation_x (Yuksel in both stages), with no cap hit and no
  timing-accounting failure.
- **Proposed (control): εM is essentially inert.** At εM = 0.010 one mesh is delayed by 1 iteration; at εM = 0.005 three meshes
  are delayed by 1 to 3. ω₁(E1) changes by at most 0.03 %, |ΔM_nd| ≤ 0.0003, and the trajectories are bit-identical
  prefixes. Only 160×20 stops on the grayness test at every εM.
- **Yuksel: tighter εM lengthens stage 1.** k₁ rises by +0 to +14 at εM = 0.010 and by +1 to +40 at 0.005, mostly on coarse meshes
  (160×20 to 400×50) and at 800×100. Stage 2 barely moves.
- **Yuksel, cost and regularity:** iteration counts become less regular (k₁ CV 0.151 → 0.205 → 0.291; k₁ + k₂ CV 0.136 →
  0.170 → 0.234). Campaign wall time rises by +5 % and +12 %.
- **Yuksel, engineering result:** M_nd falls by 0.000 to 0.008 (εM = 0.010) and by 0.000 to 0.017 (εM = 0.005). ω₁(E1) changes by
  −0.58 % to +0.30 %, with mixed sign.
- **Yuksel 320×40 is not caught.** The grey central X-braces survive at every εM. εM = 0.010 happens to make the
  thresholded design connected (7 → 1 components) with no change in ω₁ (−0.02 %). At εM = 0.005 the design has 23 components and ω₁ is +0.22 %.
  Both stay 1.3–1.5 % below the εx = 0.005 reference state. That late event happens in stage 2 (reference k₂ = 85 vs
  20–28 here). Stage 2 moves slowly there, with changes below 0.02 but above 0.005, while its grayness range is
  already small, so the grayness test cannot see it.
- **Yuksel 800×100:** εM ≤ 0.010 removes the thin end-bay X-braces, but ω₁ falls by 0.57 %. The εx = 0.005 reference
  ends at the baseline ω₁ (+0.03 %). Removing the braces is an intermediate state, not an engineering gain.

## Conclusion

The data point to outcome B: a tighter εM delays Yuksel stage 1 and adds tails, which hurts regularity. It catches
grayness drift (lower M_nd) but no engineering-relevant event, and it has no effect on Proposed.
- **εM = 0.020 is not too loose.**
- **The candidate rule W = 10, εx = 0.02, εM = 0.02 can be frozen for Du-Olhoff validation.**
- **Known limitation:** the Yuksel 320×40 late stage-2 event (+1.5 % ω₁ at εx = 0.005) is isolated, and no
  grayness setting catches it.

No production change was made.
