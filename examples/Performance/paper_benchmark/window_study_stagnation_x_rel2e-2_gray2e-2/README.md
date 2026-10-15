# stagnation_x window-length study: W = 10 / 15 / 20 (Proposed + Yuksel)

2026-10-15. Screening study for the persistence window W of `stagnation_x`
(relativeChangeTol = 2e-2, graynessTol = 2e-2; objectiveTol 2e-2 present but not read).
Du-Olhoff is not included; it is the later validation case.

## Provenance

- Host BIO-2_HELI, Windows 11 (NT 10.0.26200), MATLAB 24.1.0.2603908 (R2024a) Update 3,
  MKL 2023.2 AVX512_E1, maxNumCompThreads = 1, run sequentially (no concurrent MATLAB).
- Source: the Mac working tree of branch `benchmark-methodology-r2` at HEAD 2d57f52 plus the
  uncommitted driver edits (2e-2/2e-2, Olhoff off), transferred byte-exact (git archive + scp).
  The three trees differ only in `examples/Performance/performance_comparison.m`, and there only in
  `R1.window` and `cfg.runLabel` (W = 10 has the same two-line edit to the label):

  | W  | git tree (write-tree)                      | driver SHA-256 |
  |----|--------------------------------------------|----------------|
  | 10 | c9e0440f89ebb141e8d176329d3462882b2b6846   | ffdbb927…ae3b  |
  | 15 | e14100d56b37b1916292bc61469c05c5f9d78fd0   | 5952c63d…e234  |
  | 20 | 7613fe35bde5c1b22a460d846a170ca6d56744a3   | b6663c7e…4f58  |

  Solver code is identical to the remote's previous tree 1d3f86d (audited campaign).
- Preflight PASS (18/18) for all three trees before launch, and again inside each campaign.
- Campaigns (remote `C:\Users\ptauzow\Programming\wstudy\`): W10 10:08–10:15, W15 10:15–10:24,
  W20 10:24–10:32, all MATLAB exit 0. Results copied to
  `../campaign_remote_stagnation_x_w{10,15,20}_rel2e-2_gray2e-2/`.
- Mac reference with the same W = 10 configuration: `../campaign_mac_stagnation_x_no_olhoff_rel1e-2_gray1e-2_audited_4`
  (stale label; it ran 2e-2/2e-2).

## Files

- `analyze_window_study.py`: the analysis script; `exports/` holds the final designs and objective histories
  exported from each `benchmark_records.mat` (by `wstudy_export.m`, in this folder).
- `window_study_summary.txt`: all tables. `window_study_rows.csv`: one row per (W, method, mesh).
  `window_study_engineering_vs_w10.csv`: change in ω₁, M_nd and the design relative to W = 10.
- `fig_iterations_vs_ne.png`, `fig_walltime_vs_ne.png`, `fig_engineering_vs_ne.png`,
  `fig_topologies_{proposed,yuksel}.png`.

## Findings

1. **No anomalies.** All 54 rows NATIVE_CONVERGED on `stagnation_x_tolerance` (Yuksel: both stages). No cap hit,
   no timing-accounting failure. Remote W = 10 reproduces the Mac W = 10 run exactly: same counts at every mesh,
   ω₁(E1) identical to 1e-6 relative.
2. **Proposed: the window is a pure delay.** k_stop(W) = k_stop(10) + (W − 10) at all nine meshes, the reported
   window maximum of the relative change is identical across W, and the objective histories are bit-identical
   up to the W = 10 stop. Once the relative change first drops below 0.02 it never rises above it again, so W = 10
   never stopped on a short accidental plateau. The irregularity across meshes (38–44, std 2.0, maximum jump 4,
   4 sign changes) is unchanged in absolute terms. It comes from the iteration at which the threshold is first
   crossed, not from the window. ω₁(E1) changes by at most 0.29 % (slightly *lower* with longer W at fine meshes),
   |ΔM_nd| ≤ 0.0017, binary mismatch ≤ 2.3 %, and the topologies are visually identical.
3. **Yuksel: stage 2 is almost a pure delay (+W − 10 ± 1). Stage 1 keeps evolving after the W = 10 stop**
   (k1 grows by W − 10 plus 0–10 more, most at 160×20: 64 → 75 → 84). k1 irregularity does not improve:
   the 160×20 outlier grows (maximum jump 15 → 18 → 21) and sign changes go 2 → 5 → 3. Longer W gives crisper designs
   (M_nd −0.002 to −0.012) with ω₁(E1) within ±0.5 % and mixed sign (800×100: −0.47 % at W15, −0.41 % at W20).
   **Visible topology difference: Yuksel 800×100 at W = 10 still carries thin X-braces in both end bays, which
   are gone at W = 15 and W = 20.** Elsewhere the differences are thin-member details: at 320×40 the gray central
   X-braces fragment under the 0.5 threshold, so the binary component count goes 7 → 19 → 35.
4. **Cost.** Total wall time increases by +7 to +27 % (Proposed) and +16 to +40 % (Yuksel). The per-iteration cost
   does not change with W. The fitted exponents do not move (Proposed p ≈ 1.19, Yuksel p ≈ 1.12, R² 0.98–0.99).

Recommendation: keep W = 10. No production change was made; Du-Olhoff validation is pending.
