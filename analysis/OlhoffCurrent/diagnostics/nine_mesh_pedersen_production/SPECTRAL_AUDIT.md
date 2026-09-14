# SPECTRAL_AUDIT (Part 14)

**Sources.** `evidence/history/<mesh>.csv`, which comes from the tapped `res.hist` of the production solves, and `METRICS.json` `per_mesh`.

**Thresholds.** Only existing thresholds are used, as preregistered:

- `multiplicity.tolerance` = **0.05**, the resolved configuration row;
- coalescence gap < **0.02**, the upstream `repro/run_repro.m` `coalescenceIter`;
- ω₁ spike ω₁(k) < **0.7**·ω₁(k−1), the upstream sweep audit's `spike_events`.

**Definitions.** gap12 is (ω₂−ω₁)/ω₁ of the eigen-analysis at each outer iterate. The terminal gap is taken from the final analysis.

## 1. Per-mesh spectral record

| mesh | terminal gap (native) | terminal gap (eq. 4 re-eval) | min gap in history (@outer) | iterations gap < 0.05 | iterations gap < 0.02 | first gap < 0.02 | last gap < 0.05 | touch→separate episodes | ω₁ spikes (0.7) | largest one-step ω₁ drop | effectively bimodal at end (gap < 0.05) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 160x20 | **0.70 %** | 0.65 % | 0.259 % (@90) | 114 / 121 | 96 | 8 | 121 (terminal) | 0 | 0 | 0.71 % | **yes** |
| 240x30 | 11.84 % | 11.70 % | 0.355 % (@8) | 24 / 111 | 20 | 8 | 31 | 1 | 0 | 0.30 % | no |
| 320x40 | 17.66 % | 17.53 % | 0.048 % (@9) | 32 / 101 | 25 | 8 | 39 | 1 | 0 | 0.03 % | no |
| 400x50 | 19.03 % | 18.90 % | 0.252 % (@10) | 29 / 93 | 24 | 8 | 36 | 1 | 0 | 0.02 % | no |
| 480x60 | 22.55 % | 22.40 % | 0.043 % (@9) | 31 / 112 | 19 | 8 | 38 | 1 | 0 | 0.01 % | no |
| 560x70 | 24.47 % | 24.30 % | 0.099 % (@10) | 34 / 130 | 21 | 9 | 42 | 1 | 0 | < 0.01 % | no |
| 640x80 | 24.34 % | 24.17 % | 0.171 % (@11) | 40 / 156 | 27 | 10 | 48 | 1 | 0 | < 0.01 % | no |
| 720x90 | 22.39 % | 22.23 % | 0.715 % (@11) | 48 / 204 | 34 | 10 | 57 | 1 | 0 | < 0.01 % | no |
| 800x100 | 18.32 % | 18.16 % | 0.416 % (@12) | 76 / 246 | 58 | 11 | 86 | 1 | 0 | < 0.01 % | no |

## 2. Common trajectory

Figure: `figures/histories_spectral_box_cost.png`.

1. **Iterations 1–12.** From the uniform ρ = 0.5 start, ω₁ rises steadily, from about 68 to about 135–145 by outer 8. ω₂ first rises to about 300, then falls. The two meet between outer 8 and 12 at every mesh, with a minimum gap of 0.04–0.7 %.
2. **Separation (240x30 and finer).** After this near-coalescence the pair separates. gap12 last falls below 0.05 at outer 31 (240x30), rising steadily to outer 86 (800x100). The near-multiplicity phase is longer at finer meshes: 24 iterations at 240x30, 76 at 800x100.
3. **160x20 is the exception.** The pair never separates: gap12 < 0.05 in 114 of 121 iterations and < 0.02 in 96. The run ends at a native gap of 0.70 %, an effectively bimodal terminal state.
4. **Mode order at the gap minimum.** The histories store sorted eigenvalues without mode tracking. It therefore **cannot be determined** whether ω₁ and ω₂ exchanged order at the gap minimum, i.e. whether a true crossing occurred. The events are reported as "touch then separate".

## 3. Required checks

| check | finding |
|---|---|
| mode crossing | One touch-then-separate event per mesh from 240x30 up (§2). Crossing versus touching cannot be decided from the recorded data. |
| near multiplicity | Present in every run during outer 8–86. It is persistent, and terminal, only at 160x20. |
| localized-mode collapse | **Not observed.** See §4. |
| sudden ω₁ drop | **None.** 0 spike events at every mesh. The largest one-step relative decrease of ω₁ anywhere is 0.71 % (160x20). |
| discontinuous gap | No jump in gap12 beyond the early touch-then-separate transient. After separation, gap12 grows smoothly (figure, upper right). |
| recovery after transients | Not applicable, since no ω₁ drop occurred. ω₁ reached 99 % of its run maximum at outer 32–85 at eight meshes, and at outer 171 at 800x100. |
| suspicious frequency spikes | None in ω₁. The early ω₂ overshoot to about 300 appears at every mesh within the first 5 iterations of the uniform start, so it is a property of the start rather than of any single mesh. |

## 4. Localized low-density modes at the terminal state

| mesh | mode 1: E_x / E_y, centreline sign changes | mode 2: E_y, sign changes | mode-1 kinetic energy in ρ ≤ 0.1 | void fraction ρ ≤ 0.1 | evaluator E1/E2/E3 status, selected ordinal |
|---|---|---|---|---|---|
| 160x20 | 0.014 / 0.986, 0 | 0.965, 1 | 0.69 % | 42.1 % | PASS / PASS / PASS, 1 / 1 / 1 |
| 240x30 | 0.013 / 0.987, 0 | 0.968, 1 | 0.34 % | 42.5 % | PASS ×3, 1 / 1 / 1 |
| 320x40 | 0.012 / 0.988, 0 | 0.969, 1 | 0.34 % | 41.6 % | PASS ×3, 1 / 1 / 1 |
| 400x50 | 0.012 / 0.988, 0 | 0.970, 1 | 0.33 % | 42.8 % | PASS ×3, 1 / 1 / 1 |
| 480x60 | 0.012 / 0.988, 0 | 0.970, 1 | 0.38 % | 42.5 % | PASS ×3, 1 / 1 / 1 |
| 560x70 | 0.012 / 0.988, 0 | 0.971, 1 | 0.40 % | 42.5 % | PASS ×3, 1 / 1 / 1 |
| 640x80 | 0.012 / 0.988, 0 | 0.971, 1 | 0.38 % | 42.4 % | PASS ×3, 1 / 1 / 1 |
| 720x90 | 0.012 / 0.988, 0 | 0.970, 1 | 0.39 % | 40.8 % | PASS ×3, 1 / 1 / 1 |
| 800x100 | 0.012 / 0.988, 0 | 0.969, 1 | 0.41 % | 40.7 % | PASS ×3, 1 / 1 / 1 |

- **Mode shapes.** At every mesh mode 1 is the fundamental transverse bending mode, with no interior sign change of the centreline deflection, and mode 2 is the next bending mode, with one sign change (`res.modeTable`). The mode character is identical across all nine meshes.
- **Energy location.** The void region holds about 41 % of the elements but only 0.3–0.7 % of the mode-1 kinetic energy. This energy share is descriptive and has no threshold. It comes from one native eigen-analysis of the final design, which reproduced the solver's final ω bit for bit. No low-density mode sits at the bottom of the spectrum.
- **Common evaluator.** The runner's E1/E2/E3 evaluator picked the lowest mode, ordinal 1, as the structural mode under all three evaluator models at all nine meshes.

## 5. Does the terminal gap vary systematically with mesh?

Yes, and not monotonically:

- 0.70 % at 160x20;
- rising to 11.8, 17.7, 19.0, 22.5 and 24.5 % up to 560x70;
- flat at 640x80 (24.3 %);
- falling to 22.4 % at 720x90 and 18.3 % at 800x100.

The variation comes almost entirely from **ω₂**: 170.4 → 206.4 up to 560x70, then 206.0 → 202.5 → 195.7. ω₁ moves by at most 0.6 rad/s over the five finest meshes. The fall at 720x90 and 800x100 coincides with the grayer designs described in TOPOLOGY_AUDIT.md: extra gray end-bay members at 720x90, and left–right asymmetry at 800x100. That is a coincidence in the data, not a demonstrated cause.
