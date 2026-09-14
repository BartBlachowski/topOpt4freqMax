# MODEL_REALIZATION — Parts 5 and 13

```
OUTER_MODEL_REALIZATION_INCONCLUSIVE
```

Preregistered rule (§7): fewer than 20 eligible steps ⇒ INCONCLUSIVE. The treatment
has **14**. The rule is applied mechanically; the numbers below are reported, not
graded.

Definitions: λ₁ₖ = ω₁(k)² before step k; pred_k = β_k − λ₁ₖ, with β = bs·λ_ref from the
certified SOCP; act_k = λ₁,ₖ₊₁ − λ₁ₖ (for k = 14 from the final analysis at ρ₁₄);
eligible iff pred_k > 1e-7·λ₁ₖ; r_k = act_k/pred_k.

## Treatment, all 14 accepted steps (stage 1, move 0.04)

| k | pred | act | r_k | rel. error |
|---|---|---|---|---|
| 1 | 824.7 | 872.9 | 1.058 | +0.058 |
| 2 | 898.8 | 942.1 | 1.048 | +0.048 |
| 3 | 960.8 | 994.6 | 1.035 | +0.035 |
| 4 | 1007.3 | 1038.4 | 1.031 | +0.031 |
| 5 | 1055.2 | 1080.1 | 1.024 | +0.024 |
| 6 | 1096.5 | 1110.9 | 1.013 | +0.013 |
| 7 | 1140.7 | 1160.5 | 1.017 | +0.017 |
| 8 | 1190.8 | 1205.9 | 1.013 | +0.013 |
| 9 | 1288.4 | 1348.9 | 1.047 | +0.047 |
| 10 | 1417.6 | 1489.5 | 1.051 | +0.051 |
| 11 | 1633.6 | 2142.6 | **1.312** | +0.312 |
| 12 | 1997.8 | 1792.8 | 0.897 | −0.103 |
| 13 | 1300.5 | 1376.0 | 1.058 | +0.058 |
| 14 | 767.7 | 703.8 | 0.917 | −0.083 |

| statistic | treatment (14 steps) | control, first 14 steps | control, all 386 (comparator) |
|---|---|---|---|
| median r | **1.033** | 1.058 | 0.711 |
| r quartiles | 1.014 / 1.050 | — | 0.428 / 0.911 |
| fraction act < 0 | **0** | 0 | 0 |
| fraction r < 0.25 / r > 4 | 0 / 0 | — | 0.025 / 0.008 |
| Σact / Σpred | **1.041** | 1.056 | 0.981 |
| median \|relative error\| | 0.048 | — | 0.324 |
| sign agreement (act > 0 given pred > 0) | 14 / 14 | 14 / 14 | 362 / 362 eligible |
| λ₁ decreases | 0 of 14 | 0 of 14 | 18 of 386 |
| cumulative predicted / realized gain | 16 580 / 17 259 | — | 22 611 / 22 171 (eligible) |

The control's full-run statistics are its own (MMA β); under the treatment rule they
would grade MARGINAL (stage 2 median r 0.36). This is a comparator only.

## What this does and does not show

- For the 14 steps it covers, the exact local model was **highly trustworthy**.
  Every predicted gain was positive and realized with the right sign. Realization was
  within ±6% except at k = 11 (+31%) and k = 12 (−10%), where the design first left
  the gray band en masse (gray fraction 1.00 → 0.60 after step 11). λ₁ never
  decreased. No move-bound reversal cycling was present (see
  `MOVE_REVERSAL_ANALYSIS.md`).
- Fully saturated ±0.04 steps on every element did **not** break the first-order
  model in early stage 1.
- The steps at which the prompt's H2 would be expected to appear are exactly those
  **not observed**: late stage 1 near the double eigenvalue, and the 0.02/0.01 rungs.
  Healthy realization over the first 14 steps cannot be extrapolated.
  `OUTER_GLOBALIZATION_PROBLEM_EXPOSED` is not supported, and neither is HEALTHY.

Figures: `figures/FIG_10_predicted_vs_realized_gain`, `FIG_11_realization_ratio_vs_iteration`,
`FIG_18_model_error_by_stage`.
