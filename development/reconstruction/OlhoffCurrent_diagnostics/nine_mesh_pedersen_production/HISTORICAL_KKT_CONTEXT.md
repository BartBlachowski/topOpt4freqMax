# HISTORICAL_KKT_CONTEXT (Part 21)

## The historical verdict is unchanged

`analysis/OlhoffCurrent/diagnostics/gray_kkt_forensic_audit` records:

```
GRAY_REGIONS_NOT_KKT_STATIONARY
PERFORMANCE_CAMPAIGN_BLOCKED_BY_OPTIMALITY_ISSUE
```

This campaign did not edit, re-run or re-evaluate that study and does not change those verdicts. The study's only change since it was finalized is `EVIDENCE_AMENDMENT_1.md` (2026-09-14, owner decision). That amendment reclassified one historical manifest pin and changed no verdict.

## What that verdict is about

| | gray/KKT forensic audit | this campaign |
|---|---|---|
| implementation | pre-migration OlhoffCurrent, `+impl` tree `edbfe47e…` (75 files) | upstream 253069 promotion, `+impl` tree `4ba9a3ae…` (79 files), HEAD `b21483b` |
| stiffness | SIMP p = 3 | Pedersen (2000): p = 3, linear below ρ = 0.1 |
| mass | eq. (4b) low-density mass | linear mass, eq. (2), q = 1 |
| controller | historical global move ladder (β-stall / stage variants) | adaptive per-element move box (0.1 → floor 0.002, ×1.2 / ×0.7) |
| stop | historical ladder/guard stop | natural ‖Δρ‖₂ < 0.05·√(NE/3200), no guards |
| states studied | saved final designs at 400, 480 and 800, from the historical campaigns | nine new final designs, 160x20 … 800x100 |
| method | KKT residuals of the relaxed FE problem, finite-difference validation, filtered vs. physical sensitivity comparison, zero optimization runs | observational production solves through `performance_comparison.m` |

The verdict **remains valid historically** for those states and that formulation. It must not be read as a statement about the Pedersen/adaptive designs produced here. The reverse also holds: the historical eq. (4b) results must not be read as Pedersen results.

## What this campaign can and cannot say about stationarity

- It observes termination by the implementation's **heuristic design-change stop**. The runner's own caveat says it is *"a heuristic design-change stop and not a KKT certificate"*.
- It records native eigenvalue histories, the gap, M_nd and gray fraction, the box state, and inner convergence. **None** of these is a first-order optimality test of the physical problem.
- The structural issue the historical audit identified is still present by construction in this formulation. The nested MMA subproblem uses **sensitivity-filtered** spectral derivatives, and the filtered local model need not coincide with the physical eigenvalue derivative. This campaign did not measure the size of that mismatch for the Pedersen designs.
- Consequently, **this campaign neither proves nor disproves KKT stationarity** of the Pedersen/adaptive formulation. Smooth, monotone-looking histories and natural termination at every mesh would be consistent with a well-behaved heuristic. They would not show that the final designs are stationary points of the physical relaxed problem.

To answer that question, the forensic audit's KKT residual analysis would need to be applied, unchanged in method, to the nine designs saved here (`runs/<mesh>/SOLVER_RESULT.mat`). That is a separate, preregistered study and is not part of this campaign.
