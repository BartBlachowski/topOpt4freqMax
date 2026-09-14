# figures/

## The twelve required figures — all present, all from canary data

| # | figure | files |
|---|---|---|
| 1 | ω trajectory, 480×60 | `FIG_1_omega_480x60.{png,svg}` |
| 2 | ω trajectory, 800×100 | `FIG_2_omega_800x100.{png,svg}` |
| 3 | M_nd / grayness / mid fraction | `FIG_3_discreteness_480x60`, `FIG_3_discreteness_800x100` |
| 4 | move / stage timeline | `FIG_4_move_stage_480x60`, `FIG_4_move_stage_800x100` |
| 5 | A/B/E activity and persistence counters | `FIG_5_branch_timeline_480x60`, `FIG_5_branch_timeline_800x100` |
| 6 | gap12 / gap23 | `FIG_6_gaps_480x60`, `FIG_6_gaps_800x100` |
| 7 | next-mode (multiple-J) warning timeline | `FIG_7_multiJ_480x60`, `FIG_7_multiJ_800x100` |
| 8 | inner MMA per outer | `FIG_8_inner_480x60`, `FIG_8_inner_800x100` |
| 9 | per-outer timing decomposition | `FIG_9_timing_480x60`, `FIG_9_timing_800x100` |
| 10 | cumulative wall time | `FIG_10_cumwall_480x60`, `FIG_10_cumwall_800x100` |
| 11 | 480 vs 800 final topology | `FIG_11_topology_480_vs_800` |
| 12 | legacy vs three-rung final topology at 800×100 | `FIG_12_topology_legacy_vs_three_rung_800` |

Stage bands and the terminal declaration are marked on every trajectory figure.

## Three supporting figures from retained records

Each is labelled LEGACY or HISTORICAL in its own title, and each supports a
specific statement made elsewhere in the study. None is canary data.

| file | source | supports |
|---|---|---|
| `FIG_A_historical_three_rung_structure` | validated three-rung endpoints at 160/240/320/400 and `HISTORICAL_STAGE_WORK.csv` | the pre-run cap risk and budget projection (`PREREGISTRATION.md` §4) — which the canaries then showed to be an over-prediction |
| `FIG_B_legacy_next_mode_warning_regime` | `nine_mesh_campaign_audit/MASTER_TABLE.csv` | `MULTIPLICITY_WARNING_AUDIT.md` §4 |
| `FIG_C_legacy_runtime_inversion` | same | `PERFORMANCE_DECOMPOSITION.md` §4 |

FIG_A's 480 and 800 markers are the **pre-run extrapolated budget**, annotated
"projected". They are retained unchanged as a record of what was predicted
before the runs: the projection over-predicted S1 by 65 % at 480×60 and total
outer count by 2.4× at 800×100.
