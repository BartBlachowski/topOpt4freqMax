#!/usr/bin/env python3
"""Regenerate the Olhoff LP diagnostic figure from its retained CSV row."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
with (HERE / "olhoff_640_failure_diagnostics.csv").open(newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))

fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), facecolor="white")
fig.suptitle("Olhoff failed-attempt LP diagnostics (post-solve, no LP alteration)", fontsize=16)

axes[0].axis("off")
axes[0].set_title("Available matrix/scaling proxies", fontweight="bold")
available = (
    f"finite matrices: {'yes' if row['finite_matrices'] == '1' else 'no'}\n\n"
    f"normalized constraint-row rank: {row['constraint_row_rank']} / 5\n\n"
    f"normalized Gram rcond: {float(row['normalized_gram_rcond']):.6g}\n\n"
    f"inequality row-norm ratio: {float(row['row_norm_ratio_A']):.6g}\n\n"
    f"equality row-norm ratio: {float(row['row_norm_ratio_Aeq']):.6g}"
)
axes[0].text(0.5, 0.52, available, ha="center", va="center", fontsize=13)

axes[1].axis("off")
axes[1].set_title("Returned-point diagnostics", fontweight="bold")
axes[1].text(
    0.5,
    0.60,
    "Residual and bound-activity diagnostics unavailable",
    ha="center",
    va="center",
    fontweight="bold",
    fontsize=13,
)
axes[1].text(
    0.5,
    0.44,
    "No primal point was returned by linprog.\n"
    "Unavailable quantities are not plotted or replaced by zero.",
    ha="center",
    va="center",
    fontsize=12,
)
axes[1].text(
    0.5,
    0.25,
    f"exitflag={row['linprog_exitflag']}, reported iterations={row['lp_iterations']}",
    ha="center",
    va="center",
    fontsize=11,
)

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "figures" / "02_olhoff_640_lp_diagnostics.png", dpi=180)
plt.close(fig)
