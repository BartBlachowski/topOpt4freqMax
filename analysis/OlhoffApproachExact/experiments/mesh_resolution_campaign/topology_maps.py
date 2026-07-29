#!/usr/bin/env python3
"""Render campaign final topologies as ASCII density maps (row 0 = bottom)."""

from pathlib import Path
import sys

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RAMP = " .:-=+*#%@"


def render(grid: np.ndarray, max_cols: int = 120) -> list[str]:
    nely, nelx = grid.shape
    if nelx > max_cols:
        f = int(np.ceil(nelx / max_cols))
        fy = max(1, f * nely // nelx) if nely >= f else 1
        fy = f if nely % f == 0 else 1
        g = grid[: nely - nely % fy, : nelx - nelx % f]
        g = g.reshape(g.shape[0] // fy, fy, g.shape[1] // f, f).mean(axis=(1, 3))
    else:
        g = grid
    idx = np.clip((g * (len(RAMP) - 1)).round().astype(int), 0, len(RAMP) - 1)
    return ["".join(RAMP[v] for v in row) for row in idx[::-1]]   # top row first


def main(tags: list[str]) -> None:
    dirs = [RESULTS / t for t in tags] if tags else sorted(
        p for p in RESULTS.iterdir() if p.is_dir())
    for d in dirs:
        f = d / "rho_final.csv"
        if not f.exists():
            continue
        grid = np.loadtxt(f, delimiter=",", ndmin=2)
        print(f"\n=== {d.name}   ({grid.shape[0]} x {grid.shape[1]}) "
              f"solid_frac={float((grid>=0.5).mean()):.3f} ===")
        for line in render(grid):
            print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1:])
