#!/usr/bin/env python3
"""ASCII density maps of every final design (results/topology_maps.txt).

Same rendering convention as the mesh-resolution campaign: top row = y = H,
'@' = solid, ' ' = void.  Wide meshes are column-averaged down to 80 columns
and row-averaged down to at most 20 rows so the maps stay readable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
RAMP = " .:-=+*#%@"


def coarsen(g: np.ndarray, max_r: int = 20, max_c: int = 80) -> np.ndarray:
    r, c = g.shape
    fr = max(1, r // max_r)
    fc = max(1, c // max_c)
    r2, c2 = (r // fr) * fr, (c // fc) * fc
    return g[:r2, :c2].reshape(r2 // fr, fr, c2 // fc, fc).mean(axis=(1, 3))


def render(g: np.ndarray) -> list[str]:
    q = np.clip((coarsen(g) * (len(RAMP) - 1)).round().astype(int), 0, len(RAMP) - 1)
    return ["  " + "".join(RAMP[v] for v in row) for row in q[::-1]]


def main() -> None:
    lines: list[str] = []
    for d in sorted(p for p in RES.iterdir() if p.is_dir()):
        rf, sj = d / "rho_final.csv", d / "summary.json"
        if not rf.exists() or not sj.exists():
            continue
        s = json.loads(sj.read_text())
        g = np.loadtxt(rf, delimiter=",")
        lines.append("")
        lines.append(
            f'{d.name}   {s["nelx"]}x{s["nely"]}  status={s["stop_status"]}  '
            f'iters={s["outer_iters"]}  omega1(p=3)={s.get("omega1_final_p3", float("nan")):.2f}  '
            f'N={s.get("final_N")}  vol={s.get("final_volume", 0):.4f}'
        )
        lines += render(g)
    (RES / "topology_maps.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {RES / 'topology_maps.txt'}")


if __name__ == "__main__":
    main()
