#!/usr/bin/env python3
"""Analysis of the Du & Olhoff (2007) mesh-resolution verification campaign.

MEASUREMENT ONLY.  Operates on files written by run_mesh_campaign.m and
postprocess_modes.m.  Touches no solver code.

Produces, per run:
  * topology descriptors per outer iteration (connectivity, span, centre-span
    density, grey fraction, disconnected-member count)
  * trajectory descriptors (oscillation, collapse events, smoothness)
and, across runs, the relative topology correlation between consecutive meshes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

RESULTS = Path(__file__).resolve().parent / "results"

SOLID_THRESHOLD = 0.5      # density above which an element counts as material
GREY_LO, GREY_HI = 0.1, 0.9
MEMBER_MIN_AREA = 0.005    # component must hold >=0.5% of the domain to count
COLLAPSE_DROP = 0.5        # >50% single-iteration drop in omega_1 = collapse event

CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
CONN8 = np.ones((3, 3), dtype=bool)


def to_grid(vec: np.ndarray, nelx: int, nely: int) -> np.ndarray:
    """MATLAB column-major nEl vector -> (nely, nelx) array, row 0 = bottom."""
    return vec.reshape((nelx, nely)).T


def block_average(grid: np.ndarray, fy: int, fx: int) -> np.ndarray:
    nely, nelx = grid.shape
    return grid.reshape(nely // fy, fy, nelx // fx, fx).mean(axis=(1, 3))


def to_common_grid(grid: np.ndarray, tgt=(10, 80)) -> np.ndarray:
    """Map any campaign mesh onto the common 80x10 comparison grid.

    Refinements are volume-preserving block averages (exact).  The 40x5 control
    is upsampled by nearest replication, the only mapping available since 5
    does not divide 10 downward.
    """
    nely, nelx = grid.shape
    ty, tx = tgt
    if nely >= ty and nelx >= tx and nely % ty == 0 and nelx % tx == 0:
        return block_average(grid, nely // ty, nelx // tx)
    return np.repeat(np.repeat(grid, ty // nely, axis=0), tx // nelx, axis=1)


def topology_metrics(grid: np.ndarray) -> dict:
    nely, nelx = grid.shape
    nEl = grid.size
    solid = grid >= SOLID_THRESHOLD

    lab4, n4 = ndimage.label(solid, structure=CONN4)
    lab8, n8 = ndimage.label(solid, structure=CONN8)

    # Members = 8-connected components large enough to be structural.
    sizes = ndimage.sum(solid, lab8, range(1, n8 + 1)) if n8 else np.array([])
    members = int((sizes >= MEMBER_MIN_AREA * nEl).sum())

    # Spanning: one solid component touching both the left and the right edge.
    spanning = False
    largest_span = 0.0
    for c in range(1, n8 + 1):
        comp = lab8 == c
        if comp[:, 0].any() and comp[:, -1].any():
            spanning = True
        cols = np.where(comp.any(axis=0))[0]
        if cols.size:
            largest_span = max(largest_span, (cols.max() - cols.min() + 1) / nelx)

    lo, hi = nelx // 3, (2 * nelx) // 3
    centre = grid[:, lo:hi]

    # Symmetry about mid-height (y = H/2) and about mid-span (x = L/2).
    def sym_corr(a: np.ndarray, b: np.ndarray) -> float:
        x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
        d = math.sqrt(float((x * x).sum() * (y * y).sum()))
        return float((x * y).sum() / d) if d > 1e-12 else float("nan")

    return dict(
        n_comp_4conn=int(n4),
        n_comp_8conn=int(n8),
        n_members=members,
        extra_members=max(0, members - 1),
        spanning=bool(spanning),
        largest_component_span=float(largest_span),
        solid_frac=float(solid.mean()),
        grey_frac=float(((grid > GREY_LO) & (grid < GREY_HI)).mean()),
        mean_density=float(grid.mean()),
        centre_third_mean=float(centre.mean()),
        centre_third_solid_frac=float((centre >= SOLID_THRESHOLD).mean()),
        min_column_max=float(grid.max(axis=0).min()),
        cols_without_material=int((grid.max(axis=0) < SOLID_THRESHOLD).sum()),
        midheight_symmetry=sym_corr(grid, grid[::-1, :]),
        midspan_symmetry=sym_corr(grid, grid[:, ::-1]),
        midheight_asymmetry_l1=float(np.abs(grid - grid[::-1, :]).mean()),
    )


def trajectory_metrics(w1: np.ndarray) -> dict:
    w = np.asarray(w1, dtype=float)
    w = w[np.isfinite(w)]
    if w.size < 3:
        return {}
    ratio = w[1:] / np.maximum(w[:-1], 1e-12)
    logstep = np.abs(np.log(np.maximum(ratio, 1e-12)))
    collapses = int((ratio < COLLAPSE_DROP).sum())
    tail = w[len(w) // 2:]
    return dict(
        omega1_final=float(w[-1]),
        omega1_max=float(w.max()),
        omega1_min=float(w.min()),
        omega1_median=float(np.median(w)),
        omega1_iqr=float(np.percentile(w, 75) - np.percentile(w, 25)),
        omega1_tail_median=float(np.median(tail)),
        omega1_tail_cv=float(tail.std() / max(tail.mean(), 1e-12)),
        collapse_events=collapses,
        collapse_rate=float(collapses / len(ratio)),
        mean_abs_log_step=float(logstep.mean()),
        max_abs_log_step=float(logstep.max()),
        total_variation_norm=float(np.abs(np.diff(w)).sum() / max(w.max(), 1e-12)),
    )


def load_run(d: Path) -> dict | None:
    summ = d / "summary.csv"
    snap = d / "rho_snapshots.csv"
    if not summ.exists() or not snap.exists():
        return None

    import csv
    with summ.open() as fh:
        row = next(csv.DictReader(fh))
    nelx, nely = int(row["nelx"]), int(row["nely"])

    S = np.loadtxt(snap, delimiter=",", ndmin=2)
    if S.shape[0] != nelx * nely:
        S = S.T
    n_iter = S.shape[1]

    hist = np.genfromtxt(d / "history.csv", delimiter=",", names=True)
    w1_post = np.atleast_1d(hist["omega_post_1"])

    per_iter = [topology_metrics(to_grid(S[:, k], nelx, nely)) for k in range(n_iter)]

    # rho_final.csv is already written as reshape(rho_final, nely, nelx).
    final_grid = np.loadtxt(d / "rho_final.csv", delimiter=",", ndmin=2)
    assert final_grid.shape == (nely, nelx), final_grid.shape

    out = dict(
        tag=row["tag"], bc=row["bc"], regime=row["regime"],
        nelx=nelx, nely=nely,
        nEl=int(row["nEl"]), nDof=int(row["nDof"]),
        nFixedDof=int(row["nFixedDof"]),
        mid_y_over_H=float(row["mid_y_over_H"]),
        exact_midheight=bool(int(row["exact_midheight"])),
        outer_iters=int(row["outer_iters"]),
        final_omega1=float(row["final_omega1"]),
        final_omega2=float(row["final_omega2"]),
        final_N=float(row["final_N"]),
        final_volume=float(row["final_volume"]),
        wall_time=float(row["wall_time"]),
        cpu_time=float(row["cpu_time"]),
        final_topology=topology_metrics(final_grid),
        trajectory=trajectory_metrics(w1_post),
        n_iter=n_iter,
    )

    # Trajectory-aggregate topology: the endpoint of a non-converged oscillating
    # run is arbitrary, so report how the topology behaves over the whole run.
    for key in ("n_comp_8conn", "extra_members", "centre_third_mean",
                "grey_frac", "cols_without_material", "largest_component_span",
                "midheight_symmetry", "midheight_asymmetry_l1"):
        vals = np.array([m[key] for m in per_iter], dtype=float)
        out[f"traj_{key}_median"] = float(np.median(vals))
        out[f"traj_{key}_mean"] = float(vals.mean())
    out["traj_spanning_frac"] = float(np.mean([m["spanning"] for m in per_iter]))
    out["traj_connected_frac"] = float(np.mean([m["n_comp_8conn"] == 1 for m in per_iter]))

    # ---- modal instrumentation (postprocess_modes.m), if present ----------
    mf = d / "modes.csv"
    if mf.exists():
        M = np.genfromtxt(mf, delimiter=",", names=True)
        tm = np.atleast_1d(M["track_mac"])
        tid = np.atleast_1d(M["track_id"])
        ld1 = np.atleast_1d(M["ld_strain_frac_1"])
        ldall = np.column_stack([np.atleast_1d(M[f"ld_strain_frac_{j}"])
                                 for j in (1, 2, 3, 4)])
        fin = np.isfinite(tm)
        out["modal"] = dict(
            ld_strain_frac_per_mode_median=[float(np.nanmedian(ldall[:, j]))
                                            for j in range(4)],
            mac_mode1_median=float(np.nanmedian(tm[fin])) if fin.any() else float("nan"),
            mac_mode1_min=float(np.nanmin(tm[fin])) if fin.any() else float("nan"),
            mac_mode1_p05=float(np.nanpercentile(tm[fin], 5)) if fin.any() else float("nan"),
            # A tracking break: the continuation of mode 1 is a different index,
            # or its MAC with the previous mode 1 falls below 0.5.
            mode_swap_events=int(np.nansum(tid[fin] != 1)),
            tracking_break_events=int(np.nansum(tm[fin] < 0.5)),
            ld_strain_frac_mode1_median=float(np.nanmedian(ld1)),
            ld_strain_frac_mode1_max=float(np.nanmax(ld1)),
            ld_strain_frac_allmodes_median=float(np.nanmedian(ldall)),
            ld_strain_frac_allmodes_max=float(np.nanmax(ldall)),
            localized_mode_iters_frac=float(np.mean(np.nanmax(ldall, axis=1) > 0.5)),
        )

    out["_final_grid"] = final_grid
    out["_snapshots"] = S
    out["_per_iter"] = per_iter
    return out


def main() -> None:
    runs = {}
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        r = load_run(d)
        if r:
            runs[r["tag"]] = r

    # ---- relative topology correlation between consecutive meshes ----------
    ladder = [(40, 5), (80, 10), (160, 20), (240, 30)]
    correlations = []
    for bc in ("CC", "SS", "CS"):
        for regime in ("B", "A"):
            seq = []
            for nx, ny in ladder:
                t = f"{bc}_regime{regime}_{nx}x{ny}_s0"
                if t in runs:
                    seq.append(runs[t])
            for a, b in zip(seq[:-1], seq[1:]):
                ga = to_common_grid(a["_final_grid"]).ravel()
                gb = to_common_grid(b["_final_grid"]).ravel()
                if ga.std() < 1e-12 or gb.std() < 1e-12:
                    pear = float("nan")
                else:
                    pear = float(np.corrcoef(ga, gb)[0, 1])
                correlations.append(dict(
                    bc=bc, regime=regime,
                    mesh_a=f"{a['nelx']}x{a['nely']}",
                    mesh_b=f"{b['nelx']}x{b['nely']}",
                    pearson_r=pear,
                    l1_distance=float(np.abs(ga - gb).mean()),
                    solid_overlap_iou=float(
                        ((ga >= SOLID_THRESHOLD) & (gb >= SOLID_THRESHOLD)).sum()
                        / max(((ga >= SOLID_THRESHOLD) | (gb >= SOLID_THRESHOLD)).sum(), 1)
                    ),
                ))

    # ---- convergence towards the finest mesh -------------------------------
    to_finest = []
    for bc in ("CC", "SS", "CS"):
        for regime in ("B", "A"):
            ref_tag = f"{bc}_regime{regime}_240x30_s0"
            if ref_tag not in runs:
                continue
            gref = to_common_grid(runs[ref_tag]["_final_grid"]).ravel()
            for nx, ny in ladder[:-1]:
                t = f"{bc}_regime{regime}_{nx}x{ny}_s0"
                if t not in runs:
                    continue
                g = to_common_grid(runs[t]["_final_grid"]).ravel()
                r = float("nan") if g.std() < 1e-12 else float(np.corrcoef(g, gref)[0, 1])
                to_finest.append(dict(bc=bc, regime=regime, mesh=f"{nx}x{ny}",
                                      reference="240x30", pearson_r=r,
                                      l1_distance=float(np.abs(g - gref).mean())))

    payload = {
        "runs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in runs.items()},
        "consecutive_mesh_correlation": correlations,
        "correlation_to_finest_mesh": to_finest,
        "definitions": {
            "solid_threshold": SOLID_THRESHOLD,
            "member_min_area_frac": MEMBER_MIN_AREA,
            "collapse_drop_ratio": COLLAPSE_DROP,
            "common_comparison_grid": "80x10",
        },
    }
    out = RESULTS / "campaign_analysis.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")

    # ---- console report ---------------------------------------------------
    hdr = (f"{'tag':28} {'nDof':>6} {'it':>3} {'w1_fin':>8} {'w1_med':>8} "
           f"{'8conn':>5} {'xmem':>5} {'span%':>6} {'ctr3':>6} {'ysym':>6} "
           f"{'coll':>4} {'dlog':>6} {'sec':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for tag in sorted(runs, key=lambda t: (runs[t]["bc"], runs[t]["regime"], runs[t]["nEl"])):
        r = runs[tag]
        t = r["trajectory"]
        print(f"{tag:28} {r['nDof']:6d} {r['outer_iters']:3d} "
              f"{r['final_omega1']:8.2f} {t.get('omega1_median', float('nan')):8.2f} "
              f"{r['final_topology']['n_comp_8conn']:5d} "
              f"{r['final_topology']['extra_members']:5d} "
              f"{100*r['traj_spanning_frac']:6.1f} "
              f"{r['traj_centre_third_mean_median']:6.3f} "
              f"{r['final_topology']['midheight_symmetry']:6.3f} "
              f"{t.get('collapse_events', -1):4d} "
              f"{t.get('mean_abs_log_step', float('nan')):6.3f} "
              f"{r['wall_time']:7.1f}")

    hdr2 = (f"{'tag':28} {'MACmed':>7} {'MACmin':>7} {'swap':>5} {'break':>6} "
            f"{'ld1_med':>8} {'ld1_max':>8} {'locfrac':>8}")
    print("\n" + hdr2)
    print("-" * len(hdr2))
    for tag in sorted(runs, key=lambda t: (runs[t]["bc"], runs[t]["regime"], runs[t]["nEl"])):
        m = runs[tag].get("modal")
        if not m:
            continue
        print(f"{tag:28} {m['mac_mode1_median']:7.4f} {m['mac_mode1_min']:7.4f} "
              f"{m['mode_swap_events']:5d} {m['tracking_break_events']:6d} "
              f"{m['ld_strain_frac_mode1_median']:8.2e} {m['ld_strain_frac_mode1_max']:8.2e} "
              f"{m['localized_mode_iters_frac']:8.3f}")

    print("\nConsecutive-mesh topology correlation (final designs, common 80x10 grid)")
    print(f"{'bc':4} {'reg':4} {'A':>9} {'B':>9} {'pearson_r':>10} {'L1':>8} {'IoU':>7}")
    for c in correlations:
        print(f"{c['bc']:4} {c['regime']:4} {c['mesh_a']:>9} {c['mesh_b']:>9} "
              f"{c['pearson_r']:10.4f} {c['l1_distance']:8.4f} {c['solid_overlap_iou']:7.4f}")

    print("\nCorrelation of each mesh's final design with the finest (240x30)")
    print(f"{'bc':4} {'reg':4} {'mesh':>9} {'pearson_r':>10} {'L1':>8}")
    for c in to_finest:
        print(f"{c['bc']:4} {c['regime']:4} {c['mesh']:>9} "
              f"{c['pearson_r']:10.4f} {c['l1_distance']:8.4f}")

    print("\nLocal-mode dominance: median strain-energy fraction in rho<=0.1, per mode")
    print(f"{'tag':28} {'mode1':>9} {'mode2':>9} {'mode3':>9} {'mode4':>9}")
    for tag in sorted(runs, key=lambda t: (runs[t]["bc"], runs[t]["regime"], runs[t]["nEl"])):
        m = runs[tag].get("modal")
        if not m:
            continue
        v = m["ld_strain_frac_per_mode_median"]
        print(f"{tag:28} " + " ".join(f"{x:9.2e}" for x in v))


if __name__ == "__main__":
    main()
