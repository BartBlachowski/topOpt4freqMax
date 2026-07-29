#!/usr/bin/env python3
"""Aggregate + classify the faithful-reconstruction campaign.

Reads every ``results/<tag>/`` produced by ``run_variant.m`` and emits

    results/aggregate.json          machine-readable summary of every run
    results/tables.md               the tables quoted in the report
    results/classification.csv      Phase-7 terminal-behaviour classification
    results/gates.csv               Phase-8 acceptance gates

Topology and trajectory descriptors reuse the definitions of the preceding
mesh-resolution campaign verbatim so the two campaigns are comparable.

All classification thresholds are DECLARED here and never adjusted per run.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

# ---- descriptor thresholds (identical to the mesh-resolution campaign) ----
SOLID_THRESHOLD = 0.5
GREY_LO, GREY_HI = 0.05, 0.95
MEMBER_MIN_AREA = 0.005
CONN8 = np.ones((3, 3), dtype=int)
CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

# ---- Phase-7 classification thresholds (DECLARED, fixed for all runs) ----
TAIL_N = 40                # iterations examined at the end of a trajectory
OBJ_CV_TOL = 1e-2          # tail coefficient of variation of omega1
OBJ_SLOPE_TOL = 2e-4       # |per-iteration relative drift| of omega1
CYCLE_RATIO = 0.25         # lag-k change < this * lag-1 change  => period-k
MECHANISM_FRAC = 0.05      # omega1 below this * initial omega1  => mechanism
DECAY_TOL = 5e-3           # |slope of log10 d1_rms per iteration| below this
                           # => design change is NOT decaying
PAPER = {"SS": 174.7, "CS": 288.7, "CC": 456.4}


# ---------------------------------------------------------------- helpers
def read_csv(p: Path) -> dict[str, np.ndarray]:
    if not p.exists():
        return {}
    with p.open() as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return {}
    hdr, body = rows[0], rows[1:]
    out: dict[str, np.ndarray] = {}
    for j, name in enumerate(hdr):
        col = []
        for r in body:
            v = r[j] if j < len(r) else ""
            try:
                col.append(float(v))
            except ValueError:
                col.append(np.nan)
        out[name] = np.array(col)
    return out


def read_matrix(p: Path) -> np.ndarray | None:
    if not p.exists():
        return None
    return np.loadtxt(p, delimiter=",")


def fnum(d: dict, key: str, default: float = float("nan")) -> float:
    """MATLAB jsonencode writes NaN as null; coerce anything unusable to NaN."""
    v = d.get(key, default)
    if v is None or isinstance(v, (list, dict, str)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def slope(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = np.arange(len(y))[m]
    return float(np.polyfit(x, y[m], 1)[0])


# ------------------------------------------------------------- descriptors
def topology_metrics(grid: np.ndarray) -> dict:
    nely, nelx = grid.shape
    nEl = grid.size
    solid = grid >= SOLID_THRESHOLD
    lab4, n4 = ndimage.label(solid, structure=CONN4)
    lab8, n8 = ndimage.label(solid, structure=CONN8)
    sizes = ndimage.sum(solid, lab8, range(1, n8 + 1)) if n8 else np.array([])
    members = int((sizes >= MEMBER_MIN_AREA * nEl).sum())

    spanning, largest_span = False, 0.0
    for c in range(1, n8 + 1):
        comp = lab8 == c
        if comp[:, 0].any() and comp[:, -1].any():
            spanning = True
        cols = np.where(comp.any(axis=0))[0]
        if cols.size:
            largest_span = max(largest_span, (cols.max() - cols.min() + 1) / nelx)

    lo, hi = nelx // 3, (2 * nelx) // 3
    centre = grid[:, lo:hi]

    def sym_corr(a: np.ndarray, b: np.ndarray) -> float:
        x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
        d = math.sqrt(float((x * x).sum() * (y * y).sum()))
        return float((x * y).sum() / d) if d > 1e-12 else float("nan")

    return dict(
        n_comp_4conn=int(n4), n_comp_8conn=int(n8),
        n_members=members, extra_members=max(0, members - 1),
        spanning=bool(spanning), largest_component_span=float(largest_span),
        solid_frac=float(solid.mean()),
        grey_frac=float(((grid > GREY_LO) & (grid < GREY_HI)).mean()),
        mean_density=float(grid.mean()),
        centre_third_mean=float(centre.mean()),
        midheight_symmetry=sym_corr(grid, grid[::-1, :]),
        midspan_symmetry=sym_corr(grid, grid[:, ::-1]),
    )


def trajectory_metrics(w1: np.ndarray) -> dict:
    w = np.asarray(w1, float)
    w = w[np.isfinite(w)]
    if w.size < 3:
        return {}
    ratio = w[1:] / np.maximum(w[:-1], 1e-12)
    logstep = np.abs(np.log(np.maximum(ratio, 1e-12)))
    tail = w[max(0, len(w) - TAIL_N):]
    return dict(
        omega1_final=float(w[-1]), omega1_max=float(w.max()),
        omega1_min=float(w.min()),
        omega1_iqr=float(np.percentile(w, 75) - np.percentile(w, 25)),
        omega1_tail_cv=float(tail.std() / max(abs(tail.mean()), 1e-12)),
        omega1_tail_rel_slope=float(slope(tail) / max(abs(tail.mean()), 1e-12)),
        collapse_events=int((ratio < 0.5).sum()),
        mean_abs_log_step=float(logstep.mean()),
        total_variation_norm=float(np.abs(np.diff(w)).sum() / max(w.max(), 1e-12)),
    )


# --------------------------------------------------- Phase 7 classification
def classify(run: dict) -> dict:
    s, oh = run["summary"], run["outer"]
    n = int(s.get("outer_iters", 0))
    tol = float(s.get("outer_tol", 1e-4))
    w = np.asarray(oh.get("omega1", []), float) if oh else np.array([])
    if not len(w):
        w = np.asarray(run["eigen"].get("omega1_trial", []), float)
    w = w[np.isfinite(w)]

    d1 = np.asarray(oh.get("d1_rms", []), float)
    d2 = np.asarray(oh.get("d2_rms", []), float)
    lag3 = run.get("lag3", {})
    d3 = np.asarray(lag3.get("d3_rms", []), float)

    tail = slice(max(0, n - TAIL_N), n)
    d1t = d1[tail][np.isfinite(d1[tail])] if d1.size else np.array([])
    d2t = d2[tail][np.isfinite(d2[tail])] if d2.size else np.array([])
    d3t = d3[np.isfinite(d3)][-TAIL_N:] if d3.size else np.array([])

    tm = trajectory_metrics(w) if w.size else {}
    stationary = (
        tm.get("omega1_tail_cv", 9e9) < OBJ_CV_TOL
        and abs(tm.get("omega1_tail_rel_slope", 9e9)) < OBJ_SLOPE_TOL
    )
    m1 = float(np.median(d1t)) if d1t.size else float("nan")
    m2 = float(np.median(d2t)) if d2t.size else float("nan")
    m3 = float(np.median(d3t)) if d3t.size else float("nan")
    r2 = m2 / m1 if m1 and np.isfinite(m1) and np.isfinite(m2) else float("nan")
    r3 = m3 / m1 if m1 and np.isfinite(m1) and np.isfinite(m3) else float("nan")

    decay = slope(np.log10(np.maximum(d1t, 1e-16))) if d1t.size >= 3 else float("nan")
    design_converged = bool(d1.size and np.isfinite(d1[-1]) and d1[-1] < tol)

    w0 = fnum(s, "omega1_init")
    mech = bool(s.get("mechanism_collapse", False)) or (
        w.size and np.isfinite(w0) and float(w.min()) < MECHANISM_FRAC * w0
    )

    status = s.get("stop_status", "")
    if status == "INNER_FAILURE":
        cls = "INNER_FAILURE"
    elif mech:
        cls = "MECHANISM_COLLAPSE"
    elif design_converged:
        cls = ("CONVERGED_BIMODAL" if fnum(s, "final_N", 1.0) >= 2
               else "CONVERGED_UNIMODAL")
    elif stationary and np.isfinite(r2) and r2 < CYCLE_RATIO:
        cls = "OUTER_LIMIT_CYCLE"        # period-2
    elif stationary and np.isfinite(r3) and r3 < CYCLE_RATIO:
        cls = "OUTER_LIMIT_CYCLE"        # period-3
    elif stationary and np.isfinite(decay) and abs(decay) < DECAY_TOL:
        cls = "OBJECTIVE_STATIONARY_DESIGN_CHATTERING"
    else:
        cls = "MAX_ITERATIONS"

    return dict(
        classification=cls, objective_stationary=bool(stationary),
        design_converged=design_converged, mechanism=bool(mech),
        d1_rms_tail_median=m1, d2_rms_tail_median=m2, d3_rms_tail_median=m3,
        lag2_over_lag1=r2, lag3_over_lag1=r3,
        d1_rms_final=float(d1[-1]) if d1.size else float("nan"),
        d1_inf_final=float(np.asarray(oh.get("d1_inf", [np.nan]))[-1]) if oh else float("nan"),
        log10_d1_decay_slope=decay, **tm,
    )


# --------------------------------------------------------- Phase 8 gates
def gates(run: dict, cls: dict, topo: dict, sib: dict | None) -> dict:
    s, oh = run["summary"], run["outer"]
    fc = bool(s.get("fail_closed", False))
    acc = np.asarray(oh.get("accepted", []), float) if oh else np.array([])

    # G1  no outer step accepted from a non-converged inner problem
    if fc:
        g1 = True   # a violating iteration halts the run before any update
    else:
        g1 = bool(acc.size and np.all(acc > 0.5))

    # G2  no mechanism collapse / singular disconnected design
    g2 = (not cls["mechanism"]) and bool(topo.get("spanning", False))

    # G3  feasibility at every accepted outer iteration
    vol = np.asarray(oh.get("vol", []), float) if oh else np.array([])
    volfrac = 0.5
    vfin = vol[np.isfinite(vol)] if vol.size else np.array([])
    if vfin.size:
        g3 = bool(vfin.max() <= volfrac + 1e-4)
    else:
        # no outer step was ever accepted: the initial design is feasible
        g3 = bool(fnum(s, "final_volume", 1.0) <= volfrac + 1e-4)

    # G4  reported objective is the actual lowest eigenvalue cluster
    ot = np.asarray(run["eigen"].get("omega1_trial", []), float)
    ot = ot[np.isfinite(ot)]
    fo = fnum(s, "omega1_final")
    consistent = bool(ot.size and abs(ot[-1] - fo) <= 1e-6 * max(abs(fo), 1.0))
    loc = run.get("localization")
    g4 = consistent and (not cls["mechanism"])
    if loc is not None:
        g4 = g4 and (loc.get("mode1_local_frac", 1.0) < 0.10)

    # G5  reaches and RETAINS a defensible N = 2
    Nn = np.asarray(run["mult"].get("N_solver", []), float)
    Nt = np.asarray(run["mult"].get("N_trial", []), float)
    last10 = Nt[np.isfinite(Nt)][-10:] if Nt.size else np.array([])
    g5 = bool(last10.size == 10 and np.all(last10 >= 2))

    # G6  the terminal state is not a hidden limit cycle
    g6 = cls["classification"] in ("CONVERGED_BIMODAL", "CONVERGED_UNIMODAL")

    # G7  behaviour transfers to the finer mesh (filled in by the caller)
    g7 = None if sib is None else bool(sib.get("same_class", False))

    # G8  topological plausibility
    g8 = bool(
        topo.get("spanning", False)
        and topo.get("extra_members", 9) == 0
        and abs(topo.get("midheight_symmetry", 0.0)) > 0.9
        and topo.get("grey_frac", 1.0) < 0.75
    )
    return dict(G1=g1, G2=g2, G3=g3, G4=g4, G5=g5, G6=g6, G7=g7, G8=g8)


# ------------------------------------------------------------------- load
def load_run(d: Path) -> dict | None:
    sj = d / "summary.json"
    if not sj.exists():
        return None
    run = {"tag": d.name, "summary": json.loads(sj.read_text())}
    run["outer"] = read_csv(d / "outer_history.csv")
    run["eigen"] = read_csv(d / "eigen_history.csv")
    run["mult"] = read_csv(d / "multiplicity_history.csv")
    run["mac"] = read_csv(d / "mac_history.csv")
    run["cycle"] = read_csv(d / "convergence_cycle.csv")
    run["lag3"] = read_csv(d / "convergence_cycle_lag3.csv")
    lp = d / "localization.json"
    if lp.exists():
        run["localization"] = json.loads(lp.read_text())
    g = read_matrix(d / "rho_final.csv")
    run["grid"] = g
    return run


def main() -> None:
    dirs = sorted(p for p in RES.iterdir()
                  if p.is_dir() and (p / "summary.json").exists())
    runs = [r for r in (load_run(d) for d in dirs) if r]

    out = {}
    for r in runs:
        cls = classify(r)
        topo = topology_metrics(r["grid"]) if r["grid"] is not None else {}
        r["_cls"], r["_topo"] = cls, topo

    # mesh-transfer sibling matching: same variant + inner budget, other mesh
    def key(s):
        return (s["variant"], s.get("inner_max_iter"), s["bc"])

    by_key: dict = {}
    for r in runs:
        by_key.setdefault(key(r["summary"]), []).append(r)

    for r in runs:
        s = r["summary"]
        sibs = [q for q in by_key[key(s)] if q["summary"]["nelx"] != s["nelx"]]
        sib = None
        if sibs:
            q = sibs[0]
            sib = {"other_mesh": f'{q["summary"]["nelx"]}x{q["summary"]["nely"]}',
                   "other_class": q["_cls"]["classification"],
                   "same_class": q["_cls"]["classification"] == r["_cls"]["classification"]}
        r["_sib"] = sib
        r["_gates"] = gates(r, r["_cls"], r["_topo"], sib)

        bc = s.get("bc", "CC")
        out[r["tag"]] = {
            "summary": s, "classification": r["_cls"],
            "topology": r["_topo"], "gates": r["_gates"],
            "mesh_transfer": sib,
            "pct_of_paper_p3": (100.0 * fnum(s, "omega1_final_p3")
                                / PAPER.get(bc, 456.4)),
        }

    (RES / "aggregate.json").write_text(json.dumps(out, indent=2, default=str))

    # ---------------- CSV: classification + gates ----------------
    with (RES / "classification.csv").open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["tag", "variant", "mesh", "inner_budget", "steps",
                      "continuation", "fail_closed", "stop_status", "iters",
                      "classification", "obj_stationary", "design_converged",
                      "d1_rms_final", "d1_inf_final", "d1_tail_med",
                      "lag2/lag1", "lag3/lag1", "log10_decay_slope",
                      "omega1_tail_cv", "omega1_final", "omega1_final_p3",
                      "final_N", "min_g12", "wall_s"])
        for r in runs:
            s, c = r["summary"], r["_cls"]
            wtr.writerow([r["tag"], s["variant"],
                          f'{s["nelx"]}x{s["nely"]}', s.get("inner_max_iter"),
                          s["step_controls"], s["continuation_enabled"],
                          s["fail_closed"], s["stop_status"], s["outer_iters"],
                          c["classification"], c["objective_stationary"],
                          c["design_converged"],
                          f'{c["d1_rms_final"]:.4e}', f'{c["d1_inf_final"]:.4e}',
                          f'{c["d1_rms_tail_median"]:.4e}',
                          f'{c["lag2_over_lag1"]:.4f}', f'{c["lag3_over_lag1"]:.4f}',
                          f'{c["log10_d1_decay_slope"]:.4e}',
                          f'{c.get("omega1_tail_cv", float("nan")):.4e}',
                          f'{fnum(s, "omega1_final"):.4f}',
                          f'{fnum(s, "omega1_final_p3"):.4f}',
                          s.get("final_N"), f'{fnum(s, "min_g12"):.4e}',
                          f'{fnum(s, "wall_time_s", 0.0):.0f}'])

    with (RES / "gates.csv").open("w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["tag", "variant", "mesh", "inner_budget",
                      "G1_inner", "G2_no_mech", "G3_feasible", "G4_spectral",
                      "G5_multiplicity", "G6_trajectory", "G7_mesh",
                      "G8_topology", "n_passed"])
        for r in runs:
            g, s = r["_gates"], r["summary"]
            vals = [g[k] for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")]
            npass = sum(1 for v in vals if v is True)
            wtr.writerow([r["tag"], s["variant"], f'{s["nelx"]}x{s["nely"]}',
                          s.get("inner_max_iter")]
                         + ["PASS" if v is True else ("n/a" if v is None else "FAIL")
                            for v in vals] + [npass])

    # ---------------- markdown tables ----------------
    L: list[str] = []
    L.append("## Ablation matrix — outcome per variant\n")
    L.append("| tag | variant | mesh | inner budget | steps | cont | FC | "
             "status | iters | omega1 final (p=3) | % paper | final N | "
             "min g12 | classification |")
    L.append("|---|---|---|---:|---|:-:|:-:|---|---:|---:|---:|---:|---:|---|")
    for r in sorted(runs, key=lambda q: (q["summary"].get("inner_max_iter", 0),
                                         q["summary"]["variant"],
                                         q["summary"]["nelx"])):
        s, c = r["summary"], r["_cls"]
        L.append(
            f'| {r["tag"]} | {s["variant"]} | {s["nelx"]}x{s["nely"]} | '
            f'{s.get("inner_max_iter")} | {s["step_controls"]} | '
            f'{"Y" if s["continuation_enabled"] else "n"} | '
            f'{"Y" if s["fail_closed"] else "n"} | {s["stop_status"]} | '
            f'{s["outer_iters"]} | {fnum(s, "omega1_final_p3"):.2f} | '
            f'{100*fnum(s, "omega1_final_p3")/PAPER.get(s.get("bc","CC"),456.4):.1f}% | '
            f'{s.get("final_N")} | {fnum(s, "min_g12"):.3e} | '
            f'{c["classification"]} |')

    L.append("\n## Inner-solve validity\n")
    L.append("| tag | inner budget | converged / total | rejected outer steps | "
             "median inner iters | singular warns |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in sorted(runs, key=lambda q: q["tag"]):
        s, oh = r["summary"], r["outer"]
        ii = np.asarray(oh.get("inner_iters", []), float)
        L.append(f'| {r["tag"]} | {s.get("inner_max_iter")} | '
                 f'{s.get("n_inner_converged")}/{s.get("outer_iters")} | '
                 f'{s.get("n_rejected_outer")} | '
                 f'{np.nanmedian(ii) if ii.size else float("nan"):.0f} | '
                 f'{s.get("n_singular_warn_total")} |')

    L.append("\n## Multiplicity audit\n")
    L.append("| tag | first N=2 (pre) | first N=2 (post) | # N=2 iters | "
             "# N=2 @tol 1e-2 | min g12 | final N | final g12 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(runs, key=lambda q: q["tag"]):
        s = r["summary"]
        L.append(f'| {r["tag"]} | {s.get("first_N2_iter")} | '
                 f'{s.get("first_N2_trial_iter")} | {s.get("n_N2_trial_iters")} | '
                 f'{s.get("n_N2_tol1e2")} | {fnum(s, "min_g12_trial"):.3e} | '
                 f'{s.get("final_N")} | {fnum(s, "final_g12"):.3e} |')

    L.append("\n## Terminal-behaviour classification\n")
    L.append("| tag | class | obj stationary | d1_rms final | d1_inf final | "
             "lag2/lag1 | lag3/lag1 | log10 decay slope | omega1 tail CV |")
    L.append("|---|---|:-:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(runs, key=lambda q: q["tag"]):
        c = r["_cls"]
        L.append(f'| {r["tag"]} | {c["classification"]} | '
                 f'{"Y" if c["objective_stationary"] else "n"} | '
                 f'{c["d1_rms_final"]:.3e} | {c["d1_inf_final"]:.3e} | '
                 f'{c["lag2_over_lag1"]:.3f} | {c["lag3_over_lag1"]:.3f} | '
                 f'{c["log10_d1_decay_slope"]:.3e} | '
                 f'{c.get("omega1_tail_cv", float("nan")):.3e} |')

    L.append("\n## Topology descriptors (final design)\n")
    L.append("| tag | 8conn | extra members | spanning | centre-third rho | "
             "grey frac | y-symmetry | x-symmetry |")
    L.append("|---|---:|---:|:-:|---:|---:|---:|---:|")
    for r in sorted(runs, key=lambda q: q["tag"]):
        t = r["_topo"]
        if not t:
            continue
        L.append(f'| {r["tag"]} | {t["n_comp_8conn"]} | {t["extra_members"]} | '
                 f'{"Y" if t["spanning"] else "n"} | {t["centre_third_mean"]:.3f} | '
                 f'{t["grey_frac"]:.3f} | {t["midheight_symmetry"]:.3f} | '
                 f'{t["midspan_symmetry"]:.3f} |')

    L.append("\n## Acceptance gates\n")
    L.append("| tag | G1 inner | G2 no-mech | G3 feasible | G4 spectral | "
             "G5 multiplicity | G6 trajectory | G7 mesh | G8 topology | passed |")
    L.append("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|---:|")
    for r in sorted(runs, key=lambda q: q["tag"]):
        g = r["_gates"]
        vals = [g[k] for k in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")]
        cells = ["PASS" if v is True else ("n/a" if v is None else "FAIL") for v in vals]
        L.append(f'| {r["tag"]} | ' + " | ".join(cells) + f' | {sum(v is True for v in vals)}/8 |')

    (RES / "tables.md").write_text("\n".join(L) + "\n")
    print(f"{len(runs)} runs analysed -> aggregate.json, tables.md, "
          f"classification.csv, gates.csv")
    for r in sorted(runs, key=lambda q: q["tag"]):
        s, c = r["summary"], r["_cls"]
        print(f'  {r["tag"]:28s} {s["stop_status"]:22s} it={s["outer_iters"]:4d} '
              f'w1={fnum(s, "omega1_final_p3"):9.3f} N={s.get("final_N")} '
              f'{c["classification"]}')


if __name__ == "__main__":
    main()
