"""
topopt_elastic2D.py — Minimum compliance topology optimization (2-D plane stress).

Based on:
    Andreassen et al. (2010), "Efficient topology optimization in MATLAB
    using 88 lines of code", Struct Multidisc Optim 43(1):1-16.

Accepts a run_cfg dict (populated by tools/Python/run_topopt_from_json.py).
Supports sensitivity filter (ft=0) and density filter (ft=1).
Supports OC and MMA optimizers.
Load types: closest_node, self_weight.  Dynamic load types raise an error.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Element stiffness matrix — Q4 plane stress, arbitrary hx × hy, E = 1
# Node order: LL → LR → UR → UL  (same as ourApproach)
# ---------------------------------------------------------------------------
def _ke_plane_stress(hx: float, hy: float, nu: float) -> np.ndarray:
    D = (1.0 / (1.0 - nu ** 2)) * np.array(
        [[1.0, nu, 0.0],
         [nu, 1.0, 0.0],
         [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float,
    )
    inv_j = np.diag([2.0 / hx, 2.0 / hy])
    det_j = 0.25 * hx * hy
    gp = 1.0 / np.sqrt(3.0)
    KE = np.zeros((8, 8), dtype=float)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            dN_dxi  = 0.25 * np.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
            dN_deta = 0.25 * np.array([-(1 - xi),  -(1 + xi),   (1 + xi),  (1 - xi)])
            dN_xy = inv_j @ np.vstack((dN_dxi, dN_deta))
            dN_dx, dN_dy = dN_xy[0], dN_xy[1]
            B = np.zeros((3, 8), dtype=float)
            B[0, 0::2] = dN_dx
            B[1, 1::2] = dN_dy
            B[2, 0::2] = dN_dy
            B[2, 1::2] = dN_dx
            KE += (B.T @ D @ B) * det_j
    return KE


# ---------------------------------------------------------------------------
# Load vector assembly
# ---------------------------------------------------------------------------
def _assemble_loads(
    load_cases: list[dict] | None,
    ndof: int,
    nelx: int,
    nely: int,
    hx: float,
    hy: float,
    rho0: float,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return F (ndof × n_cases) and case_factors (n_cases,)."""
    if not load_cases:
        raise ValueError(
            "No load cases defined. Specify domain.load_cases in the JSON config."
        )

    n_cases = len(load_cases)
    F = np.zeros((ndof, n_cases), dtype=float)
    case_factors = np.ones(n_cases, dtype=float)

    # Precompute element corner nodes for self_weight (vectorised).
    el_idx  = np.arange(nelx * nely, dtype=np.int64)
    elx_arr = el_idx // nely
    ely_arr = el_idx %  nely
    sw_n1 = (nely + 1) * elx_arr + ely_arr           # LL
    sw_n2 = (nely + 1) * (elx_arr + 1) + ely_arr     # LR
    sw_n3 = sw_n2 + 1                                  # UR
    sw_n4 = sw_n1 + 1                                  # UL

    for ci, lc in enumerate(load_cases):
        case_factors[ci] = float(lc.get("factor", 1.0))
        for ld in lc["loads"]:
            lt  = ld["type"]
            ldf = float(ld.get("factor", 1.0))

            if lt == "closest_node":
                loc  = np.asarray(ld["location"], dtype=float)
                dist2 = (node_x - loc[0]) ** 2 + (node_y - loc[1]) ** 2
                n    = int(node_ids[dist2 == dist2.min()].min())
                fx, fy = float(ld["force"][0]), float(ld["force"][1])
                F[2 * n,     ci] += ldf * fx
                F[2 * n + 1, ci] += ldf * fy

            elif lt == "self_weight":
                # Lumped consistent nodal force: rho0 * factor * hx*hy / 4
                # per corner node in -y direction (gravity = -y).
                ew = ldf * rho0 * hx * hy / 4.0
                for nc in (sw_n1, sw_n2, sw_n3, sw_n4):
                    np.add.at(F[:, ci], 2 * nc + 1, -ew)

            else:
                raise ValueError(
                    f'Load type "{lt}" is not supported by elastic2D. '
                    "Supported: closest_node, self_weight."
                )

    return F, case_factors


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def topopt_elastic2D(
    nelx: int,
    nely: int,
    volfrac: float,
    penal: float,
    rmin: float,
    ft: int,
    L: float,
    H: float,
    run_cfg: Optional[dict] = None,
) -> tuple[np.ndarray, float, float, int]:
    """Topology optimization for minimum compliance (elastic 2-D plane stress).

    Parameters
    ----------
    nelx, nely : int
        Number of elements in x and y.
    volfrac : float
        Target volume fraction.
    penal : float
        SIMP penalization exponent.
    rmin : float
        Filter radius in physical units.
    ft : int
        0 = sensitivity filter, 1 = density filter.
    L, H : float
        Physical domain dimensions.
    run_cfg : dict, optional
        Solver configuration dict.  Keys:

        E0, Emin, nu          — material
        rho0                  — density (used for self_weight loads only)
        move, conv_tol, max_iters — optimization control
        optimizer             — "OC" or "MMA"
        extra_fixed_dofs      — 0-based fixed DOF array
        pas_s, pas_v          — 0-based passive solid/void element arrays
        load_cases            — normalized list from validate_load_cases()
        visualize_live        — bool

    Returns
    -------
    x_out   : (nelx*nely,) float  — final physical density field
    c_final : float               — final compliance value
    t_iter  : float               — average time per iteration [s]
    n_iter  : int                 — number of iterations executed
    """
    if run_cfg is None:
        run_cfg = {}

    # --- Material ---
    E0   = float(run_cfg.get("E0",   1.0))
    Emin = float(run_cfg.get("Emin", 1e-9))
    nu   = float(run_cfg.get("nu",   0.3))
    rho0 = float(run_cfg.get("rho0", 1.0))

    # --- Optimization ---
    move      = float(run_cfg.get("move",     0.2))
    conv_tol  = float(run_cfg.get("conv_tol", 0.01))
    max_iters = int(  run_cfg.get("max_iters", 200))
    optimizer = str(  run_cfg.get("optimizer", "OC")).upper().strip()
    visualize_live = bool(run_cfg.get("visualize_live", False))

    if optimizer not in {"OC", "MMA"}:
        raise ValueError(f'optimizer must be "OC" or "MMA" (got "{optimizer}")')

    print("elastic2D — minimum compliance")
    print(f"  mesh {nelx}×{nely},  domain {L}×{H},  volfrac {volfrac}")
    print(f"  penal {penal},  rmin {rmin:.4g}")
    print(f"  filter {'sensitivity' if ft == 0 else 'density'},  optimizer {optimizer}")

    hx = L / nelx
    hy = H / nely
    n_el    = nelx * nely
    n_nodes = (nelx + 1) * (nely + 1)
    ndof    = 2 * n_nodes

    # --- Passive elements ---
    pas_s = np.asarray(run_cfg.get("pas_s", []), dtype=np.int64).ravel()
    pas_v = np.asarray(run_cfg.get("pas_v", []), dtype=np.int64).ravel()
    act   = np.setdiff1d(np.arange(n_el, dtype=np.int64), np.union1d(pas_s, pas_v))

    # --- Boundary conditions ---
    extra_fixed = np.asarray(run_cfg.get("extra_fixed_dofs", []), dtype=np.int64).ravel()
    if extra_fixed.size == 0:
        raise ValueError(
            "No fixed DOFs provided — the structure is a mechanism. "
            "Specify bc.supports in the JSON config."
        )
    fixed = np.unique(extra_fixed)
    free  = np.setdiff1d(np.arange(ndof, dtype=np.int64), fixed)

    # --- Element stiffness matrix ---
    KE = _ke_plane_stress(hx, hy, nu)

    # --- Element connectivity (LL→LR→UR→UL, 0-based DOFs) ---
    el_idx  = np.arange(n_el, dtype=np.int64)
    elx_arr = el_idx // nely
    ely_arr = el_idx %  nely
    n1 = (nely + 1) * elx_arr + ely_arr           # LL
    n2 = (nely + 1) * (elx_arr + 1) + ely_arr     # LR
    n3 = n2 + 1                                     # UR
    n4 = n1 + 1                                     # UL
    edof_mat = np.column_stack([
        2*n1, 2*n1+1, 2*n2, 2*n2+1,
        2*n3, 2*n3+1, 2*n4, 2*n4+1,
    ]).astype(np.int64)

    iK = np.kron(edof_mat, np.ones((8, 1), dtype=np.int64)).ravel()
    jK = np.kron(edof_mat, np.ones((1, 8), dtype=np.int64)).ravel()

    # --- Filter (linear-decay, Euclidean distance in physical units) ---
    rminx = max(1, int(np.ceil(rmin / hx)))
    rminy = max(1, int(np.ceil(rmin / hy)))
    nfilt = n_el * (2 * (rminx - 1) + 1) * (2 * (rminy - 1) + 1)
    iH = np.zeros(nfilt, dtype=np.int64)
    jH = np.zeros(nfilt, dtype=np.int64)
    sH = np.zeros(nfilt, dtype=float)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k in range(max(i - rminx + 1, 0), min(i + rminx, nelx)):
                for ll in range(max(j - rminy + 1, 0), min(j + rminy, nely)):
                    col = k * nely + ll
                    d   = np.sqrt(((i - k) * hx) ** 2 + ((j - ll) * hy) ** 2)
                    iH[cc], jH[cc], sH[cc] = row, col, max(0.0, rmin - d)
                    cc += 1
    Hf = csc_array((sH[:cc], (iH[:cc], jH[:cc])), shape=(n_el, n_el))
    Hs = np.asarray(Hf.sum(axis=1)).ravel()

    # --- Node coordinates ---
    node_ids = np.arange(n_nodes, dtype=np.int64)
    node_x   = (node_ids // (nely + 1)) * hx
    node_y   = (node_ids %  (nely + 1)) * hy

    # --- Load vectors ---
    F, case_factors = _assemble_loads(
        run_cfg.get("load_cases"),
        ndof, nelx, nely, hx, hy, rho0,
        node_x, node_y, node_ids,
    )
    n_cases = F.shape[1]
    print(f"  load cases: {n_cases}")

    # --- MMA setup ---
    mmasub = None
    if optimizer == "MMA":
        mma_dir = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "OlhoffApproach", "Python"))
        if mma_dir not in sys.path:
            sys.path.insert(0, mma_dir)
        try:
            from mma import mmasub as _mmasub
            mmasub = _mmasub
        except ImportError:
            raise ImportError(
                "MMA optimizer requested but mma.py not found in "
                "analysis/OlhoffApproach/Python/."
            )
        n_act  = act.size
        xmin_mma = np.zeros(n_act)
        xmax_mma = np.ones(n_act)
        xold1    = np.full(n_act, volfrac)
        xold2    = np.full(n_act, volfrac)
        low_mma  = xmin_mma.copy()
        upp_mma  = xmax_mma.copy()

    # --- Design variables ---
    x = np.full(n_el, volfrac)
    x[pas_s] = 1.0
    x[pas_v] = 0.0
    if act.size > 0 and (pas_s.size or pas_v.size):
        x[act] = np.clip((volfrac * n_el - float(pas_s.size)) / act.size, 0.0, 1.0)

    def _physical(xv: np.ndarray) -> np.ndarray:
        xp = xv.copy() if ft == 0 else np.asarray(Hf @ xv) / Hs
        if pas_s.size: xp[pas_s] = 1.0
        if pas_v.size: xp[pas_v] = 0.0
        return xp

    xPhys = _physical(x)

    # --- Visualization setup ---
    im_live = None
    if visualize_live:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _fig, _ax = plt.subplots(
                figsize=(max(4, nelx // 10), max(2, nely // 10))
            )
            im_live = _ax.imshow(
                1.0 - xPhys.reshape(nely, nelx, order='F'),
                cmap='gray', vmin=0, vmax=1, origin='lower', aspect='equal',
            )
            _ax.axis('off')
            plt.tight_layout()
            plt.ion()
            plt.show()
        except Exception:
            visualize_live = False

    # --- Main optimization loop ---
    loop    = 0
    change  = 1.0
    c_final = np.inf
    t_total = 0.0

    print(f"  {'It':>5}  {'Compliance':>14}  {'Vol':>7}  {'Ch':>8}")

    while change > conv_tol and loop < max_iters:
        t0 = time.time()
        loop += 1

        # FE analysis — assemble and solve
        E_field = Emin + xPhys ** penal * (E0 - Emin)
        sK = np.outer(KE.ravel(), E_field).ravel(order='F')
        K  = csc_array((sK, (iK, jK)), shape=(ndof, ndof))
        K  = (K + K.T) * 0.5

        Kf = K[free][:, free]
        U  = np.zeros((ndof, n_cases), dtype=float)
        for ci in range(n_cases):
            U[free, ci] = spsolve(Kf, F[free, ci])

        # Compliance and raw sensitivity
        c  = 0.0
        ce = np.zeros(n_el, dtype=float)
        for ci in range(n_cases):
            ue   = U[edof_mat, ci]              # (n_el, 8)
            ce_k = np.sum((ue @ KE) * ue, axis=1)
            c   += case_factors[ci] * float(np.dot(E_field, ce_k))
            ce  += case_factors[ci] * ce_k
        c_final = c

        dc = -penal * (E0 - Emin) * xPhys ** (penal - 1) * ce
        dv = np.ones(n_el, dtype=float)

        # Sensitivity / density filter
        if ft == 0:
            dc_f = np.asarray(Hf @ (x * dc)) / Hs / np.maximum(1e-3, x)
            dv_f = dv
        else:
            dc_f = np.asarray(Hf @ dc) / Hs
            dv_f = np.asarray(Hf @ dv) / Hs

        # Volume target for active elements
        vol_target = volfrac * n_el - float(pas_s.size)

        if optimizer == "OC":
            dc_act = dc_f[act]
            dv_act = dv_f[act]
            x_act  = x[act].copy()
            l1, l2 = 0.0, 1e9
            while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
                lmid     = 0.5 * (l1 + l2)
                xnew_act = np.clip(
                    np.clip(x_act * np.sqrt(-dc_act / dv_act / lmid),
                            x_act - move, x_act + move),
                    0.0, 1.0,
                )
                if ft == 1:
                    x_trial        = x.copy()
                    x_trial[act]   = xnew_act
                    x_trial[pas_s] = 1.0
                    x_trial[pas_v] = 0.0
                    xp_trial       = np.asarray(Hf @ x_trial) / Hs
                    xp_trial[pas_s] = 1.0
                    xp_trial[pas_v] = 0.0
                    vol_curr = float(np.sum(xp_trial[act]))
                else:
                    vol_curr = float(np.sum(xnew_act))
                if vol_curr > vol_target:
                    l1 = lmid
                else:
                    l2 = lmid
            x_old  = x[act].copy()
            x[act] = xnew_act

        else:  # MMA
            n_act = act.size
            f0val = c
            df0dx = dc_f[act].reshape(-1, 1)
            fval  = np.array([(float(np.sum(xPhys[act])) - vol_target) / n_el])
            dfdx  = (dv_f[act] / n_el).reshape(1, -1)
            x_col = x[act].reshape(-1, 1)
            xnew_mma, _, _, _, _, _, _, _, _, low_mma, upp_mma = mmasub(
                1, n_act, loop,
                x_col, xmin_mma.reshape(-1, 1), xmax_mma.reshape(-1, 1),
                xold1.reshape(-1, 1), xold2.reshape(-1, 1),
                f0val, df0dx, fval.reshape(-1, 1), dfdx,
                low_mma.reshape(-1, 1), upp_mma.reshape(-1, 1),
                1.0, np.zeros((1, 1)), 1e3 * np.ones((1, 1)), np.ones((1, 1)),
            )
            xold2  = xold1.copy()
            xold1  = x[act].copy()
            x_old  = x[act].copy()
            x[act] = xnew_mma.ravel()

        change   = float(np.max(np.abs(x[act] - x_old)))
        x[pas_s] = 1.0
        x[pas_v] = 0.0
        xPhys    = _physical(x)

        t_total += time.time() - t0

        print(f"  {loop:>5}  {c:>14.6g}  {float(np.mean(xPhys)):>7.4f}  {change:>8.4f}")

        if visualize_live and im_live is not None:
            try:
                import matplotlib.pyplot as plt
                im_live.set_data(1.0 - xPhys.reshape(nely, nelx, order='F'))
                plt.pause(0.01)
            except Exception:
                pass

    t_iter = t_total / max(loop, 1)
    print(f"  Done: {loop} iters,  compliance = {c_final:.6g},  t/iter = {t_iter:.3f} s")

    if visualize_live:
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.close("all")
        except Exception:
            pass

    return xPhys.copy(), c_final, t_iter, loop
