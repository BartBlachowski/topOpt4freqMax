#!/usr/bin/env python3
"""Profile individual operations of the Olhoff solver to find bottleneck."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as spla

try:
    from .solver import (
        OlhoffConfig,
        _assemble_global,
        _solve_modes,
        build_supports,
        heaviside_projection,
        q4_rect_ke_m_plane_stress,
    )
    from .mma import mmasub
except ImportError:  # direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from OlhoffApproach.Python.solver import (
        OlhoffConfig,
        _assemble_global,
        _solve_modes,
        build_supports,
        heaviside_projection,
        q4_rect_ke_m_plane_stress,
    )
    from OlhoffApproach.Python.mma import mmasub


def profile_iteration():
    print(f"Python: {sys.version}")
    print(f"numpy: {np.__version__}")
    import scipy
    print(f"scipy: {scipy.__version__}")
    print()

    cfg = OlhoffConfig()
    cfg.support_type = "SS"
    cfg.nelx = 240
    cfg.nely = 30
    cfg.maxiter = 300
    cfg.rmin = 2 * cfg.L / cfg.nelx
    cfg.finalize()

    L, H = cfg.L, cfg.H
    nelx, nely = cfg.nelx, cfg.nely
    n_el = nelx * nely
    dx, dy = L / nelx, H / nely

    node_nrs = np.arange((nelx + 1) * (nely + 1), dtype=np.int64).reshape((nely + 1, nelx + 1), order="F")
    node_nrs_1 = node_nrs + 1
    n_dof = 2 * (nelx + 1) * (nely + 1)

    fixed = build_supports(cfg.support_type, node_nrs)
    free = np.setdiff1d(np.arange(n_dof, dtype=np.int64), fixed)

    Ke0, M0 = q4_rect_ke_m_plane_stress(cfg.E0, cfg.nu, cfg.rho0, cfg.thickness, dx, dy)
    Ke = Ke0 / cfg.E0
    Me = M0 / cfg.rho0

    rmin_elem = cfg.rmin / dx
    if rmin_elem < 1.2:
        rmin_elem = max(rmin_elem, 1.5)

    rg = np.arange(-int(np.ceil(rmin_elem)) + 1, int(np.ceil(rmin_elem)), dtype=np.float64)
    dxf, dyf = np.meshgrid(rg, rg, indexing="xy")
    h = np.maximum(0.0, rmin_elem - np.sqrt(dxf**2 + dyf**2))
    Hs = ndimage.convolve(np.ones((nely, nelx), dtype=np.float64), h, mode="reflect")

    def fwd(xm):
        return ndimage.convolve(xm, h, mode="reflect") / Hs

    def bwd(gm):
        return ndimage.convolve(gm / Hs, h, mode="reflect")

    c_vec = (2 * node_nrs_1[:-1, :-1] + 1).reshape(-1, order="F")
    c_mat = np.column_stack([
        c_vec, c_vec + 1, c_vec + 2 * nely + 2, c_vec + 2 * nely + 3,
        c_vec + 2 * nely, c_vec + 2 * nely + 1, c_vec - 2, c_vec - 1,
    ]).astype(np.int64) - 1

    sI_list, sII_list = [], []
    for j in range(8):
        sI_list.extend(range(j, 8))
        sII_list.extend([j] * (8 - j))
    sI = np.array(sI_list, dtype=np.int64)
    sII = np.array(sII_list, dtype=np.int64)

    iK = c_mat[:, sI].T.reshape(-1, order="F")
    jK = c_mat[:, sII].T.reshape(-1, order="F")
    Ke_l = Ke[sI, sII]
    Me_l = Me[sI, sII]

    # --- Profile each operation for 5 iterations ---
    np.random.seed(42)
    x = cfg.volfrac * np.ones((n_el,), dtype=np.float64)
    Eb = cfg.Eb0

    n = n_el + 1
    m = cfg.J + 2

    xmin = np.concatenate((1e-3 * np.ones((n_el,)), np.array([cfg.Eb_min])))
    xmax = np.concatenate((np.ones((n_el,)), np.array([cfg.Eb_max])))
    xval = np.concatenate((x, np.array([Eb])))
    xold1 = xval.copy()
    xold2 = xval.copy()
    low = xmin.copy()
    upp = xmax.copy()

    mma_a0 = 1.0
    mma_a = np.zeros((m, 1))
    mma_c = np.concatenate((100 * np.ones((cfg.J,)), np.array([10, 10]))).reshape(-1, 1)
    mma_d = 1e-3 * np.ones((m, 1))

    timings = {
        "filter_fwd": [],
        "heaviside": [],
        "material_interp": [],
        "assemble_global": [],
        "submatrix_extract": [],
        "eigsh_solve": [],
        "sensitivity": [],
        "mma_solve": [],
        "eval_trial": [],
        "total_iteration": [],
    }

    N_ITER = 5
    print(f"Profiling {N_ITER} iterations on {nelx}x{nely} mesh ({n_el} elements, {n_dof} DOFs, {free.size} free DOFs)")
    print(f"Support type: {cfg.support_type}, J={cfg.J} modes\n")

    beta = 1.0

    for it in range(1, N_ITER + 1):
        t_total = time.perf_counter()

        x = xval[:n_el]

        # 1. Filter forward
        t0 = time.perf_counter()
        xT = fwd(x.reshape((nely, nelx), order="F"))
        timings["filter_fwd"].append(time.perf_counter() - t0)

        # 2. Heaviside projection
        t0 = time.perf_counter()
        xPhysMat, dH = heaviside_projection(xT, beta, cfg.eta)
        xPhys = xPhysMat.reshape(-1, order="F")
        timings["heaviside"].append(time.perf_counter() - t0)

        # 3. Material interpolation
        t0 = time.perf_counter()
        Ee = cfg.Emin + (xPhys**cfg.penal) * (cfg.E0 - cfg.Emin)
        re = cfg.rho_min + xPhys * (cfg.rho0 - cfg.rho_min)
        timings["material_interp"].append(time.perf_counter() - t0)

        # 4. Global assembly
        t0 = time.perf_counter()
        K, M = _assemble_global(iK, jK, Ke_l, Me_l, Ee, re, n_dof)
        timings["assemble_global"].append(time.perf_counter() - t0)

        # 5. Submatrix extraction
        t0 = time.perf_counter()
        Kf = K[free][:, free]
        Mf = M[free][:, free]
        timings["submatrix_extract"].append(time.perf_counter() - t0)

        # 6. Eigsh solve
        t0 = time.perf_counter()
        lam_sorted, V_low, resvec = _solve_modes(Kf, Mf, cfg.J)
        timings["eigsh_solve"].append(time.perf_counter() - t0)

        omega_cur = float(np.sqrt(max(lam_sorted[0], 0.0)))

        # 7. Sensitivity computation
        t0 = time.perf_counter()
        dlam = np.zeros((n_el, cfg.J), dtype=np.float64)
        for j in range(cfg.J):
            v = V_low[:, j].copy()
            norm_m = np.sqrt(max(np.finfo(float).eps, float(v.T @ (Mf @ v))))
            v /= norm_m
            phi = np.zeros((n_dof,), dtype=np.float64)
            phi[free] = v
            pe = phi[c_mat]
            dlam[:, j] = (
                cfg.penal * (cfg.E0 - cfg.Emin) * (xPhys ** (cfg.penal - 1)) * np.sum((pe @ Ke) * pe, axis=1)
                - lam_sorted[j] * (cfg.rho0 - cfg.rho_min) * np.sum((pe @ Me) * pe, axis=1)
            )

        gray_measure = float(np.mean(4 * xPhys * (1 - xPhys)))
        dgray_dxPhys = 4 * (1 - 2 * xPhys) / n_el
        gray_penalty_weight = cfg.gray_penalty_base * (beta / cfg.beta_max)
        f0 = -Eb + gray_penalty_weight * gray_measure
        dgray_dxT = dgray_dxPhys.reshape((nely, nelx), order="F") * dH
        dgray_dx = bwd(dgray_dxT).reshape(-1, order="F")
        df0 = np.zeros((n,), dtype=np.float64)
        df0[:n_el] = gray_penalty_weight * dgray_dx
        df0[-1] = -1.0

        fval = np.zeros((m,), dtype=np.float64)
        dfdx = np.zeros((m, n), dtype=np.float64)
        for j in range(cfg.J):
            scale_j = max(1.0, Eb)
            fval[j] = (Eb - lam_sorted[j] / cfg.lambda_ref) / scale_j
            g = (-dlam[:, j] / cfg.lambda_ref).reshape((nely, nelx), order="F") * dH
            bg = bwd(g).reshape(-1, order="F") / scale_j
            dfdx[j, :n_el] = bg
            dfdx[j, -1] = 1.0 / scale_j
        g_up = float(np.mean(xPhys) - cfg.volfrac)
        gv = (np.ones((n_el,), dtype=np.float64) / n_el).reshape((nely, nelx), order="F") * dH
        bgv = bwd(gv).reshape(-1, order="F")
        fval[cfg.J] = g_up
        dfdx[cfg.J, :n_el] = bgv
        g_low = float(cfg.volfrac - np.mean(xPhys))
        fval[cfg.J + 1] = g_low
        dfdx[cfg.J + 1, :n_el] = -bgv
        timings["sensitivity"].append(time.perf_counter() - t0)

        # 8. MMA solve
        t0 = time.perf_counter()
        xnew, _, _, _, _, _, _, _, _, low_new, upp_new = mmasub(
            m, n, it,
            xval.reshape(-1, 1), xmin.reshape(-1, 1), xmax.reshape(-1, 1),
            xold1.reshape(-1, 1), xold2.reshape(-1, 1),
            float(f0), df0.reshape(-1, 1), fval.reshape(-1, 1), dfdx,
            low.reshape(-1, 1), upp.reshape(-1, 1),
            mma_a0, mma_a, mma_c, mma_d,
        )
        xnew = xnew.ravel()
        low = low_new.ravel()
        upp = upp_new.ravel()
        xnew = np.clip(xnew, xmin, xmax)
        timings["mma_solve"].append(time.perf_counter() - t0)

        # 9. Trial evaluation (second eigsh)
        t0 = time.perf_counter()
        x_trial = xnew[:n_el]
        xT_trial = fwd(x_trial.reshape((nely, nelx), order="F"))
        xPhys_trial, _ = heaviside_projection(xT_trial, beta, cfg.eta)
        xPhys_trial_flat = xPhys_trial.reshape(-1, order="F")
        Ee_trial = cfg.Emin + (xPhys_trial_flat**cfg.penal) * (cfg.E0 - cfg.Emin)
        re_trial = cfg.rho_min + xPhys_trial_flat * (cfg.rho0 - cfg.rho_min)
        K_t, M_t = _assemble_global(iK, jK, Ke_l, Me_l, Ee_trial, re_trial, n_dof)
        Kf_t = K_t[free][:, free]
        Mf_t = M_t[free][:, free]
        lam_t, _, _ = _solve_modes(Kf_t, Mf_t, min(3, cfg.J))
        timings["eval_trial"].append(time.perf_counter() - t0)

        timings["total_iteration"].append(time.perf_counter() - t_total)

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xnew.copy()

        print(f"  Iter {it}: omega={omega_cur:.3f} total={timings['total_iteration'][-1]:.3f}s")

    print("\n" + "=" * 70)
    print(f"{'Operation':<25} {'Mean (ms)':>10} {'Std (ms)':>10} {'% total':>10}")
    print("-" * 70)
    total_mean = np.mean(timings["total_iteration"])
    for key in ["filter_fwd", "heaviside", "material_interp", "assemble_global",
                 "submatrix_extract", "eigsh_solve", "sensitivity", "mma_solve", "eval_trial"]:
        vals = np.array(timings[key]) * 1000
        pct = np.mean(timings[key]) / total_mean * 100
        print(f"  {key:<23} {np.mean(vals):10.1f} {np.std(vals):10.1f} {pct:9.1f}%")
    print(f"  {'TOTAL':<23} {total_mean*1000:10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    profile_iteration()
