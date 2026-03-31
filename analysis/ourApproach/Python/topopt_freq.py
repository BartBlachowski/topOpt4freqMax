# TOPOLOGY OPTIMIZATION FOR FREQUENCY MAXIMIZATION
# Rewritten from the Python version (Aage & Johansen, 2013, modified)
#
# Supports aggregated compliance over multiple load cases via run_cfg["load_cases"].
# Includes semi_harmonic loads with a cached baseline (M0, Phi0, omega0).
# Passive elements (pas_s, pas_v) excluded from optimizer update.
# Supports OC and MMA optimizers.

from __future__ import annotations
import sys
import time
from typing import Optional
import numpy as np
from scipy.sparse import csc_array, coo_array
from scipy.sparse.linalg import spsolve, eigsh

import os
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'OlhoffApproach', 'Python'))
try:
    from mma import mmasub
except ImportError:
    mmasub = None

_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tools', 'Python'))
try:
    from nodal_projection import (
        build_nodal_projection_cache,
        project_q4_element_density_to_nodes,
    )
except ImportError:
    build_nodal_projection_cache = None
    project_q4_element_density_to_nodes = None


# =============================================================================
# Element matrices (LL, LR, UR, UL node order — matches Matlab edofMat)
# =============================================================================

def lk(hx: float, hy: float, nu: float = 0.3) -> np.ndarray:
    """Q4 plane-stress element stiffness matrix (E=1, node order LL-LR-UR-UL)."""
    E = 1.0
    D = (E / (1.0 - nu ** 2)) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float,
    )
    inv_j = np.array([[2.0 / hx, 0.0], [0.0, 2.0 / hy]], dtype=float)
    det_j = 0.25 * hx * hy
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = (-gp, gp)

    KE = np.zeros((8, 8), dtype=float)
    for xi in gauss_pts:
        for eta in gauss_pts:
            dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
            dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
            dN_xy = inv_j @ np.vstack((dN_dxi, dN_deta))
            dN_dx = dN_xy[0, :]
            dN_dy = dN_xy[1, :]
            B = np.zeros((3, 8), dtype=float)
            B[0, 0::2] = dN_dx
            B[1, 1::2] = dN_dy
            B[2, 0::2] = dN_dy
            B[2, 1::2] = dN_dx
            KE += (B.T @ D @ B) * det_j
    return KE


def lm(hx: float, hy: float) -> np.ndarray:
    """Q4 consistent element mass matrix (node order LL-LR-UR-UL)."""
    area = hx * hy
    Ms = (area / 36.0) * np.array(
        [[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], dtype=float
    )
    return np.kron(Ms, np.eye(2))


# =============================================================================
# Main solver
# =============================================================================

def topopt_freq(
    nelx: int,
    nely: int,
    volfrac: float,
    penal: float,
    rmin: float,
    ft: int,
    L: float,
    H: float,
    run_cfg: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, float, int, dict]:
    """Topology optimization for frequency maximization.

    Parameters
    ----------
    nelx, nely : int
        Number of elements in x and y.
    volfrac : float
        Volume fraction target.
    penal : float
        SIMP penalization exponent.
    rmin : float
        Filter radius in physical units.
    ft : int
        Filter type: 0 = sensitivity, 1 = density.
    L, H : float
        Physical domain dimensions.
    run_cfg : dict, optional
        Configuration struct (all fields optional):
            E0, Emin, nu, rho0, rho_min, pmass
            move, conv_tol, max_iters
            optimizer          : "OC" or "MMA"
            extra_fixed_dofs   : 0-based int array
            pas_s, pas_v       : 0-based int arrays (passive solid / void)
            load_cases         : normalized list from validate_load_cases()
            semi_harmonic_baseline : "solid" | "initial"
            semi_harmonic_rho_source : "x" | "xphys"
            harmonic_normalize : bool
            visualize_live     : bool
            save_frq_iterations: bool

    Returns
    -------
    x_out : (nelx*nely,) float array  — final physical density
    f_hz  : (3,) float array          — first three natural frequencies [Hz]
    t_iter : float                     — average time per iteration [s]
    n_iter : int                       — number of iterations executed
    info   : dict                      — diagnostics
    """
    if run_cfg is None:
        run_cfg = {}

    # --- Material / SIMP ---
    E0 = float(run_cfg.get("E0", 1e7))
    Emin = float(run_cfg.get("Emin", 1e-2))
    nu = float(run_cfg.get("nu", 0.3))
    rho0 = float(run_cfg.get("rho0", 1.0))
    rho_min = float(run_cfg.get("rho_min", 1e-6))
    pmass = float(run_cfg.get("pmass", 1.0))

    # --- Optimization settings ---
    move = float(run_cfg.get("move", 0.2))
    conv_tol = float(run_cfg.get("conv_tol", 0.01))
    max_iters = int(run_cfg.get("max_iters", 2000))
    optimizer_type = str(run_cfg.get("optimizer", "OC")).upper().strip()
    if optimizer_type not in {"OC", "MMA"}:
        raise ValueError(f'optimizer must be "OC" or "MMA" (got "{optimizer_type}")')
    visualize_live = bool(run_cfg.get("visualize_live", True))
    save_frq_iter = bool(run_cfg.get("save_frq_iterations", False))
    harmonic_normalize = bool(run_cfg.get("harmonic_normalize", True))
    semi_baseline = str(run_cfg.get("semi_harmonic_baseline", "solid")).lower().strip()
    semi_rho_src = str(run_cfg.get("semi_harmonic_rho_source", "x")).lower().strip()
    use_heaviside = bool(run_cfg.get("use_heaviside", False))
    beta_h_schedule = [float(b) for b in run_cfg.get("heaviside_beta_schedule", [1, 2, 4, 8, 16, 32, 64])]
    beta_h_interval = int(run_cfg.get("heaviside_beta_interval", 40))
    eta_h = float(run_cfg.get("heaviside_eta", 0.5))
    beta_h_idx = 1
    beta_h = beta_h_schedule[0]

    print(f"Compliance objective with load-case aggregation")
    print(f"mesh: {nelx} x {nely}")
    print(f"domain: L x H = {L} x {H}")
    print(f"volfrac: {volfrac}, rmin(phys): {rmin}, penal: {penal}")
    ftnames = ["Sensitivity based", "Density based"]
    print(f"Filter method: {ftnames[ft]}")
    print(f"Optimizer: {optimizer_type}")

    if L <= 0.0 or H <= 0.0:
        raise ValueError("L and H must be positive.")
    hx = L / nelx
    hy = H / nely
    print(f"element size: hx={hx:.6g}, hy={hy:.6g}")

    n_el = nelx * nely
    n_nodes = (nelx + 1) * (nely + 1)
    ndof = 2 * n_nodes

    # --- Passive elements ---
    pas_s = np.asarray(run_cfg.get("pas_s", []), dtype=np.int64).ravel()
    pas_v = np.asarray(run_cfg.get("pas_v", []), dtype=np.int64).ravel()
    act = np.setdiff1d(np.arange(n_el, dtype=np.int64), np.union1d(pas_s, pas_v))

    # --- Boundary conditions ---
    extra_fixed = np.asarray(run_cfg.get("extra_fixed_dofs", []), dtype=np.int64).ravel()
    fixed = np.unique(extra_fixed) if extra_fixed.size else np.array([], dtype=np.int64)
    # Fallback: hinge-hinge at mid-height if no BCs provided.
    if fixed.size == 0:
        j_mid = nely // 2
        nL = j_mid
        nR = nelx * (nely + 1) + j_mid
        fixed = np.array([2 * nL, 2 * nL + 1, 2 * nR, 2 * nR + 1], dtype=np.int64)
    free = np.setdiff1d(np.arange(ndof, dtype=np.int64), fixed)

    # --- Element matrices ---
    KE = lk(hx, hy, nu)
    ME = lm(hx, hy)

    # --- Build edofMat (Q4, LL LR UR UL, 0-based) ---
    edof_mat = np.zeros((n_el, 8), dtype=np.int64)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely          # LL
            n2 = (nely + 1) * (elx + 1) + ely   # LR
            n3 = n2 + 1                            # UR
            n4 = n1 + 1                            # UL
            edof_mat[el, :] = [
                2 * n1, 2 * n1 + 1,
                2 * n2, 2 * n2 + 1,
                2 * n3, 2 * n3 + 1,
                2 * n4, 2 * n4 + 1,
            ]

    iK = np.kron(edof_mat, np.ones((8, 1), dtype=np.int64)).ravel()   # (64*n_el,)
    jK = np.kron(edof_mat, np.ones((1, 8), dtype=np.int64)).ravel()

    # --- Filter matrix ---
    rminx = max(1, int(np.ceil(rmin / hx)))
    rminy = max(1, int(np.ceil(rmin / hy)))
    nfilter = n_el * (2 * (rminx - 1) + 1) * (2 * (rminy - 1) + 1)
    iH = np.zeros(nfilter, dtype=np.int64)
    jH = np.zeros(nfilter, dtype=np.int64)
    sH = np.zeros(nfilter, dtype=float)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            for k in range(max(i - rminx + 1, 0), min(i + rminx, nelx)):
                for ll in range(max(j - rminy + 1, 0), min(j + rminy, nely)):
                    col = k * nely + ll
                    dx = (i - k) * hx
                    dy = (j - ll) * hy
                    fac = rmin - np.sqrt(dx * dx + dy * dy)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = max(0.0, fac)
                    cc += 1
    Hf = csc_array((sH[:cc], (iH[:cc], jH[:cc])), shape=(n_el, n_el))
    Hs = np.asarray(Hf.sum(axis=1)).ravel()

    # --- Design variables ---
    x = volfrac * np.ones(n_el, dtype=float)
    x[pas_s] = 1.0
    x[pas_v] = 0.0
    if pas_s.size or pas_v.size:
        n_act = act.size
        if n_act > 0:
            act_target = (volfrac * n_el - pas_s.size) / n_act
            x[act] = np.clip(act_target, 0.0, 1.0)

    if use_heaviside:
        _xt0 = np.asarray(Hf @ x) / Hs if ft == 1 else x
        xPhys, _ = _heaviside_projection(_xt0, beta_h, eta_h)
        if pas_s.size: xPhys[pas_s] = 1.0
        if pas_v.size: xPhys[pas_v] = 0.0
    else:
        xPhys = x.copy()
    g = 0.0  # OC Lagrange accumulator

    # --- Node coordinates (for load assembly) ---
    node_ids_0 = np.arange(n_nodes, dtype=np.int64)
    node_i = node_ids_0 // (nely + 1)   # column
    node_j = node_ids_0 % (nely + 1)    # row
    node_x = node_i * hx
    node_y = node_j * hy
    y_down = np.zeros(ndof, dtype=float)
    y_down[1::2] = -1.0   # uy DOFs = -1 (gravity direction)

    # --- Resolve load cases ---
    (
        load_cases, using_lc, max_harmonic_mode,
        mode_update_after, max_semi_mode,
    ) = _resolve_load_cases(run_cfg, node_x, node_y, node_ids_0)

    n_cases = len(load_cases)
    print(f"Load cases: {n_cases}")
    for ci, lc in enumerate(load_cases):
        n_ld = len(lc["loads"])
        print(f'  case[{ci}] "{lc["name"]}": factor={lc["factor"]}, nLoads={n_ld}')

    has_semi = using_lc and max_semi_mode > 0

    # --- Semi-harmonic baseline setup ---
    nodal_proj_cache = None
    semi_base_vec = np.zeros((ndof, max(max_semi_mode, 0)), dtype=float)
    semi_omega0 = np.full(max(max_semi_mode, 0), np.nan, dtype=float)
    semi_phi0 = np.zeros((ndof, max(max_semi_mode, 0)), dtype=float)
    semi_M0 = None

    if has_semi:
        if build_nodal_projection_cache is None:
            raise ImportError("tools/Python/nodal_projection.py is required for semi_harmonic loads.")
        nodal_proj_cache = build_nodal_projection_cache(nelx, nely)
        print(f"[Load cases] semi_harmonic baseline={semi_baseline}, rho_source={semi_rho_src}")

        x_base = _build_semi_baseline(semi_baseline, xPhys, pas_s, pas_v, n_el)
        sK0 = (KE.ravel()[:, np.newaxis] * (Emin + x_base ** penal * (E0 - Emin))).ravel(order="F")
        K0 = csc_array((sK0, (iK, jK)), shape=(ndof, ndof))
        K0 = (K0 + K0.T) / 2

        rho0_vec = rho_min + x_base ** pmass * (rho0 - rho_min)
        sM0 = (ME.ravel()[:, np.newaxis] * rho0_vec).ravel(order="F")
        semi_M0 = csc_array((sM0, (iK, jK)), shape=(ndof, ndof))
        semi_M0 = (semi_M0 + semi_M0.T) / 2

        K0f = K0[free][:, free]
        M0f = semi_M0[free][:, free]
        omegas0, phis0 = _compute_modes(K0f, M0f, free, ndof, max_semi_mode)
        for k in range(max_semi_mode):
            if not np.isfinite(omegas0[k]):
                raise RuntimeError(f"Unable to evaluate semi_harmonic mode {k+1} from baseline model.")
            semi_omega0[k] = omegas0[k]
            semi_phi0[:, k] = phis0[:, k]
            semi_base_vec[:, k] = omegas0[k] * (semi_M0 @ phis0[:, k])
        print(f"[Load cases] semi_harmonic baseline cached up to mode {max_semi_mode}.")

    # --- Legacy one-time eigensolve (no configured load cases) ---
    omega_legacy = np.nan
    phi_legacy = np.zeros(ndof, dtype=float)

    if not using_lc:
        sK0 = (KE.ravel()[:, np.newaxis] * (Emin + xPhys ** penal * (E0 - Emin))).ravel(order="F")
        K0 = csc_array((sK0, (iK, jK)), shape=(ndof, ndof))
        K0 = (K0 + K0.T) / 2
        rho0_vec = rho_min + xPhys ** pmass * (rho0 - rho_min)
        sM0 = (ME.ravel()[:, np.newaxis] * rho0_vec).ravel(order="F")
        M0 = csc_array((sM0, (iK, jK)), shape=(ndof, ndof))
        M0 = (M0 + M0.T) / 2
        K0f = K0[free][:, free]
        M0f = M0[free][:, free]
        lam_vals, phi_free = eigsh(K0f, M=M0f, k=2, sigma=1e-6, which="LM")
        lam_vals = np.real(lam_vals)
        idx = np.argmin(lam_vals)
        lam1 = max(lam_vals[idx], 0.0)
        omega_legacy = np.sqrt(lam1)
        pf = np.real(phi_free[:, idx])
        mn = float(pf @ (M0f @ pf))
        if mn > 0:
            pf = pf / np.sqrt(mn)
        phi_legacy[free] = pf
        print(f"[Eigen] lambda1={lam1:.6e}, omega1={omega_legacy:.6e} rad/s (computed once, fixed)")

    # --- Harmonic eigenpair cache ---
    harmonic_omegas = np.full(max(max_harmonic_mode, 0), np.nan, dtype=float)
    harmonic_phi = np.zeros((ndof, max(max_harmonic_mode, 0)), dtype=float)
    harmonic_norm_ref = np.full(max(max_harmonic_mode, 0), np.nan, dtype=float)

    # --- MMA persistent state ---
    n_act = act.size
    if optimizer_type == "MMA":
        if mmasub is None:
            raise ImportError("mmasub not found. Ensure OlhoffApproach/Python/mma.py is accessible.")
        mma_low = np.zeros(n_act, dtype=float)
        mma_upp = np.ones(n_act, dtype=float)
        mma_xold1 = x[act].copy()
        mma_xold2 = x[act].copy()

    # --- Visualization ---
    backend_name = ""
    try:
        import matplotlib
        backend_name = matplotlib.get_backend().lower()
    except ImportError:
        visualize_live = False
    live_plot = visualize_live and "agg" not in backend_name
    fig = ax_img = im = None
    if live_plot:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax_img = plt.subplots(figsize=(10, 3))
            im = ax_img.imshow(
                xPhys.reshape((nelx, nely)).T, cmap="gray_r",
                interpolation="none", origin="lower",
                vmin=0, vmax=1,
            )
            ax_img.set_title("Topology (density)")
            fig.tight_layout()
            plt.show(block=False)
        except Exception:
            live_plot = False

    obj_hist: list[float] = []
    vol_hist: list[float] = []
    ch_hist: list[float] = []

    info: dict = {}
    if save_frq_iter:
        info["freq_iter_omega"] = np.full((max_iters, 3), np.nan)

    # ==========================================================================
    # Optimization loop
    # ==========================================================================
    loop = 0
    change = 1.0
    obj = np.nan
    F = np.zeros((ndof, n_cases), dtype=float)
    U = np.zeros((ndof, n_cases), dtype=float)
    obj_cases = np.zeros(n_cases, dtype=float)

    loop_tic = time.perf_counter()

    while change > conv_tol and loop < max_iters:
        loop += 1

        # --- Assemble M(x) ---
        rho_phys = rho_min + xPhys ** pmass * (rho0 - rho_min)
        sM = (ME.ravel()[:, np.newaxis] * rho_phys).ravel(order="F")
        M = csc_array((sM, (iK, jK)), shape=(ndof, ndof))
        M = (M + M.T) / 2

        # --- Assemble K(x) ---
        sK = (KE.ravel()[:, np.newaxis] * (Emin + xPhys ** penal * (E0 - Emin))).ravel(order="F")
        K = csc_array((sK, (iK, jK)), shape=(ndof, ndof))
        K = (K + K.T) / 2

        Kf = K[free][:, free]

        # --- Update harmonic eigenpair cache for due modes ---
        if using_lc and max_harmonic_mode > 0:
            due_modes = []
            for k in range(max_harmonic_mode):
                ua = mode_update_after[k]
                if ua == 0:
                    if loop == 1:
                        due_modes.append(k)
                else:
                    if (loop - 1) % ua == 0:
                        due_modes.append(k)

            if due_modes:
                max_mode_needed = max(due_modes) + 1
                Mf = M[free][:, free]
                new_omegas, new_phi = _compute_modes(Kf, Mf, free, ndof, max_mode_needed)
                for k in range(max_mode_needed):
                    if np.isfinite(new_omegas[k]):
                        harmonic_omegas[k] = new_omegas[k]
                        harmonic_phi[:, k] = new_phi[:, k]
            else:
                Mf = None
        else:
            Mf = None

        # --- Nodal density projection for semi-harmonic loads ---
        rho_nodal = None
        rho_source_vec = None
        if has_semi:
            rho_source_vec = _semi_rho_source(semi_rho_src, x, xPhys)
            rho_nodal, nodal_proj_cache = project_q4_element_density_to_nodes(
                rho_source_vec, nelx, nely, nodal_proj_cache
            )

        # --- Assemble load matrix ---
        F[:] = 0.0
        for ci, lc in enumerate(load_cases):
            Fi = np.zeros(ndof, dtype=float)
            for ld in lc["loads"]:
                ldf = float(ld.get("factor", 1.0))
                lt = ld["type"]

                if lt == "self_weight":
                    Fi += ldf * (M @ y_down)

                elif lt == "closest_node":
                    node_id = int(ld["node_id"])
                    Fi[2 * node_id] += ldf * ld["force"][0]
                    Fi[2 * node_id + 1] += ldf * ld["force"][1]

                elif lt == "harmonic":
                    mode_k = ld["mode"] - 1   # 0-based
                    if mode_k < len(harmonic_omegas) and np.isfinite(harmonic_omegas[mode_k]):
                        omega_k = harmonic_omegas[mode_k]
                        phi_k = harmonic_phi[:, mode_k]
                    elif np.isfinite(omega_legacy):
                        omega_k = omega_legacy
                        phi_k = phi_legacy
                    else:
                        raise RuntimeError(
                            f'Unable to evaluate harmonic mode {ld["mode"]} for case "{lc["name"]}".'
                        )
                    fH = (omega_k ** 2) * (M @ phi_k)
                    if harmonic_normalize:
                        n_raw = np.linalg.norm(fH)
                        if not np.isfinite(harmonic_norm_ref[mode_k]) or harmonic_norm_ref[mode_k] <= 0:
                            if n_raw > 0:
                                harmonic_norm_ref[mode_k] = n_raw
                        n_ref = harmonic_norm_ref[mode_k]
                        if np.isfinite(n_ref) and n_ref > 0 and n_raw > 0:
                            fH = fH * (n_ref / n_raw)
                    Fi += ldf * fH

                elif lt == "semi_harmonic":
                    mode_k = ld["mode"] - 1   # 0-based
                    if rho_nodal is None:
                        raise RuntimeError("rho_nodal is None but semi_harmonic load is active.")
                    if mode_k >= semi_base_vec.shape[1] or not np.isfinite(semi_omega0[mode_k]):
                        raise RuntimeError(
                            f'Unable to evaluate semi_harmonic mode {ld["mode"]} for case "{lc["name"]}".'
                        )
                    # rho_dof: same nodal density at both DOFs of each node
                    rho_dof = np.repeat(rho_nodal, 2)
                    fSemi = rho_dof * semi_base_vec[:, mode_k]
                    Fi += ldf * fSemi

            Fi *= lc["factor"]
            F[:, ci] = Fi

        # --- Solve FE ---
        U[:] = 0.0
        for ci in range(n_cases):
            U[free, ci] = spsolve(Kf, F[free, ci])

        if save_frq_iter:
            if Mf is None:
                Mf = M[free][:, free]
            info["freq_iter_omega"][loop - 1, :] = _first_n_omegas(Kf, Mf, 3)

        # --- Objective and sensitivities ---
        obj = 0.0
        obj_cases[:] = 0.0
        dc = np.zeros(n_el, dtype=float)
        stiff_scale = -penal * xPhys ** (penal - 1) * (E0 - Emin)
        dM_dx_scale = pmass * xPhys ** (pmass - 1) * (rho0 - rho_min)

        for ci, lc in enumerate(load_cases):
            Ui = U[:, ci]
            obj_case = float(Ui @ (K @ Ui))
            obj_cases[ci] = obj_case
            obj += obj_case

            ue = Ui[edof_mat]   # (n_el, 8)
            ce = (ue @ KE * ue).sum(axis=1)
            dc += stiff_scale * ce

            if using_lc:
                dc += _load_sensitivity(
                    Ui, ue, lc, dM_dx_scale, ME, y_down,
                    harmonic_phi, harmonic_omegas, phi_legacy, omega_legacy,
                    edof_mat, nodal_proj_cache, semi_base_vec,
                )

        # Passive elements excluded from volume constraint and update.
        dv = np.ones(n_el, dtype=float)
        dv[pas_s] = 0.0; dv[pas_v] = 0.0
        dc[pas_s] = 0.0; dc[pas_v] = 0.0

        # --- Heaviside chain rule (applied before filter) ---
        if use_heaviside:
            _x_tilde = np.asarray(Hf @ x) / Hs if ft == 1 else x
            _, dH_vals = _heaviside_projection(_x_tilde, beta_h, eta_h)
            dc = dc * dH_vals
            dv = dv * dH_vals
            dc[pas_s] = 0.0; dc[pas_v] = 0.0
            dv[pas_s] = 0.0; dv[pas_v] = 0.0

        # --- Filter ---
        if ft == 0:
            dc = np.asarray(Hf @ (x * dc)) / Hs / np.maximum(0.001, x)
        elif ft == 1:
            dc = np.asarray(Hf @ (dc / Hs))
            dv = np.asarray(Hf @ (dv / Hs))

        # Re-zero passive after filtering.
        dc[pas_s] = 0.0; dc[pas_v] = 0.0
        dv[pas_s] = 0.0; dv[pas_v] = 0.0

        # --- Design update ---
        xold = x.copy()

        if optimizer_type == "OC":
            x, g = _oc_update(x, volfrac, dc, dv, g, move, act)
            x[pas_s] = 1.0; x[pas_v] = 0.0

        else:  # MMA
            n_act_cur = n_act
            xmin_act = np.zeros(n_act_cur)
            xmax_act = np.ones(n_act_cur)
            fval_mma = np.array([np.mean(xPhys) - volfrac])
            dfdx_act = (dv[act] / n_el).reshape(1, -1)

            xnew_act, _, _, _, _, _, _, _, _, mma_low, mma_upp = mmasub(
                1, n_act_cur, loop,
                x[act].reshape(-1, 1), xmin_act.reshape(-1, 1), xmax_act.reshape(-1, 1),
                mma_xold1.reshape(-1, 1), mma_xold2.reshape(-1, 1),
                obj, dc[act].reshape(-1, 1), fval_mma, dfdx_act,
                mma_low, mma_upp,
                1, np.zeros((1, 1)), 1e3 * np.ones((1, 1)), np.ones((1, 1)),
            )
            xnew_act = xnew_act.ravel()
            xnew_act = np.clip(xnew_act, np.maximum(0.0, x[act] - move),
                                         np.minimum(1.0, x[act] + move))
            mma_xold2 = mma_xold1.copy()
            mma_xold1 = x[act].copy()
            x[act] = xnew_act
            x[pas_s] = 1.0; x[pas_v] = 0.0

        # --- Physical field ---
        if use_heaviside:
            _x_tilde = np.asarray(Hf @ x) / Hs if ft == 1 else x
            xPhys, _ = _heaviside_projection(_x_tilde, beta_h, eta_h)
            if pas_s.size: xPhys[pas_s] = 1.0
            if pas_v.size: xPhys[pas_v] = 0.0
        else:
            xPhys = _physical_field(x, ft, Hf, Hs, pas_s, pas_v)

        # MMA volume projection (if needed, skip when heaviside active)
        if optimizer_type == "MMA" and not use_heaviside:
            vol_residual = np.mean(xPhys) - volfrac
            if vol_residual > 1e-10:
                x, xPhys, _ = _mma_volume_project(x, xold, act, move, ft, Hf, Hs, pas_s, pas_v, volfrac)

        # --- Change and volume ---
        vol = np.mean(xPhys)
        change = np.max(np.abs(x - xold))

        obj_hist.append(obj)
        vol_hist.append(vol)
        ch_hist.append(change)

        # --- Heaviside beta advancement ---
        if use_heaviside:
            target_h_idx = min(len(beta_h_schedule), int(np.floor(loop / beta_h_interval)) + 1)
            if target_h_idx > beta_h_idx:
                beta_h_idx = target_h_idx
                beta_h = float(beta_h_schedule[beta_h_idx - 1])
                print(f"  Heaviside beta -> {beta_h} (iteration {loop})")

        # Print iteration
        if optimizer_type == "MMA":
            print(f"it.: {loop:4d} , obj: {obj:.3f} Vol.: {vol:.3f}, ch.: {change:.3f}")
        else:
            print(f"it.: {loop:4d} , obj: {obj:.3f} Vol.: {vol:.3f}, ch.: {change:.3f}")

        # Live plot
        if live_plot and fig is not None:
            try:
                import matplotlib.pyplot as plt
                im.set_data(xPhys.reshape((nelx, nely)).T)
                ax_img.set_title(f"Topology | it={loop}  obj={obj:.3e}  vol={vol:.3f}")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
            except Exception:
                pass

    loop_time = time.perf_counter() - loop_tic
    t_iter = loop_time / max(loop, 1)
    n_iter = loop

    if save_frq_iter:
        info["freq_iter_omega"] = info["freq_iter_omega"][:loop, :]

    # ==========================================================================
    # Post-analysis: first three natural frequencies of final topology
    # ==========================================================================
    sK_f = (KE.ravel()[:, np.newaxis] * (Emin + xPhys ** penal * (E0 - Emin))).ravel(order="F")
    K_f = csc_array((sK_f, (iK, jK)), shape=(ndof, ndof))
    K_f = (K_f + K_f.T) / 2
    rho_f = rho_min + xPhys ** pmass * (rho0 - rho_min)
    sM_f = (ME.ravel()[:, np.newaxis] * rho_f).ravel(order="F")
    M_f = csc_array((sM_f, (iK, jK)), shape=(ndof, ndof))
    M_f = (M_f + M_f.T) / 2
    Kff = K_f[free][:, free]
    Mff = M_f[free][:, free]
    n_req = min(3, max(1, free.size - 1))
    try:
        lam_final, _ = eigsh(Kff, M=Mff, k=n_req, sigma=1e-6, which="LM")
        lam_final = np.real(lam_final)
        lam_final = np.sort(lam_final[lam_final > 0])
    except Exception:
        lam_final = np.array([])

    f_hz = np.full(3, np.nan)
    if lam_final.size > 0:
        n_ok = min(3, lam_final.size)
        f_hz[:n_ok] = np.sqrt(lam_final[:n_ok]) / (2.0 * np.pi)

    omega1_f = np.sqrt(lam_final[0]) if lam_final.size > 0 else np.nan
    f1_f = f_hz[0]
    print(f"[Final Eigen] omega1={omega1_f:.6e} rad/s, f1={f1_f:.6e} Hz")

    if live_plot:
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()
        except Exception:
            pass

    info["last_obj"] = obj
    info["last_vol"] = float(np.mean(xPhys))

    return xPhys.copy(), f_hz, t_iter, n_iter, info


# =============================================================================
# Internal helpers
# =============================================================================

def _resolve_load_cases(
    run_cfg: dict,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_ids: np.ndarray,
) -> tuple[list, bool, int, list, int]:
    """Resolve load cases from run_cfg, returning normalized load case list."""
    if "load_cases" in run_cfg and run_cfg["load_cases"]:
        load_cases = run_cfg["load_cases"]  # already validated by caller
        using_lc = True
    else:
        # Legacy fallback: one harmonic load on mode 1.
        load_cases = [{
            "name": "legacy_harmonic_fixed_mode",
            "factor": 1.0,
            "loads": [{"type": "harmonic", "mode": 1, "factor": 1.0, "update_after": 1}],
        }]
        using_lc = False

    max_harmonic_mode = 0
    max_semi_mode = 0
    mode_update_after_raw: dict[int, float] = {}  # 1-based mode -> min ua

    for lc in load_cases:
        for ld in lc["loads"]:
            lt = ld["type"]
            if lt == "closest_node":
                # Resolve closest node id (0-based).
                loc = np.asarray(ld["location"], dtype=float)
                dist2 = (node_x - loc[0]) ** 2 + (node_y - loc[1]) ** 2
                min_d2 = dist2.min()
                node_id = int(node_ids[dist2 == min_d2].min())
                ld["node_id"] = node_id
            elif lt == "harmonic":
                mode_k = int(ld["mode"])
                max_harmonic_mode = max(max_harmonic_mode, mode_k)
                ua = int(ld.get("update_after", 1))
                prev = mode_update_after_raw.get(mode_k, float("inf"))
                mode_update_after_raw[mode_k] = min(prev, ua)
            elif lt == "semi_harmonic":
                max_semi_mode = max(max_semi_mode, int(ld["mode"]))

    if max_harmonic_mode > 0:
        mode_update_after = [
            int(mode_update_after_raw.get(k + 1, 1)) for k in range(max_harmonic_mode)
        ]
    else:
        mode_update_after = []

    return load_cases, using_lc, max_harmonic_mode, mode_update_after, max_semi_mode


def _compute_modes(
    Kf: csc_array,
    Mf: csc_array,
    free: np.ndarray,
    ndof: int,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first n_modes mass-normalized eigenpairs, returned in global DOFs."""
    omegas = np.full(n_modes, np.nan, dtype=float)
    phis = np.zeros((ndof, n_modes), dtype=float)
    n_free = free.size
    n_req = min(n_modes, n_free - 1)
    if n_req < 1:
        return omegas, phis
    try:
        lam_vals, V = eigsh(Kf, M=Mf, k=n_req, sigma=1e-6, which="LM",
                            tol=1e-8, maxiter=800)
        lam_vals = np.real(lam_vals)
        order = np.argsort(lam_vals)
        lam_vals = lam_vals[order]
        V = np.real(V[:, order])
        valid = np.isfinite(lam_vals) & (lam_vals > 0)
        lam_vals = lam_vals[valid]
        V = V[:, valid]
        n_ok = min(n_modes, lam_vals.size)
        for k in range(n_ok):
            phi = V[:, k]
            mn = float(phi @ (Mf @ phi))
            if mn > 0:
                phi = phi / np.sqrt(mn)
            omegas[k] = np.sqrt(lam_vals[k])
            phis[free, k] = phi
    except Exception:
        pass
    return omegas, phis


def _first_n_omegas(Kf: csc_array, Mf: csc_array, n: int) -> np.ndarray:
    """Return first n circular frequencies from submatrices (NaN on failure)."""
    omegas = np.full(n, np.nan, dtype=float)
    n_req = min(n, max(1, Kf.shape[0] - 1))
    if n_req < 1:
        return omegas
    try:
        lam, _ = eigsh(Kf, M=Mf, k=n_req, sigma=1e-6, which="LM")
        lam = np.sort(np.real(lam[np.real(lam) > 0]))
        n_ok = min(n, lam.size)
        omegas[:n_ok] = np.sqrt(lam[:n_ok])
    except Exception:
        pass
    return omegas


def _build_semi_baseline(kind: str, xPhys: np.ndarray, pas_s, pas_v, n_el: int) -> np.ndarray:
    if kind == "solid":
        x_base = np.ones(n_el, dtype=float)
        if len(pas_v): x_base[pas_v] = 0.0
        if len(pas_s): x_base[pas_s] = 1.0
    elif kind == "initial":
        x_base = xPhys.copy()
    else:
        raise ValueError(f'Unknown semi_harmonic baseline "{kind}".')
    return x_base


def _semi_rho_source(source: str, x: np.ndarray, xPhys: np.ndarray) -> np.ndarray:
    if source == "x":
        return x.copy()
    if source == "xphys":
        return xPhys.copy()
    raise ValueError(f'Unknown semi_harmonic rho source "{source}".')


def _load_sensitivity(
    Ui: np.ndarray,
    ue: np.ndarray,
    lc: dict,
    dM_dx_scale: np.ndarray,
    ME: np.ndarray,
    y_down: np.ndarray,
    harmonic_phi: np.ndarray,
    harmonic_omegas: np.ndarray,
    phi_legacy: np.ndarray,
    omega_legacy: float,
    edof_mat: np.ndarray,
    nodal_proj_cache,
    semi_base_vec: np.ndarray,
) -> np.ndarray:
    """Load-dependent sensitivity d(F)/dx contribution for one load case."""
    n_el = dM_dx_scale.size
    dc_load = np.zeros(n_el, dtype=float)
    lc_factor = lc["factor"]

    for ld in lc["loads"]:
        ldf = float(ld.get("factor", 1.0))
        lt = ld["type"]
        coeff = lc_factor * ldf
        if coeff == 0.0:
            continue

        if lt == "self_weight":
            vec = y_down
        elif lt == "harmonic":
            mode_k = ld["mode"] - 1
            if mode_k < len(harmonic_omegas) and np.isfinite(harmonic_omegas[mode_k]):
                omega_k = harmonic_omegas[mode_k]
                phi_k = harmonic_phi[:, mode_k]
            elif np.isfinite(omega_legacy):
                omega_k = omega_legacy
                phi_k = phi_legacy
            else:
                continue
            vec = (omega_k ** 2) * phi_k
        elif lt == "semi_harmonic":
            # Frozen-load approximation: semi_harmonic load sensitivity is zeroed.
            continue
        else:
            # closest_node has no rho dependence.
            continue

        vec_e = vec[edof_mat]   # (n_el, 8)
        uMeVec = (ue @ ME * vec_e).sum(axis=1)
        dc_load += 2.0 * coeff * dM_dx_scale * uMeVec

    return dc_load


def _physical_field(
    x: np.ndarray, ft: int,
    Hf: csc_array, Hs: np.ndarray,
    pas_s: np.ndarray, pas_v: np.ndarray,
) -> np.ndarray:
    if ft == 0:
        xPhys = x.copy()
    elif ft == 1:
        xPhys = np.asarray(Hf @ x) / Hs
    else:
        raise ValueError(f"Unsupported ft={ft}")
    if pas_s.size: xPhys[pas_s] = 1.0
    if pas_v.size: xPhys[pas_v] = 0.0
    return xPhys


def _heaviside_projection(x_tilde: np.ndarray, beta: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    denom = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    x_phys = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom
    dH = (beta * (1.0 - np.tanh(beta * (x_tilde - eta)) ** 2)) / denom
    return x_phys, dH


def _oc_update(
    x: np.ndarray,
    volfrac: float,
    dc: np.ndarray,
    dv: np.ndarray,
    g: float,
    move: float,
    act: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Optimality criteria update on active elements."""
    l1 = 0.0
    l2 = 1e9
    xnew = x.copy()
    eps = 1e-30
    dv_safe = np.maximum(dv, 1e-12)
    dc_safe = np.minimum(dc, -1e-12)

    for _ in range(200):
        lmid = 0.5 * (l1 + l2)
        denom = max(lmid, 1e-30)
        B = np.maximum(-dc_safe / dv_safe / denom, 1e-30)
        x_cand = x * np.sqrt(B)
        xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x_cand))))
        # Only enforce volume on active elements.
        gt = g + float(dv_safe[act] @ (xnew[act] - x[act]))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        if (l2 - l1) / max(l1 + l2, eps) < 1e-3:
            break
    return xnew, gt


def _mma_volume_project(
    x: np.ndarray,
    xold: np.ndarray,
    act: np.ndarray,
    move: float,
    ft: int,
    Hf: csc_array,
    Hs: np.ndarray,
    pas_s: np.ndarray,
    pas_v: np.ndarray,
    volfrac: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Monotone bisection to enforce volume feasibility after MMA update."""
    x_proj = x.copy()
    xPhys_proj = _physical_field(x_proj, ft, Hf, Hs, pas_s, pas_v)
    residual = float(np.mean(xPhys_proj) - volfrac)
    if residual <= 0 or act.size == 0:
        return x_proj, xPhys_proj, residual

    lb = np.maximum(0.0, xold[act] - move)
    ub = np.minimum(1.0, xold[act] + move)
    x_act_base = np.clip(x_proj[act], lb, ub)

    x_low = x_proj.copy()
    x_low[act] = lb
    if pas_s.size: x_low[pas_s] = 1.0
    if pas_v.size: x_low[pas_v] = 0.0
    xPhys_low = _physical_field(x_low, ft, Hf, Hs, pas_s, pas_v)
    res_low = float(np.mean(xPhys_low) - volfrac)
    if res_low > 0:
        return x_low, xPhys_low, res_low

    tau_lo, tau_hi = 0.0, max(float(np.max(x_act_base - lb)), 1e-12)
    x_best = x_low.copy(); xPhys_best = xPhys_low.copy(); res_best = res_low

    for _ in range(40):
        tau = 0.5 * (tau_lo + tau_hi)
        x_try = x_proj.copy()
        x_try[act] = np.clip(x_act_base - tau, lb, ub)
        if pas_s.size: x_try[pas_s] = 1.0
        if pas_v.size: x_try[pas_v] = 0.0
        xPhys_try = _physical_field(x_try, ft, Hf, Hs, pas_s, pas_v)
        res_try = float(np.mean(xPhys_try) - volfrac)
        if res_try > 0:
            tau_lo = tau
        else:
            tau_hi = tau
            x_best = x_try; xPhys_best = xPhys_try; res_best = res_try

    return x_best, xPhys_best, res_best


# =============================================================================
# Stand-alone entry point
# =============================================================================

if __name__ == "__main__":
    nelx = 240
    nely = 30
    volfrac = 0.4
    rmin = 0.05
    penal = 3.0
    ft = 0
    L = 8.0
    H = 1.0

    args = sys.argv[1:]
    if len(args) > 0: nelx = int(args[0])
    if len(args) > 1: nely = int(args[1])
    if len(args) > 2: volfrac = float(args[2])
    if len(args) > 3: rmin = float(args[3])
    if len(args) > 4: penal = float(args[4])
    if len(args) > 5: ft = int(args[5])
    if len(args) > 6: L = float(args[6])
    if len(args) > 7: H = float(args[7])

    topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H)
