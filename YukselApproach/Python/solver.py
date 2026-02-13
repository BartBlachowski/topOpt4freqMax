from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as spla


@dataclass
class YukselResult:
    xPhys_stage2: np.ndarray
    U_stage2: np.ndarray
    info: dict


def _matlab_round(x: float) -> int:
    """MATLAB-compatible round for positive values (ties away from zero)."""
    return int(np.floor(x + 0.5))


def _lower_col_major(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    out = []
    for j in range(n):
        for i in range(j, n):
            out.append(A[i, j])
    return np.asarray(out, dtype=np.float64)


def _sym_from_lower_col_major(v: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.float64)
    k = 0
    for j in range(n):
        for i in range(j, n):
            out[i, j] = v[k]
            k += 1
    out = out + out.T - np.diag(np.diag(out))
    return out


def _local_assemble(i: np.ndarray, j: np.ndarray, s: np.ndarray, shape: tuple[int, int]) -> sparse.csr_matrix:
    return sparse.coo_matrix((s, (i, j)), shape=shape).tocsr()


def _solve_smallest_modes(Kff: sparse.spmatrix, Mff: sparse.spmatrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = Kff.shape[0]
    if n <= 2:
        raise RuntimeError("Not enough DOFs for eigen solve")
    k = max(1, min(k, n - 2))

    vals, vecs = spla.eigsh(Kff, k=k, M=Mff, sigma=0.0, which="LM", tol=1e-8, maxiter=6000)
    ord_idx = np.argsort(np.real(vals))
    return np.real(vals[ord_idx]), np.real(vecs[:, ord_idx])


def _physical_setup(bc_type: str) -> tuple[float, float, float]:
    b = bc_type.lower()
    if b == "cantilever":
        return 15.0, 10.0, 0.20
    if b in {"simply", "fixedpinned"}:
        return 8.0, 1.0, 0.0
    raise ValueError(f"Unsupported bcType '{bc_type}'")


def _build_ke_ke0(nu: float) -> tuple[np.ndarray, np.ndarray]:
    c1 = np.array(
        [
            12,
            3,
            -6,
            -3,
            -6,
            -3,
            0,
            3,
            12,
            3,
            0,
            -3,
            -6,
            -3,
            -6,
            12,
            -3,
            0,
            -3,
            -6,
            3,
            12,
            3,
            -6,
            3,
            -6,
            12,
            3,
            -6,
            -3,
            12,
            3,
            0,
            12,
            -3,
            12,
        ],
        dtype=np.float64,
    )
    c2 = np.array(
        [
            -4,
            3,
            -2,
            9,
            2,
            -3,
            4,
            -9,
            -4,
            -9,
            4,
            -3,
            2,
            9,
            -2,
            -4,
            -3,
            4,
            9,
            2,
            3,
            -4,
            -9,
            -2,
            3,
            2,
            -4,
            3,
            -2,
            9,
            -4,
            -9,
            4,
            -4,
            -3,
            -4,
        ],
        dtype=np.float64,
    )
    ke_lower = 1.0 / (1 - nu**2) / 24.0 * (c1 + nu * c2)

    Ke0 = _sym_from_lower_col_major(ke_lower, 8)
    return ke_lower, Ke0


def _build_me0(beamL: float, beamH: float, nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray]:
    elem_area = (beamL / nelx) * (beamH / nely)
    MeS = (elem_area / 36.0) * np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], dtype=np.float64)
    Me0 = np.kron(MeS, np.eye(2))
    me_lower = _lower_col_major(Me0)
    return me_lower, Me0


def _bc_and_load(node_nrs_1: np.ndarray, nely: int, nelx: int, bc_type: str) -> tuple[np.ndarray, int, Optional[int]]:
    b = bc_type.lower()
    tip_mass_node = None

    def ux(node_id_1: int) -> int:
        return 2 * node_id_1 - 2

    def uy(node_id_1: int) -> int:
        return 2 * node_id_1 - 1

    if b == "simply":
        mid_row = _matlab_round(nely / 2.0) + 1
        left_mid = int(node_nrs_1[mid_row - 1, 0])
        right_mid = int(node_nrs_1[mid_row - 1, -1])
        fixed = np.array([ux(left_mid), uy(left_mid), ux(right_mid), uy(right_mid)], dtype=np.int64)

        mid_col = _matlab_round((nelx + 1) / 2.0)
        lc_node = int(node_nrs_1[mid_row - 1, mid_col - 1])
        lc_dof = uy(lc_node)
    elif b == "cantilever":
        left_nodes = node_nrs_1[:, 0].astype(np.int64)
        fixed = np.unique(np.concatenate((2 * left_nodes - 2, 2 * left_nodes - 1))).astype(np.int64)

        mid_row = _matlab_round((nely + 1) / 2.0)
        lc_node = int(node_nrs_1[mid_row - 1, -1])
        lc_dof = uy(lc_node)
        tip_mass_node = lc_node
    elif b == "fixedpinned":
        left_nodes = node_nrs_1[:, 0].astype(np.int64)
        fixed = np.unique(np.concatenate((2 * left_nodes - 2, 2 * left_nodes - 1))).astype(np.int64)

        mid_row = _matlab_round(nely / 2.0) + 1
        right_mid = int(node_nrs_1[mid_row - 1, -1])
        fixed = np.unique(np.concatenate((fixed, np.array([ux(right_mid), uy(right_mid)], dtype=np.int64))))

        mid_col = _matlab_round((nelx + 1) / 2.0)
        lc_node = int(node_nrs_1[mid_row - 1, mid_col - 1])
        lc_dof = uy(lc_node)
    else:
        raise ValueError(f"Unsupported bcType '{bc_type}'")

    return fixed, int(lc_dof), tip_mass_node


def _fixed_pinned_load_from_solid_mode(
    fixed: np.ndarray,
    node_nrs_1: np.ndarray,
    n_el: int,
    n_dof: int,
    Iar: np.ndarray,
    ke_lower: np.ndarray,
    me_lower: np.ndarray,
    E0: float,
    Emin: float,
    penal: float,
    rho0: float,
    rho_min: float,
    dMass: float,
    xMassCut: float,
) -> int:
    try:
        x_solid = np.ones((n_el,), dtype=np.float64)

        sK = Emin + (x_solid**penal) * (E0 - Emin)
        K = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(sK, ke_lower), (n_dof, n_dof))
        K = K + K.T - sparse.diags(K.diagonal())

        rhoe = rho_min + (rho0 - rho_min) * x_solid
        low = x_solid <= xMassCut
        rhoe[low] = rho_min + (rho0 - rho_min) * (x_solid[low] ** dMass)

        M = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(rhoe, me_lower), (n_dof, n_dof))
        M = M + M.T - sparse.diags(M.diagonal())

        free = np.setdiff1d(np.arange(n_dof, dtype=np.int64), fixed)
        Kff = K[free][:, free]
        Mff = M[free][:, free]
        vals, vecs = _solve_smallest_modes(Kff, Mff, 1)
        _ = vals

        U = np.zeros((n_dof,), dtype=np.float64)
        U[free] = vecs[:, 0]

        all_nodes = node_nrs_1.reshape(-1, order="F").astype(np.int64)
        v_dofs = 2 * all_nodes - 1
        free_v_dofs = v_dofs[~np.isin(v_dofs, fixed)]
        idx = int(np.argmax(np.abs(U[free_v_dofs])))
        return int(free_v_dofs[idx])
    except Exception:
        nely = node_nrs_1.shape[0] - 1
        mid_row = _matlab_round(nely / 2.0) + 1
        mid_col = _matlab_round(node_nrs_1.shape[1] / 2.0)
        return int(2 * int(node_nrs_1[mid_row - 1, mid_col - 1]) - 1)


def _first_n_omegas(
    xPhys: np.ndarray,
    free: np.ndarray,
    n_el: int,
    n_dof: int,
    Iar: np.ndarray,
    ke_lower: np.ndarray,
    me_lower: np.ndarray,
    E0: float,
    Emin: float,
    penal: float,
    rho0: float,
    rho_min: float,
    dMass: float,
    xMassCut: float,
    tip_mass_dofs: np.ndarray,
    tip_mass_val: float,
    n_modes: int,
) -> np.ndarray:
    out = np.full((n_modes,), np.nan, dtype=np.float64)

    Ee = Emin + (xPhys**penal) * (E0 - Emin)
    K = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(Ee, ke_lower), (n_dof, n_dof))
    K = K + K.T - sparse.diags(K.diagonal())

    rhoe = rho_min + (rho0 - rho_min) * xPhys
    low = xPhys <= xMassCut
    rhoe[low] = rho_min + (rho0 - rho_min) * (xPhys[low] ** dMass)
    M = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(rhoe, me_lower), (n_dof, n_dof))
    M = M + M.T - sparse.diags(M.diagonal())

    if tip_mass_val > 0 and tip_mass_dofs.size > 0:
        M = M.tolil()
        for d in tip_mass_dofs:
            M[d, d] += tip_mass_val
        M = M.tocsr()

    Kff = K[free][:, free]
    Mff = M[free][:, free]

    vals, _ = _solve_smallest_modes(Kff, Mff, n_modes)
    vals = vals[vals > 0]
    n_ok = min(vals.size, n_modes)
    if n_ok > 0:
        out[:n_ok] = np.sqrt(vals[:n_ok])
    return out


def _oc_update(
    x: np.ndarray,
    act: np.ndarray,
    dc: np.ndarray,
    dV0: np.ndarray,
    move: float,
    target_mean: float,
) -> tuple[np.ndarray, float]:
    x_new = x.copy()
    xT = x[act]
    xU = np.minimum(1.0, xT + move)
    xL = np.maximum(0.0, xT - move)

    denom = np.maximum(dV0[act], 1e-30)
    ocArg = -dc[act] / denom
    ocArg[~np.isfinite(ocArg)] = 1e-30
    ocArg = np.maximum(ocArg, 1e-30)
    ocP = xT * np.sqrt(ocArg)

    l1 = 0.0
    l2 = float(max(np.mean(ocP) / max(np.mean(x), np.finfo(float).eps), 1.0))

    # Robust bracket: increase l2 until mean(x_new) <= target.
    for _ in range(60):
        x_trial = np.clip(np.clip(ocP / l2, xL, xU), 0.0, 1.0)
        x_new[act] = x_trial
        if np.mean(x_new) <= target_mean + 1e-12:
            break
        l2 *= 2.0

    while (l2 - l1) / max(l2 + l1, np.finfo(float).eps) > 1e-4:
        lmid = 0.5 * (l1 + l2)
        x_new[act] = np.clip(np.clip(ocP / lmid, xL, xU), 0.0, 1.0)
        if np.mean(x_new) > target_mean:
            l1 = lmid
        else:
            l2 = lmid

    ch = float(np.max(np.abs(x_new[act] - xT)))
    return x_new, ch


def _create_live_plot(enabled: bool, window_title: str, figsize: tuple[float, float]) -> Optional[tuple]:
    if not enabled:
        return None
    try:
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=figsize)
        if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
            try:
                fig.canvas.manager.set_window_title(window_title)
            except Exception:
                pass
        return plt, fig, ax
    except Exception:
        return None


def _update_live_plot(plot_state: Optional[tuple], x_phys: np.ndarray, nely: int, nelx: int, title: str) -> None:
    if plot_state is None:
        return
    plt, fig, ax = plot_state
    ax.clear()
    ax.imshow(1 - x_phys.reshape((nely, nelx), order="F"), cmap="gray", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title)
    fig.canvas.draw_idle()
    plt.pause(0.001)


def top99neo_inertial_freq(
    nelx: int = 300,
    nely: int = 100,
    volfrac: float = 0.5,
    penal: float = 3.0,
    rmin: float = 8.75,
    ft: int = 3,
    ftBC: str = "N",
    eta: float = 0.5,
    beta: float = 1.0,
    move: float = 0.2,
    maxit: int = 100,
    stage1_maxit: Optional[int] = None,
    bcType: str = "simply",
    nHistModes: int = 0,
    output_dir: Optional[Path] = None,
    snapshot_every: int = 0,
    verbose: bool = True,
    plot_iterations: bool = False,
) -> YukselResult:
    if stage1_maxit is None:
        stage1_maxit = maxit

    nHistModes = max(0, int(nHistModes))
    stage2_tol = 1e-2
    if bcType.lower() == "fixedpinned":
        stage2_tol = 1e-3

    E0 = 1e7
    Emin = 1e-9 * E0
    nu = 0.3

    rho0 = 1.0
    rho_min = 1e-9 * rho0
    dMass = 6.0
    xMassCut = 0.1

    penalCnt = (1, 1, 25, 0.25)
    betaCnt = (1, 1, 25, 2)
    mode_bc = "reflect" if str(ftBC).upper() == "N" else "constant"

    beamL, beamH, tipMassFrac = _physical_setup(bcType)

    n_el = nelx * nely
    node_nrs_1 = np.arange(1, (nelx + 1) * (nely + 1) + 1, dtype=np.int64).reshape((nely + 1, nelx + 1), order="F")

    c_vec_1 = (2 * node_nrs_1[:-1, :-1] + 1).reshape(-1, order="F")
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1], dtype=np.int64)
    c_mat = (c_vec_1[:, None] + offsets[None, :] - 1).astype(np.int64)

    n_dof = 2 * (nely + 1) * (nelx + 1)

    sI: list[int] = []
    sII: list[int] = []
    for j in range(8):
        sI.extend(range(j, 8))
        sII.extend([j] * (8 - j))
    sI = np.array(sI, dtype=np.int64)
    sII = np.array(sII, dtype=np.int64)

    iK = c_mat[:, sI].T.reshape(-1, order="F")
    jK = c_mat[:, sII].T.reshape(-1, order="F")
    Iar = np.column_stack((np.maximum(iK, jK), np.minimum(iK, jK)))

    ke_lower, Ke0 = _build_ke_ke0(nu)
    me_lower, Me0 = _build_me0(beamL, beamH, nelx, nely)

    fixed, lc_dof, tip_mass_node = _bc_and_load(node_nrs_1, nely, nelx, bcType)
    if bcType.lower() == "fixedpinned":
        lc_dof = _fixed_pinned_load_from_solid_mode(
            fixed,
            node_nrs_1,
            n_el,
            n_dof,
            Iar,
            ke_lower,
            me_lower,
            E0,
            Emin,
            penal,
            rho0,
            rho_min,
            dMass,
            xMassCut,
        )

    F_point = np.zeros((n_dof,), dtype=np.float64)
    F_point[lc_dof] = -1.0

    tip_mass_dofs = np.array([], dtype=np.int64)
    tip_mass_val = 0.0
    if tipMassFrac > 0 and tip_mass_node is not None:
        permitted_mass = volfrac * beamL * beamH * rho0
        tip_mass_val = tipMassFrac * permitted_mass
        tip_mass_dofs = np.array([2 * tip_mass_node - 2, 2 * tip_mass_node - 1], dtype=np.int64)

    free = np.setdiff1d(np.arange(n_dof, dtype=np.int64), fixed)
    act = np.arange(n_el, dtype=np.int64)

    prj = lambda v, eta_, beta_: (np.tanh(beta_ * eta_) + np.tanh(beta_ * (v - eta_))) / (
        np.tanh(beta_ * eta_) + np.tanh(beta_ * (1 - eta_))
    )
    deta = lambda v, eta_, beta_: (
        -beta_
        * (1.0 / np.sinh(beta_))
        * (1.0 / np.cosh(beta_ * (v - eta_)) ** 2)
        * np.sinh(v * beta_)
        * np.sinh((1 - v) * beta_)
    )
    dprj = lambda v, eta_, beta_: beta_ * (1 - np.tanh(beta_ * (v - eta_)) ** 2) / (
        np.tanh(beta_ * eta_) + np.tanh(beta_ * (1 - eta_))
    )

    def cnt(v: float, spec: tuple[float, float, int, float], loop: int) -> float:
        return v + (loop >= spec[0]) * (v < spec[1]) * ((loop % spec[2]) == 0) * spec[3]

    rg = np.arange(-int(np.ceil(rmin)) + 1, int(np.ceil(rmin)), dtype=np.float64)
    dy, dx = np.meshgrid(rg, rg, indexing="xy")
    h = np.maximum(0.0, rmin - np.sqrt(dx**2 + dy**2))
    Hs = ndimage.convolve(np.ones((nely, nelx), dtype=np.float64), h, mode=mode_bc)

    x = np.zeros((n_el,), dtype=np.float64)
    dV = np.zeros((n_el,), dtype=np.float64)
    dV[act] = 1.0 / (n_el * volfrac)
    x[act] = volfrac

    info = {
        "stage1": {"c": [], "v": [], "ch": [], "xHist": [], "omegaHist": [], "loadDof": int(lc_dof)},
        "stage2": {"c": [], "v": [], "ch": [], "xHist": [], "omegaHist": []},
    }

    snapshots_dir = None
    if output_dir is not None and snapshot_every > 0:
        snapshots_dir = Path(output_dir) / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
    live_plot = _create_live_plot(plot_iterations, f"Yuksel {bcType} iteration", (10, 3))
    if plot_iterations and live_plot is None and verbose:
        print("WARN could not initialize interactive plotting backend; continuing without live iteration plot")

    xPhys = x.copy()
    U = np.zeros((n_dof,), dtype=np.float64)

    tolX1 = 1e-2
    for loop in range(1, stage1_maxit + 1):
        if ft == 1:
            xPhys[act] = x[act]
        else:
            xTilde = ndimage.convolve(x.reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs
            xPhys[act] = xTilde.reshape(-1, order="F")[act]

        dHs = Hs.copy()
        if ft > 1:
            xTilde = xPhys.reshape((nely, nelx), order="F")
            if ft == 3:
                f_eta = np.mean(prj(xPhys, eta, beta)) - np.mean(xPhys)
                while abs(f_eta) > 1e-6:
                    eta -= f_eta / max(np.mean(deta(xPhys, eta, beta)), np.finfo(float).eps)
                    f_eta = np.mean(prj(xPhys, eta, beta)) - np.mean(xPhys)
            dHs = Hs / dprj(xTilde, eta, beta)
            xPhys = prj(xPhys, eta, beta)

        Ee = Emin + (xPhys**penal) * (E0 - Emin)
        dsK = np.zeros((n_el,), dtype=np.float64)
        dsK[act] = -penal * (E0 - Emin) * (xPhys[act] ** (penal - 1))

        K = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(Ee, ke_lower), (n_dof, n_dof))
        K = K + K.T - sparse.diags(K.diagonal())

        U[:] = 0.0
        U[free] = spla.spsolve(K[free][:, free], F_point[free])

        dc = dsK * np.sum((U[c_mat] @ Ke0) * U[c_mat], axis=1)

        if ft == 1:
            xMat = np.maximum(1e-3, x).reshape((nely, nelx), order="F")
            dcF = ndimage.convolve((x * dc).reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs / xMat
            dV0 = ndimage.convolve((x * dV).reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs / xMat
        else:
            dcF = ndimage.convolve((dc.reshape((nely, nelx), order="F") / dHs), h, mode=mode_bc)
            dV0 = ndimage.convolve((dV.reshape((nely, nelx), order="F") / dHs), h, mode=mode_bc)

        x, ch = _oc_update(
            x,
            act,
            dcF.reshape(-1, order="F"),
            dV0.reshape(-1, order="F"),
            move,
            volfrac,
        )

        penal = cnt(penal, penalCnt, loop)
        beta = cnt(beta, betaCnt, loop)

        cVal = float(F_point @ U)
        info["stage1"]["c"].append(cVal)
        info["stage1"]["v"].append(float(np.mean(xPhys)))
        info["stage1"]["ch"].append(ch)
        if nHistModes > 0:
            info["stage1"]["xHist"].append(xPhys.copy())

        if verbose:
            print(f"S1 It.:{loop:5d} C:{cVal:10.4e} V:{np.mean(xPhys):7.3f} ch:{ch:0.2e} penal:{penal:5.2f} beta:{beta:5.1f} eta:{eta:6.3f}")
        _update_live_plot(
            live_plot,
            xPhys,
            nely,
            nelx,
            f"S1 It.:{loop:5d} C:{cVal:10.4e} V:{np.mean(xPhys):7.3f}",
        )

        if snapshots_dir is not None and snapshot_every > 0 and (loop % snapshot_every == 0):
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.imshow(1 - xPhys.reshape((nely, nelx), order="F"), cmap="gray", origin="lower", interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"stage1 it {loop}")
                fig.tight_layout()
                fig.savefig(snapshots_dir / f"stage1_{loop:04d}.png", dpi=120)
                plt.close(fig)
            except Exception:
                pass

        if loop > 1 and ch < tolX1:
            break

    info["stage1"]["xFinal"] = xPhys.copy()
    info["stage1"]["UFinal"] = U.copy()
    omega1_stage1 = _first_n_omegas(
        xPhys,
        free,
        n_el,
        n_dof,
        Iar,
        ke_lower,
        me_lower,
        E0,
        Emin,
        penal,
        rho0,
        rho_min,
        dMass,
        xMassCut,
        tip_mass_dofs,
        tip_mass_val,
        1,
    )
    info["stage1"]["omega1"] = float(omega1_stage1[0])

    U_est = U.copy()
    x_stage2 = xPhys.copy()
    xPhys2 = x_stage2.copy()
    U2 = U.copy()

    for loop in range(1, maxit + 1):
        if ft == 1:
            xPhys2[act] = x_stage2[act]
        else:
            xTilde = ndimage.convolve(x_stage2.reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs
            xPhys2[act] = xTilde.reshape(-1, order="F")[act]

        dHs = Hs.copy()
        if ft > 1:
            xTilde = xPhys2.reshape((nely, nelx), order="F")
            if ft == 3:
                f_eta = np.mean(prj(xPhys2, eta, beta)) - np.mean(xPhys2)
                while abs(f_eta) > 1e-6:
                    eta -= f_eta / max(np.mean(deta(xPhys2, eta, beta)), np.finfo(float).eps)
                    f_eta = np.mean(prj(xPhys2, eta, beta)) - np.mean(xPhys2)
            dHs = Hs / dprj(xTilde, eta, beta)
            xPhys2 = prj(xPhys2, eta, beta)

        Ee = Emin + (xPhys2**penal) * (E0 - Emin)
        dsK = np.zeros((n_el,), dtype=np.float64)
        dsK[act] = -penal * (E0 - Emin) * (xPhys2[act] ** (penal - 1))
        K = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(Ee, ke_lower), (n_dof, n_dof))
        K = K + K.T - sparse.diags(K.diagonal())

        rhoe = rho_min + (rho0 - rho_min) * xPhys2
        low = xPhys2 <= xMassCut
        rhoe[low] = rho_min + (rho0 - rho_min) * (xPhys2[low] ** dMass)
        M = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(rhoe, me_lower), (n_dof, n_dof))
        M = M + M.T - sparse.diags(M.diagonal())
        if tip_mass_val > 0 and tip_mass_dofs.size > 0:
            M = M.tolil()
            for d in tip_mass_dofs:
                M[d, d] += tip_mass_val
            M = M.tocsr()

        uhat = U_est.copy()
        nrm = np.linalg.norm(uhat[free])
        if nrm == 0:
            nrm = 1.0
        uhat /= nrm

        F = M @ uhat
        F[fixed] = 0.0

        U2[:] = 0.0
        U2[free] = spla.spsolve(K[free][:, free], F[free])

        uhat_new = U2.copy()
        nrm_new = np.linalg.norm(uhat_new[free])
        if nrm_new == 0:
            nrm_new = 1.0
        uhat_new /= nrm_new
        sgn = np.sign(float(uhat[free] @ uhat_new[free]))
        if sgn == 0:
            sgn = 1.0
        uhat_new *= sgn
        du = np.linalg.norm(uhat_new[free] - uhat[free]) / max(1.0, np.linalg.norm(uhat[free]))
        U_est = uhat_new.copy()

        dc = dsK * np.sum((U2[c_mat] @ Ke0) * U2[c_mat], axis=1)

        if ft == 1:
            xMat = np.maximum(1e-3, x_stage2).reshape((nely, nelx), order="F")
            dcF = ndimage.convolve((x_stage2 * dc).reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs / xMat
            dV0 = ndimage.convolve((x_stage2 * dV).reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs / xMat
        else:
            dcF = ndimage.convolve((dc.reshape((nely, nelx), order="F") / dHs), h, mode=mode_bc)
            dV0 = ndimage.convolve((dV.reshape((nely, nelx), order="F") / dHs), h, mode=mode_bc)

        x_stage2, ch = _oc_update(
            x_stage2,
            act,
            dcF.reshape(-1, order="F"),
            dV0.reshape(-1, order="F"),
            move,
            volfrac,
        )

        penal = cnt(penal, penalCnt, loop)
        beta = cnt(beta, betaCnt, loop)

        cVal = float(F @ U2)
        info["stage2"]["c"].append(cVal)
        info["stage2"]["v"].append(float(np.mean(xPhys2)))
        info["stage2"]["ch"].append(ch)
        if nHistModes > 0:
            info["stage2"]["xHist"].append(xPhys2.copy())

        if verbose:
            print(
                f"S2 It.:{loop:5d} C:{cVal:10.4e} V:{np.mean(xPhys2):7.3f} ch:{ch:0.2e} du:{du:0.2e} |F|:{np.linalg.norm(F[free]):9.2e} "
                f"penal:{penal:5.2f} beta:{beta:5.1f} eta:{eta:6.3f}"
            )
        _update_live_plot(
            live_plot,
            xPhys2,
            nely,
            nelx,
            f"S2 It.:{loop:5d} C:{cVal:10.4e} V:{np.mean(xPhys2):7.3f}",
        )

        if snapshots_dir is not None and snapshot_every > 0 and (loop % snapshot_every == 0):
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.imshow(1 - xPhys2.reshape((nely, nelx), order="F"), cmap="gray", origin="lower", interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"stage2 it {loop}")
                fig.tight_layout()
                fig.savefig(snapshots_dir / f"stage2_{loop:04d}.png", dpi=120)
                plt.close(fig)
            except Exception:
                pass

        if loop > 1 and ch < stage2_tol:
            break

    info["stage2"]["xFinal"] = xPhys2.copy()
    info["stage2"]["UFinal"] = U2.copy()
    omega1_stage2 = _first_n_omegas(
        xPhys2,
        free,
        n_el,
        n_dof,
        Iar,
        ke_lower,
        me_lower,
        E0,
        Emin,
        penal,
        rho0,
        rho_min,
        dMass,
        xMassCut,
        tip_mass_dofs,
        tip_mass_val,
        1,
    )
    info["stage2"]["omega1"] = float(omega1_stage2[0])

    if nHistModes > 0:
        stage1_hist = np.asarray(info["stage1"]["xHist"], dtype=np.float64)
        stage2_hist = np.asarray(info["stage2"]["xHist"], dtype=np.float64)

        om1 = []
        for xh in stage1_hist:
            om1.append(
                _first_n_omegas(
                    xh,
                    free,
                    n_el,
                    n_dof,
                    Iar,
                    ke_lower,
                    me_lower,
                    E0,
                    Emin,
                    penal,
                    rho0,
                    rho_min,
                    dMass,
                    xMassCut,
                    tip_mass_dofs,
                    tip_mass_val,
                    nHistModes,
                )
            )
        om2 = []
        for xh in stage2_hist:
            om2.append(
                _first_n_omegas(
                    xh,
                    free,
                    n_el,
                    n_dof,
                    Iar,
                    ke_lower,
                    me_lower,
                    E0,
                    Emin,
                    penal,
                    rho0,
                    rho_min,
                    dMass,
                    xMassCut,
                    tip_mass_dofs,
                    tip_mass_val,
                    nHistModes,
                )
            )
        info["stage1"]["omegaHist"] = np.asarray(om1, dtype=np.float64)
        info["stage2"]["omegaHist"] = np.asarray(om2, dtype=np.float64)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / "results_inertial.npz",
            xPhys_stage1=info["stage1"]["xFinal"],
            xPhys_stage2=xPhys2,
            U_stage2=U2,
            omega1_stage1=info["stage1"]["omega1"],
            omega1_stage2=info["stage2"]["omega1"],
            stage1_c=np.asarray(info["stage1"]["c"], dtype=np.float64),
            stage2_c=np.asarray(info["stage2"]["c"], dtype=np.float64),
            stage1_v=np.asarray(info["stage1"]["v"], dtype=np.float64),
            stage2_v=np.asarray(info["stage2"]["v"], dtype=np.float64),
            stage1_ch=np.asarray(info["stage1"]["ch"], dtype=np.float64),
            stage2_ch=np.asarray(info["stage2"]["ch"], dtype=np.float64),
            nelx=nelx,
            nely=nely,
        )

    return YukselResult(xPhys_stage2=xPhys2, U_stage2=U2, info=info)


def top99neo_dynamic_freq(
    nelx: int = 320,
    nely: int = 40,
    volfrac: float = 0.5,
    penal: float = 3.0,
    rmin: float = 2.5,
    ft: int = 1,
    ftBC: str = "N",
    move: float = 0.01,
    maxit: int = 200,
    bcType: str = "simply",
    output_dir: Optional[Path] = None,
    snapshot_every: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    if ft != 1:
        raise ValueError("top99neo_dynamic_freq supports ft=1 only")

    E0 = 1e7
    Emin = 1e-9 * E0
    nu = 0.3
    rho0 = 1.0
    rho_min = 1e-9 * rho0
    dMass = 6.0
    xMassCut = 0.1
    repTol = 0.04

    n_el = nelx * nely
    node_nrs_1 = np.arange(1, (nelx + 1) * (nely + 1) + 1, dtype=np.int64).reshape((nely + 1, nelx + 1), order="F")

    c_vec_1 = (2 * node_nrs_1[:-1, :-1] + 1).reshape(-1, order="F")
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1], dtype=np.int64)
    c_mat = (c_vec_1[:, None] + offsets[None, :] - 1).astype(np.int64)

    n_dof = 2 * (nely + 1) * (nelx + 1)

    sI: list[int] = []
    sII: list[int] = []
    for j in range(8):
        sI.extend(range(j, 8))
        sII.extend([j] * (8 - j))
    sI = np.array(sI, dtype=np.int64)
    sII = np.array(sII, dtype=np.int64)

    iK = c_mat[:, sI].T.reshape(-1, order="F")
    jK = c_mat[:, sII].T.reshape(-1, order="F")
    Iar = np.column_stack((np.maximum(iK, jK), np.minimum(iK, jK)))

    ke_lower, Ke0 = _build_ke_ke0(nu)
    me_lower, Me0 = _build_me0(*_physical_setup(bcType)[:2], nelx, nely)

    fixed, _, tip_mass_node = _bc_and_load(node_nrs_1, nely, nelx, bcType)
    free = np.setdiff1d(np.arange(n_dof, dtype=np.int64), fixed)

    beamL, beamH, tipMassFrac = _physical_setup(bcType)
    tip_mass_dofs = np.array([], dtype=np.int64)
    tip_mass_val = 0.0
    if tipMassFrac > 0 and tip_mass_node is not None:
        permitted_mass = volfrac * beamL * beamH * rho0
        tip_mass_val = tipMassFrac * permitted_mass
        tip_mass_dofs = np.array([2 * tip_mass_node - 2, 2 * tip_mass_node - 1], dtype=np.int64)

    mode_bc = "reflect" if str(ftBC).upper() == "N" else "constant"
    rg = np.arange(-int(np.ceil(rmin)) + 1, int(np.ceil(rmin)), dtype=np.float64)
    dy, dx = np.meshgrid(rg, rg, indexing="xy")
    h = np.maximum(0.0, rmin - np.sqrt(dx**2 + dy**2))
    Hs = ndimage.convolve(np.ones((nely, nelx), dtype=np.float64), h, mode=mode_bc)

    x = volfrac * np.ones((n_el,), dtype=np.float64)
    xPhys = x.copy()
    dV = np.ones((n_el,), dtype=np.float64) / (n_el * volfrac)

    info = {
        "omegaHist": np.full((maxit, 3), np.nan, dtype=np.float64),
        "chHist": np.full((maxit,), np.nan, dtype=np.float64),
        "repActive": np.zeros((maxit,), dtype=bool),
    }

    snapshots_dir = None
    if output_dir is not None and snapshot_every > 0:
        snapshots_dir = Path(output_dir) / "snapshots_dynamic"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    for it in range(1, maxit + 1):
        xPhys = x.copy()

        Ee = Emin + (xPhys**penal) * (E0 - Emin)
        K = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(Ee, ke_lower), (n_dof, n_dof))
        K = K + K.T - sparse.diags(K.diagonal())

        rhoe = rho_min + (rho0 - rho_min) * xPhys
        low = xPhys <= xMassCut
        rhoe[low] = rho_min + (rho0 - rho_min) * (xPhys[low] ** dMass)
        M = _local_assemble(Iar[:, 0], Iar[:, 1], np.kron(rhoe, me_lower), (n_dof, n_dof))
        M = M + M.T - sparse.diags(M.diagonal())
        if tip_mass_val > 0 and tip_mass_dofs.size > 0:
            M = M.tolil()
            for d in tip_mass_dofs:
                M[d, d] += tip_mass_val
            M = M.tocsr()

        Kff = K[free][:, free]
        Mff = M[free][:, free]

        lam, V = _solve_smallest_modes(Kff, Mff, 3)
        lam = np.maximum(lam, np.finfo(float).eps)
        omega = np.sqrt(lam)
        info["omegaHist"][it - 1, : omega.size] = omega

        drho = (rho0 - rho_min) * np.ones((n_el,), dtype=np.float64)
        drho[low] = dMass * (rho0 - rho_min) * (xPhys[low] ** (dMass - 1))

        dlam = np.zeros((n_el, 2), dtype=np.float64)
        for j in range(2):
            vj = V[:, j].copy()
            vj /= np.sqrt(max(np.finfo(float).eps, float(vj.T @ (Mff @ vj))))

            phi = np.zeros((n_dof,), dtype=np.float64)
            phi[free] = vj
            pe = phi[c_mat]

            dlam[:, j] = (
                penal * (E0 - Emin) * (xPhys ** (penal - 1)) * np.sum((pe @ Ke0) * pe, axis=1)
                - lam[j] * drho * np.sum((pe @ Me0) * pe, axis=1)
            )

        if (omega[1] - omega[0]) / max(omega[0], np.finfo(float).eps) < repTol:
            dlam_obj = np.maximum(dlam[:, 0], dlam[:, 1])
            info["repActive"][it - 1] = True
        else:
            dlam_obj = dlam[:, 0]

        dlam_obj = dlam_obj - np.min(dlam_obj) + 1e-12
        dc = -dlam_obj

        xMat = np.maximum(1e-3, x).reshape((nely, nelx), order="F")
        dcF = ndimage.convolve((x * dc).reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs / xMat
        dV0 = ndimage.convolve((x * dV).reshape((nely, nelx), order="F"), h, mode=mode_bc) / Hs / xMat

        x_old = x.copy()
        x, ch = _oc_update(
            x,
            np.arange(n_el, dtype=np.int64),
            dcF.reshape(-1, order="F"),
            dV0.reshape(-1, order="F"),
            move,
            volfrac,
        )

        info["chHist"][it - 1] = ch
        if verbose:
            print(
                f"Dyn It.:{it:4d} w1:{omega[0]:8.3f} w2:{omega[1]:8.3f} w3:{omega[2]:8.3f} ch:{ch:0.2e} rep:{int(info['repActive'][it-1])}"
            )

        if snapshots_dir is not None and snapshot_every > 0 and (it % snapshot_every == 0):
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.imshow(1 - xPhys.reshape((nely, nelx), order="F"), cmap="gray", origin="lower", interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"dynamic it {it}")
                fig.tight_layout()
                fig.savefig(snapshots_dir / f"dynamic_{it:04d}.png", dpi=120)
                plt.close(fig)
            except Exception:
                pass

        _ = x_old

    info["xFinal"] = xPhys.copy()
    info["omega1"] = float(info["omegaHist"][maxit - 1, 0])
    info["tipMassVal"] = tip_mass_val

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_dir / "results_dynamic.npz",
            xPhys=xPhys,
            omegaHist=info["omegaHist"],
            chHist=info["chHist"],
            repActive=info["repActive"],
            omega1=info["omega1"],
            nelx=nelx,
            nely=nely,
        )

    return xPhys, info
