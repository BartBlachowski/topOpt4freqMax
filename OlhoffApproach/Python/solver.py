from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as spla

from .mma import mmasub


@dataclass
class OlhoffConfig:
    L: float = 8.0
    H: float = 1.0
    nelx: int = 240
    nely: int = 30

    volfrac: float = 0.5
    penal: float = 3.0
    rmin: float = 2.0*L/nelx #Optional[float] = None
    maxiter: int = 300

    support_type: str = "CC"
    J: int = 3

    E0: float = 1e7
    Emin: Optional[float] = None
    rho0: float = 1.0
    rho_min: float = 1e-6
    nu: float = 0.3
    thickness: float = 1.0

    eta: float = 0.5
    beta_max: float = 64.0
    beta_schedule: tuple[float, ...] = (1, 2, 4, 8, 16, 32, 64)
    beta_interval: int = 40
    beta_start_idx: int = 1
    beta_safe_iters: int = 5
    move_safe: float = 0.05
    gray_tol: float = 0.10
    move_reduce: float = 0.02
    n_polish: int = 40
    gray_penalty_base: float = 0.5

    lambda_ref: float = 2e4
    Eb0: float = 1.0
    Eb_min: float = 0.0
    Eb_max: float = 50.0

    move: float = 0.2
    move_hist_len: int = 10

    seed: Optional[int] = None

    def finalize(self) -> None:
        if self.rmin is None:
            self.rmin = 2 * self.L / self.nelx
        if self.Emin is None:
            self.Emin = max(1e-6 * self.E0, 1e-3)


@dataclass
class OlhoffResult:
    omega_best: float
    xPhys_best: np.ndarray
    objective_history: np.ndarray
    volume_history: np.ndarray
    grayness_history: np.ndarray
    freq_iter_omega: np.ndarray  # shape (n_iters, 3): ω₁, ω₂, ω₃ at each iteration
    diagnostics: dict
    config: OlhoffConfig


def q4_rect_ke_m_plane_stress(E: float, nu: float, rho: float, t: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    C = E / (1 - nu**2) * np.array(
        [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]], dtype=np.float64
    )

    gp = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)], dtype=np.float64)
    xiN = np.array([-1, 1, 1, -1], dtype=np.float64)
    etaN = np.array([-1, -1, 1, 1], dtype=np.float64)

    detJ = (dx * dy) / 4.0
    invJ = np.array([[2 / dx, 0], [0, 2 / dy]], dtype=np.float64)

    Ke = np.zeros((8, 8), dtype=np.float64)
    Me = np.zeros((8, 8), dtype=np.float64)

    for xi in gp:
        for eta in gp:
            N = 0.25 * (1 + xi * xiN) * (1 + eta * etaN)
            dN_dxi = 0.25 * xiN * (1 + eta * etaN)
            dN_deta = 0.25 * etaN * (1 + xi * xiN)

            grads = invJ @ np.vstack((dN_dxi, dN_deta))
            dN_dx, dN_dy = grads[0], grads[1]

            B = np.zeros((3, 8), dtype=np.float64)
            Nmat = np.zeros((2, 8), dtype=np.float64)
            for a in range(4):
                B[0, 2 * a] = dN_dx[a]
                B[1, 2 * a + 1] = dN_dy[a]
                B[2, 2 * a] = dN_dy[a]
                B[2, 2 * a + 1] = dN_dx[a]
                Nmat[0, 2 * a] = N[a]
                Nmat[1, 2 * a + 1] = N[a]

            Ke += (B.T @ C @ B) * (t * detJ)
            Me += (Nmat.T @ Nmat) * (rho * t * detJ)

    return Ke, Me


def build_supports(support_type: str, node_nrs: np.ndarray) -> np.ndarray:
    nely = node_nrs.shape[0] - 1
    left_nodes = node_nrs[:, 0]
    right_nodes = node_nrs[:, -1]

    left_bot = node_nrs[0, 0]
    right_bot = node_nrs[0, -1]
    mid_idx = int(round(nely / 2.0))
    left_mid = node_nrs[mid_idx, 0]
    right_mid = node_nrs[mid_idx, -1]

    def ux(nodes: np.ndarray) -> np.ndarray:
        return 2 * nodes

    def uy(nodes: np.ndarray) -> np.ndarray:
        return 2 * nodes + 1

    s = support_type.upper()
    if s in {"CF", "CLAMPED-FREE"}:
        fixed = np.concatenate((ux(left_nodes), uy(left_nodes)))
    elif s in {"CC", "CLAMPED-CLAMPED"}:
        fixed = np.concatenate((ux(left_nodes), uy(left_nodes), ux(right_nodes), uy(right_nodes)))
    elif s in {"CH", "CS", "CLAMPED-SIMPLY"}:
        fixed = np.concatenate((ux(left_nodes), uy(left_nodes), np.array([ux(np.array([right_mid]))[0], uy(np.array([right_mid]))[0]])))
    elif s in {"HH", "SS", "SIMPLY-SUPPORTED"}:
        fixed = np.array([ux(np.array([left_mid]))[0], uy(np.array([left_mid]))[0], ux(np.array([right_mid]))[0], uy(np.array([right_mid]))[0]])
    elif s == "SS_CORNER":
        fixed = np.array([ux(np.array([left_bot]))[0], uy(np.array([left_bot]))[0], uy(np.array([right_bot]))[0]])
    else:
        raise ValueError(f"Unknown support_type '{support_type}'")

    return np.unique(fixed.astype(np.int64))


def heaviside_projection(x_tilde: np.ndarray, beta: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    denom = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    x_phys = (np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))) / denom
    dH = (beta * (1 - np.tanh(beta * (x_tilde - eta)) ** 2)) / denom
    return x_phys, dH


def _assemble_global(
    iK: np.ndarray,
    jK: np.ndarray,
    Ke_l: np.ndarray,
    Me_l: np.ndarray,
    Ee: np.ndarray,
    re: np.ndarray,
    n_dof: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    sK = np.kron(Ee, Ke_l)
    sM = np.kron(re, Me_l)

    K = sparse.coo_matrix((sK, (iK, jK)), shape=(n_dof, n_dof)).tocsr()
    M = sparse.coo_matrix((sM, (iK, jK)), shape=(n_dof, n_dof)).tocsr()

    K = K + K.T - sparse.diags(K.diagonal())
    M = M + M.T - sparse.diags(M.diagonal())
    return K, M


def _solve_modes(Kf: sparse.spmatrix, Mf: sparse.spmatrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = Kf.shape[0]
    if n <= 2:
        raise RuntimeError("Not enough free DOFs")
    k = max(1, min(k, n - 2))

    eigvals, eigvecs = spla.eigsh(Kf, k=k, M=Mf, sigma=0.0, which="LM", tol=1e-8, maxiter=4000)
    order = np.argsort(np.real(eigvals))
    eigvals = np.real(eigvals[order])
    eigvecs = np.real(eigvecs[:, order])

    res = np.zeros((k,), dtype=np.float64)
    for j in range(k):
        v = eigvecs[:, j]
        Kv = Kf @ v
        Mv = Mf @ v
        res[j] = np.linalg.norm(Kv - eigvals[j] * Mv) / max(np.linalg.norm(Kv), np.finfo(float).eps)

    return eigvals, eigvecs, res


def _eval_modes(
    x_phys: np.ndarray,
    num_modes: int,
    cfg: OlhoffConfig,
    Ke_l: np.ndarray,
    Me_l: np.ndarray,
    iK: np.ndarray,
    jK: np.ndarray,
    n_dof: int,
    free: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ee = cfg.Emin + (x_phys**cfg.penal) * (cfg.E0 - cfg.Emin)
    re = cfg.rho_min + x_phys * (cfg.rho0 - cfg.rho_min)
    K, M = _assemble_global(iK, jK, Ke_l, Me_l, Ee, re, n_dof)

    Kf = K[free][:, free]
    Mf = M[free][:, free]
    lam, _, _ = _solve_modes(Kf, Mf, num_modes)

    omega = np.sqrt(np.maximum(lam, 0.0))
    freq = omega / (2 * np.pi)
    return lam, omega, freq


def run_optimization(
    cfg: OlhoffConfig,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    do_diagnostic: bool = True,
    diag_modes: int = 5,
    plot_binary: bool = False,
    snapshot_every: int = 0,
) -> OlhoffResult:
    cfg.finalize()
    if cfg.seed is not None:
        np.random.seed(cfg.seed)

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

    def fwd(xm: np.ndarray) -> np.ndarray:
        return ndimage.convolve(xm, h, mode="reflect") / Hs

    def bwd(gm: np.ndarray) -> np.ndarray:
        return ndimage.convolve(gm / Hs, h, mode="reflect")

    c_vec = (2 * node_nrs_1[:-1, :-1] + 1).reshape(-1, order="F")
    c_mat = np.column_stack(
        [
            c_vec,
            c_vec + 1,
            c_vec + 2 * nely + 2,
            c_vec + 2 * nely + 3,
            c_vec + 2 * nely,
            c_vec + 2 * nely + 1,
            c_vec - 2,
            c_vec - 1,
        ]
    ).astype(np.int64) - 1

    sI: list[int] = []
    sII: list[int] = []
    for j in range(8):
        sI.extend(range(j, 8))
        sII.extend([j] * (8 - j))
    sI = np.array(sI, dtype=np.int64)
    sII = np.array(sII, dtype=np.int64)

    iK = c_mat[:, sI].T.reshape(-1, order="F")
    jK = c_mat[:, sII].T.reshape(-1, order="F")

    Ke_l = Ke[sI, sII]
    Me_l = Me[sI, sII]

    beta_list = np.array(cfg.beta_schedule, dtype=np.float64)
    beta_idx = int(max(1, cfg.beta_start_idx))
    beta_prev = -np.inf

    n = n_el + 1
    m = cfg.J + 2

    xmin = np.concatenate((1e-3 * np.ones((n_el,)), np.array([cfg.Eb_min], dtype=np.float64)))
    xmax = np.concatenate((np.ones((n_el,)), np.array([cfg.Eb_max], dtype=np.float64)))

    xval = np.concatenate((cfg.volfrac * np.ones((n_el,)), np.array([cfg.Eb0], dtype=np.float64)))
    xold1 = xval.copy()
    xold2 = xval.copy()
    low = xmin.copy()
    upp = xmax.copy()

    mma_a0 = 1.0
    mma_a = np.zeros((m, 1), dtype=np.float64)
    mma_c = np.concatenate((100 * np.ones((cfg.J,)), np.array([10, 10], dtype=np.float64))).reshape(-1, 1)
    mma_d = 1e-3 * np.ones((m, 1), dtype=np.float64)

    omega_best = -np.inf
    xPhys_best = xval[:n_el].copy()

    objective_hist: list[float] = []
    volume_hist: list[float] = []
    gray_hist: list[float] = []
    omega_iter_hist: list[np.ndarray] = []  # ω₁..ω₃ per iteration
    iteration_log: list[list[float]] = []

    diagnostics: dict = {}

    if do_diagnostic:
        x0 = cfg.volfrac * np.ones((n_el,), dtype=np.float64)
        xT0 = fwd(x0.reshape((nely, nelx), order="F"))
        xPhys0, _ = heaviside_projection(xT0, 1.0, cfg.eta)
        lam0, omega0, freq0 = _eval_modes(
            xPhys0.reshape(-1, order="F"),
            diag_modes,
            cfg,
            Ke_l,
            Me_l,
            iK,
            jK,
            n_dof,
            free,
        )
        diagnostics["initial"] = {"lam": lam0, "omega": omega0, "freq": freq0}

    move = cfg.move
    safe_counter = 0
    reduce_counter = 0
    prev_V_low: Optional[np.ndarray] = None

    snapshots_dir = None
    if output_dir is not None and snapshot_every > 0:
        snapshots_dir = Path(output_dir) / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    x_prev: Optional[np.ndarray] = None
    omega_prev: Optional[float] = None

    for it in range(1, cfg.maxiter + 1):
        beta_target = beta_list[min(beta_idx - 1, beta_list.size - 1)]
        beta = min(cfg.beta_max, beta_target)

        x = xval[:n_el]
        Eb = xval[-1]

        xT = fwd(x.reshape((nely, nelx), order="F"))
        xPhysMat, dH = heaviside_projection(xT, beta, cfg.eta)
        xPhys = xPhysMat.reshape(-1, order="F")

        Ee = cfg.Emin + (xPhys**cfg.penal) * (cfg.E0 - cfg.Emin)
        re = cfg.rho_min + xPhys * (cfg.rho0 - cfg.rho_min)

        K, M = _assemble_global(iK, jK, Ke_l, Me_l, Ee, re, n_dof)
        Kf = K[free][:, free]
        Mf = M[free][:, free]

        try:
            lam_sorted, V_low, resvec = _solve_modes(Kf, Mf, cfg.J)
        except Exception:
            reduce_counter = 5
            if verbose:
                print(f"WARN eigsh failed at iteration {it}; reducing move for 5 iterations")
            continue

        resmax = np.max(resvec)
        if resmax > 1e-3:
            try:
                if prev_V_low is not None:
                    v0 = prev_V_low[:, 0]
                else:
                    v0 = None
                eigvals, eigvecs = spla.eigsh(
                    Kf,
                    k=min(cfg.J, Kf.shape[0] - 2),
                    M=Mf,
                    sigma=0.0,
                    which="LM",
                    tol=1e-10,
                    maxiter=12000,
                    v0=v0,
                )
                ord_idx = np.argsort(np.real(eigvals))
                lam_sorted = np.real(eigvals[ord_idx])
                V_low = np.real(eigvecs[:, ord_idx])
                resvec = np.zeros((lam_sorted.size,), dtype=np.float64)
                for jj in range(lam_sorted.size):
                    vv = V_low[:, jj]
                    Kv = Kf @ vv
                    Mv = Mf @ vv
                    resvec[jj] = np.linalg.norm(Kv - lam_sorted[jj] * Mv) / max(np.linalg.norm(Kv), np.finfo(float).eps)
                resmax = np.max(resvec)
            except Exception:
                reduce_counter = 5
                if verbose:
                    print(f"WARN eigsh retry failed at iteration {it}; reducing move for 5 iterations")
                continue

        if resmax > 1e-3:
            reduce_counter = 5
            if verbose:
                print(f"WARN eig residual too large ({resmax:.2e}) at iteration {it}; reducing move")
            continue

        prev_V_low = V_low.copy()
        omega_cur = float(np.sqrt(max(lam_sorted[0], 0.0)))
        row3 = np.full(3, np.nan)
        for _j in range(min(3, lam_sorted.size)):
            row3[_j] = np.sqrt(max(lam_sorted[_j], 0.0))
        omega_iter_hist.append(row3)

        if beta != beta_prev:
            Eb_feas = float(np.min(lam_sorted[: cfg.J])) / cfg.lambda_ref
            Eb = min(Eb, Eb_feas - 1e-8)
            Eb = min(xmax[-1], max(xmin[-1], Eb))
            xval[-1] = Eb
            xold1 = xval.copy()
            xold2 = xval.copy()
            low = xmin.copy()
            upp = xmax.copy()
            safe_counter = cfg.beta_safe_iters
            move = min(move, 0.05)

        beta_prev = beta

        if omega_cur > omega_best:
            omega_best = omega_cur
            xPhys_best = xPhys.copy()

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

        xnew, _, _, _, _, _, _, _, _, low_new, upp_new = mmasub(
            m,
            n,
            it,
            xval.reshape(-1, 1),
            xmin.reshape(-1, 1),
            xmax.reshape(-1, 1),
            xold1.reshape(-1, 1),
            xold2.reshape(-1, 1),
            float(f0),
            df0.reshape(-1, 1),
            fval.reshape(-1, 1),
            dfdx,
            low.reshape(-1, 1),
            upp.reshape(-1, 1),
            mma_a0,
            mma_a,
            mma_c,
            mma_d,
        )

        xnew = xnew.ravel()
        low = low_new.ravel()
        upp = upp_new.ravel()

        Eb_old = xval[-1]
        dEb_max = 0.2 * max(1.0, Eb_old)
        Eb_new = float(np.clip(xnew[-1], Eb_old - dEb_max, Eb_old + dEb_max))
        xnew[-1] = np.clip(Eb_new, xmin[-1], xmax[-1])

        move_lim = move
        if safe_counter > 0:
            move_lim = min(move_lim, cfg.move_safe)
        if reduce_counter > 0:
            move_lim = min(move_lim, cfg.move_reduce)

        xnew[:n_el] = np.clip(xnew[:n_el], xval[:n_el] - move_lim, xval[:n_el] + move_lim)
        xnew = np.clip(xnew, xmin, xmax)

        if safe_counter > 0:
            safe_counter -= 1
        if reduce_counter > 0:
            reduce_counter -= 1

        x_trial = xnew[:n_el]
        xT_trial = fwd(x_trial.reshape((nely, nelx), order="F"))
        xPhys_trial, _ = heaviside_projection(xT_trial, beta, cfg.eta)
        lam_trial, _, _ = _eval_modes(
            xPhys_trial.reshape(-1, order="F"),
            min(3, cfg.J),
            cfg,
            Ke_l,
            Me_l,
            iK,
            jK,
            n_dof,
            free,
        )
        omega_trial = float(np.sqrt(max(lam_trial[0], 0.0)))
        if omega_trial < omega_cur * (1 - 0.01):
            move = max(move * 0.5, 0.01)
            xnew[:n_el] = np.clip(xnew[:n_el], xval[:n_el] - move, xval[:n_el] + move)
            xnew = np.clip(xnew, xmin, xmax)

        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xnew.copy()

        grayness = float(np.mean(4 * xPhys * (1 - xPhys)))
        frac_gray = float(np.mean((xPhys > 0.1) & (xPhys < 0.9)))

        if x_prev is not None:
            change_x = float(np.linalg.norm(x - x_prev) / np.sqrt(n_el))
        else:
            change_x = np.inf

        if omega_prev is not None:
            rel_change_obj = abs(omega_cur - omega_prev) / max(omega_prev, np.finfo(float).eps)
        else:
            rel_change_obj = np.inf

        objective_hist.append(omega_cur)
        volume_hist.append(float(np.mean(xPhys)))
        gray_hist.append(grayness)

        target_beta_idx = min(beta_list.size, int(np.floor(it / cfg.beta_interval) + 1))
        if target_beta_idx > beta_idx:
            beta_idx = target_beta_idx
            safe_counter = cfg.beta_safe_iters
            move = min(move, cfg.move_safe)

        if grayness < cfg.gray_tol and beta_idx < beta_list.size:
            beta_idx += 1
            safe_counter = cfg.beta_safe_iters
            move = min(move, cfg.move_safe)

        if beta_idx == beta_list.size:
            polish_left = max(0, cfg.n_polish - (it - cfg.beta_interval * (beta_list.size - 1)))
            if rel_change_obj < 1e-3 and change_x < 1e-3 and grayness < 0.05 and polish_left <= 0:
                if verbose:
                    print("Converged in max-beta polish regime")
                break
        else:
            if rel_change_obj < 1e-3 and change_x < 1e-3 and grayness < 0.05:
                if verbose:
                    print("Converged before max-beta stage")
                break

        x_prev = x.copy()
        omega_prev = omega_cur

        iteration_log.append([it, beta, omega_cur, float(np.mean(xPhys)), grayness, frac_gray, g_up, g_low, float(np.max(fval))])

        if verbose:
            print(
                f"It:{it:3d} beta:{int(beta):2d} omega:{omega_cur:9.3f} vol:{np.mean(xPhys):6.3f} gray:{grayness:6.3f} "
                f"fracGray:{frac_gray:6.3f} g_up:{g_up:+8.2e} g_low:{g_low:+8.2e} maxg:{np.max(fval):+8.2e}"
            )

        if snapshots_dir is not None and (it % snapshot_every == 0):
            try:
                import matplotlib.pyplot as plt

                img = xPhys.reshape((nely, nelx), order="F")
                if plot_binary:
                    img = (img > 0.5).astype(float)
                fig, ax = plt.subplots(figsize=(9, 2.5))
                ax.imshow(1 - img, cmap="gray", vmin=0.0, vmax=1.0, origin="lower", interpolation="nearest")
                ax.set_axis_off()
                ax.set_title(f"it {it} | omega {omega_cur:.2f}")
                fig.tight_layout()
                fig.savefig(snapshots_dir / f"iter_{it:04d}.png", dpi=120)
                plt.close(fig)
            except Exception:
                pass

    lam_best, omega_vec_best, freq_vec_best = _eval_modes(
        xPhys_best,
        diag_modes,
        cfg,
        Ke_l,
        Me_l,
        iK,
        jK,
        n_dof,
        free,
    )
    omega_best = float(omega_vec_best[0])

    diagnostics["final"] = {"lam": lam_best, "omega": omega_vec_best, "freq": freq_vec_best}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_dir / "results.npz",
            omega_best=omega_best,
            xPhys_best=xPhys_best,
            objective_history=np.asarray(objective_hist, dtype=np.float64),
            volume_history=np.asarray(volume_hist, dtype=np.float64),
            grayness_history=np.asarray(gray_hist, dtype=np.float64),
            diagnostics_initial_lam=diagnostics.get("initial", {}).get("lam", np.array([])),
            diagnostics_initial_omega=diagnostics.get("initial", {}).get("omega", np.array([])),
            diagnostics_final_lam=lam_best,
            diagnostics_final_omega=omega_vec_best,
            nelx=nelx,
            nely=nely,
        )

        if iteration_log:
            np.savetxt(
                output_dir / "iteration_log.csv",
                np.asarray(iteration_log, dtype=np.float64),
                delimiter=",",
                header="iter,beta,omega,vol,gray,frac_gray,g_up,g_low,max_constraint",
                comments="",
            )

    return OlhoffResult(
        omega_best=omega_best,
        xPhys_best=xPhys_best,
        objective_history=np.asarray(objective_hist, dtype=np.float64),
        volume_history=np.asarray(volume_hist, dtype=np.float64),
        grayness_history=np.asarray(gray_hist, dtype=np.float64),
        freq_iter_omega=np.asarray(omega_iter_hist, dtype=np.float64) if omega_iter_hist else np.empty((0, 3)),
        diagnostics=diagnostics,
        config=cfg,
    )
