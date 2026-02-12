"""
topFreqOptimization_MMA.py - Frequency maximization via MMA (BOUND formulation).

Translated from the Matlab implementation of Olhoff & Du (2014) approach.
Maximize min_{j=1..J} lambda_j using variable Eb = E/lambda_ref.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags as spdiags, csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.ndimage import correlate
import matplotlib.pyplot as plt

from mmasub import mmasub


# =========================================================================
# Main function
# =========================================================================

def topFreqOptimization_MMA(cfg=None, opts=None):
    """
    Frequency maximization via MMA.

    Returns
    -------
    omega_best : float
    xPhys_best : ndarray
    diagnostics : dict
    """
    if cfg is None:
        cfg = {}
    if opts is None:
        opts = {}
    cfg = _apply_defaults(cfg)
    opts = _apply_default_opts(opts, cfg)
    diagnostics = {}

    L = cfg['L'];  H = cfg['H']
    nelx = cfg['nelx'];  nely = cfg['nely']
    volfrac = cfg['volfrac'];  penal = cfg['penal']
    rmin = cfg['rmin'];  maxiter = cfg['maxiter']
    supportType = cfg['supportType'];  J = cfg['J']
    E0 = cfg['E0'];  Emin = cfg['Emin']
    rho0 = cfg['rho0'];  rho_min = cfg['rho_min']
    nu = cfg['nu'];  t = cfg['t']

    dx = L / nelx;  dy_m = H / nely
    nEl = nelx * nely
    nodeNrs = np.arange((nelx+1)*(nely+1)).reshape(nely+1, nelx+1, order='F')
    nDof = 2 * (nelx+1) * (nely+1)

    fixed = _build_supports(supportType, nodeNrs)
    free = np.setdiff1d(np.arange(nDof), fixed)

    # Element matrices
    Ke0, M0 = _q4_rect_KeM(E0, nu, rho0, t, dx, dy_m)
    Ke = Ke0 / E0
    Me = M0 / rho0

    # Filter
    rmin_elem = rmin / dx
    if rmin_elem < 1.2:
        print(f"Warning: rmin={rmin:.4f} gives only {rmin_elem:.2f} elements")
        rmin_elem = max(rmin_elem, 1.5)
    print(f"Filter: rmin={rmin:.4f} (physical) = {rmin_elem:.2f} elements")

    r = int(np.ceil(rmin_elem))
    dyf, dxf = np.meshgrid(np.arange(-r+1, r), np.arange(-r+1, r))
    h = np.maximum(0.0, rmin_elem - np.sqrt(dxf**2 + dyf**2))
    Hs = correlate(np.ones((nely, nelx)), h, mode='reflect')

    def fwd(x_mat):
        return correlate(x_mat, h, mode='reflect') / Hs

    def bwd(g_mat):
        return correlate(g_mat / Hs, h, mode='reflect')

    # Projection
    eta = cfg['eta'];  betaMax = cfg['betaMax']
    beta_prev = -np.inf
    beta_list = np.array(cfg['beta_schedule'], dtype=float)
    beta_interval = cfg['beta_interval']
    beta_idx = cfg['beta_start_idx']
    Nsafe = cfg['beta_safe_iters']
    move_safe = cfg['move_safe']
    safe_counter = 0
    gray_tol = cfg['gray_tol']
    reduce_counter = 0
    move_reduce = cfg['move_reduce']
    prev_V_low = None
    Npolish = cfg['Npolish']
    gray_penalty_base = cfg['gray_penalty_base']

    # Assembly indices (0-based)
    cVec = (2 * nodeNrs[:nely, :nelx] + 2).ravel(order='F')
    offsets = np.array([0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1],
                       dtype=np.int32)
    cMat = cVec[:, None] + offsets[None, :]

    Il, Jl = np.where(np.tril(np.ones((8, 8))))
    iK = cMat[:, Il].T.ravel(order='F')
    jK = cMat[:, Jl].T.ravel(order='F')
    Ke_l = Ke[Il, Jl]
    Me_l = Me[Il, Jl]

    # MMA setup
    lambda_ref = cfg['lambda_ref']
    n_mma = nEl + 1
    m_mma = J + 2

    xmin = np.concatenate([1e-3 * np.ones(nEl), [cfg['Eb_min']]])
    xmax = np.concatenate([np.ones(nEl), [cfg['Eb_max']]])
    xval = np.concatenate([volfrac * np.ones(nEl), [cfg['Eb0']]])
    xold1 = xval.copy();  xold2 = xval.copy()
    low = xmin.copy();  upp = xmax.copy()

    a0 = 1.0
    a_mma = np.zeros(m_mma)
    c_mma = np.concatenate([100.0 * np.ones(J), [10.0, 10.0]])
    d_mma = 1e-3 * np.ones(m_mma)

    omega_best = -np.inf
    xPhys_best = xval[:nEl].copy()
    move = cfg['move']
    move_hist_len = cfg['move_hist_len']
    omega_hist = []
    dx_hist = []

    # Diagnostic
    if opts['doDiagnostic']:
        x0 = volfrac * np.ones(nEl)
        xT0 = fwd(x0.reshape(nely, nelx))
        xPhys0, _ = _heaviside_projection(xT0, 1.0, eta)
        xPhys0_flat = xPhys0.ravel()
        lam0, omega0, freq0 = _eval_eigen(
            xPhys0_flat, min(opts['diagModes'], len(free)),
            penal, E0, Emin, rho0, rho_min, Ke_l, Me_l, iK, jK, nDof, free)
        diagnostics['initial'] = dict(lam=lam0, omega=omega0, freq=freq0)
        print(f"--- Diagnostic: uniform design, support={supportType}, vol={volfrac:.2f} ---")
        print(f"  omega [rad/s]: {' '.join(f'{w:8.3f}' for w in omega0[:3])}")
        print(f"  f     [Hz]   : {' '.join(f'{f:8.3f}' for f in freq0[:3])}")
        if opts.get('diagnosticOnly', False):
            omega_best = omega0[0]
            xPhys_best = xPhys0_flat
            diagnostics['final'] = diagnostics['initial']
            print("Diagnostic-only mode, skipping optimization.")
            return omega_best, xPhys_best, diagnostics

    # === OPT LOOP ===
    x_prev = None;  omega_prev = None
    fig_topo = None
    ax_topo = None
    topo_artist = None
    live_plot = bool(opts.get('showTopology', True))
    non_interactive_backends = ('agg', 'pdf', 'ps', 'svg', 'template')
    if live_plot and plt.get_backend().lower().endswith(non_interactive_backends):
        print(f"Live iteration plot disabled for non-interactive backend '{plt.get_backend()}'.")
        live_plot = False
    if live_plot:
        plt.ion()
        fig_topo, ax_topo = plt.subplots(figsize=(10, 2))
        try:
            fig_topo.canvas.manager.set_window_title('Olhoff Optimization Progress')
        except Exception:
            pass
        plt.show(block=False)

    for it in range(1, maxiter + 1):
        beta_target = beta_list[min(beta_idx, len(beta_list)) - 1]
        beta = min(betaMax, beta_target)

        x = xval[:nEl].copy()
        Eb = xval[-1]

        # Filter + projection
        xT = fwd(x.reshape(nely, nelx))
        xPhysMat, dH = _heaviside_projection(xT, beta, eta)
        xPhys = xPhysMat.ravel()

        # Assemble K, M
        Ee = Emin + xPhys**penal * (E0 - Emin)
        re = rho_min + xPhys * (rho0 - rho_min)
        sK = np.kron(Ee, Ke_l)
        sM = np.kron(re, Me_l)
        K = coo_matrix((sK, (iK, jK)), shape=(nDof, nDof)).tocsc()
        M = coo_matrix((sM, (iK, jK)), shape=(nDof, nDof)).tocsc()
        K = K + K.T - spdiags(K.diagonal(), offsets=0, shape=(nDof, nDof))
        M = M + M.T - spdiags(M.diagonal(), offsets=0, shape=(nDof, nDof))

        Kf = K[np.ix_(free, free)]
        Mf = M[np.ix_(free, free)]

        # Eigensolve strategy:
        # - shift-invert (LM with sigma=0): typically fastest in SciPy for low modes
        # - SM (Matlab-like): available via cfg['eigs_strategy'] = 'sm'
        eigs_strategy = str(cfg.get('eigs_strategy', 'shift-invert')).lower()
        v0 = None
        if prev_V_low is not None and prev_V_low.shape[0] == Kf.shape[0]:
            v0 = np.real(prev_V_low[:, 0])

        def _mode_residual_max(Kff, Mff, lam_vals, vecs):
            rmax = 0.0
            for jj in range(len(lam_vals)):
                vtmp = vecs[:, jj]
                Kv = Kff @ vtmp
                Mv = Mff @ vtmp
                rmax = max(rmax, np.linalg.norm(Kv - lam_vals[jj] * Mv) /
                           (np.linalg.norm(Kv) + 1e-30))
            return rmax

        def _solve_modes(mode_name, maxiter, tol, ncv):
            kwargs = dict(M=Mf, k=J, maxiter=maxiter, tol=tol)
            if ncv > J + 1:
                kwargs['ncv'] = ncv
            if v0 is not None:
                kwargs['v0'] = v0
            if mode_name == 'sm':
                kwargs['which'] = 'SM'
            else:
                kwargs['which'] = 'LM'
                kwargs['sigma'] = 0.0
            return eigsh(Kf, **kwargs)

        flag_eigs = 1
        resmax = 1.0
        ncv_primary = min(max(2 * J + 1, 20), Kf.shape[0] - 1)
        ncv_retry = min(max(4 * J, 20), Kf.shape[0] - 1)
        primary_mode = 'sm' if eigs_strategy in ('sm', 'matlab', 'matlab_sm') else 'shift'
        secondary_mode = 'shift' if primary_mode == 'sm' else 'sm'

        try:
            lam_low, V_low = _solve_modes(primary_mode, maxiter=400, tol=1e-8, ncv=ncv_primary)
            idx_sort = np.argsort(np.real(lam_low))
            lam_sorted = np.real(lam_low[idx_sort])
            V_low = np.real(V_low[:, idx_sort])
            resmax = _mode_residual_max(Kf, Mf, lam_sorted, V_low)
            flag_eigs = 0
        except Exception:
            flag_eigs = 1

        if flag_eigs != 0 or resmax > 1e-3:
            try:
                lam_low, V_low = _solve_modes(primary_mode, maxiter=1200, tol=1e-10, ncv=ncv_retry)
                idx_sort = np.argsort(np.real(lam_low))
                lam_sorted = np.real(lam_low[idx_sort])
                V_low = np.real(V_low[:, idx_sort])
                resmax = _mode_residual_max(Kf, Mf, lam_sorted, V_low)
                flag_eigs = 0
            except Exception:
                flag_eigs = 1

        if flag_eigs != 0 or resmax > 1e-3:
            try:
                lam_low, V_low = _solve_modes(secondary_mode, maxiter=800, tol=1e-8, ncv=ncv_retry)
                idx_sort = np.argsort(np.real(lam_low))
                lam_sorted = np.real(lam_low[idx_sort])
                V_low = np.real(V_low[:, idx_sort])
                resmax = _mode_residual_max(Kf, Mf, lam_sorted, V_low)
                flag_eigs = 0
            except Exception:
                flag_eigs = 1

        if flag_eigs != 0 or resmax > 1e-3:
            reduce_counter = 5
            print(f"WARN eigs failed (flag={flag_eigs}, res={resmax:.2e})")
            beta_prev = beta
            continue

        prev_V_low = V_low.copy()
        omega_cur = np.sqrt(lam_sorted[0])

        # Handle beta change
        if beta != beta_prev:
            Eb_feas = np.min(lam_sorted[:J]) / lambda_ref
            Eb = min(Eb, Eb_feas - 1e-8)
            Eb = np.clip(Eb, xmin[-1], xmax[-1])
            xval[-1] = Eb
            xold1 = xval.copy();  xold2 = xval.copy()
            low = xmin.copy();  upp = xmax.copy()
            safe_counter = Nsafe
            move = min(move, 0.05)
        beta_prev = beta

        if omega_cur > omega_best:
            omega_best = omega_cur
            xPhys_best = xPhys.copy()

        # Sensitivities
        dlam = np.zeros((nEl, J))
        for j in range(J):
            v = V_low[:, j]
            v = v / np.sqrt(float(v @ (Mf @ v)))
            phi = np.zeros(nDof);  phi[free] = v
            pe = phi[cMat]
            dlam[:, j] = (penal * (E0-Emin) * xPhys**(penal-1)
                          * np.sum((pe @ Ke) * pe, axis=1)
                          - lam_sorted[j] * (rho0-rho_min)
                          * np.sum((pe @ Me) * pe, axis=1))

        # Grayness penalty
        gray_measure = np.mean(4 * xPhys * (1 - xPhys))
        dgray_dxPhys = 4 * (1 - 2*xPhys) / nEl
        gray_penalty_weight = gray_penalty_base * (beta / betaMax)

        f0 = -Eb + gray_penalty_weight * gray_measure
        dgray_dxT = dgray_dxPhys.reshape(nely, nelx) * dH
        dgray_dx = bwd(dgray_dxT)
        df0 = np.zeros(n_mma)
        df0[:nEl] = gray_penalty_weight * dgray_dx.ravel()
        df0[-1] = -1.0

        # Constraints
        fval = np.zeros(m_mma)
        dfdx = np.zeros((m_mma, n_mma))
        for j in range(J):
            scale_j = max(1.0, Eb)
            fval[j] = (Eb - lam_sorted[j] / lambda_ref) / scale_j
            g = (-dlam[:, j] / lambda_ref).reshape(nely, nelx) * dH
            bg = bwd(g) / scale_j
            dfdx[j, :nEl] = bg.ravel()
            dfdx[j, -1] = 1.0 / scale_j

        # Volume constraints
        g_up = np.mean(xPhys) - volfrac
        gv = (np.ones(nEl) / nEl).reshape(nely, nelx) * dH
        bgv = bwd(gv)
        fval[J] = g_up
        dfdx[J, :nEl] = bgv.ravel()
        g_low = volfrac - np.mean(xPhys)
        fval[J+1] = g_low
        dfdx[J+1, :nEl] = -bgv.ravel()

        # MMA step
        xmma, _, _, _, _, _, _, _, _, low, upp = mmasub(
            m_mma, n_mma, it, xval, xmin, xmax, xold1, xold2,
            f0, df0, fval, dfdx, low, upp, a0, a_mma, c_mma, d_mma)
        xnew = xmma.copy()

        # Damp Eb step
        Eb_old = xval[-1]
        dEbMax = 0.2 * max(1.0, Eb_old)
        Eb_new = np.clip(xnew[-1], Eb_old - dEbMax, Eb_old + dEbMax)
        Eb_new = np.clip(Eb_new, xmin[-1], xmax[-1])
        xnew[-1] = Eb_new

        # Move limit
        move_lim = move
        if safe_counter > 0:
            move_lim = min(move_lim, move_safe)
        if reduce_counter > 0:
            move_lim = min(move_lim, move_reduce)
        xnew[:nEl] = np.clip(xnew[:nEl], xval[:nEl] - move_lim,
                              xval[:nEl] + move_lim)
        xnew = np.clip(xnew, xmin, xmax)
        if safe_counter > 0:
            safe_counter -= 1
        if reduce_counter > 0:
            reduce_counter -= 1

        # Guard against objective drop
        x_trial = xnew[:nEl]
        xT_trial = fwd(x_trial.reshape(nely, nelx))
        xPhys_trial, _ = _heaviside_projection(xT_trial, beta, eta)
        lam_trial, _, _ = _eval_eigen(
            xPhys_trial.ravel(), min(3, J),
            penal, E0, Emin, rho0, rho_min, Ke_l, Me_l, iK, jK, nDof, free)
        omega_trial = np.sqrt(lam_trial[0]) if len(lam_trial) > 0 else 0
        if omega_trial < omega_cur * (1 - 0.01):
            move = max(move * 0.5, 0.01)
            xnew[:nEl] = np.clip(xnew[:nEl], xval[:nEl] - move,
                                  xval[:nEl] + move)
            xnew = np.clip(xnew, xmin, xmax)

        xold2 = xold1.copy();  xold1 = xval.copy();  xval = xnew.copy()

        # Convergence metrics
        grayness = np.mean(4 * xPhys * (1 - xPhys))
        if x_prev is not None:
            change_x = np.linalg.norm(x - x_prev) / np.sqrt(nEl)
        else:
            change_x = np.inf
        if omega_prev is not None:
            rel_change_obj = abs(omega_cur - omega_prev) / max(omega_prev, 1e-30)
        else:
            rel_change_obj = np.inf

        omega_hist.append(omega_cur)
        dx_hist.append(change_x)
        if len(omega_hist) > move_hist_len:
            omega_hist.pop(0)
            dx_hist.pop(0)

        # Beta continuation
        target_beta_idx = min(len(beta_list), it // beta_interval + 1)
        if target_beta_idx > beta_idx:
            beta_idx = target_beta_idx
            safe_counter = Nsafe
            move = min(move, move_safe)
            print(f"  -> Advancing beta to {beta_list[beta_idx-1]:.0f} (iteration {it})")

        if grayness < gray_tol and beta_idx < len(beta_list):
            beta_idx += 1
            safe_counter = Nsafe
            move = min(move, move_safe)
            print(f"  -> Advancing beta to {beta_list[beta_idx-1]:.0f} "
                  f"(grayness={grayness:.3f} < {gray_tol:.3f})")

        # Termination
        if beta_idx == len(beta_list):
            polish_left = max(0, Npolish - (it - beta_interval*(len(beta_list)-1)))
            if rel_change_obj < 1e-3 and change_x < 1e-3 and grayness < 0.05 and polish_left <= 0:
                print(f"Converged: rel dOmega={rel_change_obj:.2e}, "
                      f"dx={change_x:.2e}, gray={grayness:.3f}")
                break
        else:
            if rel_change_obj < 1e-3 and change_x < 1e-3 and grayness < 0.05:
                print(f"Converged early: rel dOmega={rel_change_obj:.2e}, "
                      f"dx={change_x:.2e}, gray={grayness:.3f}")
                break

        x_prev = x.copy();  omega_prev = omega_cur
        max_g_eig = max(fval[:J])

        print(f"It:{it:3d}  beta:{beta:2.0f}  omega:{omega_cur:.3f}  "
              f"vol:{np.mean(xPhys):.3f}  gray:{grayness:.3f}  "
              f"g_up:{g_up:+.2e}  maxg_eig:{max_g_eig:+.2e}")

        # Plot (Matlab drawnow-style update each iteration)
        if live_plot and fig_topo is not None and plt.fignum_exists(fig_topo.number):
            img = xPhys.reshape(nely, nelx)
            if opts.get('plotBinary', False):
                img = (img > 0.5).astype(float)
            if topo_artist is None:
                topo_artist = ax_topo.imshow(
                    1 - img, cmap='gray', aspect='equal', vmin=0, vmax=1,
                    interpolation='nearest')
                ax_topo.axis('off')
            else:
                topo_artist.set_data(1 - img)
            ax_topo.set_title(f"It {it} | beta {beta:.0f} | omega1 {omega_cur:.3f} rad/s")
            fig_topo.canvas.draw()
            fig_topo.canvas.flush_events()
            plt.pause(max(1e-4, float(opts.get('iterPause', 0.01))))

    if live_plot:
        plt.ioff()

    # Final diagnostics
    lam_best, omega_vec_best, freq_vec_best = _eval_eigen(
        xPhys_best, min(opts['diagModes'], len(free)),
        penal, E0, Emin, rho0, rho_min, Ke_l, Me_l, iK, jK, nDof, free)
    omega_best = omega_vec_best[0]
    diagnostics['final'] = dict(lam=lam_best, omega=omega_vec_best, freq=freq_vec_best)

    print(f"\nBest design: omega1 = {omega_best:.4f} rad/s ({omega_best/(2*np.pi):.4f} Hz)")
    print(f"Best design volume = {np.mean(xPhys_best):.4f} (target {volfrac:.4f})")

    return omega_best, xPhys_best, diagnostics


# =========================================================================
# Helpers
# =========================================================================

def _apply_defaults(cfg):
    defaults = dict(
        L=8, H=1, nelx=240, nely=30, volfrac=0.5, penal=3.0, rmin=None,
        maxiter=300, supportType="CC", J=3, E0=1e7, Emin=None,
        rho0=1.0, rho_min=1e-6, nu=0.3, t=1.0, eta=0.5, betaMax=64,
        beta_schedule=[1, 2, 4, 8, 16, 32, 64], beta_interval=40,
        beta_start_idx=1, beta_safe_iters=5, move_safe=0.05, gray_tol=0.10,
        move_reduce=0.02, Npolish=40, gray_penalty_base=0.5,
        lambda_ref=2e4, Eb0=1, Eb_min=0, Eb_max=50, move=0.2, move_hist_len=10,
        eigs_strategy='shift-invert')
    for k, v in defaults.items():
        if k not in cfg:
            cfg[k] = v
    if cfg['rmin'] is None:
        cfg['rmin'] = 2 * cfg['L'] / cfg['nelx']
    if cfg['Emin'] is None:
        cfg['Emin'] = max(1e-6 * cfg['E0'], 1e-3)
    cfg['supportType'] = str(cfg['supportType']).upper()
    return cfg


def _apply_default_opts(opts, cfg):
    defaults = dict(doDiagnostic=True, diagnosticOnly=False,
                    diagModes=max(3, cfg['J']), plotBinary=False,
                    showTopology=True, iterPause=0.0)
    for k, v in defaults.items():
        if k not in opts:
            opts[k] = v
    return opts


def _heaviside_projection(xTilde, beta, eta):
    denom = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    xPhys = (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / denom
    dH = (beta * (1 - np.tanh(beta * (xTilde - eta))**2)) / denom
    return xPhys, dH


def _build_supports(supportType, nodeNrs):
    nely = nodeNrs.shape[0] - 1
    leftNodes = nodeNrs[:, 0]
    rightNodes = nodeNrs[:, -1]
    midIdx = round(nely / 2)
    leftMid = nodeNrs[midIdx, 0]
    rightMid = nodeNrs[midIdx, -1]

    u = lambda n: 2 * n      # x-DOF (0-based)
    v = lambda n: 2 * n + 1  # y-DOF (0-based)

    s = supportType.upper()
    if s in ("CF", "CLAMPED-FREE"):
        fixed = np.concatenate([u(leftNodes), v(leftNodes)])
    elif s in ("CC", "CLAMPED-CLAMPED"):
        fixed = np.concatenate([u(leftNodes), v(leftNodes),
                                u(rightNodes), v(rightNodes)])
    elif s in ("CH", "CS", "CLAMPED-SIMPLY"):
        fixed = np.concatenate([u(leftNodes), v(leftNodes),
                                [u(rightMid), v(rightMid)]])
    elif s in ("HH", "SS", "SIMPLY-SUPPORTED"):
        fixed = np.array([u(leftMid), v(leftMid), u(rightMid), v(rightMid)])
    else:
        raise ValueError(f"Unknown supportType '{s}'.")
    return np.unique(fixed)


def _q4_rect_KeM(E, nu, rho, t, dx, dy):
    C = E / (1 - nu**2) * np.array([[1, nu, 0],
                                      [nu, 1, 0],
                                      [0, 0, (1-nu)/2]])
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    Ke = np.zeros((8, 8))
    Me = np.zeros((8, 8))

    xiN = np.array([-1, 1, 1, -1], dtype=float)
    etaN = np.array([-1, -1, 1, 1], dtype=float)
    detJ = (dx * dy) / 4.0
    invJ = np.array([[2/dx, 0], [0, 2/dy]])

    for i in range(2):
        for j in range(2):
            xi = gp[i];  et = gp[j]
            N = 0.25 * (1 + xi*xiN) * (1 + et*etaN)
            dN_dxi = 0.25 * xiN * (1 + et*etaN)
            dN_deta = 0.25 * etaN * (1 + xi*xiN)
            grads = invJ @ np.vstack([dN_dxi, dN_deta])
            dN_dx = grads[0, :];  dN_dy = grads[1, :]

            B = np.zeros((3, 8))
            Nmat = np.zeros((2, 8))
            for a in range(4):
                B[0, 2*a] = dN_dx[a]
                B[1, 2*a+1] = dN_dy[a]
                B[2, 2*a] = dN_dy[a]
                B[2, 2*a+1] = dN_dx[a]
                Nmat[0, 2*a] = N[a]
                Nmat[1, 2*a+1] = N[a]

            Ke += (B.T @ C @ B) * (t * detJ)
            Me += (Nmat.T @ Nmat) * (rho * t * detJ)
    return Ke, Me


def _eval_eigen(xPhys, numModes, penal, E0, Emin, rho0, rho_min,
                Ke_l, Me_l, iK, jK, nDof, free):
    numModes = min(numModes, len(free) - 1)
    if numModes < 1:
        return np.array([]), np.array([]), np.array([])
    Ee = Emin + xPhys**penal * (E0 - Emin)
    re = rho_min + xPhys * (rho0 - rho_min)
    sK = np.kron(Ee, Ke_l)
    sM = np.kron(re, Me_l)
    K = coo_matrix((sK, (iK, jK)), shape=(nDof, nDof)).tocsc()
    M = coo_matrix((sM, (iK, jK)), shape=(nDof, nDof)).tocsc()
    K = K + K.T - spdiags(K.diagonal(), offsets=0, shape=(nDof, nDof))
    M = M + M.T - spdiags(M.diagonal(), offsets=0, shape=(nDof, nDof))
    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    try:
        vals, _ = eigsh(Kf, M=Mf, k=numModes, sigma=0.0, which='LM')
    except Exception:
        vals, _ = eigsh(Kf, M=Mf, k=numModes, which='SM')
    lam = np.sort(np.real(vals))
    omega = np.sqrt(np.maximum(lam, 0))
    freq = omega / (2 * np.pi)
    return lam, omega, freq
