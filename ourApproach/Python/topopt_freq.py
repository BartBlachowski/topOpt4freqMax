# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# MODIFIED:
# (1) Compute (omega1, Phi1) once on the DESIGN DOMAIN (free DOFs)
# (2) Use harmonic-type load: F(x) = omega1^2 * M(x) * Phi1
# (3) During TO, update F only through M(x) (SIMP mass); Phi1, omega1 stay fixed

from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, eigsh
from matplotlib import colors
import matplotlib.pyplot as plt


# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft, L, H):
    print("Compliance with harmonic-type inertial load (fixed mode)")
    print("mesh: " + str(nelx) + " x " + str(nely))
    print("domain: L x H = {0} x {1}".format(L, H))
    print("volfrac: " + str(volfrac) + ", rmin(phys): " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", "Density based"][ft])

    if L <= 0.0 or H <= 0.0:
        raise ValueError("L and H must be positive.")
    hx = L / float(nelx)
    hy = H / float(nely)
    print("element size: hx={0:.6g}, hy={1:.6g}".format(hx, hy))

    # --- Material / SIMP ---
    Emin = 1e-2
    Emax = 1.0E7

    # Mass SIMP (you can tune)
    rho_min = 1e-6
    rho0 = 1.0
    pmass = 1.0  # SIMP exponent for mass (often 1)

    # dofs:
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely * nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0  # must be initialized to use the NGuyen/Paulino OC approach
    dc = np.zeros((nely, nelx), dtype=float)

    # FE: Build the index vectors for the for coo matrix format.
    KE = lk(hx, hy)
    ME = lm(hx, hy)  # element mass matrix (consistent)
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]
            )

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    rminx = max(1, int(np.ceil(rmin / hx)))
    rminy = max(1, int(np.ceil(rmin / hy)))
    nfilter = int(nelx * nely * (2 * (rminx - 1) + 1) * (2 * (rminy - 1) + 1))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = max(i - (rminx - 1), 0)
            kk2 = min(i + rminx, nelx)
            ll1 = max(j - (rminy - 1), 0)
            ll2 = min(j + rminy, nely)
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    dx = (i - k) * hx
                    dy = (j - l) * hy
                    fac = rmin - np.sqrt(dx * dx + dy * dy)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc = cc + 1

    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)

    # BC's and support: hinge–hinge (pin–pin) at mid-height
    dofs = np.arange(ndof)

    j_mid = nely // 2                          # middle of height
    nL = j_mid                                 # node at (x=0, y=mid)
    nR = (nelx)*(nely + 1) + j_mid             # node at (x=L, y=mid)

    fixed = np.array([2*nL, 2*nL+1, 2*nR, 2*nR+1], dtype=int)  # ux,uy at both pins
    free = np.setdiff1d(dofs, fixed)


    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))

    # -------------------------------------------------------------------------
    # (1) EIGENANALYSIS ON DESIGN DOMAIN (free DOFs) ONCE, using initial xPhys
    # -------------------------------------------------------------------------
    sK0 = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order="F")
    K0 = coo_matrix((sK0, (iK, jK)), shape=(ndof, ndof)).tocsc()

    rhoPhys0 = rho_min + (xPhys) ** pmass * (rho0 - rho_min)
    sM0 = ((ME.flatten()[np.newaxis]).T * rhoPhys0).flatten(order="F")
    M0 = coo_matrix((sM0, (iK, jK)), shape=(ndof, ndof)).tocsc()

    K0f = K0[free, :][:, free]
    M0f = M0[free, :][:, free]

    # Smallest eigenpair for K phi = lambda M phi (shift-invert near 0)
    lam1, phi1_free = eigsh(K0f, M=M0f, k=2, sigma=0.0, which="LM")
    lam1 = float(lam1[0])
    omega1 = np.sqrt(max(lam1, 0.0))

    # Mass-normalize phi1: phi^T M phi = 1 (recommended for stability)
    phi1_free = phi1_free[:, 0]
    mn = float(phi1_free.T @ (M0f @ phi1_free))
    if mn > 0:
        phi1_free = phi1_free / np.sqrt(mn)

    Phi1 = np.zeros(ndof, dtype=float)
    Phi1[free] = phi1_free

    print(f"[Eigen] lambda1={lam1:.6e}, omega1={omega1:.6e} rad/s (computed once, fixed)")

    # Initialize interactive visualization
    backend_name = plt.get_backend().lower()
    live_plot = "agg" not in backend_name
    if live_plot:
        plt.ion()
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.35)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_obj = fig.add_subplot(gs[0, 1])
    ax_conv = fig.add_subplot(gs[1, 1])

    im = ax_img.imshow(
        xPhys.reshape((nelx, nely)).T,
        cmap="gray_r",
        interpolation="none",
        norm=colors.Normalize(vmin=0, vmax=1),
        origin="lower",
    )
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, label="Density")
    ax_img.set_title("Topology (density)")
    ax_img.set_xlabel("Element x")
    ax_img.set_ylabel("Element y")

    (line_obj,) = ax_obj.plot([], [], lw=1.6, color="tab:blue")
    ax_obj.set_title("Objective history")
    ax_obj.set_xlabel("Iteration")
    ax_obj.set_ylabel("C = f^T u")
    ax_obj.grid(True, alpha=0.25)

    (line_vol,) = ax_conv.plot([], [], lw=1.6, color="tab:green", label="Volume")
    (line_change,) = ax_conv.plot([], [], lw=1.6, color="tab:red", label="Change")
    ax_conv.set_title("Convergence history")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Value")
    ax_conv.grid(True, alpha=0.25)
    ax_conv.legend(loc="best")

    obj_hist = []
    vol_hist = []
    ch_hist = []
    it_hist = []

    if live_plot:
        plt.show(block=False)

    # Set loop counter and gradient vectors
    loop = 0
    change = 1
    dv = np.ones(nely * nelx)
    dc = np.ones(nely * nelx)
    ce = np.ones(nely * nelx)

    # Pre-allocate for load sensitivity term
    u_e = np.zeros((nelx * nely, 8), dtype=float)
    phi_e = np.zeros((nelx * nely, 8), dtype=float)

    while change > 0.01 and loop < 2000:
        loop = loop + 1

        # ---------------------------------------------------------------------
        # (2,3) Update load using CURRENT M(x):  F = omega1^2 * M(x) * Phi1
        #     Phi1, omega1 are FIXED (from initial eigenanalysis).
        # ---------------------------------------------------------------------
        rhoPhys = rho_min + (xPhys) ** pmass * (rho0 - rho_min)
        sM = ((ME.flatten()[np.newaxis]).T * rhoPhys).flatten(order="F")
        M = coo_matrix((sM, (iK, jK)), shape=(ndof, ndof)).tocsc()

        f[:, 0] = (omega1 ** 2) * (M @ Phi1)

        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

        # Remove constrained dofs from matrix
        Kf = K[free, :][:, free]

        # Solve system
        u[:, 0] = 0.0
        u[free, 0] = spsolve(Kf, f[free, 0])

        # Objective (compliance): C = f^T u
        obj = float(f[:, 0].T @ u[:, 0])

        # Element strain energy term (as in original) for -u^T dK u part
        ue = u[edofMat].reshape(nelx * nely, 8)
        ce[:] = (np.dot(ue, KE) * ue).sum(1)

        # Sensitivities:
        # dC/dx = -u^T (dK/dx) u  +  (df/dx)^T u
        # where f = omega^2 * M(x) Phi1, so df/dx = omega^2 * (dM/dx) Phi1
        dc_stiff = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce

        # Load term: omega^2 * u_e^T (dM_e/dx) phi_e
        # dM_e/dx = pmass * x^(pmass-1) * (rho0-rho_min) * ME
        u_e[:, :] = ue
        phi_e[:, :] = Phi1[edofMat].reshape(nelx * nely, 8)

        # scalar per element: u_e^T * ME * phi_e
        uMephi = (np.dot(u_e, ME) * phi_e).sum(1)
        dMdx_scale = pmass * (xPhys ** (pmass - 1)) * (rho0 - rho_min)
        dc_load = (omega1 ** 2) * dMdx_scale * uMephi

        dc[:] = dc_stiff #+ dc_load
        dv[:] = np.ones(nely * nelx)

        # Sensitivity filtering:
        if ft == 0:
            dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

        # Optimality criteria
        xold[:] = x
        (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)

        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]

        # Compute current volume and change by the inf. norm
        vol = (g + volfrac * nelx * nely) / (nelx * nely)
        change = np.linalg.norm(x.reshape(nelx * nely, 1) - xold.reshape(nelx * nely, 1), np.inf)

        # Update live visualization
        it_hist.append(loop)
        obj_hist.append(obj)
        vol_hist.append(vol)
        ch_hist.append(change)

        if live_plot:
            im.set_array(xPhys.reshape((nelx, nely)).T)
            ax_img.set_title(f"Topology (density) | it={loop}  obj={obj:.3e}  vol={vol:.3f}  ch={change:.3f}")

            line_obj.set_data(it_hist, obj_hist)
            line_vol.set_data(it_hist, vol_hist)
            line_change.set_data(it_hist, ch_hist)

            ax_obj.set_xlim(1, max(2, loop))
            ax_conv.set_xlim(1, max(2, loop))

            obj_min = float(np.min(obj_hist))
            obj_max = float(np.max(obj_hist))
            obj_pad = max(1e-12, 0.05 * max(abs(obj_min), abs(obj_max), 1e-12))
            ax_obj.set_ylim(obj_min - obj_pad, obj_max + obj_pad)

            conv_min = float(min(np.min(vol_hist), np.min(ch_hist)))
            conv_max = float(max(np.max(vol_hist), np.max(ch_hist)))
            conv_pad = max(1e-6, 0.05 * max(abs(conv_min), abs(conv_max), 1e-6))
            ax_conv.set_ylim(conv_min - conv_pad, conv_max + conv_pad)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        print(
            "it.: {0} , obj(C=f^T u): {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
                loop, obj, vol, change
            )
        )

    # Post-analysis: first circular frequency for final topology
    sK_final = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order="F")
    K_final = coo_matrix((sK_final, (iK, jK)), shape=(ndof, ndof)).tocsc()

    rhoPhys_final = rho_min + (xPhys) ** pmass * (rho0 - rho_min)
    sM_final = ((ME.flatten()[np.newaxis]).T * rhoPhys_final).flatten(order="F")
    M_final = coo_matrix((sM_final, (iK, jK)), shape=(ndof, ndof)).tocsc()

    Kf_final = K_final[free, :][:, free]
    Mf_final = M_final[free, :][:, free]
    lam1_final, _ = eigsh(Kf_final, M=Mf_final, k=1, sigma=0.0, which="LM")
    lam1_final = float(lam1_final[0])
    omega1_final = np.sqrt(max(lam1_final, 0.0))
    f1_final = omega1_final / (2.0 * np.pi)
    print(f"[Final Eigen] lambda1={lam1_final:.6e}, omega1={omega1_final:.6e} rad/s, f1={f1_final:.6e} Hz")

    if live_plot:
        plt.ioff()
        plt.show()
    try:
        input("Press any key...")
    except EOFError:
        pass


# element stiffness matrix (Q4, plane stress) for rectangular element hx x hy
def lk(hx, hy):
    E = 1.0
    nu = 0.3
    D = (E / (1.0 - nu ** 2)) * np.array(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)],
        ],
        dtype=float,
    )

    invJ = np.array([[2.0 / hx, 0.0], [0.0, 2.0 / hy]], dtype=float)
    detJ = 0.25 * hx * hy
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = (-gp, gp)

    # ccw node order: [LL, LR, UR, UL]
    KE_ccw = np.zeros((8, 8), dtype=float)
    for xi in gauss_pts:
        for eta in gauss_pts:
            dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)], dtype=float)
            dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)], dtype=float)
            dN_xy = invJ @ np.vstack((dN_dxi, dN_deta))
            dN_dx = dN_xy[0, :]
            dN_dy = dN_xy[1, :]

            B = np.zeros((3, 8), dtype=float)
            B[0, 0::2] = dN_dx
            B[1, 1::2] = dN_dy
            B[2, 0::2] = dN_dy
            B[2, 1::2] = dN_dx

            KE_ccw += (B.T @ D @ B) * detJ

    # topopt edof order: [UL, UR, LR, LL]
    perm = np.array([6, 7, 4, 5, 2, 3, 0, 1], dtype=int)
    KE = KE_ccw[np.ix_(perm, perm)]
    return KE


# element consistent mass matrix (Q4, 2 dof/node) for rectangular hx x hy
def lm(hx, hy):
    area = hx * hy
    # ccw node order: [LL, LR, UR, UL]
    Ms_ccw = (area / 36.0) * np.array(
        [[4, 2, 1, 2],
         [2, 4, 2, 1],
         [1, 2, 4, 2],
         [2, 1, 2, 4]],
        dtype=float
    )
    ME_ccw = np.kron(Ms_ccw, np.eye(2))
    # topopt edof order: [UL, UR, LR, LL]
    perm = np.array([6, 7, 4, 5, 2, 3, 0, 1], dtype=int)
    ME = ME_ccw[np.ix_(perm, perm)]
    return ME


# Optimality criterion
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1 = 0.0
    l2 = 1e9
    move = 0.2
    xnew = np.zeros_like(x)

    eps = 1e-30
    # Enforce OC assumptions: dv > 0 and dc <= 0 (avoid sqrt of negative)
    dv_safe = np.maximum(dv, 1e-12)
    dc_safe = np.minimum(dc, -1e-12)  # clamp positives to small negative

    for _ in range(200):  # hard cap
        lmid = 0.5 * (l1 + l2)

        # Avoid division by zero at lmid=0
        denom = max(lmid, 1e-30)

        B = -dc_safe / dv_safe / denom
        B = np.maximum(B, 1e-30)  # keep positive

        x_candidate = x * np.sqrt(B)
        xnew[:] = np.maximum(0.0,
                    np.maximum(x - move,
                        np.minimum(1.0,
                            np.minimum(x + move, x_candidate))))

        gt = g + np.sum(dv_safe * (xnew - x))

        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid

        # Safe convergence check (no (l1+l2) division)
        if (l2 - l1) / max(l1 + l2, eps) < 1e-3:
            break

    return xnew, gt




# The real main driver
if __name__ == "__main__":
    nelx = 240
    nely = 30
    volfrac = 0.4
    rmin = 0.05
    penal = 3.0
    ft = 0  # ft==0 -> sens, ft==1 -> dens
    L = 8.0
    H = 1.0

    import sys

    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        rmin = float(sys.argv[4])
    if len(sys.argv) > 5:
        penal = float(sys.argv[5])
    if len(sys.argv) > 6:
        ft = int(sys.argv[6])
    if len(sys.argv) > 7:
        L = float(sys.argv[7])
    if len(sys.argv) > 8:
        H = float(sys.argv[8])

    main(nelx, nely, volfrac, penal, rmin, ft, L, H)
