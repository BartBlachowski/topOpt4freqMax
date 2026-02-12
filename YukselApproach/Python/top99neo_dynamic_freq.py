"""
top99neo_dynamic_freq.py - Dynamic-code frequency maximization (Section 3 style).

Translated from the Matlab implementation of Yuksel & Yilmaz (2025).
- direct eigenvalue-based optimization at each iteration
- sensitivity filtering (ft = 1)
- mass interpolation with low-density penalization (Eq. 10, d = 6, x_cut = 0.1)
- repeated-mode handling via a 4% proximity threshold between omega1 and omega2
"""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.ndimage import correlate


def top99neo_dynamic_freq(nelx=320, nely=40, volfrac=0.5, penal=3.0,
                          rmin=2.5, ft=1, ftBC='N', move=0.01,
                          maxit=200, bcType="simply"):
    if ft != 1:
        raise ValueError("Only ft=1 (sensitivity filtering) is supported.")

    E0 = 1e7;  Emin = 1e-9 * E0;  nu = 0.3
    rho0 = 1.0;  rho_min = 1e-9 * rho0
    dMass = 6;  xMassCut = 0.1;  repTol = 0.04

    nEl = nelx * nely
    nodeNrs = np.arange((1 + nelx) * (1 + nely), dtype=np.int32).reshape(
        1 + nely, 1 + nelx, order='F')
    cVec = (2 * nodeNrs[:-1, :-1] + 2).ravel(order='F')
    offsets = np.array([0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1],
                       dtype=np.int32)
    cMat = cVec[:, None] + offsets[None, :]
    nDof = (1 + nely) * (1 + nelx) * 2

    # Lower-triangle assembly indices
    sI, sII = [], []
    for j in range(8):
        sI.extend(range(j, 8))
        sII.extend([j] * (8 - j))
    sI = np.array(sI);  sII = np.array(sII)
    iK = cMat[:, sI].T;  jK = cMat[:, sII].T
    Iar = np.column_stack([np.maximum(iK.ravel('F'), jK.ravel('F')),
                           np.minimum(iK.ravel('F'), jK.ravel('F'))])

    # Q4 element stiffness (lower triangle coefficients)
    c1 = np.array([12,3,-6,-3,-6,-3,0,3,12,3,0,-3,-6,-3,-6,12,-3,0,-3,-6,3,
                   12,3,-6,3,-6,12,3,-6,-3,12,3,0,12,-3,12], dtype=float)
    c2 = np.array([-4,3,-2,9,2,-3,4,-9,-4,-9,4,-3,2,9,-2,-4,-3,4,9,2,3,-4,
                   -9,-2,3,2,-4,3,-2,9,-4,-9,4,-4,-3,-4], dtype=float)
    Ke_vec = 1.0 / (1.0 - nu**2) / 24.0 * (c1 + nu * c2)
    Ke0 = np.zeros((8, 8))
    Ke0[np.tril_indices(8)] = Ke_vec
    Ke0 = Ke0 + Ke0.T - np.diag(np.diag(Ke0))

    # Element mass matrix
    beamL, beamH, tipMassFrac = _physical_setup(bcType)
    elemArea = (beamL / nelx) * (beamH / nely)
    MeS = (elemArea / 36.0) * np.array([[4,2,1,2],[2,4,2,1],
                                         [1,2,4,2],[2,1,2,4]], dtype=float)
    Me0 = np.kron(MeS, np.eye(2))

    fixed, tipMassDofs = _bc_and_tip_mass(nodeNrs, nely, bcType)
    free = np.setdiff1d(np.arange(nDof), fixed)
    act = np.arange(nEl)

    tipMassVal = 0.0
    if tipMassFrac > 0 and len(tipMassDofs) > 0:
        tipMassVal = tipMassFrac * volfrac * beamL * beamH * rho0

    # Filter
    bcF = 'reflect' if ftBC.upper() == 'N' else 'constant'
    r = int(np.ceil(rmin))
    dy, dx = np.meshgrid(np.arange(-r+1, r), np.arange(-r+1, r))
    h = np.maximum(0.0, rmin - np.sqrt(dx**2 + dy**2))
    Hs = correlate(np.ones((nely, nelx)), h, mode=bcF)

    x = volfrac * np.ones(nEl)
    xPhys = x.copy()
    dV = np.ones(nEl) / (nEl * volfrac)

    info = dict(omegaHist=np.full((maxit, 3), np.nan),
                chHist=np.full(maxit, np.nan),
                repActive=np.zeros(maxit, dtype=bool))

    ltMask = np.tril(np.ones((8, 8), dtype=bool))
    meLower = Me0[ltMask]

    for it in range(maxit):
        xPhys[act] = x[act]

        # Assemble K
        Ee = Emin + xPhys**penal * (E0 - Emin)
        sK = np.outer(Ke_vec, Ee).ravel('F')
        K = _assemble(Iar[:, 0], Iar[:, 1], sK, nDof)
        K = K + K.T - diags(K.diagonal(), 0, shape=(nDof, nDof), format='csc')

        # Assemble M with low-density penalization
        rhoe = rho_min + (rho0 - rho_min) * xPhys.copy()
        low = xPhys <= xMassCut
        rhoe[low] = rho_min + (rho0 - rho_min) * (xPhys[low]**dMass)
        sM = np.outer(meLower, rhoe).ravel('F')
        M = _assemble(Iar[:, 0], Iar[:, 1], sM, nDof)
        M = M + M.T - diags(M.diagonal(), 0, shape=(nDof, nDof), format='csc')
        if tipMassVal > 0 and len(tipMassDofs) > 0:
            M = M + coo_matrix(
                (tipMassVal * np.ones(len(tipMassDofs)),
                 (tipMassDofs, tipMassDofs)), shape=(nDof, nDof)).tocsc()

        Kff = K[np.ix_(free, free)]
        Mff = M[np.ix_(free, free)]

        vals, vecs = eigsh(Kff, M=Mff, k=3, sigma=0.0, which='LM',
                           maxiter=1200, tol=1e-10)
        idx = np.argsort(vals)
        lam = np.maximum(vals[idx], np.finfo(float).eps)
        V = vecs[:, idx]
        omega = np.sqrt(lam)
        info['omegaHist'][it, :len(omega)] = omega

        # Element sensitivities for first two modes
        drho = (rho0 - rho_min) * np.ones(nEl)
        drho[low] = dMass * (rho0 - rho_min) * xPhys[low]**(dMass - 1)
        dlam = np.zeros((nEl, 2))
        for j in range(2):
            vj = V[:, j].copy()
            mn = max(np.finfo(float).eps, float(np.real(vj @ (Mff @ vj))))
            vj /= np.sqrt(mn)
            phi = np.zeros(nDof);  phi[free] = vj
            pe = phi[cMat]
            dlam[:, j] = (penal * (E0 - Emin) * xPhys**(penal - 1)
                          * np.sum((pe @ Ke0) * pe, axis=1)
                          - lam[j] * drho * np.sum((pe @ Me0) * pe, axis=1))

        # Repeated eigenvalue handling
        if (omega[1] - omega[0]) / max(omega[0], np.finfo(float).eps) < repTol:
            dlamObj = np.maximum(dlam[:, 0], dlam[:, 1])
            info['repActive'][it] = True
        else:
            dlamObj = dlam[:, 0]

        dlamObj = dlamObj - dlamObj.min() + 1e-12
        dc = -dlamObj
        xMat = np.maximum(1e-3, x).reshape(nely, nelx)
        dcF = correlate((x * dc).reshape(nely, nelx), h, mode=bcF) / Hs / xMat
        dV0 = correlate((x * dV).reshape(nely, nelx), h, mode=bcF) / Hs / xMat

        xOld = x.copy()
        xT = x[act].copy()
        xU = xT + move;  xL = xT - move
        ocArg = np.maximum(-dcF.ravel()[act] / dV0.ravel()[act], 1e-30)
        ocP = xT * np.sqrt(ocArg)
        lb = [0.0, float(np.mean(ocP) / np.mean(x))]
        while (lb[1] - lb[0]) / (lb[1] + lb[0]) > 1e-4:
            lmid = 0.5 * (lb[0] + lb[1])
            x[act] = np.clip(np.minimum(ocP / lmid, xU), xL, 1.0)
            x[act] = np.maximum(x[act], 0.0)
            if np.mean(x) > np.mean(xPhys):
                lb[0] = lmid
            else:
                lb[1] = lmid

        ch = np.max(np.abs(x - xOld))
        info['chHist'][it] = ch
        print(f"Dyn It.:{it+1:4d} w1:{omega[0]:8.3f} w2:{omega[1]:8.3f} "
              f"w3:{omega[2]:8.3f} ch:{ch:.2e} rep:{int(info['repActive'][it])}")

    info['xFinal'] = xPhys.copy()
    info['omega1'] = info['omegaHist'][maxit - 1, 0]
    info['tipMassVal'] = tipMassVal
    print(f"\nDynamic code final: omega1 = {info['omega1']:.4f} rad/s")
    return xPhys, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _physical_setup(bcType):
    bc = bcType.lower()
    if bc == "cantilever":
        return 15.0, 10.0, 0.20
    elif bc in ("simply", "fixedpinned"):
        return 8.0, 1.0, 0.0
    raise ValueError(f"Unsupported bcType: {bcType}")


def _bc_and_tip_mass(nodeNrs, nely, bcType):
    """Return (fixed, tipMassDofs) with 0-based DOF indices."""
    tipMassDofs = np.array([], dtype=np.int32)
    bc = bcType.lower()
    if bc == "simply":
        midRow = round(nely / 2)
        lm = nodeNrs[midRow, 0];  rm = nodeNrs[midRow, -1]
        fixed = np.array([2*lm, 2*lm+1, 2*rm, 2*rm+1])
    elif bc == "cantilever":
        ln = nodeNrs[:, 0]
        fixed = np.union1d(2*ln, 2*ln+1)
        midRow = round((nely + 1) / 2) - 1
        tn = nodeNrs[midRow, -1]
        tipMassDofs = np.array([2*tn, 2*tn+1], dtype=np.int32)
    elif bc == "fixedpinned":
        ln = nodeNrs[:, 0]
        fixed = np.union1d(2*ln, 2*ln+1)
        midRow = round(nely / 2)
        rm = nodeNrs[midRow, -1]
        fixed = np.union1d(fixed, [2*rm, 2*rm+1])
    else:
        raise ValueError(f"Unsupported bcType: {bcType}")
    return np.unique(fixed), tipMassDofs


def _assemble(i, j, s, n):
    return coo_matrix((s, (i.astype(np.int64), j.astype(np.int64))),
                      shape=(n, n)).tocsc()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xPhys, info = top99neo_dynamic_freq()
    plt.figure()
    plt.imshow(1 - xPhys.reshape(40, 320), cmap='gray', aspect='equal',
               vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f"omega1 = {info['omega1']:.1f} rad/s")
    plt.tight_layout()
    plt.show()
