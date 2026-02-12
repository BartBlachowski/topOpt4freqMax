"""
top99neo_inertial_freq.py - Two-stage inertial-load frequency maximization.

Translated from the Matlab implementation of Yuksel & Yilmaz (2025).
  Stage 1: standard compliance minimization with a unit point load.
  Stage 2: compliance minimization with design-dependent inertial load F = M(x)*u_hat.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh, spsolve
from scipy.ndimage import correlate
import matplotlib.pyplot as plt


# =========================================================================
# Main function
# =========================================================================

def top99neo_inertial_freq(nelx=300, nely=100, volfrac=0.5, penal=3.0,
                           rmin=8.75, ft=3, ftBC='N', eta=0.5, beta=1.0,
                           move=0.2, maxit=100, stage1_maxit=None,
                           bcType="simply", nHistModes=0):
    if stage1_maxit is None:
        stage1_maxit = maxit
    nHistModes = max(0, int(nHistModes))
    stage2Tol = 1e-3 if bcType.lower() == 'fixedpinned' else 1e-2

    # Material / continuation
    E0 = 1e7;  Emin = 1e-9 * E0;  nu = 0.3
    rho0 = 1.0;  rho_min = 1e-9 * rho0
    dMass = 6;  xMassCut = 0.1
    penalCnt = [1, 1, 25, 0.25]
    betaCnt  = [1, 1, 25, 2]
    bcF = 'reflect' if ftBC.upper() == 'N' else 'constant'

    # Mesh
    beamL, beamH, tipMassFrac = _physical_setup(bcType)
    nEl = nelx * nely
    nodeNrs = np.arange((1+nelx)*(1+nely), dtype=np.int32).reshape(
        1+nely, 1+nelx, order='F')
    cVec = (2 * nodeNrs[:-1, :-1] + 2).ravel(order='F')
    offsets = np.array([0,1,2*nely+2,2*nely+3,2*nely,2*nely+1,-2,-1], dtype=np.int32)
    cMat = cVec[:, None] + offsets[None, :]
    nDof = (1+nely) * (1+nelx) * 2

    sI, sII = [], []
    for j in range(8):
        sI.extend(range(j, 8))
        sII.extend([j]*(8-j))
    sI = np.array(sI);  sII = np.array(sII)
    iK = cMat[:, sI].T;  jK = cMat[:, sII].T
    Iar = np.column_stack([np.maximum(iK.ravel('F'), jK.ravel('F')),
                           np.minimum(iK.ravel('F'), jK.ravel('F'))])

    # Element stiffness
    c1 = np.array([12,3,-6,-3,-6,-3,0,3,12,3,0,-3,-6,-3,-6,12,-3,0,-3,-6,3,
                   12,3,-6,3,-6,12,3,-6,-3,12,3,0,12,-3,12], dtype=float)
    c2 = np.array([-4,3,-2,9,2,-3,4,-9,-4,-9,4,-3,2,9,-2,-4,-3,4,9,2,3,-4,
                   -9,-2,3,2,-4,3,-2,9,-4,-9,4,-4,-3,-4], dtype=float)
    Ke_vec = 1.0/(1.0 - nu**2)/24.0 * (c1 + nu*c2)
    Ke0 = np.zeros((8,8))
    Ke0[np.tril_indices(8)] = Ke_vec
    Ke0 = Ke0 + Ke0.T - np.diag(np.diag(Ke0))

    # Element mass
    elemArea = (beamL / nelx) * (beamH / nely)
    MeS = (elemArea/36.0) * np.array([[4,2,1,2],[2,4,2,1],
                                       [1,2,4,2],[2,1,2,4]], dtype=float)
    Me0 = np.kron(MeS, np.eye(2))

    # BCs and loads
    pasS = np.array([], dtype=int);  pasV = np.array([], dtype=int)
    fixed, lcDof, tipMassNode = _bc_and_load(nodeNrs, nely, nelx, nDof, bcType)
    if bcType.lower() == 'fixedpinned':
        lcDof = _fixed_pinned_load_from_solid_mode(
            fixed, nodeNrs, nEl, nDof, Iar, Ke_vec, Me0,
            E0, Emin, penal, rho0, rho_min, dMass, xMassCut)

    F_point = np.zeros(nDof)
    F_point[lcDof] = -1.0

    tipMassDofs = np.array([], dtype=np.int32)
    tipMassVal = 0.0
    if tipMassFrac > 0 and tipMassNode is not None:
        permittedMass = volfrac * beamL * beamH * rho0
        tipMassVal = tipMassFrac * permittedMass
        tipMassDofs = np.array([2*tipMassNode, 2*tipMassNode+1], dtype=np.int32)

    free = np.setdiff1d(np.arange(nDof), fixed)
    act = np.setdiff1d(np.arange(nEl), np.union1d(pasS, pasV))

    # Projection helpers
    def prj(v, et, bt):
        return (np.tanh(bt*et) + np.tanh(bt*(v - et))) / (np.tanh(bt*et) + np.tanh(bt*(1-et)))

    def deta_fn(v, et, bt):
        return (-bt / np.sinh(bt) * (1.0/np.cosh(bt*(v - et)))**2
                * np.sinh(v*bt) * np.sinh((1-v)*bt))

    def dprj(v, et, bt):
        return bt*(1 - np.tanh(bt*(v - et))**2) / (np.tanh(bt*et) + np.tanh(bt*(1-et)))

    def cnt(v, vc, l):
        return v + (l >= vc[0]) * (v < vc[1]) * ((l+1) % vc[2] == 0) * vc[3]

    # Filter
    r = int(np.ceil(rmin))
    dy_f, dx_f = np.meshgrid(np.arange(-r+1, r), np.arange(-r+1, r))
    h = np.maximum(0.0, rmin - np.sqrt(dx_f**2 + dy_f**2))
    Hs = correlate(np.ones((nely, nelx)), h, mode=bcF)

    # Initialization
    x = np.zeros(nEl)
    dsK = np.zeros(nEl)
    dV = np.zeros(nEl)
    dV[act] = 1.0 / nEl / volfrac
    x[act] = (volfrac * (nEl - len(pasV)) - len(pasS)) / len(act)
    x[pasS] = 1.0

    ltMask = np.tril(np.ones((8,8), dtype=bool))
    meLower = Me0[ltMask]

    info = dict(
        stage1=dict(c=[], v=[], ch=[], xHist=[], omegaHist=[], loadDof=lcDof),
        stage2=dict(c=[], v=[], ch=[], xHist=[], omegaHist=[]))

    # === STAGE 1: compliance minimization ===
    xPhys = x.copy();  U = np.zeros(nDof)
    xPhys, U, eta, penal, beta, info['stage1'] = _compliance_loop(
        x.copy(), xPhys, U, F_point, fixed, free, act,
        nelx, nely, nEl, nDof, cMat, Iar, Ke_vec, Ke0,
        E0, Emin, penal, rmin, h, Hs, bcF, ft, eta, beta, move,
        stage1_maxit, penalCnt, betaCnt, dsK.copy(), dV, info['stage1'],
        True, nHistModes, prj, deta_fn, dprj, cnt)
    info['stage1']['xFinal'] = xPhys.copy()
    info['stage1']['UFinal'] = U.copy()
    info['stage1']['omega1'] = _first_omega(
        xPhys, free, nEl, nDof, Iar, Ke_vec, Me0, E0, Emin, penal,
        rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, ltMask, meLower)

    x = xPhys.copy();  U_est = U.copy()

    # === STAGE 2: inertial load loop ===
    xPhys2 = xPhys.copy();  U2 = U.copy()
    xPhys2, U2, eta, penal, beta, info['stage2'] = _inertial_loop(
        x.copy(), xPhys2, U_est, fixed, free, act,
        nelx, nely, nEl, nDof, cMat, Iar, Ke_vec, Ke0, Me0,
        E0, Emin, rho0, rho_min, dMass, xMassCut,
        tipMassDofs, tipMassVal,
        penal, rmin, h, Hs, bcF, ft, eta, beta, move, maxit,
        penalCnt, betaCnt, dsK.copy(), dV, info['stage2'], stage2Tol, nHistModes,
        prj, deta_fn, dprj, cnt, ltMask, meLower)
    info['stage2']['xFinal'] = xPhys2.copy()
    info['stage2']['UFinal'] = U2.copy()
    info['stage2']['omega1'] = _first_omega(
        xPhys2, free, nEl, nDof, Iar, Ke_vec, Me0, E0, Emin, penal,
        rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal, ltMask, meLower)

    if nHistModes > 0:
        info['stage1']['omegaHist'] = _mode_history(
            info['stage1']['xHist'], free, nEl, nDof, Iar, Ke_vec, Me0,
            E0, Emin, penal, rho0, rho_min, dMass, xMassCut,
            tipMassDofs, tipMassVal, nHistModes, ltMask, meLower)
        info['stage2']['omegaHist'] = _mode_history(
            info['stage2']['xHist'], free, nEl, nDof, Iar, Ke_vec, Me0,
            E0, Emin, penal, rho0, rho_min, dMass, xMassCut,
            tipMassDofs, tipMassVal, nHistModes, ltMask, meLower)

    if np.isfinite(info['stage2']['omega1']):
        print(f"\nFinal design: omega1 = {info['stage2']['omega1']:.4f} rad/s")

    return xPhys2, U2, info


# =========================================================================
# Internal loops
# =========================================================================

def _compliance_loop(x, xPhys, U, F, fixed, free, act,
                     nelx, nely, nEl, nDof, cMat, Iar, Ke_vec, Ke0,
                     E0, Emin, penal, rmin, h, Hs, bcF, ft, eta, beta,
                     move, maxit, penalCnt, betaCnt, dsK, dV, stageInfo,
                     doPlot, nHistModes, prj, deta_fn, dprj, cnt):
    tolX = 1e-2
    for loop in range(1, maxit + 1):
        # Physical density
        if ft == 1:
            xPhys[act] = x[act]
        else:
            xTilde = correlate(x.reshape(nely, nelx), h, mode=bcF) / Hs
            xPhys[act] = xTilde.ravel()[act]

        dHs = Hs.copy()
        if ft > 1:
            xTilde_flat = xTilde.ravel() if ft > 1 else x
            if ft == 3:
                f_val = np.mean(prj(xPhys, eta, beta)) - np.mean(xPhys)
                while abs(f_val) > 1e-6:
                    eta = eta - f_val / np.mean(deta_fn(xPhys, eta, beta))
                    f_val = np.mean(prj(xPhys, eta, beta)) - np.mean(xPhys)
            dHs = Hs / dprj(xTilde.ravel(), eta, beta).reshape(nely, nelx)
            xPhys[:] = prj(xPhys, eta, beta)

        # FE solve
        sK_val = Emin + xPhys**penal * (E0 - Emin)
        dsK[act] = -penal * (E0 - Emin) * xPhys[act]**(penal - 1)
        sK = np.outer(Ke_vec, sK_val).ravel('F')
        K = _assemble(Iar[:, 0], Iar[:, 1], sK, nDof)
        K = K + K.T - diags(K.diagonal(), 0, shape=(nDof, nDof), format='csc')
        U[:] = 0.0
        U[free] = spsolve(K[np.ix_(free, free)], F[free])

        # Sensitivities
        pe = U[cMat]
        dc = dsK * np.sum((pe @ Ke0) * pe, axis=1)
        if ft == 1:
            xMat = np.maximum(1e-3, x).reshape(nely, nelx)
            dc = correlate((x * dc).reshape(nely, nelx), h, mode=bcF) / Hs / xMat
            dV0 = correlate((x * dV).reshape(nely, nelx), h, mode=bcF) / Hs / xMat
        else:
            dc = correlate(dc.reshape(nely, nelx) / dHs, h, mode=bcF)
            dV0 = correlate(dV.reshape(nely, nelx) / dHs, h, mode=bcF)

        # OC update
        xT = x[act].copy()
        xU = xT + move;  xL = xT - move
        ocArg = np.maximum(-dc.ravel()[act] / dV0.ravel()[act], 1e-30)
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
        ch = float(np.max(np.abs(x[act] - xT)))

        penal = cnt(penal, penalCnt, loop)
        beta = cnt(beta, betaCnt, loop)

        cVal = float(F @ U)
        stageInfo['c'].append(cVal)
        stageInfo['v'].append(float(np.mean(xPhys)))
        stageInfo['ch'].append(ch)
        if nHistModes > 0:
            stageInfo['xHist'].append(xPhys.copy())

        if doPlot:
            print(f"S1 It.:{loop:5d} C:{cVal:10.4e} V:{np.mean(xPhys):7.3f} "
                  f"ch:{ch:.2e} penal:{penal:5.2f} beta:{beta:5.1f} eta:{eta:6.3f}")
            plt.clf()
            plt.imshow(1 - xPhys.reshape(nely, nelx), cmap='gray', aspect='equal',
                       vmin=0, vmax=1)
            plt.axis('off')
            plt.pause(0.01)

        if loop > 1 and ch < tolX:
            break

    return xPhys, U, eta, penal, beta, stageInfo


def _inertial_loop(x, xPhys, U_est, fixed, free, act,
                   nelx, nely, nEl, nDof, cMat, Iar, Ke_vec, Ke0, Me0,
                   E0, Emin, rho0, rho_min, dMass, xMassCut,
                   tipMassDofs, tipMassVal,
                   penal, rmin, h, Hs, bcF, ft, eta, beta, move, maxit,
                   penalCnt, betaCnt, dsK, dV, stageInfo, stage2Tol,
                   nHistModes, prj, deta_fn, dprj, cnt, ltMask, meLower):
    tolX = stage2Tol
    U = U_est.copy()

    for loop in range(1, maxit + 1):
        # Physical density
        if ft == 1:
            xPhys[act] = x[act]
        else:
            xTilde = correlate(x.reshape(nely, nelx), h, mode=bcF) / Hs
            xPhys[act] = xTilde.ravel()[act]

        dHs = Hs.copy()
        if ft > 1:
            if ft == 3:
                f_val = np.mean(prj(xPhys, eta, beta)) - np.mean(xPhys)
                while abs(f_val) > 1e-6:
                    eta = eta - f_val / np.mean(deta_fn(xPhys, eta, beta))
                    f_val = np.mean(prj(xPhys, eta, beta)) - np.mean(xPhys)
            dHs = Hs / dprj(xTilde.ravel(), eta, beta).reshape(nely, nelx)
            xPhys[:] = prj(xPhys, eta, beta)

        # Assemble K
        sK_val = Emin + xPhys**penal * (E0 - Emin)
        dsK[act] = -penal * (E0 - Emin) * xPhys[act]**(penal - 1)
        sK = np.outer(Ke_vec, sK_val).ravel('F')
        K = _assemble(Iar[:, 0], Iar[:, 1], sK, nDof)
        K = K + K.T - diags(K.diagonal(), 0, shape=(nDof, nDof), format='csc')

        # Assemble M
        rhoe = rho_min + (rho0 - rho_min) * xPhys.copy()
        low = xPhys <= xMassCut
        rhoe[low] = rho_min + (rho0 - rho_min) * xPhys[low]**dMass
        sM = np.outer(meLower, rhoe).ravel('F')
        M = _assemble(Iar[:, 0], Iar[:, 1], sM, nDof)
        M = M + M.T - diags(M.diagonal(), 0, shape=(nDof, nDof), format='csc')
        if tipMassVal > 0 and len(tipMassDofs) > 0:
            M = M + coo_matrix(
                (tipMassVal * np.ones(len(tipMassDofs)),
                 (tipMassDofs, tipMassDofs)), shape=(nDof, nDof)).tocsc()

        # Inertial load
        uhat = U.copy()
        nrm = np.linalg.norm(uhat[free])
        if nrm == 0:
            nrm = 1.0
        uhat /= nrm
        F = M @ uhat
        F[fixed] = 0.0

        # Solve
        U[:] = 0.0
        U[free] = spsolve(K[np.ix_(free, free)], F[free])
        uhatNew = U.copy()
        nrmNew = np.linalg.norm(uhatNew[free])
        if nrmNew == 0:
            nrmNew = 1.0
        uhatNew /= nrmNew
        sgn = np.sign(uhat[free] @ uhatNew[free])
        if sgn == 0:
            sgn = 1.0
        uhatNew *= sgn
        du = np.linalg.norm(uhatNew[free] - uhat[free]) / max(1.0, np.linalg.norm(uhat[free]))

        # Sensitivities
        pe = U[cMat]
        dc = dsK * np.sum((pe @ Ke0) * pe, axis=1)
        if ft == 1:
            xMat = np.maximum(1e-3, x).reshape(nely, nelx)
            dc = correlate((x * dc).reshape(nely, nelx), h, mode=bcF) / Hs / xMat
            dV0 = correlate((x * dV).reshape(nely, nelx), h, mode=bcF) / Hs / xMat
        else:
            dc = correlate(dc.reshape(nely, nelx) / dHs, h, mode=bcF)
            dV0 = correlate(dV.reshape(nely, nelx) / dHs, h, mode=bcF)

        # OC update
        xT = x[act].copy()
        xU = xT + move;  xL = xT - move
        ocArg = np.maximum(-dc.ravel()[act] / dV0.ravel()[act], 1e-30)
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
        ch = float(np.max(np.abs(x[act] - xT)))

        penal = cnt(penal, penalCnt, loop)
        beta = cnt(beta, betaCnt, loop)

        cVal = float(F @ U)
        stageInfo['c'].append(cVal)
        stageInfo['v'].append(float(np.mean(xPhys)))
        stageInfo['ch'].append(ch)
        if nHistModes > 0:
            stageInfo['xHist'].append(xPhys.copy())

        print(f"S2 It.:{loop:5d} C:{cVal:10.4e} V:{np.mean(xPhys):7.3f} "
              f"ch:{ch:.2e} du:{du:.2e} |F|:{np.linalg.norm(F[free]):9.2e} "
              f"penal:{penal:5.2f} beta:{beta:5.1f} eta:{eta:6.3f}")
        plt.clf()
        plt.imshow(1 - xPhys.reshape(nely, nelx), cmap='gray', aspect='equal',
                   vmin=0, vmax=1)
        plt.axis('off')
        plt.pause(0.01)

        if loop > 1 and ch < tolX:
            break

    return xPhys, U, eta, penal, beta, stageInfo


# =========================================================================
# Helpers
# =========================================================================

def _physical_setup(bcType):
    bc = bcType.lower()
    if bc == "cantilever":
        return 15.0, 10.0, 0.20
    elif bc in ("simply", "fixedpinned"):
        return 8.0, 1.0, 0.0
    raise ValueError(f"Unsupported bcType: {bcType}")


def _bc_and_load(nodeNrs, nely, nelx, nDof, bcType):
    """Return (fixed, lcDof, tipMassNode) with 0-based indices."""
    tipMassNode = None
    bc = bcType.lower()
    if bc == "simply":
        midRow = round(nely / 2)
        lm = nodeNrs[midRow, 0];  rm = nodeNrs[midRow, -1]
        fixed = np.array([2*lm, 2*lm+1, 2*rm, 2*rm+1])
        midCol = round(nelx / 2)
        lcNode = nodeNrs[midRow, midCol]
        lcDof = 2*lcNode + 1  # vertical DOF
    elif bc == "cantilever":
        ln = nodeNrs[:, 0]
        fixed = np.union1d(2*ln, 2*ln+1)
        midRow = round((nely + 1) / 2) - 1
        lcNode = nodeNrs[midRow, -1]
        lcDof = 2*lcNode + 1
        tipMassNode = int(lcNode)
    elif bc == "fixedpinned":
        ln = nodeNrs[:, 0]
        fixed = np.union1d(2*ln, 2*ln+1)
        midRow = round(nely / 2)
        rm = nodeNrs[midRow, -1]
        fixed = np.union1d(fixed, [2*rm, 2*rm+1])
        midCol = round(nelx / 2)
        lcNode = nodeNrs[midRow, midCol]
        lcDof = 2*lcNode + 1
    else:
        raise ValueError(f"Unsupported bcType: {bcType}")
    return np.unique(fixed), int(lcDof), tipMassNode


def _fixed_pinned_load_from_solid_mode(fixed, nodeNrs, nEl, nDof, Iar,
                                        Ke_vec, Me0, E0, Emin, penal,
                                        rho0, rho_min, dMass, xMassCut):
    """Determine stage-1 load location from first eigenmode of solid beam."""
    nely = nodeNrs.shape[0] - 1
    ltMask = np.tril(np.ones((8, 8), dtype=bool))
    meLower = Me0[ltMask]
    try:
        xSolid = np.ones(nEl)
        sK_val = Emin + xSolid**penal * (E0 - Emin)
        sK = np.outer(Ke_vec, sK_val).ravel('F')
        K = _assemble(Iar[:, 0], Iar[:, 1], sK, nDof)
        K = K + K.T - diags(K.diagonal(), 0, shape=(nDof, nDof), format='csc')

        rhoe = rho_min + (rho0 - rho_min) * xSolid
        sM = np.outer(meLower, rhoe).ravel('F')
        M = _assemble(Iar[:, 0], Iar[:, 1], sM, nDof)
        M = M + M.T - diags(M.diagonal(), 0, shape=(nDof, nDof), format='csc')

        free = np.setdiff1d(np.arange(nDof), fixed)
        vals, vecs = eigsh(K[np.ix_(free, free)], M=M[np.ix_(free, free)],
                           k=1, sigma=0.0, which='LM', maxiter=1000)
        Ufull = np.zeros(nDof)
        Ufull[free] = np.real(vecs[:, 0])

        allNodes = nodeNrs.ravel()
        vDofs = 2 * allNodes + 1  # vertical DOFs
        freeVDofs = vDofs[~np.isin(vDofs, fixed)]
        idx = np.argmax(np.abs(Ufull[freeVDofs]))
        return int(freeVDofs[idx])
    except Exception:
        midRow = round(nely / 2)
        midCol = round((nodeNrs.shape[1]) / 2)
        return int(2 * nodeNrs[midRow, midCol] + 1)


def _assemble(i, j, s, n):
    return coo_matrix((s, (i.astype(np.int64), j.astype(np.int64))),
                      shape=(n, n)).tocsc()


def _first_omega(xPhys, free, nEl, nDof, Iar, Ke_vec, Me0, E0, Emin, penal,
                 rho0, rho_min, dMass, xMassCut, tipMassDofs, tipMassVal,
                 ltMask, meLower):
    omegas = _first_n_omegas(xPhys, free, nEl, nDof, Iar, Ke_vec, Me0,
                             E0, Emin, penal, rho0, rho_min, dMass, xMassCut,
                             tipMassDofs, tipMassVal, 1, ltMask, meLower)
    return omegas[0]


def _first_n_omegas(xPhys, free, nEl, nDof, Iar, Ke_vec, Me0,
                    E0, Emin, penal, rho0, rho_min, dMass, xMassCut,
                    tipMassDofs, tipMassVal, nModes, ltMask, meLower):
    omegas = np.full(nModes, np.nan)
    if nModes < 1:
        return omegas
    try:
        sK = np.outer(Ke_vec, Emin + xPhys**penal*(E0-Emin)).ravel('F')
        K = _assemble(Iar[:, 0], Iar[:, 1], sK, nDof)
        K = K + K.T - diags(K.diagonal(), 0, shape=(nDof, nDof), format='csc')

        rhoe = rho_min + (rho0 - rho_min) * xPhys.copy()
        low = xPhys <= xMassCut
        rhoe[low] = rho_min + (rho0 - rho_min) * xPhys[low]**dMass
        sM = np.outer(meLower, rhoe).ravel('F')
        M = _assemble(Iar[:, 0], Iar[:, 1], sM, nDof)
        M = M + M.T - diags(M.diagonal(), 0, shape=(nDof, nDof), format='csc')
        if tipMassVal > 0 and len(tipMassDofs) > 0:
            M = M + coo_matrix(
                (tipMassVal*np.ones(len(tipMassDofs)),
                 (tipMassDofs, tipMassDofs)), shape=(nDof, nDof)).tocsc()

        Kff = K[np.ix_(free, free)]
        Mff = M[np.ix_(free, free)]
        nReq = min(nModes, max(1, Kff.shape[0] - 1))
        vals, _ = eigsh(Kff, M=Mff, k=nReq, sigma=0.0, which='LM', maxiter=1000)
        vals = np.sort(np.real(vals))
        vals = vals[vals > 0]
        nOk = min(nReq, len(vals))
        if nOk > 0:
            omegas[:nOk] = np.sqrt(vals[:nOk])
    except Exception:
        omegas[:] = np.nan
    return omegas


def _mode_history(xHist, free, nEl, nDof, Iar, Ke_vec, Me0,
                  E0, Emin, penal, rho0, rho_min, dMass, xMassCut,
                  tipMassDofs, tipMassVal, nModes, ltMask, meLower):
    if not xHist:
        return np.zeros((0, nModes))
    nIter = len(xHist)
    omegaHist = np.full((nIter, nModes), np.nan)
    for k in range(nIter):
        omegaHist[k, :] = _first_n_omegas(
            xHist[k], free, nEl, nDof, Iar, Ke_vec, Me0,
            E0, Emin, penal, rho0, rho_min, dMass, xMassCut,
            tipMassDofs, tipMassVal, nModes, ltMask, meLower)
    return omegaHist


if __name__ == "__main__":
    xPhys, _, info = top99neo_inertial_freq(
        nelx=320, nely=40, volfrac=0.5, penal=3, rmin=2.5, ft=1,
        maxit=200, stage1_maxit=200, bcType="simply")
    plt.figure()
    plt.imshow(1 - xPhys.reshape(40, 320), cmap='gray', aspect='equal',
               vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f"omega1 = {info['stage2']['omega1']:.1f} rad/s")
    plt.tight_layout()
    plt.show()
