"""Python mirror of the Du & Olhoff (2007) 2D beam FE model used by analysis/Olhoff.

Mirrors, line for line where it matters:
  * +impl/fem/model2D.m      geometry, top88 node numbering, BCs ('a' SS / 'b' CS / 'c' CC,
                             support 'mid'|'corner'|'face', axial 'one'|'both')
  * +impl/fem/elemMats2D.m   Q4 plane stress, 2x2 Gauss, consistent mass
  * +impl/fem/assemble2D.m   SIMP or Pedersen stiffness, eq.(2)/(4)/(4a)/(4b) mass
  * +impl/algo/genGrad.m     generalized gradients f_sk = phi_s'(dK - lam dM)phi_k
  * +impl/filter/prepFilter.m, applyFilter.m   top88 sensitivity filter
  * +impl/fem/eigSolve.m     shift-invert ARPACK, M-orthonormal modes

Element ordering is column-major e = elx*nely + ely (0-based), as in MATLAB.
Equivalence with the MATLAB model is checked by run_olhoff_nso_vs_mma.py against the
initial eigenfrequencies recorded by the production solver.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# --------------------------------------------------------------------------- #
def elem_mats_q4(dx, dy, E, nu, rhom, t):
    D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    g = 1 / np.sqrt(3)
    gp = np.array([[-g, -g], [g, -g], [g, g], [-g, g]])
    detJ = dx * dy / 4
    xn = np.array([-1, 1, 1, -1]); yn = np.array([-1, -1, 1, 1])
    K = np.zeros((8, 8)); M = np.zeros((8, 8))
    for xi, eta in gp:
        N = 0.25 * (1 + xn * xi) * (1 + yn * eta)
        dNdx = 0.25 * xn * (1 + yn * eta) * (2 / dx)
        dNdy = 0.25 * (1 + xn * xi) * yn * (2 / dy)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx; B[1, 1::2] = dNdy; B[2, 0::2] = dNdy; B[2, 1::2] = dNdx
        K += B.T @ D @ B * detJ * t
        Nm = np.zeros((2, 8)); Nm[0, 0::2] = N; Nm[1, 1::2] = N
        M += Nm.T @ Nm * rhom * detJ * t
    return 0.5 * (K + K.T), 0.5 * (M + M.T)


class OlhoffBeam:
    """FE model + material laws + filter + eigen-solver + generalized gradients."""

    def __init__(self, nelx=160, nely=20, a=8.0, b=1.0, t=1.0, E=1e7, nu=0.3, rhom=1.0,
                 bc="a", support="mid", axial="both",
                 stiffness="simp", p=3.0, linear_below=0.1,
                 mass="eq4b", mass_cutoff=0.1, mass_r=6,
                 rmin_phys=None, rmin_el=3.0, rhomin=1e-3):
        self.nelx, self.nely = nelx, nely
        self.dx, self.dy = a / nelx, b / nely
        self.nele = nelx * nely
        self.nnode = (nelx + 1) * (nely + 1)
        self.ndof = 2 * self.nnode
        self.Ve = self.dx * self.dy * t
        self.stiffness, self.p, self.linear_below = stiffness, p, linear_below
        self.mass, self.mass_cutoff, self.mass_r = mass, mass_cutoff, mass_r
        self.rhomin = rhomin

        # top88 connectivity (1-based in MATLAB; 0-based here)
        nodenrs = np.arange(1, self.nnode + 1).reshape((nely + 1, nelx + 1), order="F")
        edofVec = (2 * nodenrs[:-1, :-1] + 1).reshape(-1, order="F")
        self.edofMat = (edofVec[:, None] + np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely + 0, 2 * nely + 1, -2, -1])[None, :]) - 1
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).ravel().astype(int)
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).ravel().astype(int)
        self.nodenrs = nodenrs

        self.K0, self.M0 = elem_mats_q4(self.dx, self.dy, E, nu, rhom, t)

        # boundary conditions (model2D.m)
        def dofs(nodes, comp):  # nodes 1-based, comp 1=ux 2=uy -> 0-based dof
            return 2 * np.asarray(nodes).ravel() - 2 + comp - 1
        if support == "mid":
            assert nely % 2 == 0
            rowSS = nely // 2          # 0-based row index
        elif support == "corner":
            rowSS = nely
        else:
            rowSS = None
        left, right = 0, nelx
        clamp = lambda col: np.concatenate((dofs(nodenrs[:, col], 1), dofs(nodenrs[:, col], 2)))
        if support == "face":
            ssY = lambda col: dofs(nodenrs[:, col], 2); ssX = lambda col: dofs(nodenrs[:, col], 1)
        else:
            ssY = lambda col: dofs(nodenrs[rowSS, col], 2); ssX = lambda col: dofs(nodenrs[rowSS, col], 1)
        if bc == "a":
            fixed = [ssY(left), ssY(right), ssX(left)] + ([ssX(right)] if axial == "both" else [])
        elif bc == "b":
            fixed = [clamp(left), ssY(right)] + ([ssX(right)] if axial == "both" else [])
        elif bc == "c":
            fixed = [clamp(left), clamp(right)]
        else:
            raise ValueError(bc)
        self.fixed = np.unique(np.concatenate([np.atleast_1d(f) for f in fixed]))
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)

        # filter (prepFilter.m), radius in element units
        rmin = rmin_el if rmin_phys is None else rmin_phys / self.dy
        self.rmin = rmin
        self._prep_filter(rmin)

    # ---- material laws ----------------------------------------------------
    def stiff_interp(self, rho):
        p = self.p
        g, dg = rho**p, p * rho**(p - 1)
        if self.stiffness == "pedersen":
            lo = rho < self.linear_below
            c = self.linear_below**(p - 1)
            g = np.where(lo, c * rho, g); dg = np.where(lo, c, dg)
        return g, dg

    def mass_interp(self, rho):
        c = self.mass_cutoff
        if self.mass in ("lin", "eq2"):
            return rho.copy(), np.ones_like(rho)
        lo = rho <= c
        g, dg = rho.copy(), np.ones_like(rho)
        if self.mass == "eq4":
            r = self.mass_r
            g[lo] = rho[lo]**r; dg[lo] = r * rho[lo]**(r - 1)
        elif self.mass == "eq4a":
            g[lo] = 1e5 * rho[lo]**6; dg[lo] = 6e5 * rho[lo]**5
        elif self.mass == "eq4b":
            g[lo] = 6e5 * rho[lo]**6 - 5e6 * rho[lo]**7
            dg[lo] = 36e5 * rho[lo]**5 - 35e6 * rho[lo]**6
        else:
            raise ValueError(self.mass)
        return g, dg

    # ---- assembly / eigen -------------------------------------------------
    def assemble(self, rho):
        gK, _ = self.stiff_interp(rho); gM, _ = self.mass_interp(rho)
        sK = (self.K0.reshape(-1, 1) * gK[None, :]).ravel(order="F")
        sM = (self.M0.reshape(-1, 1) * gM[None, :]).ravel(order="F")
        K = sp.coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        M = sp.coo_matrix((sM, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        K = 0.5 * (K + K.T); M = 0.5 * (M + M.T)
        f = self.free
        return K[f][:, f], M[f][:, f]

    def eig(self, rho, J=6, sigma=1e-6):
        K, M = self.assemble(rho)
        n = K.shape[0]
        v0 = np.sin(np.arange(1, n + 1) * 0.7071067811865476) + 0.5    # eigSolve.m fixed start
        lam, Phi = spla.eigsh(K, k=J, M=M, sigma=sigma, which="LM", v0=v0, tol=1e-12,
                              maxiter=5000, ncv=min(n, max(20, 4 * J)))
        order = np.argsort(lam); lam = lam[order]; Phi = Phi[:, order]
        for j in range(J):
            Phi[:, j] /= np.sqrt(Phi[:, j] @ (M @ Phi[:, j]))
        return lam, Phi

    # ---- generalized gradients (genGrad.m) --------------------------------
    def gen_grad(self, rho, Phi, lam_tilde, idx):
        """F[:, s, k] = phi_s'(dK/drho_e - lam_tilde dM/drho_e) phi_k, elementwise."""
        idx = np.atleast_1d(idx)
        nI = len(idx)
        U = np.zeros((self.ndof, nI)); U[self.free, :] = Phi[:, idx]
        _, dgK = self.stiff_interp(rho); _, dgM = self.mass_interp(rho)
        sM = lam_tilde * dgM
        Ue = [U[self.edofMat, s] for s in range(nI)]            # nele x 8 each
        F = np.zeros((self.nele, nI, nI))
        for s in range(nI):
            UsK = Ue[s] @ self.K0; UsM = Ue[s] @ self.M0
            for k in range(s, nI):
                f = dgK * np.sum(UsK * Ue[k], 1) - sM * np.sum(UsM * Ue[k], 1)
                F[:, s, k] = f; F[:, k, s] = f
        return F

    # ---- filter -----------------------------------------------------------
    def _prep_filter(self, rmin):
        nelx, nely = self.nelx, self.nely
        r = int(np.ceil(rmin)) - 1
        rows, cols, vals = [], [], []
        for i1 in range(nelx):
            for j1 in range(nely):
                e1 = i1 * nely + j1
                for i2 in range(max(i1 - r, 0), min(i1 + r, nelx - 1) + 1):
                    for j2 in range(max(j1 - r, 0), min(j1 + r, nely - 1) + 1):
                        rows.append(e1); cols.append(i2 * nely + j2)
                        vals.append(max(0.0, rmin - np.hypot(i1 - i2, j1 - j2)))
        self.H = sp.csr_matrix((vals, (rows, cols)), shape=(self.nele, self.nele))
        self.Hs = np.asarray(self.H.sum(1)).ravel()

    def sens_filter(self, rho, df):
        """top88 ft=1 sensitivity filter with the published max(1e-3, rho) guard."""
        den = self.Hs * np.maximum(1e-3, rho)
        return (self.H @ (rho * df)) / den

    # ---- misc ---------------------------------------------------------------
    def grayness(self, rho):
        return float(np.mean(4 * rho * (1 - rho)))

    def rho_image(self, rho):
        return rho.reshape((self.nely, self.nelx), order="F")
