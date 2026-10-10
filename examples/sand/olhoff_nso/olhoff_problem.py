"""Du & Olhoff eigenfrequency maximization as a nullspace_optimizer `Optimizable`.

Bound formulation (Du & Olhoff 2007, problem (25) without the nested increment
subproblem):

    x = [rho_1..rho_NE, beta]
    min  J(x) = -beta
    s.t. h_j = beta - lambda_j(rho)/lamref <= 0,     j = 1..Jc      (eigenvalue bounds)
         rhomin - rho_e <= 0,  rho_e - 1 <= 0                        (box)
         (sum rho_e - volfrac*NE)/(volfrac*NE) <= 0                  (volume, (25e))
    optional equalities  f_sk' xi = 0  for near-degenerate pairs (s,k):
         the Krog & Olhoff (1999) route, eq. (22), imposed on the DIRECTION by
         declaring g_sk = 0 with Jacobian row f_sk (value identically zero, so the
         range-space step ignores it and the null-space step is tangent to it).

Sensitivities are the generalized gradients f_jj of eq. (19) with the Sigmund (1997)
sensitivity filter, exactly as analysis/Olhoff applies them (filterMode 'diag').
The eigenvalue Jacobian is DENSE (Jc x NE); nothing here benefits from the sparse
machinery of the SAND paper.  That is the point of the experiment.

The same class drives the package's MMA (`mma_solve`) for a single-loop control.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import scipy.sparse as sp
from nullspace_optimizer import Optimizable

from olhoff_fe import OlhoffBeam


class OlhoffFreqProblem(Optimizable):
    def __init__(self, model: OlhoffBeam, volfrac=0.5, rho0=0.5, Jc=4, n_target=1,
                 filter_sens=True, bounds_in_H=True, offdiag_equality=False, gap_tol=0.02,
                 metric="euclid", beta_weight=1.0, helmholtz_radius_el=None, subN=2):
        self.m = model
        self.NE = model.nele
        self.volfrac = volfrac
        self.Jc = Jc
        self.n_target = n_target          # maximize the n-th eigenfrequency (1 = fundamental)
        self.filter_sens = filter_sens
        self.bounds_in_H = bounds_in_H
        self.offdiag_equality = offdiag_equality
        self.gap_tol = gap_tol
        self.subN = subN
        self.rhomin = model.rhomin
        self.rho_init = rho0 * np.ones(self.NE)
        lam0, _ = model.eig(self.rho_init, Jc)
        self.lamref = float(lam0[n_target - 1])     # fixed normalization, as innerLoop.m's lamref
        self.beta_weight = beta_weight
        self._cache = {}
        self.n_eig = 0
        self.t_eig = 0.0
        # metric on (rho, beta)
        if metric == "euclid":
            A_rho = sp.eye(self.NE, format="csc")
        elif metric == "helmholtz":
            r = helmholtz_radius_el if helmholtz_radius_el is not None else model.rmin / 2
            A_rho = self._helmholtz(r)
        else:
            raise ValueError(metric)
        self.A = sp.block_diag((A_rho, sp.csc_matrix([[beta_weight]])), format="csc")

    # ---- helpers ------------------------------------------------------------
    def _helmholtz(self, r_el):
        """(I + r^2 L) on the element grid, L the 5-point graph Laplacian (element units)."""
        m = self.m
        nelx, nely = m.nelx, m.nely
        e = np.arange(m.nele).reshape((nely, nelx), order="F")
        rows, cols = [], []
        for di, dj in ((0, 1), (1, 0)):
            a = e[: nely - dj, : nelx - di].ravel(); b = e[dj:, di:].ravel()
            rows += [a, b]; cols += [b, a]
        rows = np.concatenate(rows); cols = np.concatenate(cols)
        W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(m.nele, m.nele))
        L = sp.diags(np.asarray(W.sum(1)).ravel()) - W
        return (sp.eye(m.nele) + r_el**2 * L).tocsc()

    def split(self, x):
        return x[: self.NE], float(x[self.NE])

    def _solve(self, x):
        key = hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()
        if key not in self._cache:
            rho, _ = self.split(x)
            t = time.time()
            lam, Phi = self.m.eig(rho, self.Jc)
            self.t_eig += time.time() - t; self.n_eig += 1
            idx = np.arange(self.Jc)
            F = self.m.gen_grad(rho, Phi, None, idx) if False else None
            # diagonal gradients with each mode's own eigenvalue (eq. 24 / genGrad with lamTilde = lam_j)
            fdiag = np.zeros((self.NE, self.Jc))
            for j in range(self.Jc):
                fdiag[:, j] = self.m.gen_grad(rho, Phi, lam[j], [j])[:, 0, 0]
            foff = {}
            if self.offdiag_equality:
                # FIXED pair set among the first subN modes (production preset: N fixed at 2,
                # no detection), because nlspace_solve requires a constant constraint count.
                lam_t = lam[self.n_target - 1]
                for s in range(self.subN):
                    for k in range(s + 1, self.subN):
                        foff[(s, k)] = self.m.gen_grad(rho, Phi, lam_t, [s, k])[:, 0, 1]
            if self.filter_sens:
                for j in range(self.Jc):
                    fdiag[:, j] = self.m.sens_filter(rho, fdiag[:, j])
                foff = {k: self.m.sens_filter(rho, v) for k, v in foff.items()}
            if len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = dict(lam=lam, fdiag=fdiag, foff=foff)
        return self._cache[key]

    def omegas(self, x):
        return np.sqrt(np.maximum(self._solve(x)["lam"], 0))

    # ---- Optimizable interface ---------------------------------------------
    def set_initial(self, rho):
        """Warm start from a given density; beta starts at that design's lambda_1/lamref."""
        self.rho_init = np.asarray(rho, dtype=float).copy()
        lam, _ = self.m.eig(self.rho_init, self.Jc)
        self.beta_init = float(lam[self.n_target - 1] / self.lamref)

    def x0(self):
        return np.concatenate((self.rho_init, [getattr(self, "beta_init", 1.0)]))     # beta = lambda_1(rho0)/lamref

    J_OFFSET = 10.0   # J = 10 - beta: keeps J > 0 (mma_solve divides by J/10, which flips the sign if J < 0)

    def J(self, x):
        return self.J_OFFSET - self.split(x)[1]

    def dJ(self, x):
        d = np.zeros(self.NE + 1); d[-1] = -1.0
        return d

    def G(self, x):
        return np.zeros(len(self._solve(x)["foff"]))

    def dG(self, x):
        foff = self._solve(x)["foff"]
        if not foff:
            return sp.csr_matrix((0, self.NE + 1))
        rows = [np.concatenate((v / self.lamref, [0.0])) for v in foff.values()]
        return sp.csr_matrix(np.vstack(rows))

    def H(self, x):
        rho, beta = self.split(x)
        lam = self._solve(x)["lam"]
        h = [beta - lam / self.lamref]
        if self.bounds_in_H:
            h += [self.rhomin - rho, rho - 1.0]
        h += [[(rho.sum() - self.volfrac * self.NE) / (self.volfrac * self.NE)]]
        return np.concatenate(h)

    def dH(self, x):
        fdiag = self._solve(x)["fdiag"]
        blocks = [sp.hstack((sp.csr_matrix(-fdiag.T / self.lamref), sp.csr_matrix(np.ones((self.Jc, 1)))))]
        if self.bounds_in_H:
            I = sp.eye(self.NE, format="csr"); z = sp.csr_matrix((self.NE, 1))
            blocks += [sp.hstack((-I, z)), sp.hstack((I, z))]
        blocks += [sp.csr_matrix(np.concatenate((np.ones(self.NE) / (self.volfrac * self.NE), [0.0]))[None, :])]
        return sp.vstack(blocks, format="csc")

    def inner_product(self, x):
        return self.A

    def retract(self, x, dx):
        y = x + dx
        y[: self.NE] = np.clip(y[: self.NE], self.rhomin, 1.0)
        return y

    stop_omega1 = None     # predictor mode: stop as soon as omega_1 >= stop_omega1

    def accept(self, params, results):
        x = results["x"][-1]
        rho, beta = self.split(x)
        w = self.omegas(x)
        params["normalisation_norm"] = lambda v: np.linalg.norm(v[: self.NE], np.inf)
        if self.stop_omega1 is not None and w[0] >= self.stop_omega1:
            params["maxit"] = results["it"][-1]       # the loop tests it < maxit: stops now
        for k, v in (("omega1", w[0]), ("omega2", w[1]), ("omega3", w[2] if len(w) > 2 else np.nan),
                     ("beta_omega", np.sqrt(max(beta * self.lamref, 0))), ("vol", rho.mean()),
                     ("Mnd", self.m.grayness(rho)), ("n_eig", self.n_eig), ("t_eig", self.t_eig)):
            results.setdefault(k, []).append(float(v))
        if len(results["x"]) >= 2:
            results.setdefault("drho_l2", []).append(float(np.linalg.norm(results["x"][-1][: self.NE] - results["x"][-2][: self.NE])))
            results.setdefault("drho_max", []).append(float(np.max(np.abs(results["x"][-1][: self.NE] - results["x"][-2][: self.NE]))))
        else:
            results.setdefault("drho_l2", []).append(np.nan); results.setdefault("drho_max", []).append(np.nan)
