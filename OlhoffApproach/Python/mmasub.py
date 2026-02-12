"""
mmasub.py - MMA main subroutine.

Version September 2007 (and a small change August 2008).
Krister Svanberg <krille@math.kth.se>
Department of Mathematics, SE-10044 Stockholm, Sweden.

Translated from Matlab to Python.
"""

import numpy as np
from scipy.sparse import diags as spdiags
try:
    from .subsolv import subsolv
except ImportError:
    from subsolv import subsolv


def mmasub(m, n, iter_, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d):
    """
    Perform one MMA iteration.

    Parameters
    ----------
    m : int       - number of constraints
    n : int       - number of design variables
    iter_ : int   - current iteration (1 on first call)
    xval, xmin, xmax, xold1, xold2 : 1-D arrays of length n
    f0val : float - objective value
    df0dx : 1-D array of length n - objective gradient
    fval  : 1-D array of length m - constraint values
    dfdx  : ndarray (m, n) or scipy sparse - constraint Jacobian
    low, upp : 1-D arrays of length n - previous asymptotes
    a0 : float
    a, c, d : 1-D arrays of length m

    Returns
    -------
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
    """
    xval = xval.ravel();  xmin = xmin.ravel();  xmax = xmax.ravel()
    xold1 = xold1.ravel();  xold2 = xold2.ravel()
    df0dx = df0dx.ravel()
    fval = fval.ravel()
    a = a.ravel();  c = c.ravel();  d = d.ravel()
    low = low.ravel();  upp = upp.ravel()

    epsimin = 1e-7
    raa0 = 0.00001
    move = 1.0
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.7
    eeen = np.ones(n)
    zeron = np.zeros(n)

    # Asymptotes
    if iter_ < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 0.2 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 0.2 * (xmax - xmin)
        low = np.maximum(low, lowmin);  low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax);  upp = np.maximum(upp, uppmin)

    # Bounds alfa, beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)

    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)

    # p0, q0
    xmami = np.maximum(xmax - xmin, 0.00001 * eeen)
    xmamiinv = eeen / xmami
    ux1 = upp - xval;  ux2 = ux1 * ux1
    xl1 = xval - low;  xl2 = xl1 * xl1
    uxinv = eeen / ux1;  xlinv = eeen / xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    # P, Q  (sparse m x n)
    if hasattr(dfdx, 'toarray'):
        dfdx_dense = dfdx.toarray()
    else:
        dfdx_dense = np.asarray(dfdx)
    P = np.maximum(dfdx_dense, 0.0)
    Q = np.maximum(-dfdx_dense, 0.0)
    eeem = np.ones(m)
    PQ = 0.001 * (P + Q) + raa0 * np.outer(eeem, xmamiinv)
    P = P + PQ;  Q = Q + PQ

    from scipy.sparse import csr_matrix
    P_sp = csr_matrix(P) @ spdiags(ux2, offsets=0, shape=(n, n))
    Q_sp = csr_matrix(Q) @ spdiags(xl2, offsets=0, shape=(n, n))
    b = P_sp @ uxinv + Q_sp @ xlinv - fval

    xmma, ymma, zmma, lam, xsi, eta_out, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P_sp, Q_sp, a0, a, b, c, d)

    return xmma, ymma, zmma, lam, xsi, eta_out, mu, zet, s, low, upp
