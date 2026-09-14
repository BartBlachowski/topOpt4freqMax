"""
Independent re-implementation of the COMMON POST-HOC EVALUATOR spectral core.

Written from the mathematical specification in
analysis/three_method_parametric_study/study_evaluate_design.m (frozen, Eq. 4)
and analysis/.../+ie2d/study_evaluate_design_eq4a.m (amended, Eq. 4a),
in a different language, with a different sparse eigensolver, for the
Phase-2E independent delta audit.  READ-ONLY.  No optimizer is involved.

Only omega_raw_E{1,2,3}(1) is reproduced -- the quantity Q(k) that the frozen
methodology actually consumes.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import lru_cache


@lru_cache(maxsize=8)
def mesh_data(nelx, nely):
    hx, hy, nu, t = 8.0 / nelx, 1.0 / nely, 0.3, 1.0
    D = (1.0 / (1 - nu ** 2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 0.5 * (1 - nu)]])
    invJ = np.array([[2.0 / hx, 0.0], [0.0, 2.0 / hy]])
    detJ = 0.25 * hx * hy
    gp = 1.0 / np.sqrt(3.0)
    KE = np.zeros((8, 8))
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            a = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            b = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            d = invJ @ np.vstack([a, b])
            B = np.zeros((3, 8))
            B[0, 0::2] = d[0, :]
            B[1, 1::2] = d[1, :]
            B[2, 0::2] = d[1, :]
            B[2, 1::2] = d[0, :]
            KE = KE + B.T @ D @ B * detJ
    KE = t * KE
    Ms = (hx * hy / 36.0) * np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], float)
    ME = t * np.kron(Ms, np.eye(2))

    nEl = nelx * nely
    edof = np.zeros((nEl, 8), dtype=np.int64)
    for ex in range(nelx):
        for ey in range(nely):
            e = ey + ex * nely
            n1 = (nely + 1) * ex + ey
            n2 = (nely + 1) * (ex + 1) + ey
            edof[e, :] = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                          2 * (n2 + 1), 2 * (n2 + 1) + 1, 2 * (n1 + 1), 2 * (n1 + 1) + 1]

    ndof = 2 * (nelx + 1) * (nely + 1)
    jMid = int(np.round(nely / 2.0))
    nL = jMid
    nR = nelx * (nely + 1) + jMid
    fixed = np.array([2 * nL, 2 * nL + 1, 2 * nR, 2 * nR + 1])
    free = np.setdiff1d(np.arange(ndof), fixed)

    rows = np.repeat(edof, 8, axis=1).ravel()          # d[c] pattern
    cols = np.tile(edof, (1, 8)).ravel()               # d[r] pattern
    # value order must match: element-major, then KE column-major
    KEv = KE.ravel(order='F')                          # KE(:)  (r fastest)
    MEv = ME.ravel(order='F')
    # rows/cols above pair (repeat, tile); KE(:) index m -> r=m%8, c=m//8
    # repeat gives d[m//8] and tile gives d[m%8]  => (d[c], d[r], KE(r,c)) : symmetric, OK
    return dict(KE=KE, ME=ME, KEv=KEv, MEv=MEv, rows=rows, cols=cols,
                ndof=ndof, free=free, nEl=nEl)


def mass_g(z, eq4a):
    """Du & Olhoff (2007) Eq. (4) or Eq. (4a) applied to a density vector."""
    g = z.copy()
    low = z <= 0.1
    g[low] = (1e5 if eq4a else 1.0) * z[low] ** 6
    return g


def interp(z, model, eq4a):
    if model == 'E1':
        return 1e7 * (1e-6 + (1 - 1e-6) * z ** 3), 1e-6 + (1 - 1e-6) * z
    if model == 'E2':
        return 1e7 * (1e-9 + (1 - 1e-9) * z ** 3), 1e-9 + (1 - 1e-9) * mass_g(z, eq4a)
    if model == 'E3':
        z3 = np.maximum(z, 1e-3)
        return 1e7 * z3 ** 3, mass_g(z3, eq4a)
    raise ValueError(model)


def omega1(x, nelx, nely, model, eq4a=False, k=3):
    md = mesh_data(nelx, nely)
    z = np.clip(np.asarray(x, dtype=np.float64).ravel(), 0.0, 1.0)
    assert z.size == md['nEl']
    Ee, rr = interp(z, model, eq4a)
    ndof, free = md['ndof'], md['free']
    Kv = (md['KEv'][None, :] * Ee[:, None]).ravel()
    Mv = (md['MEv'][None, :] * rr[:, None]).ravel()
    K = sp.coo_matrix((Kv, (md['rows'], md['cols'])), shape=(ndof, ndof)).tocsr()
    M = sp.coo_matrix((Mv, (md['rows'], md['cols'])), shape=(ndof, ndof)).tocsr()
    K = ((K + K.T) * 0.5)[free][:, free].tocsc()
    M = ((M + M.T) * 0.5)[free][:, free].tocsc()
    # shift-invert at sigma=0 -> smallest eigenvalues, direct factorisation,
    # deliberately a different algorithm from the MATLAB reference eigs() call.
    v0 = np.random.default_rng(20260830).standard_normal(K.shape[0])
    v0 /= np.linalg.norm(v0)
    vals = spla.eigsh(K, k=k, M=M, sigma=0.0, which='LM', v0=v0,
                      return_eigenvectors=False, tol=0.0, maxiter=100000)
    lam = np.sort(np.real(vals))
    return float(np.sqrt(max(lam[0], 0.0)))


def Qvec(x, nelx, nely, eq4a=False):
    return np.array([omega1(x, nelx, nely, m, eq4a) for m in ('E1', 'E2', 'E3')])
