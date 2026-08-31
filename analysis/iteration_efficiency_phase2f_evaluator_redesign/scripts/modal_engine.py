"""
Phase-2F modal diagnostic engine.

Built on the mathematical specification of the frozen common evaluator
(analysis/three_method_parametric_study/study_evaluate_design.m), extended to
return EIGENVECTORS and per-mode energy-localisation diagnostics that the frozen
evaluator does not compute.

READ-ONLY.  No optimizer.  No methodology change.  Nothing here is a proposal:
every threshold that appears is a sweep variable, never a frozen constant.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import lru_cache

# ---------------------------------------------------------------- mesh / FE ---

@lru_cache(maxsize=16)
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
            B[0, 0::2] = d[0, :]; B[1, 1::2] = d[1, :]
            B[2, 0::2] = d[1, :]; B[2, 1::2] = d[0, :]
            KE += B.T @ D @ B * detJ
    KE *= t
    Ms = (hx * hy / 36.0) * np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], float)
    ME = t * np.kron(Ms, np.eye(2))

    nEl = nelx * nely
    ex = np.repeat(np.arange(nelx), nely)
    ey = np.tile(np.arange(nely), nelx)
    n1 = (nely + 1) * ex + ey
    n2 = (nely + 1) * (ex + 1) + ey
    edof = np.column_stack([2*n1, 2*n1+1, 2*n2, 2*n2+1,
                            2*(n2+1), 2*(n2+1)+1, 2*(n1+1), 2*(n1+1)+1]).astype(np.int64)

    ndof = 2 * (nelx + 1) * (nely + 1)
    jMid = int(np.round(nely / 2.0))
    fixed = np.array([2*jMid, 2*jMid+1,
                      2*(nelx*(nely+1)+jMid), 2*(nelx*(nely+1)+jMid)+1])
    free = np.setdiff1d(np.arange(ndof), fixed)

    rows = np.repeat(edof, 8, axis=1).ravel()
    cols = np.tile(edof, (1, 8)).ravel()
    return dict(KE=KE, ME=ME, KEv=KE.ravel(order='F'), MEv=ME.ravel(order='F'),
                rows=rows, cols=cols, edof=edof, ndof=ndof, free=free, nEl=nEl,
                element_area=(8.0 * 1.0) / (nelx * nely))

# --------------------------------------------------------- interpolation laws --

MASS_LAWS = {
    'linear': lambda x: x,
    'eq4':    lambda x: np.where(x <= 0.1, x ** 6, x),
    'eq4a':   lambda x: np.where(x <= 0.1, 1e5 * x ** 6, x),
    'eq4b':   lambda x: np.where(x <= 0.1, 6e5 * x ** 6 - 5e6 * x ** 7, x),
}

# The mass law each evaluator carries in the FROZEN common evaluator.
# E1 is linear and has no low-density branch at all, so it is untouched by any
# Eq. (4)/(4a)/(4b) question.  Passing a different law to E1 produces a
# DIAGNOSTIC VARIANT that is not the frozen E1; callers must label it as such.
FROZEN_LAW = {'E1': 'linear', 'E2': 'eq4', 'E3': 'eq4'}

def interp(z, model, law):
    """Frozen stiffness/floor conventions; only the low-density mass law varies."""
    if model == 'E1':
        return 1e7 * (1e-6 + (1 - 1e-6) * z ** 3), 1e-6 + (1 - 1e-6) * MASS_LAWS[law](z), z
    if model == 'E2':
        return 1e7 * (1e-9 + (1 - 1e-9) * z ** 3), 1e-9 + (1 - 1e-9) * MASS_LAWS[law](z), z
    if model == 'E3':
        z3 = np.maximum(z, 1e-3)
        return 1e7 * z3 ** 3, MASS_LAWS[law](z3), z3
    raise ValueError(model)

def exact_count_binary(x, volfrac=0.5):
    """Frozen rule: exact solid count, density descending, ties by increasing index."""
    x = np.asarray(x, float).ravel(); n = x.size
    nS = int(round(volfrac * n))
    order = np.lexsort((np.arange(n), -x))
    xb = np.zeros(n); xb[order[:nS]] = 1.0
    return xb

# ------------------------------------------------------------- modal solution --

def modes(x, nelx, nely, model, law, k=6, return_vectors=False):
    """
    Lowest k eigenpairs plus per-mode energy diagnostics.

    Returns a dict with, per mode: omega, lambda, frequency (Hz), and the
    element-wise kinetic- and strain-energy distributions reduced to the
    localisation measures defined in MODAL_DIAGNOSTIC_DEFINITIONS.md.
    """
    md = mesh_data(nelx, nely)
    z = np.clip(np.asarray(x, float).ravel(), 0.0, 1.0)
    Ee, rr, zeff = interp(z, model, law)
    ndof, free = md['ndof'], md['free']

    K = sp.coo_matrix(((md['KEv'][None, :] * Ee[:, None]).ravel(),
                       (md['rows'], md['cols'])), shape=(ndof, ndof)).tocsr()
    M = sp.coo_matrix(((md['MEv'][None, :] * rr[:, None]).ravel(),
                       (md['rows'], md['cols'])), shape=(ndof, ndof)).tocsr()
    Kf = ((K + K.T) * 0.5)[free][:, free].tocsc()
    Mf = ((M + M.T) * 0.5)[free][:, free].tocsc()

    v0 = np.random.default_rng(20260830).standard_normal(Kf.shape[0]); v0 /= np.linalg.norm(v0)
    lam, V = spla.eigsh(Kf, k=k, M=Mf, sigma=0.0, which='LM', v0=v0,
                        tol=0.0, maxiter=200000)
    order = np.argsort(lam.real)
    lam = lam.real[order]; V = V[:, order]

    edof = md['edof']; KEe = md['KE']; MEe = md['ME']
    U = np.zeros((ndof, V.shape[1])); U[free, :] = V
    Ue = U[edof, :]                                   # (nEl, 8, k)
    # per-element kinetic and strain energy of each mode
    ke = rr[:, None] * np.einsum('eik,ij,ejk->ek', Ue, MEe, Ue)
    se = Ee[:, None] * np.einsum('eik,ij,ejk->ek', Ue, KEe, Ue)
    ke = np.maximum(ke, 0.0); se = np.maximum(se, 0.0)
    ke_tot = ke.sum(axis=0); se_tot = se.sum(axis=0)
    ke_n = ke / np.where(ke_tot > 0, ke_tot, 1.0)
    se_n = se / np.where(se_tot > 0, se_tot, 1.0)

    out = dict(lam=lam, omega=np.sqrt(np.maximum(lam, 0.0)),
               freq=np.sqrt(np.maximum(lam, 0.0)) / (2 * np.pi),
               ke_tot=ke_tot, se_tot=se_tot, ke_n=ke_n, se_n=se_n,
               z=z, zeff=zeff, mass=rr, stiff=Ee, nEl=md['nEl'])
    if return_vectors:
        out['U'] = U
    return out

# ------------------------------------------------- localisation diagnostics ---

def localisation(res, tau):
    """
    Fraction of modal energy carried by elements with effective density <= tau.
    `tau` is a SWEEP VARIABLE, never a frozen constant.  Returns, per mode:
      ke_low  kinetic-energy share below tau
      se_low  strain-energy share below tau
    """
    low = res['zeff'] <= tau
    return res['ke_n'][low, :].sum(axis=0), res['se_n'][low, :].sum(axis=0)

def density_weighted_participation(res):
    """
    Threshold-free alternative: the kinetic-energy-weighted mean effective
    density of a mode.  A mode riding on solid material scores near 1; a mode
    confined to void scores near the void density.  No cutoff is involved.
    """
    return res['ke_n'].T @ res['zeff']

def inverse_participation_ratio(res):
    """Spatial concentration of modal kinetic energy: 1/N (uniform) .. 1 (one element)."""
    return (res['ke_n'] ** 2).sum(axis=0)
