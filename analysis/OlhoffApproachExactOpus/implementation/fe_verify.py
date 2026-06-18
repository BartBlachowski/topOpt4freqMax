"""
Clean-room FE verification for Du & Olhoff (2007), Section 4.1 initial designs.

Purpose (Phase 4, decision gate): independently reconstruct ONLY the finite-element
core from the paper-derived specification (no reuse of OlhoffApproachExact code) and
check whether the published INITIAL fundamental frequencies of the uniform rho=0.5
beam are reproduced:

    SS (simply supported both ends)        omega_1 = 68.7   rad/s   (paper Fig. 2a)
    CS (one clamped, one simply supported) omega_1 = 104.1  rad/s   (paper Fig. 2b)
    CC (clamped both ends)                 omega_1 = 146.1  rad/s   (paper Fig. 2c)

This is the paper's own "first validation criterion" (it pins down element matrices,
units, mass model, mesh, boundary conditions BEFORE any optimizer differences matter).
At the uniform rho=0.5 start every mass model (eq. 2/4/4a/4b) coincides because
rho=0.5 > 0.1, so this isolates FE + BC + mesh only.

Spec reference: specification/du_olhoff_2007_spec.md, sections 1, 2, 8.1.
Material:  E=1e7, nu=0.3, rho_m=1, t=1 ; domain L=8, H=1 ; rho=0.5 uniform.
Element:   Q4 bilinear plane stress, 2x2 Gauss, consistent mass.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

# ----------------------------------------------------------------------
# Q4 plane-stress element matrices for E=1, rho=1 (unit material).
# DOF order per element: [ux1 uy1 ux2 uy2 ux3 uy3 ux4 uy4],
# local nodes 1=LL 2=LR 3=UR 4=UL (xi,eta) = (-1,-1)(+1,-1)(+1,+1)(-1,+1).
# ----------------------------------------------------------------------
def q4_matrices(nu, t, dx, dy):
    C = 1.0 / (1 - nu**2) * np.array([[1, nu, 0],
                                      [nu, 1, 0],
                                      [0, 0, (1 - nu) / 2]])
    gp = np.array([-1, 1]) / np.sqrt(3.0)
    xiN = np.array([-1, 1, 1, -1])
    etaN = np.array([-1, -1, 1, 1])
    detJ = dx * dy / 4.0
    invJ = np.array([[2 / dx, 0], [0, 2 / dy]])
    Ke = np.zeros((8, 8))
    Me = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            N = 0.25 * (1 + xi * xiN) * (1 + eta * etaN)
            dN_dxi = 0.25 * xiN * (1 + eta * etaN)
            dN_deta = 0.25 * etaN * (1 + xi * xiN)
            grads = invJ @ np.vstack([dN_dxi, dN_deta])
            dN_dx, dN_dy = grads[0], grads[1]
            B = np.zeros((3, 8))
            Nmat = np.zeros((2, 8))
            for a in range(4):
                B[0, 2 * a] = dN_dx[a]
                B[1, 2 * a + 1] = dN_dy[a]
                B[2, 2 * a] = dN_dy[a]
                B[2, 2 * a + 1] = dN_dx[a]
                Nmat[0, 2 * a] = N[a]
                Nmat[1, 2 * a + 1] = N[a]
            Ke += (B.T @ C @ B) * (t * detJ)
            Me += (Nmat.T @ Nmat) * (t * detJ)
    return Ke, Me


def build_mesh(nelx, nely):
    """Column-major node numbering: node(i_col, j_row) = i_col*(nely+1)+j_row.
    Returns edofMat (nEl x 8) of 0-based global DOF indices and node coords helper.
    """
    nEl = nelx * nely
    edof = np.zeros((nEl, 8), dtype=int)
    def node(ic, jr):
        return ic * (nely + 1) + jr
    e = 0
    for ic in range(nelx):
        for jr in range(nely):
            n_ll = node(ic, jr)
            n_lr = node(ic + 1, jr)
            n_ur = node(ic + 1, jr + 1)
            n_ul = node(ic, jr + 1)
            dofs = []
            for n in (n_ll, n_lr, n_ur, n_ul):
                dofs += [2 * n, 2 * n + 1]
            edof[e] = dofs
            e += 1
    return edof


def assemble(rho, Ke, Me, edof, nDof, p, E0, rho0):
    """K = sum rho^p E0 Ke ; M = sum rho^q rho0 Me with q=1 (rho>=0.5 here)."""
    nEl = edof.shape[0]
    iK = np.kron(edof, np.ones((8, 1), dtype=int)).flatten()
    jK = np.kron(edof, np.ones((1, 8), dtype=int)).flatten()
    Ke_phys = (E0 * Ke).flatten()
    Me_phys = (rho0 * Me).flatten()
    sK = (rho ** p)[:, None] * Ke_phys[None, :]
    sM = (rho)[:, None] * Me_phys[None, :]   # q=1, rho>0.1 -> m=rho
    K = coo_matrix((sK.flatten(), (iK, jK)), shape=(nDof, nDof)).tocsc()
    M = coo_matrix((sM.flatten(), (iK, jK)), shape=(nDof, nDof)).tocsc()
    return K, M


def supports(kind, nelx, nely):
    """Return fixed 0-based DOFs. Node(ic,jr)=ic*(nely+1)+jr.
    Interpretations tested (paper Fig.2 is ambiguous for a 2D continuum):
      'SS'        : pin (ux,uy) at mid-height node of each vertical end edge
      'SS_corner' : pin (ux,uy) at the two bottom corners
      'CS'        : left edge fully clamped, right end pinned at mid-height
      'CC'        : both vertical end edges fully clamped
    """
    def node(ic, jr):
        return ic * (nely + 1) + jr
    mid = nely // 2
    left = [node(0, jr) for jr in range(nely + 1)]
    right = [node(nelx, jr) for jr in range(nely + 1)]
    fixed = []
    if kind == 'SS':
        for n in (node(0, mid), node(nelx, mid)):
            fixed += [2 * n, 2 * n + 1]
    elif kind == 'SS_corner':
        for n in (node(0, 0), node(nelx, 0)):
            fixed += [2 * n, 2 * n + 1]
    elif kind == 'CS':
        for n in left:
            fixed += [2 * n, 2 * n + 1]
        n = node(nelx, mid)
        fixed += [2 * n, 2 * n + 1]
    elif kind == 'CC':
        for n in left + right:
            fixed += [2 * n, 2 * n + 1]
    return np.unique(fixed)


def omega1(kind, nelx, nely, p=3.0):
    L, H = 8.0, 1.0
    E0, nu, rho0, t = 1e7, 0.3, 1.0, 1.0
    dx, dy = L / nelx, H / nely
    Ke, Me = q4_matrices(nu, t, dx, dy)
    edof = build_mesh(nelx, nely)
    nDof = 2 * (nelx + 1) * (nely + 1)
    rho = 0.5 * np.ones(nelx * nely)
    K, M = assemble(rho, Ke, Me, edof, nDof, p, E0, rho0)
    fixed = supports(kind, nelx, nely)
    free = np.setdiff1d(np.arange(nDof), fixed)
    Kf = K[free][:, free]      # two-step indexing (scipy 1.17 np.ix_ regression)
    Mf = M[free][:, free]
    vals = eigsh(Kf, k=6, M=Mf, sigma=1e-3, which='LM', return_eigenvectors=False)
    vals = np.sort(vals[vals > 1e-9])
    return np.sqrt(vals[0])


if __name__ == "__main__":
    targets = {'SS': 68.7, 'CS': 104.1, 'CC': 146.1}
    print("Initial fundamental frequency at uniform rho=0.5 (note: rho=0.5 makes")
    print("the SIMP power p irrelevant for K only via 0.5^p scaling -> p matters!).")
    print("Paper targets: SS=68.7  CS=104.1  CC=146.1\n")
    for nelx, nely in [(40, 5), (48, 6), (64, 8), (80, 10), (32, 4)]:
        print(f"--- mesh {nelx}x{nely} ---")
        for kind in ['SS', 'SS_corner', 'CS', 'CC']:
            for p in (1.0, 3.0):
                w = omega1(kind, nelx, nely, p=p)
                tgt = targets.get(kind.replace('_corner', ''), None)
                tag = f"(target {tgt})" if tgt else ""
                print(f"  {kind:10s} p={p:.0f}: omega_1 = {w:8.3f} {tag}")
        print()
