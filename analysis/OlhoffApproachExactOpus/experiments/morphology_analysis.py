"""
Phase-5 Task 2/3: morphological + modal analysis of EXISTING reconstructed designs.

Reads the candidate designs already stored in
  analysis/OlhoffApproachExact/Matlab/results/*.mat
(no solver run, no sweep, no modification of that code) and quantifies:
  - connected components (solid, threshold 0.5),
  - central-span bridge solidity (the connected<->disconnected discriminator),
  - density distribution,
  - omega_1 recomputed with the clean-room FE (fe_verify) to test hypothesis A
    (is the disconnected branch genuinely higher-frequency at equal volume?).

Mesh is 40x5 (200 elements), CC supports, p=3, mass model du2007_c1 -- matches
the audit runs. Element ordering e = elx*nely + ely (0-based) matches fe_verify.
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.ndimage import label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))
from fe_verify import q4_matrices, build_mesh, supports
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

def mass_coeff(rho, mode):
    """Element mass coefficient m(rho) matching mass_interp.m."""
    rho = np.asarray(rho, float)
    m = np.empty_like(rho)
    if mode == 'linear':
        return rho.copy()
    hi = rho > 0.1; lo = ~hi
    m[hi] = rho[hi]
    if mode == 'du2007_c1':
        m[lo] = 6e5 * rho[lo]**6 - 5e6 * rho[lo]**7
    elif mode == 'du2007_c0':
        m[lo] = 1e5 * rho[lo]**6
    elif mode == 'du2007_step':
        m[lo] = rho[lo]**6
    else:
        raise ValueError(mode)
    return m

def assemble_mm(rho, Ke, Me, edof, nDof, p, E0, rho0, mass_mode):
    nEl = edof.shape[0]
    iK = np.kron(edof, np.ones((8, 1), dtype=int)).flatten()
    jK = np.kron(edof, np.ones((1, 8), dtype=int)).flatten()
    Ke_phys = (E0 * Ke).flatten(); Me_phys = (rho0 * Me).flatten()
    sK = (rho ** p)[:, None] * Ke_phys[None, :]
    sM = mass_coeff(rho, mass_mode)[:, None] * Me_phys[None, :]
    K = coo_matrix((sK.flatten(), (iK, jK)), shape=(nDof, nDof)).tocsc()
    M = coo_matrix((sM.flatten(), (iK, jK)), shape=(nDof, nDof)).tocsc()
    return K, M

RES = os.path.join(os.path.dirname(__file__), '..', '..', 'OlhoffApproachExact',
                   'Matlab', 'results')
NELX, NELY = 40, 5

def grid(rho):
    return np.reshape(np.asarray(rho).ravel(), (NELY, NELX), order='F')

def n_components(rho, thr=0.5):
    so01 = (grid(rho) > thr).astype(int)
    lab, n = label(so01)
    sizes = [int((lab == i).sum()) for i in range(1, n + 1)]
    return n, sorted(sizes, reverse=True), so01

def central_bridge(rho, thr=0.5):
    """For each column, count solid (>thr) cells; report the min over the
    central third of the span -- a thin/empty center = disconnected bridge."""
    g = grid(rho) > thr
    colsolid = g.sum(axis=0)           # solid cells per column (0..NELY)
    c0, c1 = NELX // 3, 2 * NELX // 3
    center = colsolid[c0:c1]
    return int(center.min()), float(colsolid.mean()), colsolid

def omega1_of(rho, p=3.0, support='CC', mass_mode='du2007_c1'):
    L, H = 8.0, 1.0
    E0, nu, rho0, t = 1e7, 0.3, 1.0, 1.0
    dx, dy = L / NELX, H / NELY
    Ke, Me = q4_matrices(nu, t, dx, dy)
    edof = build_mesh(NELX, NELY)
    nDof = 2 * (NELX + 1) * (NELY + 1)
    rho = np.clip(np.asarray(rho).ravel(), 1e-3, 1.0)
    K, M = assemble_mm(rho, Ke, Me, edof, nDof, p, E0, rho0, mass_mode)
    fixed = supports(support, NELX, NELY)
    free = np.setdiff1d(np.arange(nDof), fixed)
    Kf = K[free][:, free]; Mf = M[free][:, free]
    vals = eigsh(Kf, k=6, M=Mf, sigma=1e-3, which='LM', return_eigenvectors=False)
    vals = np.sort(vals[vals > 1e-9])
    return np.sqrt(vals[:3])

def center_density(rho):
    g = grid(rho); c0, c1 = NELX // 3, 2 * NELX // 3
    cen = g[:, c0:c1]
    return float(cen.mean()), float(cen.max())

def report(name, rho):
    nC, sizes, _ = n_components(rho)
    cmin, cmean, colsolid = central_bridge(rho)
    w_c1 = omega1_of(rho, mass_mode='du2007_c1')
    w_lin = omega1_of(rho, mass_mode='linear')
    cden_mean, cden_max = center_density(rho)
    vol = float(np.mean(rho))
    grey = float(np.mean((rho > 0.1) & (rho < 0.9)))
    print(f"\n## {name}")
    print(f"   volume(mean rho) = {vol:.3f}   grey fraction = {grey:.3f}")
    print(f"   solid components (rho>0.5): {nC}   sizes={sizes}")
    print(f"   central-third: min>0.5 col solidity = {cmin}/{NELY};  "
          f"mean rho = {cden_mean:.3f};  max rho = {cden_max:.3f}")
    print(f"   column solidity profile (cells>0.5 per column, x->):")
    print("   " + "".join(str(c) for c in colsolid))
    print(f"   omega_1,2,3  du2007_c1 mass = "
          f"{w_c1[0]:.1f}, {w_c1[1]:.1f}, {w_c1[2]:.1f}   gap2/1={w_c1[1]/w_c1[0]:.4f}")
    print(f"   omega_1,2,3  LINEAR    mass = "
          f"{w_lin[0]:.1f}, {w_lin[1]:.1f}, {w_lin[2]:.1f}   "
          f"(ratio c1/lin w1 = {w_c1[0]/w_lin[0]:.2f})")
    return dict(name=name, vol=vol, nC=nC, cmin=cmin, cden=cden_mean,
                w1_c1=w_c1[0], w2_c1=w_c1[1], w1_lin=w_lin[0])

def main():
    rows = []
    # --- baseline trace: near-paper (iter24) and final (iter80) designs ---
    bt = sio.loadmat(os.path.join(RES, 'optimizer_audit', 'baseline_trace.mat'),
                     squeeze_me=True, struct_as_record=False)
    tr = bt['base_trace']
    rho_post = np.asarray(tr.rho_post)   # (niter x 200) or (200 x niter)
    if rho_post.shape[0] == 200:
        rho_post = rho_post.T
    omega_post = np.asarray(tr.omega_post)
    if omega_post.ndim == 2 and omega_post.shape[0] != rho_post.shape[0]:
        omega_post = omega_post.T
    niter = rho_post.shape[0]
    print(f"baseline trace: {niter} iterations, design dim {rho_post.shape[1]}")
    # iter indices are 1-based in MATLAB; near-paper ~24, final ~80
    for it in [1, 5, 10, 20, 24, 40, 60, niter]:
        if it <= niter:
            rows.append(report(f"baseline iter {it}", rho_post[it-1]))

    # --- penalty continuation best/final designs ---
    pc = sio.loadmat(os.path.join(RES, 'penalty_continuation',
                                  'cc_penalty_continuation.mat'),
                     squeeze_me=True, struct_as_record=False)
    for key, lab in [('rho_cont', 'continuation FINAL'),
                     ('rho_cont_best', 'continuation BEST'),
                     ('rho_ctrl', 'control(fixed-p) FINAL'),
                     ('rho_ctrl_best', 'control(fixed-p) BEST')]:
        if key in pc:
            rows.append(report(f"penalty {lab}", pc[key]))

    # --- summary table ---
    print("\n\n===== SUMMARY =====")
    print(f"{'design':30s} {'vol':>5s} {'#cmp':>4s} {'cden':>5s} "
          f"{'w1_c1':>7s} {'w2_c1':>7s} {'w1_lin':>7s}")
    for r in rows:
        print(f"{r['name']:30s} {r['vol']:5.3f} {r['nC']:4d} {r['cden']:5.3f} "
              f"{r['w1_c1']:7.1f} {r['w2_c1']:7.1f} {r['w1_lin']:7.1f}")
    print("\nPaper Fig.3c target: omega_1 = 456.4 (bimodal, CONNECTED).")
    print("cden = mean density in central third (grey-bridge probe).")
    print("w1_c1 = du2007_c1 void-mass model (paper); w1_lin = linear/physical mass.")

if __name__ == '__main__':
    main()
