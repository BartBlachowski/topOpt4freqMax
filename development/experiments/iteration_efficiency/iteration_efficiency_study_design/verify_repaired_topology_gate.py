"""Read-only verification of the Phase-1C repaired topology gate on frozen Olhoff
trajectories. No optimizer is run; no frozen artifact is modified."""
import sys, h5py, numpy as np
from scipy.ndimage import label

STRUCT4 = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
VF = 0.5
AE0 = 8.0/(160*20)          # coarsest-mesh element area
ASIG = 4*AE0                # 2x2 coarsest-mesh patch

def binarize(rho, nely, nelx):
    """Exact-count projection, ties broken by increasing global element index."""
    ne = rho.size
    nsolid = int(round(VF*ne))
    order = np.lexsort((np.arange(ne), -rho))   # primary -rho desc, secondary index asc
    b = np.zeros(ne, dtype=bool)
    b[order[:nsolid]] = True
    return b.reshape((nely, nelx), order='F')

def support_footprints(nely, nelx):
    jy = nely//2                                 # mid-height node row
    left  = [(jy-1,0),(jy,0)]
    right = [(jy-1,nelx-1),(jy,nelx-1)]
    return left, right

def gate(b, a_sig):
    lab, n = label(b, structure=STRUCT4)
    if n == 0:
        return False, 0, 0, 0
    nely, nelx = b.shape
    left, right = support_footprints(nely, nelx)
    lset = {lab[y,x] for y,x in left  if lab[y,x] > 0}
    rset = {lab[y,x] for y,x in right if lab[y,x] > 0}
    common = lset & rset
    c_req = len(common) > 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    det = np.delete(np.arange(sizes.size), list(common) + [0]) if common else np.arange(1, sizes.size)
    det_sizes = sizes[det]
    det_sizes = det_sizes[det_sizes > 0]
    det_max = int(det_sizes.max()) if det_sizes.size else 0
    det_tot = int(det_sizes.sum())
    ok = c_req and det_max < a_sig
    return ok, det_max, det_tot, c_req

def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur+1 if v else 0
        best = max(best, cur)
    return best

def run(path, nelx, nely):
    f = h5py.File(path, 'r')
    snaps = f['res/rho_snapshots']
    ne = nelx*nely
    a_e = 8.0/ne
    a_sig = int(np.ceil(ASIG/a_e))
    n = snaps.shape[0]
    okv = np.zeros(n, bool); supv = np.zeros(n, bool)
    dmax = np.zeros(n, int)
    for i in range(n):
        b = binarize(np.asarray(snaps[i], dtype=np.float64), nely, nelx)
        ok, dm, dt, cr = gate(b, a_sig)
        okv[i], supv[i], dmax[i] = ok, bool(cr), dm
    print(f"{nelx}x{nely}  states={n}  a_sig={a_sig}  "
          f"support={100*supv.mean():.2f}%  repaired={100*okv.mean():.2f}%  "
          f"longest_run={longest_run(okv)}  final_detmax={dmax[-1]}  final_pass={okv[-1]}")

if __name__ == '__main__':
    for nelx, nely in [(int(sys.argv[1]), int(sys.argv[2]))]:
        run(f"examples/Performance/final_campaign/raw/olhoff/s1_{nelx}x{nely}.mat", nelx, nely)

# Reproduced Phase-1C results (read-only, no optimizer invoked):
#   python3 verify_repaired_topology_gate.py 160 20
#     160x20  states=1601  a_sig=4   support=98.88%  repaired=66.52%  longest_run=957   final_detmax=0
#   python3 verify_repaired_topology_gate.py 240 30
#     240x30  states=1601  a_sig=9   support=98.88%  repaired=93.07%  longest_run=1319  final_detmax=0
#   python3 verify_repaired_topology_gate.py 640 80
#     640x80  states=1067  a_sig=64  support=98.50%  repaired=95.03%  longest_run=925   final_detmax=4
#   python3 verify_repaired_topology_gate.py 720 90
#     720x90  states=1601  a_sig=81  support=99.00%  repaired=97.81%  longest_run=1517  final_detmax=8
#
# These match the available rows in TOPOLOGY_SANITY_SPEC.md Sec 6. The 800x100 artifact is
# zero bytes and the frozen endpoint is RUN_ERROR/E1 N/A, so its topology evidence is
# UNVERIFIABLE_AT_PRESENT and no lower bound is inferred.
# Requires h5py and scipy. Run from the repository root.
