"""Shared loaders and preregistered metric definitions for the scientific-delta audit.

Every metric here follows AUDIT_PREREGISTRATION.md sec. 7/12/13 and, where a prior
target study froze a definition, reproduces that definition exactly:
  M_nd = 4 mean(rho(1-rho))                       (source run_repro.m; target x100)
  gray = mean(0.1 < rho < 0.9), mid = mean(0.4 <= rho <= 0.6)   (dr_telemetry.m)
  broad core = gray & physical distance-to-non-gray > 0.06       (gray_kkt geometry.py)
  gray-fit KKT residual                                          (gray_kkt stationarity.py)
"""
import hashlib, json, pathlib
import numpy as np
import h5py
from scipy import ndimage as ndi

AUDIT = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path('/Users/piotrek/Programming/topOpt4freqMax')
OC = REPO / 'analysis' / 'OlhoffCurrent'
SNAP = AUDIT / 'source_snapshot' / '+olhoff_6b08708'
EVAL = AUDIT / 'evaluations'
FIG = AUDIT / 'figures'
S480 = SNAP / 'repro' / 'results' / 'S480x60' / 'res.mat'
C480 = OC / 'evidence' / 'three_rung_canary_preflight' / 'C480x60_three_rung_trajectory.mat'
M1 = EVAL / 'm1_run' / 'M1_480x60_res.mat'
NELX, NELY = 480, 60
NE = NELX * NELY
EPS = 0.05 * np.sqrt(NE / 3200)


def sha256_double(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x, dtype='<f8')).tobytes()).hexdigest()


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def _vec(ds):
    return np.asarray(ds[()]).squeeze()


def load_hist(group):
    return {k: np.asarray(group[k][()]) for k in group.keys()}


def load_c480():
    """Target three-rung canary: RHO[k-1] is the design AFTER outer iteration k."""
    f = h5py.File(C480, 'r')
    h = load_hist(f['hist'])
    out = {
        'RHO': f['RHO'],            # lazy (386, NE)
        'DRHO': f['DRHO'],
        'omega': h['omega'].T if h['omega'].shape[0] != 5 else h['omega'],
        'hist': h,
        'file': f,
    }
    om = np.asarray(f['hist']['omega'][()])
    out['omega'] = om if om.shape[1] == 5 else om.T      # (n, 5)
    out['n'] = out['omega'].shape[0]
    out['move'] = _vec(f['hist']['move'])                 # scalar per iteration
    return out


def load_s480():
    f = h5py.File(S480, 'r')
    h = load_hist(f['res/hist'])
    om = np.asarray(f['res/hist/omega'][()])
    om = om if om.shape[1] == 5 else om.T
    return {
        'rho': _vec(f['res/rho']), 'omega': om, 'hist': h, 'n': om.shape[0],
        'Mnd': _vec(f['res/aux/Mnd']), 'moveMean': _vec(f['res/aux/moveMean']),
        'move': _vec(f['res/hist/move']), 'file': f,
        'omega_final': _vec(f['res/omega']),
    }


def fields(rho, nx=NELX, ny=NELY):
    im = np.asarray(rho).reshape(nx, ny).T
    gray = (im > .1) & (im < .9)
    h = 1.0 / ny
    depth = ndi.distance_transform_edt(gray, sampling=(h, h)) if not gray.all() else np.full(im.shape, np.inf)
    depth[~gray] = 0
    return im, gray, depth


def discreteness(rho):
    r = np.asarray(rho, dtype=float)
    im, gray, depth = fields(r)
    return {
        'Mnd': float(4 * np.mean(r * (1 - r))),
        'gray': float(np.mean((r > .1) & (r < .9))),
        'mid': float(np.mean((r >= .4) & (r <= .6))),
        'broad_core_fraction': float((gray & (depth > .06)).mean()),
        'broad_core_area': float((gray & (depth > .06)).mean() * 8),
        'max_depth_over_R': float(depth.max() / .06),
        'void_lt_0p1': float(np.mean(r < .1)),
        'at_rhomin': float(np.mean(r <= .0010001)),
        'solid_gt_0p9': float(np.mean(r > .9)),
        'at_one': float(np.mean(r >= .9999999)),
    }


def kkt_grayfit(g, rho, lam):
    """Exact re-implementation of gray_kkt_forensic_audit/scripts/stationarity.py kkt()
    with the gray fit mask and the free-element raw scale."""
    r = np.asarray(rho, dtype=float)
    v = np.asarray(g, dtype=float)
    free = (r > .0010001) & (r < .9999999)
    gray = (r > .1) & (r < .9)
    scale = np.sqrt(np.mean((v[free] / lam) ** 2))
    Vtot = .5 * len(r)
    mu = max(0.0, float(np.mean(v[gray] / lam)) * Vtot)
    red = -v / lam + mu / Vtot
    return float(np.sqrt(np.mean((red[gray] / scale) ** 2))), float(scale), float(mu)


def spikes(omega1):
    w = np.asarray(omega1, dtype=float)
    return np.flatnonzero(w[1:] < 0.7 * w[:-1]) + 2      # 1-based iteration numbers


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()]
        if isinstance(o, float):
            return None if not np.isfinite(o) else o
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        return o
    pathlib.Path(path).write_text(json.dumps(conv(obj), indent=1))
