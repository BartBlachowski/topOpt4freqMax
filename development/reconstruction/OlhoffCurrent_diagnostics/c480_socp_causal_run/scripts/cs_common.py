"""Shared paths and FROZEN metric definitions for c480_socp_causal_run.

`fields`, `stats` and `kkt` are copied verbatim (whitespace only reformatted
where noted) from
  gray_kkt_forensic_audit/scripts/geometry.py::fields
  gray_kkt_forensic_audit/scripts/stationarity.py::stats, ::kkt
so the endpoint metrics use the SAME definitions as the prior grayness/KKT audit.
cs_control_identity.py proves the copy reproduces that audit's retained 480 values.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]                    # analysis/OlhoffCurrent
REPO = ROOT.parents[1]
DIAG = ROOT / 'diagnostics'
EV = STUDY / 'evaluations'
FIG = STUDY / 'figures'
RUN = STUDY / 'run'
EVROOT = ROOT / 'evidence' / 'c480_socp_causal_run'
CONTROL_TRAJ = ROOT / 'evidence' / 'three_rung_canary_preflight' / 'C480x60_three_rung_trajectory.mat'
CANARY = DIAG / 'three_rung_canary_preflight'
GRAYKKT = DIAG / 'gray_kkt_forensic_audit'
TREAT_TRAJ = EVROOT / 'C480x60_socp_trajectory.mat'
NX, NY = 480, 60
R_FILTER = 0.06


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_vec(v):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(v, dtype='<f8')).tobytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_default))


def _default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


# ---------------------------------------------------------------------------
# VERBATIM: gray_kkt_forensic_audit/scripts/geometry.py::fields
def fields(rho, nx, ny):
    im = rho.reshape(nx, ny).T; gray = (im > .1) & (im < .9); mid = (im >= .4) & (im <= .6); h = 1 / ny
    depth = ndi.distance_transform_edt(gray, sampling=(h, h)); depth[~gray] = 0
    lab, n = ndi.label(gray); sizes = np.bincount(lab.ravel())[1:]; lab8, n8 = ndi.label(gray, np.ones((3, 3)))
    binary = im >= .5; interface = np.zeros_like(gray)
    interface[1:] |= binary[1:] != binary[:-1]; interface[:-1] |= binary[1:] != binary[:-1]
    interface[:, 1:] |= binary[:, 1:] != binary[:, :-1]; interface[:, :-1] |= binary[:, 1:] != binary[:, :-1]
    di = ndi.distance_transform_edt(~interface, sampling=(h, h))
    return im, gray, mid, depth, lab, sizes, n, n8, di


# VERBATIM: gray_kkt_forensic_audit/scripts/stationarity.py::stats
def stats(v):
    a = np.abs(np.asarray(v)); return {'n': len(a), 'max': float(a.max()) if len(a) else None, 'median': float(np.median(a)) if len(a) else None, 'RMS': float(np.sqrt(np.mean(a * a))) if len(a) else None, 'p90': float(np.quantile(a, .9)) if len(a) else None, 'p95': float(np.quantile(a, .95)) if len(a) else None, 'p99': float(np.quantile(a, .99)) if len(a) else None}


# VERBATIM: gray_kkt_forensic_audit/scripts/stationarity.py::kkt
def kkt(v, r, lam, fitmask, scale):
    # objective -lambda/lambda_ref, volume <=0, nonnegative multiplier
    Vtot = .5 * len(r); mu = max(0, float(np.mean(v[fitmask] / lam)) * Vtot); red = -v / lam + mu / Vtot
    low = r <= .0010001; high = r >= .9999999; inter = ~(low | high)
    violation = red.copy(); violation[low] = np.minimum(red[low], 0); violation[high] = np.maximum(red[high], 0)
    return red, violation, {'mu_volume': mu, 'volume_constraint': float((r.sum() - Vtot) / Vtot), 'volume_complementarity': float(abs(mu * (r.sum() - Vtot) / Vtot)), 'scale_raw_objective_RMS_interior': scale, 'global_projected_residual_normalized': stats(violation / scale), 'lower_count': int(low.sum()), 'upper_count': int(high.sum()), 'interior_count': int(inter.sum()), 'lower_sign_violation': stats(violation[low] / scale), 'upper_sign_violation': stats(violation[high] / scale)}
# ---------------------------------------------------------------------------


def geometry_metrics(rho, nx=NX, ny=NY):
    """Endpoint metrics, same expressions as geometry.py main() for one design."""
    im, gr, mid, depth, lab, sizes, n, n8, di = fields(rho, nx, ny)
    broad = gr & (depth > .06)
    comps = []
    for j, size in enumerate(sizes, 1):
        w = np.where(lab == j)
        comps.append({'id': int(j), 'elements': int(size), 'area': float(size / (nx * ny) * 8),
                      'max_depth': float(depth[lab == j].max()),
                      'bbox_x_span': float((w[1].max() - w[1].min() + 1) * 8 / nx),
                      'bbox_y_span': float((w[0].max() - w[0].min() + 1) / ny)})
    comps.sort(key=lambda a: a['elements'], reverse=True)
    q = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    m = {'Mnd_percent': float(400 * np.mean(rho * (1 - rho))), 'gray_fraction': float(gr.mean()),
         'mid_fraction': float(mid.mean()), 'gray_area': float(gr.mean() * 8), 'mid_area': float(mid.mean() * 8),
         'broad_core_fraction': float(broad.mean()), 'broad_core_area': float(broad.mean() * 8),
         'gray_components_4': n, 'gray_components_8': n8,
         'largest_gray_component_area': comps[0]['area'] if comps else 0.0,
         'max_depth': float(depth.max()), 'max_depth_over_R': float(depth.max() / R_FILTER),
         'gray_depth_p50_p90_p95_p99': np.quantile(depth[gr], [.5, .9, .95, .99]).tolist() if gr.any() else None,
         'gray_interface_distance_p50_p90_p95_p99': np.quantile(di[gr], [.5, .9, .95, .99]).tolist() if gr.any() else None,
         'rho_quantiles': dict(zip(map(str, q), np.quantile(rho, np.array(q) / 100).tolist())),
         'rho_lt_001': float((rho < .01).mean()), 'rho_gt_099': float((rho > .99).mean()),
         'components': comps}
    return m, dict(im=im, gray=gr, mid=mid, depth=depth, broad=broad, labels=lab, interface_distance=di)


def per_iteration_gray(RHO_rows, nx=NX, ny=NY):
    """Same per-iteration expressions as geometry.py's trajectory rows."""
    out = np.zeros((RHO_rows.shape[0], 4))
    for i, r in enumerate(RHO_rows):
        a = r.reshape(nx, ny).T; g = (a > .1) & (a < .9); mi = (a >= .4) & (a <= .6)
        de = ndi.distance_transform_edt(g, sampling=(1 / ny, 1 / ny)) if not g.all() else np.full_like(a, np.inf)
        out[i] = [400 * np.mean(r * (1 - r)), g.mean(), mi.mean(), (de > .06).mean()]
    return out


def load_traj(path, keys=('RHO', 'DRHO')):
    """h5py view of a MATLAB v7.3 trajectory: RHO[k-1] is rho after outer k."""
    f = h5py.File(path, 'r')
    return f


def hist_field(f, key):
    return f['hist'][key][()].squeeze()
