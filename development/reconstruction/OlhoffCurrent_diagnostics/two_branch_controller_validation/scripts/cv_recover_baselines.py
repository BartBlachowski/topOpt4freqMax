#!/usr/bin/env python3
"""Additively record recovered production density fields in baselines.json.

PREREGISTRATION section 9 marked the 160x20 and 320x40 production final densities
UNAVAILABLE because the raw .mat files were absent from the machine at freeze
time.  They are present now (diagnostics/move_stop/runs/), and the 400x50 pair
was recomputed after being lost.

This script ONLY adds recovery fields.  It never edits a frozen baseline scalar.
Before adding a pointer it re-derives M_nd, gray, mid and volume from the file's
final density column and refuses to record it unless they reproduce the frozen
values to 1e-9 absolute -- so a mislabelled or wrong-run file cannot be adopted.
"""
import json, hashlib, os, sys
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(STUDY))
REPO = os.path.dirname(os.path.dirname(ROOT))

CAND = {
    'm160': 'analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_160x20.mat',
    'm320': 'analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_320x40.mat',
    'm400': 'analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat',
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sha_vec(v):
    return hashlib.sha256(np.asarray(v, dtype='<f8').tobytes()).hexdigest()


def main():
    bf = os.path.join(STUDY, 'evidence', 'baselines.json')
    B = json.load(open(bf))
    report = {}
    for k, rel in CAND.items():
        p = os.path.join(REPO, rel)
        r = {'file': rel, 'exists': os.path.isfile(p)}
        if not r['exists']:
            report[k] = r
            print(f'  {k}: MISSING {rel}')
            continue
        with h5py.File(p, 'r') as f:
            R = f['RHO']
            rho = np.array(R[-1, :])
            r['nOuter_in_file'] = int(R.shape[0])
            r['NE_in_file'] = int(R.shape[1])
        chk = {'Mnd': 100 * np.mean(4 * rho * (1 - rho)),
               'gray': float(np.mean((rho > 0.1) & (rho < 0.9))),
               'mid': float(np.mean((rho >= 0.4) & (rho <= 0.6))),
               'volume': float(rho.mean())}
        frozen = {kk: B[k][kk] for kk in chk}
        dev = {kk: abs(chk[kk] - frozen[kk]) for kk in chk}
        ok = all(v <= 1e-9 for v in dev.values()) and r['nOuter_in_file'] == B[k]['nOuter']
        r.update({'recomputed': chk, 'frozen': frozen,
                  'maxAbsDeviation': max(dev.values()), 'verified': bool(ok)})
        if ok:
            r['rho_sha256'] = sha_vec(rho)
            r['trajectorySha'] = sha_file(p)
            B[k]['rho_recovered'] = True
            B[k]['rho_recovered_file'] = rel
            B[k]['rho_recovered_sha256'] = r['rho_sha256']
            B[k]['rho_recovered_trajectorySha'] = r['trajectorySha']
            B[k]['rho_recovered_note'] = (
                'Recovered after the freeze; frozen scalars re-derived from this '
                'file and matched to <=1e-9. No frozen value was edited.')
            if B[k].get('rho_sha256', '').startswith('UNAVAILABLE'):
                B[k]['rho_sha256_frozen_state'] = B[k]['rho_sha256']
            print(f"  {k}: VERIFIED  maxdev={r['maxAbsDeviation']:.2e}  rho={r['rho_sha256'][:16]}...")
            if 'rho_sha256' in B[k] and not B[k]['rho_sha256'].startswith('UNAVAILABLE'):
                same = B[k]['rho_sha256'] == r['rho_sha256']
                r['matchesFrozenRhoHash'] = same
                print(f"        frozen rho hash present: match={same}")
        else:
            print(f"  {k}: REFUSED   maxdev={r['maxAbsDeviation']:.3e} "
                  f"nOuter {r['nOuter_in_file']} vs {B[k]['nOuter']}")
        report[k] = r
    json.dump(B, open(bf, 'w'), indent=1)
    out = os.path.join(STUDY, 'evidence', 'baseline_recovery.json')
    json.dump(report, open(out, 'w'), indent=1, default=float)
    print('wrote', out)


if __name__ == '__main__':
    main()
