#!/usr/bin/env python3
"""cv_manifest -- DATA_MANIFEST.json and FINAL_SHA256.txt for this study.

Every tracked artifact in the study directory is hashed.  Every untracked raw
trajectory the study depends on is declared with its size and hash and marked
present or missing, so a fresh clone is told exactly what it does not have.
"""
import hashlib, json, os, datetime
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(STUDY))
REPO = os.path.dirname(os.path.dirname(ROOT))
REL = 'analysis/OlhoffCurrent/diagnostics/two_branch_controller_validation'

RAW = [
    ('analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C160x20_trajectory.mat',
     'C160x20', 'candidate 160x20 element-level trajectory (RHO, DRHO, hist, cfg, exh)'),
    ('analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C320x40_trajectory.mat',
     'C320x40', 'candidate 320x40 element-level trajectory'),
    ('analysis/OlhoffCurrent/evidence/two_branch_controller_validation/C400x50_trajectory.mat',
     'C400x50', 'candidate 400x50 element-level trajectory'),
    ('analysis/OlhoffCurrent/evidence/move_activity_400/P400_400x50_trajectory.mat',
     'P400', 'production 400x50 baseline trajectory (move_activity_400)'),
    ('analysis/OlhoffCurrent/evidence/move_activity_400/F400_400x50_trajectory.mat',
     'F400', 'fixed-move 0.04 400x50 arm (move_activity_400); sec. 6 prefix check'),
    ('analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_160x20.mat',
     'P160', 'production 160x20 baseline trajectory (move_stop); recovered'),
    ('analysis/OlhoffCurrent/diagnostics/move_stop/runs/baseline_320x40.mat',
     'P320', 'production 320x40 baseline trajectory (move_stop); recovered'),
    ('analysis/OlhoffCurrent/diagnostics/move_stop/runs/fixedmove_160x20.mat',
     'F160', 'fixed-move 0.04 160x20 arm (move_stop); stage-1 prefix check'),
    ('analysis/OlhoffCurrent/diagnostics/move_stop/runs/fixedmove_320x40.mat',
     'F320', 'fixed-move 0.04 320x40 arm (move_stop); stage-1 prefix check'),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dims(p):
    try:
        with h5py.File(p, 'r') as f:
            return {k: [int(x) for x in f[k].shape]
                    for k in f.keys() if isinstance(f.get(k), h5py.Dataset)}
    except Exception:
        return {}


def main():
    tracked = []
    for dp, dn, fn in os.walk(STUDY):
        dn[:] = [d for d in dn if d not in ('__pycache__',)]
        for f in sorted(fn):
            if f in ('DATA_MANIFEST.json', 'FINAL_SHA256.txt', '.DS_Store'):
                continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, STUDY).replace(os.sep, '/')
            tracked.append(dict(path=f'{REL}/{rel}', location='tracked in git',
                                status='required-tracked',
                                bytes=os.path.getsize(p), sha256=sha(p)))

    raw = []
    for rel, run, desc in RAW:
        p = os.path.join(REPO, rel)
        e = os.path.isfile(p)
        raw.append(dict(path=rel, producingRun=run, usedHereFor=desc,
                        location='durable evidence root, untracked by git'
                                 if '/evidence/' in rel else
                                 'study runs directory, untracked by git',
                        status='required-raw', present=e,
                        bytes=os.path.getsize(p) if e else None,
                        sha256=sha(p) if e else None,
                        dimensions=dims(p) if e else {}))

    M = dict(schema='olhoff_current_data_manifest/1',
             study='two_branch_controller_validation',
             generated=datetime.datetime.now(datetime.timezone.utc)
                       .strftime('%Y-%m-%dT%H:%M:%SZ'),
             nTracked=len(tracked), nRaw=len(raw),
             nRawPresent=sum(1 for r in raw if r['present']),
             nRawMissing=sum(1 for r in raw if not r['present']),
             tracked=tracked, raw=raw)
    json.dump(M, open(os.path.join(STUDY, 'DATA_MANIFEST.json'), 'w'), indent=1)

    lines = [f"{t['sha256']}  {t['path']}" for t in tracked]
    lines += [f"{r['sha256'] or 'MISSING':64s}  {r['path']}" for r in raw]
    open(os.path.join(STUDY, 'FINAL_SHA256.txt'), 'w').write('\n'.join(lines) + '\n')
    print(f"tracked={len(tracked)} raw={len(raw)} present={M['nRawPresent']} missing={M['nRawMissing']}")
    for r in raw:
        if not r['present']:
            print('  MISSING RAW:', r['path'])


if __name__ == '__main__':
    main()
