#!/usr/bin/env python3
"""ma4_manifest -- DATA_MANIFEST.json (brief sec. C17).

Derived from EVIDENCE.json (which measured everything from the files themselves)
plus the compact tracked artifacts, so the manifest cannot drift from what the
gate actually checks.
"""
import hashlib, json, os, sys

REPO = '/Users/piotrek/Programming/topOpt4freqMax'
OUT = os.path.join(REPO, 'analysis/OlhoffCurrent/diagnostics/move_activity_400')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    E = json.load(open(os.path.join(OUT, 'EVIDENCE.json')))
    arts = E['artifacts'] if isinstance(E['artifacts'], list) else [E['artifacts']]

    entries = []
    for a in arts:
        vs = a.get('variables', [])
        if isinstance(vs, dict): vs = [vs]
        prod = 'ARM P400' if a['path'].startswith('P400') else 'ARM F400'
        dims = {}
        precision = set()
        for v in vs:
            dims[v['name']] = [int(x) for x in (v['size'] if isinstance(v['size'], list)
                                                else [v['size']])]
            precision.add(v['class'])
        entries.append(dict(
            path=os.path.join(E['evidenceRoot'], a['path']).replace(os.sep, '/'),
            location='durable evidence root, untracked by git, declared and hash-gated',
            status=a['class'], bytes=a['bytes'], sha256=a['sha256'],
            format=a.get('format', ''), dimensions=dims,
            precision=sorted(precision), producingRun=prod,
            description=a.get('description', ''),
            gatedBy='olhoffcurrent_evidence_gate via EVIDENCE.json'))

    # compact tracked artifacts living beside the report
    tracked = []
    for sub in ['runs', 'figures', '.']:
        d = os.path.join(OUT, sub)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if not os.path.isfile(p): continue
            if f in ('DATA_MANIFEST.json', 'FINAL_SHA256.txt'): continue
            if f.endswith('.mat'): 
                cls = 'scratch'   # per-run summary structs, regenerable
            elif f.endswith(('.csv', '.json', '.md', '.png', '.py', '.m')):
                cls = 'optional' if f.endswith(('.py', '.m')) else 'required-tracked'
            else:
                continue
            rel = os.path.relpath(p, OUT).replace(os.sep, '/')
            tracked.append(dict(path=f'analysis/OlhoffCurrent/diagnostics/move_activity_400/{rel}',
                                location='tracked in git beside the report',
                                status=cls, bytes=os.path.getsize(p), sha256=sha(p)))

    M = dict(schema='olhoff_move_activity_400_data_manifest/1',
             study='move_activity_400',
             preregistration_sha256=E.get('preregistration_sha256', ''),
             evidenceRoot=E['evidenceRoot'],
             sourceTree=E.get('sourceTree', ''), matlab=E.get('matlab', ''),
             note=('Raw element-level trajectories are deliberately untracked (about '
                   '100 MB each) but DECLARED: EVIDENCE.json carries their SHA-256 and '
                   'olhoffcurrent_evidence_gate fails this study if either is missing '
                   'or altered.  Untracked is a storage decision; undeclared was the '
                   'defect that destroyed the earlier studies.'),
             rawEvidence=entries, trackedArtifacts=tracked)
    json.dump(M, open(os.path.join(OUT, 'DATA_MANIFEST.json'), 'w'), indent=2)
    print(f'wrote DATA_MANIFEST.json  ({len(entries)} raw, {len(tracked)} tracked)')
    for e in entries:
        print(f"  {e['status']:>8}  {e['bytes']/1e6:8.1f} MB  {e['path']}")
        print(f"            dims={e['dimensions']}  precision={e['precision']}")


if __name__ == '__main__':
    main()
