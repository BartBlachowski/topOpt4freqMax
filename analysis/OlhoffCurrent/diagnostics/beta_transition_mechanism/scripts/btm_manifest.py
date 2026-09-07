#!/usr/bin/env python3
"""btm_manifest -- DATA_MANIFEST.json for the beta mechanism audit."""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btm_common as B


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    E = json.load(open(os.path.join(B.OUT, 'EVIDENCE.json')))
    arts = E['artifacts'] if isinstance(E['artifacts'], list) else [E['artifacts']]
    raw = []
    for a in arts:
        vs = a.get('variables', [])
        if isinstance(vs, dict): vs = [vs]
        dims, prec = {}, set()
        for v in vs:
            sz = v['size'] if isinstance(v['size'], list) else [v['size']]
            dims[v['name']] = [int(x) for x in sz]; prec.add(v['class'])
        raw.append(dict(
            path=os.path.join(E['evidenceRoot'], a['path']).replace(os.sep, '/'),
            location=('durable evidence root, untracked by git, SHARED with '
                      'move_activity_400 (referenced, not duplicated)'),
            status=a['class'], bytes=a['bytes'], sha256=a['sha256'],
            format=a.get('format', ''), dimensions=dims, precision=sorted(prec),
            producingRun=('ARM P400' if a['path'].startswith('P400') else 'ARM F400'),
            producingStudy='move_activity_400',
            usedHereFor=a.get('description', ''),
            gatedBy='olhoffcurrent_evidence_gate via this study\'s EVIDENCE.json'))

    tracked, inputs = [], []
    for dp, dn, fn in os.walk(B.OUT):
        dn[:] = [d for d in dn if d != '__pycache__']
        for f in sorted(fn):
            if f in ('DATA_MANIFEST.json', 'FINAL_SHA256.txt', '.DS_Store'): continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, B.OUT).replace(os.sep, '/')
            tracked.append(dict(
                path=f'analysis/OlhoffCurrent/diagnostics/beta_transition_mechanism/{rel}',
                location='tracked in git', status='required-tracked',
                bytes=os.path.getsize(p), sha256=sha(p)))

    # the committed telemetry this audit READS but does not own
    for mesh, spec in B.RUNS.items():
        for kind in ('prod', 'fixed'):
            p = spec[kind]
            inputs.append(dict(path=os.path.relpath(p, B.REPO).replace(os.sep, '/'),
                               mesh=mesh, role=kind, status='input (owned by another study)',
                               bytes=os.path.getsize(p), sha256=sha(p)))

    M = dict(schema='olhoff_beta_transition_mechanism_data_manifest/1',
             study='beta_transition_mechanism',
             starting_HEAD='cb6c0eae31a25521f7c5fed1c4a89564ed63344e',
             sourceTree=E.get('sourceTree', ''), matlab=E.get('matlab', ''),
             evidenceRoot=E['evidenceRoot'],
             note=('This audit ran no optimisation and produced no new raw evidence. It '
                   'nevertheless DECLARES the two 400x50 trajectories required, because its '
                   'bound-activity result is computed from DRHO and is not derivable from any '
                   'CSV; the gate therefore fails this study too if they are lost. They are '
                   'referenced in the single durable evidence root, not duplicated.'),
             rawEvidence=raw, trackedArtifacts=tracked, inputTelemetry=inputs)
    json.dump(M, open(os.path.join(B.OUT, 'DATA_MANIFEST.json'), 'w'), indent=2)
    print(f'wrote DATA_MANIFEST.json ({len(raw)} raw, {len(tracked)} tracked, {len(inputs)} inputs)')
    for e in raw:
        print(f"  {e['status']:>8} {e['bytes']/1e6:8.1f} MB  {e['path']}")


if __name__ == '__main__':
    main()
