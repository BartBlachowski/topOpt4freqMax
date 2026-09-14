import pathlib,hashlib,json,subprocess,datetime
root=pathlib.Path(__file__).resolve().parents[3]
study=pathlib.Path(__file__).resolve().parents[1]
ref=study.parent/'frozen_problem25_reference'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((ref/'DATA_MANIFEST.json').read_text()); checks=[]
for f in m['files']:
 p=ref/f['path']; checks.append({'path':str(p),'expected':f['sha256'],'actual':hashfile(p),'ok':hashfile(p)==f['sha256']})
assert all(x['ok'] for x in checks)
repo=root.parents[1]
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0')
protected={str((repo/p).resolve()):hashfile(repo/p) for p in tracked if p and (repo/p).is_file()}
for p in ref.rglob('*'):
 if p.is_file(): protected[str(p)]=hashfile(p)
p=root/'evidence/three_rung_canary_preflight/C480x60_three_rung_trajectory.mat'; protected[str(p)]=hashfile(p)
out={'timestamp':datetime.datetime.now().astimezone().isoformat(),'reference_manifest_checks':checks,'protected':protected,'git_status_before':subprocess.check_output(['git','status','--short'],cwd=repo).decode(),'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip(),'upstream_url':'https://www.smoptit.se/GCMMA-MMA-code-1.5.zip','upstream_sha256':hashfile(study/'scripts/GCMMA-MMA-code-1.5.zip')}
(study/'evaluations/provenance_before.json').write_text(json.dumps(out,indent=2));print('Protected',len(protected),'files; reference hashes PASS')
