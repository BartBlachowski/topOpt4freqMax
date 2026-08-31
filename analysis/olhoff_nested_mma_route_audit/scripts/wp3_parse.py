#!/usr/bin/env python3
"""
WP3 - parse the complete BASE_mma_160x20 trajectory, with integrity checks.
READ-ONLY.  The log is the only surviving artifact of this run.
"""
import os, re, csv, hashlib, json
LOG='/Volumes/HP911Pro/Combobulating/Olhoff/results/BASE_mma_160x20.log'
OUT='/Users/piotrek/Programming/topOpt4freqMax/analysis/olhoff_nested_mma_route_audit'

raw=open(LOG,'r',errors='replace').read()
lines=raw.splitlines()
print(f'log lines: {len(lines)}   bytes: {len(raw.encode())}')
h=hashlib.sha256(open(LOG,'rb').read()).hexdigest()
print(f'sha256: {h}')

# header config block
cfg={}
for ln in lines[:40]:
    m=re.match(r'\s*(\w+):\s*(.+?)\s*$',ln)
    if m: cfg[m.group(1)]=m.group(2).strip()
print(f'config keys parsed: {len(cfg)}')

ROW=re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(yes|NO)\s*$')
rows=[]; noise=[]
for i,ln in enumerate(lines):
    m=ROW.match(ln)
    if m:
        rows.append(dict(outer=int(m.group(1)),omega1=float(m.group(2)),omega2=float(m.group(3)),
            omega3=float(m.group(4)),N=int(m.group(5)),sqrt_beta=float(m.group(6)),
            nInner=int(m.group(7)),cumInner=int(m.group(8)),maxdrho=float(m.group(9)),
            vol=float(m.group(10)),innerConv=(m.group(11)=='yes')))
    elif ln.strip() and not re.match(r'^\s*\w+:\s',ln) and 'it ' not in ln and '===' not in ln:
        noise.append((i+1,ln))

print(f'\nparsed rows: {len(rows)}')
print(f'first outer: {rows[0]["outer"]}   last outer: {rows[-1]["outer"]}')
its=[r['outer'] for r in rows]
missing=sorted(set(range(1,its[-1]+1))-set(its))
dup=[x for x in set(its) if its.count(x)>1]
print(f'missing iterations within range: {missing if missing else "none"}')
print(f'duplicate iterations: {dup if dup else "none"}')
print(f'declared maxOuter: {cfg.get("maxOuter")}   -> shortfall: {int(cfg.get("maxOuter",0))-its[-1]}')
print(f'\nnon-table lines ({len(noise)}):')
for i,ln in noise: print(f'  L{i}: {ln[:150]}')

# cumInner consistency
bad=0
for j,r in enumerate(rows):
    exp=rows[j-1]['cumInner']+r['nInner'] if j else r['nInner']
    if exp!=r['cumInner']: bad+=1; print(f'  cumInner mismatch at outer {r["outer"]}: expected {exp}, got {r["cumInner"]}')
print(f'cumInner arithmetic consistent: {bad==0} ({bad} mismatches)')

with open(os.path.join(OUT,'BASE_MMA_HISTORY.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json.dump({'log_path':LOG,'log_sha256':h,'n_lines':len(lines),'n_rows':len(rows),
           'first_outer':rows[0]['outer'],'last_outer':rows[-1]['outer'],
           'declared_maxOuter':int(cfg.get('maxOuter',0)),'missing':missing,'duplicates':dup,
           'noise_lines':[{'line':i,'text':t} for i,t in noise],'config':cfg},
          open(os.path.join(OUT,'scripts','parse_provenance.json'),'w'),indent=1)
print(f'\nwrote BASE_MMA_HISTORY.csv ({len(rows)} rows)')
