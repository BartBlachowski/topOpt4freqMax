#!/usr/bin/env python3
"""Collect AUDITRESULT lines + WP2 certificates into the audit report table."""
import re,sys,glob,os
LOGS='analysis/OlhoffRegularized/audit/logs'
def parse(f):
    for L in open(f):
        if L.startswith('AUDITRESULT'):
            d=dict(re.findall(r'(\w+)=(\S+)',L))
            d['_log']=os.path.basename(f); return d
    return None
rows=[r for r in (parse(f) for f in sorted(glob.glob(LOGS+'/*.log'))) if r]
cols=['tag','status','outer','accepted','rejected','inner','contractions','w1','w2','N',
      'trust','ceil','dxInf','dxRms','relObj','slope','vol','gray','wall']
w={c:max(len(c),*(len(r.get(c,'-')) for r in rows)) for c in cols}
print(' | '.join(c.ljust(w[c]) for c in cols))
print('-|-'.join('-'*w[c] for c in cols))
for r in sorted(rows,key=lambda r:r['tag']):
    print(' | '.join(str(r.get(c,'-')).ljust(w[c]) for c in cols))
