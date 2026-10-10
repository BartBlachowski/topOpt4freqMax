"""Offline replay of stopping rules on recorded histories (numpy only)."""
import sys, os, csv
import numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'out')
MESHES = ['160x20','240x30','320x40','400x50','480x60','560x70','640x80','720x90','800x100']

def ffill(a):
    a = np.array(a, float); last = np.nan
    for i in range(len(a)):
        if np.isnan(a[i]): a[i] = last
        else: last = a[i]
    return a

def load(method, mesh):
    """Return per-iteration arrays indexed by update k (state AFTER update k).
    Proposed/Yuksel histories hold the PRE-update design of each iteration, so
    row k+1 is the state after update k; the solver's own max|dx| of update k is
    the recorded d_inf_design[k].  Olhoff rows are replayed post-update designs."""
    fn = os.path.join(OUT, f'{method}_{mesh}.csv')
    if not os.path.exists(fn): return None
    A = np.genfromtxt(fn, delimiter=',', names=True)
    R = {k: np.array(A[k], float) for k in A.dtype.names}
    if method == 'olhoff':
        T = dict(R)
        T['w1'] = np.r_[R['omega1'][1:], np.nan]   # eigen of the design after update k = omega at start of outer k+1
        T['w1'] = ffill(T['w1']); T['w1'][-1] = T['w1'][-2]
        T['obj'] = T['w1']
        T['dmax_native'] = R['dxOuter']
    else:
        n = len(R['dmax']) - 1
        T = {k: v[1:] for k, v in R.items()}           # row k+1 -> update k
        T['dmax_native'] = R['dinfDesign'][:n]
        T['w1'] = ffill(T['w1'])
        if np.isnan(T['w1'][0]): T['w1'][0] = T['w1'][~np.isnan(T['w1'])][0]
        if 'obj' in R: T['obj'] = R['obj'][1:]          # objective evaluated on the pre-update design of k+1
    if 'stage' not in T: T['stage'] = np.ones(len(T['w1']))
    T['n'] = len(T['w1'])
    return T

def sub(T, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == (T['n'],) else v) for k, v in T.items()} | {'n': int(mask.sum())}

def first_true(mask):
    idx = np.where(mask)[0]
    return int(idx[0]) if len(idx) else None

def windowed(x, W, tol, strict=False):
    x = np.asarray(x); n = len(x); ok = np.zeros(n, bool)
    for k in range(W-1, n):
        seg = x[k-W+1:k+1]
        ok[k] = np.all(seg < tol) if strict else np.all(seg <= tol)
    return ok

def win_range(f, W, relative=True):
    f = np.asarray(f, float); n = len(f); r = np.full(n, np.inf)
    for k in range(W, n):
        seg = f[k-W:k+1]
        r[k] = (seg.max() - seg.min()) / (max(abs(f[k]), 1e-300) if relative else 1.0)
    return r

def stagnation(T, w_tol=0.005, mnd_tol=0.01):
    w = T['w1']; m = T['Mnd']; st = T['stage']
    ok = (np.abs(w - w[-1])/w[-1] <= w_tol) & (np.abs(m - m[-1]) <= mnd_tol) & (st == st[-1])
    bad = np.where(~ok)[0]
    return int(bad[-1] + 1) if len(bad) else 0

RULES = {}
def rule(name):
    def deco(f): RULES[name] = f; return f
    return deco
for tol in [0.01, 0.025, 0.04]:
    RULES[f'max|dx|<={tol}'] = (lambda tol: lambda T: first_true(T['dmax_native'] <= tol))(tol)
RULES['max|dx|<=0.02 W10'] = lambda T: first_true(windowed(T['dmax_native'], 10, 0.02))
RULES['max|dx|<=0.04 W10'] = lambda T: first_true(windowed(T['dmax_native'], 10, 0.04))
for tol in [1e-3, 2e-3, 5e-3, 1e-2]:
    RULES[f'relL2<{tol:g}'] = (lambda tol: lambda T: first_true(T['drel'] < tol))(tol)
RULES['rms<8.8e-4 (Olh c=.05)'] = lambda T: first_true(T['drms'] < 0.05/np.sqrt(3200))
RULES['rms<3.5e-3 (c=.2)'] = lambda T: first_true(T['drms'] < 0.2/np.sqrt(3200))
RULES['flip<1e-3'] = lambda T: first_true(T['flip'] < 1e-3)
RULES['flip<1e-3 W5'] = lambda T: first_true(windowed(T['flip'], 5, 1e-3, True))
RULES['obj rel W10 <1e-3'] = lambda T: first_true(win_range(T['obj'], 10) < 1e-3)
RULES['obj rel W10 <5e-4'] = lambda T: first_true(win_range(T['obj'], 10) < 5e-4)
RULES['obj rel W20 <1e-3'] = lambda T: first_true(win_range(T['obj'], 20) < 1e-3)
RULES['w1 rel W10 <1e-3'] = lambda T: first_true(win_range(T['w1'], 10) < 1e-3)
RULES['Mnd abs W10 <5e-3'] = lambda T: first_true(win_range(T['Mnd'], 10, False) < 5e-3)
RULES['obj<1e-3 & Mnd<5e-3 W10'] = lambda T: first_true((win_range(T['obj'], 10) < 1e-3) & (win_range(T['Mnd'], 10, False) < 5e-3))
RULES['obj<1e-3 & Mnd<1e-2 W10'] = lambda T: first_true((win_range(T['obj'], 10) < 1e-3) & (win_range(T['Mnd'], 10, False) < 1e-2))
RULES['obj<5e-4 & Mnd<5e-3 W10'] = lambda T: first_true((win_range(T['obj'], 10) < 5e-4) & (win_range(T['Mnd'], 10, False) < 5e-3))
RULES['obj<5e-4 & Mnd<5e-3 W20'] = lambda T: first_true((win_range(T['obj'], 20) < 5e-4) & (win_range(T['Mnd'], 20, False) < 5e-3))

def evaluate(method):
    rows = []
    for mesh in MESHES:
        T = load(method, mesh)
        if T is None: continue
        n = T['n']; kstag = stagnation(T); wEnd = T['w1'][-1]; mEnd = T['Mnd'][-1]
        last = T['stage'][-1]; mask = T['stage'] == last
        if method == 'yuksel':
            first = int(np.where(mask)[0][0]); mask[first:first+2] = False
        off = int((~mask).sum()) if method == 'yuksel' else 0
        Tf = sub(T, mask) if off else T
        for name, f in RULES.items():
            k = f(Tf)
            base = dict(method=method, mesh=mesh, rule=name, n=n, kstag=kstag, wEnd=wEnd, Mnd_end=mEnd)
            if k is None: rows.append(base | dict(k=None)); continue
            kk = k + off
            rows.append(base | dict(k=kk+1, w1=T['w1'][kk], w1_loss_pct=100*(wEnd-T['w1'][kk])/wEnd,
                                    Mnd=T['Mnd'][kk], dMnd=T['Mnd'][kk]-mEnd, ratio=(kk+1)/max(kstag,1)))
    return rows

if __name__ == '__main__':
    allrows = []
    for method in ['proposed','yuksel','olhoff']:
        R = evaluate(method)
        if not R: continue
        allrows += R
        print(f'\n===== {method} =====')
        for name in RULES:
            d = [r for r in R if r['rule'] == name]
            if not d: continue
            print(f'--- {name}')
            print('  mesh     k    n  kstag  k/kstag     w1   loss%    Mnd  Mnd_end')
            for r in d:
                if r['k'] is None: print(f"  {r['mesh']:8s} NONE {r['n']:4d} {r['kstag']:5d}"); continue
                print(f"  {r['mesh']:8s} {r['k']:4d} {r['n']:4d} {r['kstag']:5d} {r['ratio']:7.2f} {r['w1']:7.2f} {r['w1_loss_pct']:6.2f} {r['Mnd']:6.3f} {r['Mnd_end']:6.3f}")
    if allrows:
        keys = ['method','mesh','rule','k','n','kstag','w1','wEnd','w1_loss_pct','Mnd','Mnd_end','dMnd','ratio']
        with open(os.path.join(OUT,'replay_all.csv'),'w',newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(allrows)
        print('\n===== summary over all methods and meshes =====')
        print(f"{'rule':30s} runs miss  worst_loss%  worst_dMnd  min_ratio  med_ratio  max_ratio")
        summ = []
        for name in RULES:
            d = [r for r in allrows if r['rule'] == name]; hit = [r for r in d if r['k'] is not None]
            if not hit: print(f'{name:30s} {len(d):4d} {len(d):4d}'); continue
            summ.append((max(r['w1_loss_pct'] for r in hit), name, len(d), len(d)-len(hit), max(r['dMnd'] for r in hit),
                         min(r['ratio'] for r in hit), float(np.median([r['ratio'] for r in hit])), max(r['ratio'] for r in hit)))
        for s in sorted(summ):
            print(f"{s[1]:30s} {s[2]:4d} {s[3]:4d} {s[0]:11.2f} {s[4]:11.3f} {s[5]:10.2f} {s[6]:10.2f} {s[7]:10.2f}")

def levels_table():
    """Median metric levels over the 20 iterations after stagnation and over the last 20 recorded."""
    print('\n===== metric levels: median over iterations [kstag, kstag+20) | last 20 recorded =====')
    print(f"{'method':9s}{'mesh':9s}{'kstag':>6s} {'n':>4s} | {'max|dx|':>8s} {'relL2':>8s} {'rms':>8s} {'flip':>8s} | {'max|dx|':>8s} {'relL2':>8s} {'rms':>8s} {'flip':>8s} | {'w1(kstag)':>9s} {'w1 max':>7s} {'w1 end':>7s} {'Mnd stag':>8s} {'Mnd end':>7s}")
    for method in ['proposed','yuksel','olhoff']:
        for mesh in MESHES:
            T = load(method, mesh)
            if T is None: continue
            ks = stagnation(T); n = T['n']
            a = slice(ks, min(ks+20, n)); b = slice(max(0, n-20), n)
            med = lambda key, sl: float(np.nanmedian(T[key][sl]))
            print(f"{method:9s}{mesh:9s}{ks:6d} {n:4d} | {med('dmax_native',a):8.4f} {med('drel',a):8.5f} {med('drms',a):8.5f} {med('flip',a):8.5f} | {med('dmax_native',b):8.4f} {med('drel',b):8.5f} {med('drms',b):8.5f} {med('flip',b):8.5f} | {T['w1'][ks]:9.2f} {np.nanmax(T['w1']):7.2f} {T['w1'][-1]:7.2f} {T['Mnd'][ks]:8.3f} {T['Mnd'][-1]:7.3f}")

if __name__ == '__main__':
    levels_table()
