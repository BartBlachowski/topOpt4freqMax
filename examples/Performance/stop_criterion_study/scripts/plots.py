import sys, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay
OUT = sys.argv[1]; FIG = sys.argv[2]; os.makedirs(FIG, exist_ok=True)
replay.OUT = OUT
MESHES = replay.MESHES
cmap = plt.get_cmap('viridis')
LABEL = {'proposed':'Proposed (OC, move 0.2)', 'yuksel':'Yuksel (two-stage OC, move 0.2)', 'olhoff':'Du-Olhoff (nested MMA, adaptive box)'}
for method in ['proposed','yuksel','olhoff']:
    data = [(m, replay.load(method, m)) for m in MESHES]
    data = [(m, T) for m, T in data if T is not None]
    if not data: continue
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for i, (m, T) in enumerate(data):
        c = cmap(i/max(1, len(MESHES)-1)); k = np.arange(1, T['n']+1)
        ax[0,0].semilogy(k, T['dmax_native'], color=c, lw=1, label=m)
        ax[0,1].semilogy(k, T['drel'], color=c, lw=1)
        ax[1,0].plot(k, T['Mnd'], color=c, lw=1)
        ax[1,1].plot(k, T['w1'], color=c, lw=1)
        if method == 'yuksel':
            s2 = np.where(T['stage'] == 2)[0]
            if len(s2): 
                for a in ax.ravel(): a.axvline(s2[0]+1, color=c, lw=0.5, ls=':')
    ax[0,0].set_title(r'max$_e|\Delta x_e|$ per iteration (solver-native)'); ax[0,0].axhline(0.01, color='k', ls='--', lw=0.8); ax[0,0].axhline(0.04, color='k', ls=':', lw=0.8)
    ax[0,1].set_title(r'$\|\Delta x\|_2/\|x\|_2$ per iteration'); ax[0,1].axhline(1e-3, color='k', ls='--', lw=0.8); ax[0,1].axhline(5e-3, color='k', ls=':', lw=0.8)
    ax[1,0].set_title(r'$M_{nd}=4\,\mathrm{mean}(x(1-x))$'); ax[1,1].set_title(r'$\omega_1$ [rad/s] (E1 structural; Olhoff native)')
    for a in ax[1]: a.set_xlabel('iteration')
    ax[0,0].legend(fontsize=8, ncol=3); fig.suptitle(f'{LABEL[method]} - per-iteration stop metrics vs mesh (stop disabled / extended)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f'traces_{method}.png'), dpi=130); plt.close(fig)
    print('wrote', method, [m for m, _ in data])
