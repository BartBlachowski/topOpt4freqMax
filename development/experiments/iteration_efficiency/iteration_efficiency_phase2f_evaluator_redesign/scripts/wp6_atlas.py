#!/usr/bin/env python3
"""
WP6 - mode-shape atlas.  Renders representative artificial, structural and
boundary-case modes with identical plotting conventions, as supporting physical
evidence for the energy-based classification.  READ-ONLY.
"""
import sys, os, h5py, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_engine import modes, mesh_data
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')
FIG=os.path.join(OUT,'figures')

def panel(ax, arr, nelx, nely, title, cmap, norm=None, vmin=None, vmax=None):
    im=ax.imshow(arr.reshape(nelx,nely).T, origin='lower', aspect='auto',
                 cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    return im

def atlas(mesh,nx,ny,state,model,law,nmodes,fname,note=''):
    with h5py.File(os.path.join(REPO,f'examples/Performance/final_campaign/raw/olhoff/s1_{mesh}.mat'),'r') as f:
        z=np.clip(np.float64(f['res/rho_snapshots'][state]),0,1)
    r=modes(z,nx,ny,model,law,k=nmodes)
    low=r['zeff']<=0.1
    v=r['ke_n'][low,:].sum(axis=0)
    dwp=r['ke_n'].T@r['zeff']
    fig,axes=plt.subplots(nmodes+1,1,figsize=(9,1.35*(nmodes+1)))
    panel(axes[0], z, nx, ny, f'{mesh} state k={state} — density field  '
          f'(rho<=0.1 in {100*(z<=0.1).mean():.0f}% of elements)', 'gray_r', vmin=0, vmax=1)
    for j in range(nmodes):
        ke=np.maximum(r['ke_n'][:,j],1e-12)
        cls='ARTIFICIAL (void-localised)' if v[j]>0.5 else 'structural'
        panel(axes[j+1], ke, nx, ny,
              f'mode {j+1}:  omega={r["omega"][j]:.3f}   voidKE={v[j]:.6f}   '
              f'density-participation={dwp[j]:.3f}   [{cls}]',
              'inferno', norm=LogNorm(vmin=1e-8, vmax=1.0))
    fig.suptitle(f'{model} / {law} — modal kinetic-energy density (log scale, identical convention). {note}',
                 fontsize=9, y=0.998)
    fig.tight_layout(rect=[0,0,1,0.985])
    fig.savefig(os.path.join(FIG,fname), dpi=110); plt.close(fig)
    return r,v,dwp

os.makedirs(FIG,exist_ok=True)
print('rendering atlas...')
atlas('160x20',160,20,252,'E2','eq4a',8,'atlas_k252_E2_eq4a.png',
      'Three void modes precede the structural pair.')
atlas('160x20',160,20,252,'E2','eq4',8,'atlas_k252_E2_eq4.png',
      'Eq. (4) suppression pushes every void mode out of the low spectrum.')
atlas('160x20',160,20,252,'E3','eq4a',8,'atlas_k252_E3_eq4a.png',
      'Four void modes precede the structural pair.')
atlas('160x20',160,20,252,'E1','linear',8,'atlas_k252_E1_linear.png',
      'Frozen E1: void modes exist but sit ABOVE the structural pair.')
atlas('160x20',160,20,1600,'E2','eq4a',6,'atlas_k1600_E2_eq4a.png',
      'Converged state: no void mode below the structure.')
print('done')
