#!/usr/bin/env python3
"""WP1 - independent reproduction of the Phase-2E k=252 modal finding, with a
dense-LAPACK cross-check of the iterative solver.  READ-ONLY."""
import sys, os, csv, h5py, numpy as np, scipy.sparse as sp, scipy.linalg as la
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modal_engine import (mesh_data, interp, modes, localisation,
                          density_weighted_participation, inverse_participation_ratio)
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')

with h5py.File(os.path.join(REPO,'examples/Performance/final_campaign/raw/olhoff/s1_160x20.mat'),'r') as f:
    x=np.float64(f['res/rho_snapshots'][252])
z=np.clip(x,0,1); nelx,nely=160,20
print(f'state k=252, mesh 160x20: {z.size} elements, {int((z<=0.1).sum())} with rho<=0.1 '
      f'({100*(z<=0.1).mean():.1f}%), min={z.min():g} max={z.max():g} mean={z.mean():.6f}')

NM=14
rows=[]
CONFIGS=[('E1','linear','FROZEN E1 (linear mass, no branch)'),
         ('E2','eq4','FROZEN E2 = candidate A'),('E2','eq4a','candidate B/C E2'),
         ('E3','eq4','FROZEN E3 = candidate A'),('E3','eq4a','candidate B/C E3'),
         ('E1','eq4a','DIAGNOSTIC VARIANT - not frozen E1')]
for model,law,_role in CONFIGS:
    if True:
        r=modes(z,nelx,nely,model,law,k=NM)
        ke01,se01=localisation(r,0.1)
        dwp=density_weighted_participation(r); ipr=inverse_participation_ratio(r)
        for j in range(NM):
            rows.append(dict(state=252,mesh='160x20',model=model,mass_law=law,role=_role,mode_ordinal=j+1,
                eigenvalue=r['lam'][j],omega=r['omega'][j],frequency_Hz=r['freq'][j],
                total_modal_kinetic_energy=r['ke_tot'][j],total_modal_strain_energy=r['se_tot'][j],
                void_KE_share_tau0p1=ke01[j],solid_KE_share_tau0p1=1-ke01[j],
                void_SE_share_tau0p1=se01[j],
                density_weighted_participation=dwp[j],inverse_participation_ratio=ipr[j]))
with open(os.path.join(OUT,'K252_MODAL_REPRODUCTION.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def show(model,law,n=8):
    rr=[x for x in rows if x['model']==model and x['mass_law']==law][:n]
    print(f'\n{model} / {law}:')
    print(f"  {'m':>2} {'omega':>10} {'freq_Hz':>10} {'voidKE':>9} {'voidSE':>9} {'dens-part':>10} {'IPR':>9}")
    for q in rr:
        tag='  <-- void-localised' if q['void_KE_share_tau0p1']>0.5 else ''
        print(f"  {q['mode_ordinal']:2d} {q['omega']:10.4f} {q['frequency_Hz']:10.4f} "
              f"{q['void_KE_share_tau0p1']:9.6f} {q['void_SE_share_tau0p1']:9.6f} "
              f"{q['density_weighted_participation']:10.6f} {q['inverse_participation_ratio']:9.6f}{tag}")
for m,l,role in CONFIGS:
    print(f'\n=== {role} ==='); show(m,l)

# ---- dense LAPACK cross-check of the iterative solver, E2/eq4a --------------
md=mesh_data(nelx,nely); Ee,rr_,_=interp(z,'E2','eq4a'); ndof,free=md['ndof'],md['free']
K=sp.coo_matrix(((md['KEv'][None,:]*Ee[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
M=sp.coo_matrix(((md['MEv'][None,:]*rr_[:,None]).ravel(),(md['rows'],md['cols'])),shape=(ndof,ndof)).tocsr()
Kd=np.asarray((((K+K.T)*.5)[free][:,free]).todense()); Md=np.asarray((((M+M.T)*.5)[free][:,free]).todense())
lam=la.eigh(Kd,Md,eigvals_only=True)
w_dense=np.sqrt(np.maximum(lam[:NM],0))
w_arp=np.array([q['omega'] for q in rows if q['model']=='E2' and q['mass_law']=='eq4a'])
print('\ndense LAPACK cross-check, E2/eq4a (no iteration, no shift):')
print('  ARPACK :',np.round(w_arp,6))
print('  LAPACK :',np.round(w_dense,6))
print(f'  max relative difference: {np.max(np.abs(w_arp-w_dense)/w_dense):.3e}')

# ---- Phase-2E / Phase-2D comparison targets --------------------------------
P=list(csv.DictReader(open(os.path.join(REPO,
  'analysis/iteration_efficiency_phase2d_evaluator_amendment/AMENDED_OLHOFF_TRAJECTORY_EVALUATION.csv'))))
p=P[251]
print('\ncomparison with the frozen-pipeline values at k=252:')
SEL=lambda m,l:[q for q in rows if q['model']==m and q['mass_law']==l][0]['omega']
for tag,key,mine in (('E1 frozen (linear)','old_E1',SEL('E1','linear')),
                     ('E2 Eq.(4)         ','old_E2',SEL('E2','eq4')),
                     ('E3 Eq.(4)         ','old_E3',SEL('E3','eq4')),
                     ('E1 frozen (linear)','new_E1',SEL('E1','linear')),
                     ('E2 Eq.(4a)        ','new_E2',SEL('E2','eq4a')),
                     ('E3 Eq.(4a)        ','new_E3',SEL('E3','eq4a'))):
    ref=float(p[key]); print(f'  {tag}: Phase-2D MATLAB {ref:12.6f}   this phase {mine:12.6f}   rel {abs(mine-ref)/ref:.3e}')
