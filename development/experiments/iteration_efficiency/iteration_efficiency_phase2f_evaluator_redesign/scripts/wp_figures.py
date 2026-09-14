#!/usr/bin/env python3
"""Required figures 1-9 for Phase 2F (figure 10, the mode-shape atlas, is
produced by wp6_atlas.py).  Consumes survey.npz and the CSV outputs.  READ-ONLY."""
import os, csv, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/iteration_efficiency_phase2f_evaluator_redesign')
FIG=os.path.join(OUT,'figures'); os.makedirs(FIG,exist_ok=True)
Z=np.load(os.path.join(OUT,'scripts','survey.npz'))
GF=np.load(os.path.join(OUT,'scripts','gate_full.npz'))   # full-coverage hard gate
TAUS=Z['TAUS']
IT=int(np.argmin(np.abs(TAUS-0.10)))
G=lambda c,f: Z[f'{c}|{f}']
def ordinals(cfg,cut=0.5,ti=IT):
    v=G(cfg,'keLow')[:,ti,:]; om=G(cfg,'omega')
    o=np.full(v.shape[0],np.nan); ws=np.full(v.shape[0],np.nan)
    for s in range(v.shape[0]):
        w=np.flatnonzero(np.isfinite(v[s])&(v[s]<cut))
        if w.size: o[s]=w[0]+1; ws[s]=om[s,w[0]]
    return o,ws,om[:,0]

# ---- fig 1 : k=252 spectrum, Eq.(4) vs Eq.(4a) -------------------------------
R=list(csv.DictReader(open(os.path.join(OUT,'K252_MODAL_REPRODUCTION.csv'))))
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,law,ttl in ((axes[0],'eq4','Eq. (4) — original'),(axes[1],'eq4a','Eq. (4a) — continuous')):
    r=[q for q in R if q['model']=='E2' and q['mass_law']==law]
    om=np.array([float(q['omega']) for q in r]); vk=np.array([float(q['void_KE_share_tau0p1']) for q in r])
    art=vk>0.5
    ax.bar(np.arange(1,len(om)+1)[~art],om[~art],color='#2166ac',label='structural (void KE < 0.5)')
    ax.bar(np.arange(1,len(om)+1)[art],om[art],color='#b2182b',label='void-localised (void KE > 0.5)')
    for i,(o,v) in enumerate(zip(om,vk),1):
        ax.text(i,o+8,f'{v:.3f}',ha='center',fontsize=6,rotation=90)
    ax.set_title(f'E2, {ttl}',fontsize=10); ax.set_xlabel('mode ordinal'); ax.grid(alpha=.25,axis='y')
axes[0].set_ylabel(r'$\omega$'); axes[0].legend(fontsize=8)
fig.suptitle('Figure 1 — 160x20 state k=252: modal spectrum and void kinetic-energy share',fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig01_k252_spectrum.png'),dpi=130); plt.close(fig)

# ---- fig 2 : void-KE distribution, artificial vs structural -------------------
cfgs=[c[:-6] for c in Z.files if c.endswith('|omega')]
eq4a=[c for c in cfgs if c.endswith('eq4a')]; eq4=[c for c in cfgs if c.endswith('|eq4')]
v_all=np.concatenate([G(c,'keLow')[:,IT,:].ravel() for c in eq4a])
v_all=v_all[np.isfinite(v_all)]
fig,ax=plt.subplots(figsize=(9,4))
ax.hist(np.clip(v_all,1e-12,1),bins=np.logspace(-12,0,120),color='#4d4d4d')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'void kinetic-energy share of a mode, $\tau=0.1$'); ax.set_ylabel('modes')
ax.axvspan(0.05,0.95,color='#fdae61',alpha=.35,label='gap region (0.05 – 0.95)')
ax.legend(fontsize=9); ax.grid(alpha=.25)
ax.set_title(f'Figure 2 — void kinetic-energy share over all Eq. (4a) modes surveyed (n={v_all.size:,})',fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig02_voidKE_distribution.png'),dpi=130); plt.close(fig)

# ---- fig 3 : first structural ordinal vs iteration ---------------------------
sel=[c for c in eq4a if c.startswith('160x20')]+[c for c in cfgs if c=='160x20|E1|linear']
fig,ax=plt.subplots(figsize=(11,4))
for c in sel:
    o,_,_=ordinals(c); ks=G(c,'k')
    ax.step(ks,o,where='mid',lw=1.1,label=c.replace('|','  '))
ax.set_xlabel('iteration k'); ax.set_ylabel('ordinal of first structural mode')
ax.set_yticks(range(1,9)); ax.grid(alpha=.25); ax.legend(fontsize=8)
ax.set_title('Figure 3 — 160x20: ordinal of the first structural mode along the trajectory',fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig03_first_structural_ordinal.png'),dpi=130); plt.close(fig)

# ---- fig 4 : lowest algebraic vs first structural -----------------------------
c='160x20|E2|eq4a'; o,ws,low=ordinals(c); ks=G(c,'k')
c4='160x20|E2|eq4'; _,_,low4=ordinals(c4); ks4=G(c4,'k')
fig,ax=plt.subplots(figsize=(11,4.2))
ax.plot(ks,low,lw=.9,color='#b2182b',label='candidate B: Eq. (4a) lowest algebraic mode')
ax.plot(ks,ws,lw=1.1,color='#2166ac',label='candidate C: Eq. (4a) first structural mode')
ax.plot(ks4,low4,lw=.9,color='#1a9850',ls='--',label='candidate A: Eq. (4) lowest mode')
ax.set_xlabel('iteration k'); ax.set_ylabel(r'$\omega_1$'); ax.grid(alpha=.25); ax.legend(fontsize=8)
ax.set_title('Figure 4 — 160x20 E2: lowest algebraic vs first structural eigenfrequency',fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig04_lowest_vs_structural.png'),dpi=130); plt.close(fig)

# ---- fig 5 : threshold-stability map -----------------------------------------
S=list(csv.DictReader(open(os.path.join(OUT,'STRUCTURAL_MODE_THRESHOLD_SWEEP.csv'))))
S=[r for r in S if r['mesh']=='160x20' and r['model']=='E2']
taus=sorted({float(r['tau']) for r in S}); cuts=sorted({float(r['cut']) for r in S})
Mmap=np.full((len(taus),len(cuts)),np.nan)
for r in S:
    Mmap[taus.index(float(r['tau'])),cuts.index(float(r['cut']))]=int(r['states_with_valid_mode'])
fig,ax=plt.subplots(figsize=(11,4))
im=ax.imshow(Mmap,aspect='auto',origin='lower',cmap='viridis')
ax.set_xticks(range(0,len(cuts),2)); ax.set_xticklabels([f'{cuts[i]:g}' for i in range(0,len(cuts),2)],rotation=90,fontsize=7)
ax.set_yticks(range(len(taus))); ax.set_yticklabels([f'{t:g}' for t in taus],fontsize=8)
ax.set_xlabel('modal-validity cut on void KE'); ax.set_ylabel(r'diagnostic partition $\tau$')
fig.colorbar(im,ax=ax,label='states with a valid structural mode')
ax.set_title('Figure 5 — 160x20 E2 Eq. (4a): threshold-stability map',fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig05_threshold_map.png'),dpi=130); plt.close(fig)

# ---- fig 6 : gray structural vs binary ---------------------------------------
gp=os.path.join(OUT,'GRAY_VS_BINARY_QUALITY.csv')
if os.path.exists(gp):
    B=[r for r in csv.DictReader(open(gp)) if r['mesh']=='160x20']
    ks=np.array([int(r['state']) for r in B])
    gs=np.array([float(r['gray_struct_omega_E2']) for r in B])
    bo=np.array([float(r['binary_omega_E2']) for r in B])
    gl=np.array([float(r['gray_lowest_omega_E2']) for r in B])
    fig,axes=plt.subplots(2,1,figsize=(11,6.5),sharex=True)
    axes[0].plot(ks,gs,lw=1,color='#2166ac',label='gray structural (candidate C)')
    axes[0].plot(ks,bo,lw=1,color='#d6604d',label='binary exact-count (candidate D)')
    axes[0].plot(ks,gl,lw=.7,color='#999999',ls=':',label='gray lowest algebraic (candidate B)')
    axes[0].set_ylabel(r'$\omega_1$'); axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)
    axes[1].plot(ks,(bo-gs)/gs*100,lw=1,color='#762a83')
    axes[1].axhline(0,color='k',lw=.6)
    for _bnd,_col in ((0.5,'#fdae61'),(1,'#f46d43'),(2,'#d73027')):
        axes[1].axhline(_bnd,color=_col,lw=.7,ls='--'); axes[1].axhline(-_bnd,color=_col,lw=.7,ls='--')
    axes[1].set_ylabel('(binary - gray structural) / gray structural  [%]')
    axes[1].set_xlabel('iteration k'); axes[1].grid(alpha=.25)
    fig.suptitle('Figure 6 — 160x20 E2: gray structural vs exact-count binary quality',fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig06_gray_vs_binary.png'),dpi=130); plt.close(fig)

# ---- fig 7 : candidate A/B/C/D evolution -------------------------------------
if os.path.exists(gp):
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.plot(ks4,low4,lw=.9,color='#1a9850',label='A: Eq. (4) lowest')
    ax.plot(G(c,'k'),low,lw=.9,color='#b2182b',label='B: Eq. (4a) lowest')
    ax.plot(G(c,'k'),ws,lw=1.1,color='#2166ac',label='C: Eq. (4a) first structural')
    ax.plot(ks,bo,lw=1.1,color='#d6604d',ls='--',label='D: exact-count binary')
    ax.set_xlabel('iteration k'); ax.set_ylabel(r'$\omega_1$'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    ax.set_title('Figure 7 — 160x20 E2: candidate A/B/C/D frequency evolution',fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig07_candidate_evolution.png'),dpi=130); plt.close(fig)

# ---- fig 8 : hard-gate PASS states with invalid lowest mode -------------------
hk=GF['160x20|GATE2|k']; hard=GF['160x20|GATE2|hard'].astype(bool)
o,_,_=ordinals('160x20|E2|eq4a'); ks2=G('160x20|E2|eq4a','k')
m={int(a):b for a,b in zip(ks2,o)}
inv=np.array([m.get(int(a),np.nan)>1 for a in hk])
fig,ax=plt.subplots(figsize=(11,2.6))
ax.fill_between(hk,0,1,where=hard,color='#c7e9c0',step='mid',label='hard gate PASS')
ax.fill_between(hk,0,1,where=(hard&inv),color='#b2182b',step='mid',label='hard gate PASS but lowest mode void-localised')
ax.set_yticks([]); ax.set_xlabel('iteration k'); ax.legend(fontsize=8,loc='upper right')
ax.set_title(f'Figure 8 — 160x20 E2 Eq. (4a): {int((hard&inv).sum())} of {int(hard.sum())} '
             f'hard-gate-passing states have a void-localised lowest mode',fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig08_hardgate_vs_modal.png'),dpi=130); plt.close(fig)

# ---- fig 9 : mode-count sufficiency ------------------------------------------
fig,ax=plt.subplots(figsize=(9,4))
allo=[]
for cc in eq4a:
    oo,_,_=ordinals(cc); allo.append(oo[np.isfinite(oo)])
allo=np.concatenate(allo)
vals,cnts=np.unique(allo,return_counts=True)
cum=np.cumsum(cnts)/cnts.sum()*100
ax.bar(vals,cnts,color='#4393c3'); ax.set_yscale('log')
ax.set_xlabel('ordinal of the first structural mode'); ax.set_ylabel('states (log)')
ax2=ax.twinx(); ax2.plot(vals,cum,color='#b2182b',marker='o',ms=3); ax2.set_ylabel('cumulative % of states',color='#b2182b')
ax2.set_ylim(0,101)
for v,cu in zip(vals,cum):
    if cu<100: ax2.annotate(f'{cu:.3f}%',(v,cu),fontsize=7,color='#b2182b',xytext=(2,-10),textcoords='offset points')
ax.set_title(f'Figure 9 — mode-count sufficiency: first structural ordinal, all Eq. (4a) states surveyed (n={allo.size:,})',fontsize=10)
ax.grid(alpha=.25,axis='y')
fig.tight_layout(); fig.savefig(os.path.join(FIG,'fig09_mode_count_sufficiency.png'),dpi=130); plt.close(fig)
print('figures written')
