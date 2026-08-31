#!/usr/bin/env python3
"""Required figures for the nested-MMA route audit."""
import os, csv, h5py, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO='/Users/piotrek/Programming/topOpt4freqMax'
OUT=os.path.join(REPO,'analysis/olhoff_nested_mma_route_audit'); FIG=os.path.join(OUT,'figures')
D='/Volumes/HP911Pro/Combobulating/Olhoff/results'
R=list(csv.DictReader(open(os.path.join(OUT,'BASE_MMA_HISTORY.csv'))))
g=lambda k,t=float: np.array([t(r[k]) for r in R])
it=g('outer',int); w1=g('omega1'); w2=g('omega2'); w3=g('omega3'); N=g('N',int)
ni=g('nInner',int); cum=g('cumInner',int); mdr=g('maxdrho')
conv=np.array([r['innerConv']=='True' for r in R]); gap=(w2-w1)/w1*100

# fig 1: spectrum + N
fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True,height_ratios=[3,1,1])
ax[0].plot(it,w1,lw=.9,label=r'$\omega_1$',color='#2166ac')
ax[0].plot(it,w2,lw=.9,label=r'$\omega_2$',color='#d6604d')
ax[0].plot(it,w3,lw=.9,label=r'$\omega_3$',color='#1a9850')
ax[0].axhline(174.7,ls=':',color='k',lw=.8,label='paper reported max 174.7')
ax[0].axvline(56,ls='--',color='#888',lw=.8); ax[0].axvline(231,ls='--',color='#555',lw=.8)
ax[0].annotate('first N=2 (56)',(56,90),fontsize=7,rotation=90)
ax[0].annotate('persistent N=2 (231)',(231,90),fontsize=7,rotation=90)
ax[0].set_ylabel(r'$\omega$'); ax[0].legend(fontsize=8,ncol=4); ax[0].grid(alpha=.25)
ax[0].set_title('BASE_mma_160x20 — spectral trajectory (752 of declared 800 outer iterations)',fontsize=11)
ax[1].step(it,N,where='mid',lw=.9,color='#762a83'); ax[1].set_ylabel('N'); ax[1].set_yticks([1,2]); ax[1].grid(alpha=.25)
ax[2].plot(it,gap,lw=.8,color='#333'); ax[2].axhline(5,ls='--',color='#d73027',lw=.8,label='tolMult = 5%')
ax[2].axhline(1,ls=':',color='#4575b4',lw=.8,label='1% near-bimodal')
ax[2].set_ylabel(r'rel. gap $(\omega_2-\omega_1)/\omega_1$ [%]'); ax[2].set_xlabel('outer iteration')
ax[2].legend(fontsize=8); ax[2].grid(alpha=.25); ax[2].set_ylim(0,8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig1_spectrum_N_gap.png',dpi=130); plt.close(fig)

# fig 2: inner work
fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
c=np.where(conv,'#4393c3','#d73027')
ax[0].scatter(it,ni,s=4,c=c,linewidths=0)
ax[0].axhline(300,ls='--',color='#d73027',lw=.8,label='maxInner = 300 (cap)')
ax[0].axvline(56,ls='--',color='#888',lw=.8)
ax[0].set_ylabel('inner MMA iterations'); ax[0].grid(alpha=.25)
ax[0].legend(fontsize=8)
ax[0].set_title('Inner MMA work per outer iteration (blue = declared converged, red = cap-hit / not converged)',fontsize=10)
ax[1].plot(it,cum,lw=1.1,color='#2166ac')
ax[1].set_ylabel('cumulative inner MMA iterations'); ax[1].set_xlabel('outer iteration'); ax[1].grid(alpha=.25)
ax[1].annotate(f'total {cum[-1]:,} sub-iterates\nover {it[-1]} outer iterations',
               (it[-1]*0.55,cum[-1]*0.35),fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/fig2_inner_work.png',dpi=130); plt.close(fig)

# fig 3: inner work distribution N=1 vs N=2
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].hist(ni[N==1],bins=np.arange(20,320,10),color='#4393c3',label=f'N=1 (n={int((N==1).sum())})')
ax[0].hist(ni[N==2],bins=np.arange(20,320,10),color='#d6604d',alpha=.65,label=f'N=2 (n={int((N==2).sum())})')
ax[0].set_xlabel('inner MMA iterations'); ax[0].set_ylabel('outer iterations'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
ax[0].set_title('Inner effort by multiplicity regime',fontsize=10)
bp=ax[1].boxplot([ni[N==1],ni[N==2]],labels=['N=1','N=2'],showmeans=True)
ax[1].axhline(300,ls='--',color='#d73027',lw=.8); ax[1].set_ylabel('inner MMA iterations'); ax[1].grid(alpha=.25)
ax[1].set_title(f'mean {ni[N==1].mean():.1f} vs {ni[N==2].mean():.1f}  (ratio {ni[N==2].mean()/ni[N==1].mean():.2f})',fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/fig3_inner_distribution.png',dpi=130); plt.close(fig)

# fig 4: LP vs MMA trajectory
with h5py.File(f'{D}/lprmin1.2.mat','r') as h:
    lw=np.array(h['res/hist/omega'][()]); lp_w1=lw[:,0]; lp_w2=lw[:,1]
with h5py.File(f'{D}/fm_mma_diag.mat','r') as h:
    mw=np.array(h['res/hist/omega'][()])
fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
ax[0].plot(np.arange(1,len(lp_w1)+1),lp_w1,lw=.9,color='#2166ac',label=r'LP $\omega_1$ (rmin=1.2, move=0.005, 1600 outer)')
ax[0].plot(np.arange(1,len(lp_w2)+1),lp_w2,lw=.7,color='#92c5de',label=r'LP $\omega_2$')
ax[0].plot(it,w1,lw=.9,color='#b2182b',label=r'MMA $\omega_1$ (BASE_mma, move=0.01, 752 outer)')
ax[0].plot(it,w2,lw=.7,color='#f4a582',label=r'MMA $\omega_2$')
ax[0].set_ylabel(r'$\omega$'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.25); ax[0].set_ylim(60,200)
ax[0].set_title('LP vs nested-MMA at 160x20 with the SAME effective filter radius (1.2 elements)',fontsize=11)
lgap=(lp_w2-lp_w1)/lp_w1*100
ax[1].plot(np.arange(1,len(lgap)+1),lgap,lw=.8,color='#2166ac',label='LP relative gap')
ax[1].plot(it,gap,lw=.8,color='#b2182b',label='MMA relative gap')
ax[1].axhline(5,ls='--',color='#999',lw=.8); ax[1].axhline(1,ls=':',color='#4575b4',lw=.8)
ax[1].set_ylabel('rel. gap [%]'); ax[1].set_xlabel('outer iteration'); ax[1].legend(fontsize=8)
ax[1].grid(alpha=.25); ax[1].set_ylim(0,8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_lp_vs_mma.png',dpi=130); plt.close(fig)
print('figures written')
