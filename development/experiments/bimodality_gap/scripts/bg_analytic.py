#!/usr/bin/env python3
"""Analytic tables for the bimodality-gap study (no solver runs).

1. material_laws.csv   : stiffness/mass interpolation factors of the two implemented laws
                         (olh.material.stiffnessInterpolation / massInterpolation), and the
                         mass/stiffness ratio, at selected densities.
2. filter_kernel.csv   : the discrete Sigmund kernel prepFilter builds for R = 0.06 at each
                         mesh (rmin = R/h elements): self-weight share, number of neighbours,
                         RMS radius in physical units, vs the continuous cone (0.5477 R).
"""
import numpy as np, csv, os
here = os.path.dirname(os.path.abspath(__file__)); data = os.path.join(os.path.dirname(here), 'data')

# ---- 1. material laws (p = 3, rho0 = 0.1, eq.(4b) c1 = 6e5, c2 = -5e6) -------------
def gK_simp(r): return r**3
def gK_ped(r):  return np.where(r < 0.1, r*0.1**2, r**3)
def gM_lin(r):  return r
def gM_4b(r):   return np.where(r <= 0.1, 6e5*r**6 - 5e6*r**7, r)
rhos = [1.0, 0.5, 0.2, 0.1, 0.09, 0.05, 0.02, 0.01, 0.001]
with open(os.path.join(data, 'material_laws.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['rho','gK_simp','gK_pedersen','gM_eq4b','gM_linear','ratio_simp_eq4b','ratio_pedersen_linear'])
    for r in rhos:
        r = np.float64(r)
        w.writerow([r, gK_simp(r), gK_ped(r), gM_4b(r), gM_lin(r), gM_4b(r)/gK_simp(r), gM_lin(r)/gK_ped(r)])
# maximum of the SIMP/eq4b ratio below 0.1
rr = np.linspace(1e-4, 0.1, 100001); q = gM_4b(rr)/gK_simp(rr)
print('max mass/stiffness ratio SIMP+eq4b below 0.1: %.3f at rho=%.4f; Pedersen+linear: %.1f (constant)' % (q.max(), rr[q.argmax()], 100.0))

# ---- 2. discrete filter kernel --------------------------------------------------------
R = 0.06; b = 1.0
rows = []
for nelx, nely in [(160,20),(240,30),(320,40),(400,50),(480,60),(560,70),(640,80),(720,90),(800,100)]:
    h = b/nely; rmin = R/h
    s = int(np.ceil(rmin)) - 1
    ii, jj = np.mgrid[-s:s+1, -s:s+1]
    d = np.sqrt(ii**2 + jj**2)
    wgt = np.maximum(0.0, rmin - d)          # interior element, full stencil
    self_share = wgt[s, s]/wgt.sum()
    nnb = int((wgt > 0).sum() - 1)
    rms_el = np.sqrt((wgt*d**2).sum()/wgt.sum())
    rows.append([nelx, nely, h, rmin, nnb, self_share, rms_el, rms_el*h, rms_el*h/R])
cont = np.sqrt(0.3)   # RMS radius / R of the continuous 2-D cone kernel
with open(os.path.join(data, 'filter_kernel.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['nelx','nely','h','rmin_el','n_neighbours','self_weight_share','rms_radius_el','rms_radius_phys','rms_radius_over_R'])
    for r in rows: w.writerow(r)
    w.writerow(['continuous','cone','','','', 0.0, '', cont*R, cont])
for r in rows: print('%dx%d rmin=%.2f nnb=%d self=%.3f rms_phys=%.4f (cone %.4f)' % (r[0], r[1], r[3], r[4], r[5], r[7], cont*R))
