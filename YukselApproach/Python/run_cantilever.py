"""
Yuksel & Yilmaz (2025) - Figure 9 benchmark (cantilever with tip mass).
E0 = 1e7 Pa, rho0 = 1 kg/m^3, nu = 0.3, L:H = 15:10, volfrac = 0.5, mesh = 150x100.
"""

import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from top99neo_inertial_freq import top99neo_inertial_freq

nelx, nely = 150, 100
volfrac = 0.5;  penal = 3;  rmin = 2.3;  ft = 1;  ftBC = 'N'
eta = 0.5;  beta = 1;  move = 0.2
maxit = 200;  stage1_maxit = 200;  bcType = "cantilever"

xPhysStage2, _, info = top99neo_inertial_freq(
    nelx, nely, volfrac, penal, rmin, ft, ftBC, eta, beta, move,
    maxit, stage1_maxit, bcType, nHistModes=3)

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
fig.canvas.manager.set_window_title('Yuksel Figure 9 benchmark')

axes[0].imshow(1 - info['stage1']['xFinal'].reshape(nely, nelx),
               cmap='gray', aspect='equal', vmin=0, vmax=1)
axes[0].axis('off')
axes[0].set_title(r"Figure 9(b): $\omega_1$ = {:.1f} rad/s".format(
    info['stage1']['omega1']))

axes[1].imshow(1 - xPhysStage2.reshape(nely, nelx),
               cmap='gray', aspect='equal', vmin=0, vmax=1)
axes[1].axis('off')
axes[1].set_title(r"Figure 9(c): $\omega_1$ = {:.1f} rad/s".format(
    info['stage2']['omega1']))
fig.tight_layout()

print(f"\nFigure 9 material constants used internally:")
print(f"  E0 = 1e7 Pa, nu = 0.3, rho0 = 1 kg/m^3")
print(f"  Emin = 1e-9*E0, rho_min = 1e-9*rho0, d = 6, x_cut = 0.1")
print(f"  Concentrated mass: 20% of permitted material mass at right mid-edge")
print(f"  Stage 1 frequency: omega1 = {info['stage1']['omega1']:.4f} rad/s (paper: 94.1)")
print(f"  Stage 2 frequency: omega1 = {info['stage2']['omega1']:.4f} rad/s (paper: 101.5)")

plt.show()
