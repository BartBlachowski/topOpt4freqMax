"""
Yuksel & Yilmaz (2025) - Figure 8 benchmark (fixed-pinned beam).
E0 = 1e7 Pa, rho0 = 1 kg/m^3, nu = 0.3, L:H = 8:1, volfrac = 0.5, mesh = 320x40.
"""

import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from top99neo_inertial_freq import top99neo_inertial_freq

nelx, nely = 320, 40
volfrac = 0.5;  penal = 3;  rmin = 2.0;  ft = 1;  ftBC = 'N'
eta = 0.5;  beta = 1;  move = 0.2
maxit = 400;  stage1_maxit = 200;  bcType = "fixedPinned"

xPhysStage2, _, info = top99neo_inertial_freq(
    nelx, nely, volfrac, penal, rmin, ft, ftBC, eta, beta, move,
    maxit, stage1_maxit, bcType, nHistModes=3)

fig, axes = plt.subplots(2, 1, figsize=(10, 4))
fig.canvas.manager.set_window_title('Yuksel Figure 8 benchmark')

axes[0].imshow(1 - info['stage1']['xFinal'].reshape(nely, nelx),
               cmap='gray', aspect='equal', vmin=0, vmax=1)
axes[0].axis('off')
axes[0].set_title(r"Figure 8(b): $\omega_1$ = {:.1f} rad/s".format(
    info['stage1']['omega1']))

axes[1].imshow(1 - xPhysStage2.reshape(nely, nelx),
               cmap='gray', aspect='equal', vmin=0, vmax=1)
axes[1].axis('off')
axes[1].set_title(r"Figure 8(c): $\omega_1$ = {:.1f} rad/s".format(
    info['stage2']['omega1']))
fig.tight_layout()

print(f"\nFigure 8 material constants used internally:")
print(f"  E0 = 1e7 Pa, nu = 0.3, rho0 = 1 kg/m^3")
print(f"  Emin = 1e-9*E0, rho_min = 1e-9*rho0, d = 6, x_cut = 0.1")
print(f"  Stage 1 load DOF (auto-selected): {info['stage1']['loadDof']}")
print(f"  Stage 1 frequency: omega1 = {info['stage1']['omega1']:.4f} rad/s (paper: 224.6)")
print(f"  Stage 2 frequency: omega1 = {info['stage2']['omega1']:.4f} rad/s (paper: 255.6)")

plt.show()
