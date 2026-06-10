"""
Yuksel & Yilmaz (2025) - Figure 9 benchmark (cantilever with tip mass).
E0 = 1e7 Pa, rho0 = 1 kg/m^3, nu = 0.3, L:H = 15:10, volfrac = 0.5, mesh = 150x100.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from YukselApproach.Python.solver import top99neo_inertial_freq


def main() -> None:
    import matplotlib.pyplot as plt

    nelx, nely = 150, 100
    volfrac = 0.5
    penal = 3.0
    rmin = 2.3
    ft = 1
    ftBC = "N"
    eta = 0.5
    beta = 1.0
    move = 0.2
    maxit = 200
    stage1_maxit = 200
    bcType = "cantilever"

    res = top99neo_inertial_freq(
        nelx=nelx,
        nely=nely,
        volfrac=volfrac,
        penal=penal,
        rmin=rmin,
        ft=ft,
        ftBC=ftBC,
        eta=eta,
        beta=beta,
        move=move,
        maxit=maxit,
        stage1_maxit=stage1_maxit,
        bcType=bcType,
        nHistModes=3,
        plot_iterations=True,
        verbose=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    fig.canvas.manager.set_window_title("Yuksel Figure 9 benchmark")

    axes[0].imshow(1 - res.info["stage1"]["xFinal"].reshape((nely, nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[0].axis("off")
    axes[0].set_title(r"Figure 9(b): $\omega_1$ = {:.1f} rad/s".format(res.info["stage1"]["omega1"]))

    axes[1].imshow(1 - res.xPhys_stage2.reshape((nely, nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[1].axis("off")
    axes[1].set_title(r"Figure 9(c): $\omega_1$ = {:.1f} rad/s".format(res.info["stage2"]["omega1"]))
    fig.tight_layout()

    print("\nFigure 9 material constants used internally:")
    print("  E0 = 1e7 Pa, nu = 0.3, rho0 = 1 kg/m^3")
    print("  Emin = 1e-9*E0, rho_min = 1e-9*rho0, d = 6, x_cut = 0.1")
    print("  Concentrated mass: 20% of permitted material mass at right mid-edge")
    print(f"  Stage 1 frequency: omega1 = {res.info['stage1']['omega1']:.4f} rad/s (paper: 94.1)")
    print(f"  Stage 2 frequency: omega1 = {res.info['stage2']['omega1']:.4f} rad/s (paper: 101.5)")

    plt.show()


if __name__ == "__main__":
    main()
