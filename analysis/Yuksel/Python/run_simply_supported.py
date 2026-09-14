"""
Yuksel & Yilmaz (2025) - Figure 4 benchmark (simply supported beam).
E0 = 1e7 Pa, rho0 = 1 kg/m^3, nu = 0.3, L:H = 8:1, volfrac = 0.5, mesh = 320x40.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from YukselApproach.Python.solver import top99neo_dynamic_freq, top99neo_inertial_freq


def _plot_dynamic_frequency_convergence(omega_hist, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if omega_hist is None:
        return
    om = np.asarray(omega_hist, dtype=float)
    if om.ndim != 2 or om.size == 0:
        return
    it = np.arange(1, om.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j in range(min(3, om.shape[1])):
        ax.plot(it, om[:, j], linewidth=2.0, label=f"Mode {j + 1}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()


def main() -> None:
    import matplotlib.pyplot as plt

    nelx, nely = 320, 40
    volfrac = 0.5
    penal = 3.0
    rmin = 2.5
    ft = 1
    ftBC = "N"
    eta = 0.5
    beta = 1.0
    move = 0.2
    maxit = 200
    stage1_maxit = 200
    bcType = "simply"

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
        plot_iterations=True,
        verbose=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    fig.canvas.manager.set_window_title("Yuksel Figure 4 benchmark")

    axes[0].imshow(1 - res.info["stage1"]["xFinal"].reshape((nely, nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[0].axis("off")
    axes[0].set_title(r"Figure 4(b): $\omega_1$ = {:.1f} rad/s".format(res.info["stage1"]["omega1"]))

    axes[1].imshow(1 - res.xPhys_stage2.reshape((nely, nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    axes[1].axis("off")
    axes[1].set_title(r"Figure 4(c): $\omega_1$ = {:.1f} rad/s".format(res.info["stage2"]["omega1"]))
    fig.tight_layout()

    dyn_maxit = 200
    dyn_move = 0.01
    _, dyn_info = top99neo_dynamic_freq(
        nelx=nelx,
        nely=nely,
        volfrac=volfrac,
        penal=penal,
        rmin=rmin,
        ft=ft,
        ftBC=ftBC,
        move=dyn_move,
        maxit=dyn_maxit,
        bcType=bcType,
        verbose=True,
    )
    _plot_dynamic_frequency_convergence(dyn_info.get("omegaHist"), "Figure 6 (simply supported)")

    print("\nFigure 4 material constants used internally:")
    print("  E0 = 1e7 Pa, nu = 0.3, rho0 = 1 kg/m^3")
    print("  Emin = 1e-9*E0, rho_min = 1e-9*rho0, d = 6, x_cut = 0.1")
    print(f"  Stage 1 frequency: omega1 = {res.info['stage1']['omega1']:.4f} rad/s")
    print(f"  Stage 2 frequency: omega1 = {res.info['stage2']['omega1']:.4f} rad/s")
    print(f"  Dynamic code frequency (200 iters): omega1 = {dyn_info['omega1']:.4f} rad/s")

    plt.show()


if __name__ == "__main__":
    main()
