"""
Olhoff & Du (2014) - Simply-Simply beam (SS)
Natural frequency maximization with SIMP + MMA.
"""

import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from topFreqOptimization_MMA import topFreqOptimization_MMA


def main():
    cfg = dict(
        L=8,
        H=1,
        nelx=240,
        nely=30,
        E0=1e7,
        rho0=1.0,
        rho_min=1e-6,
        nu=0.3,
        t=1.0,
        volfrac=0.5,
        penal=3.0,
        maxiter=300,
        J=3,
        supportType="SS",
        beta_schedule=[1, 2, 4, 8, 16, 32, 64],
        beta_interval=40,
    )
    cfg["Emin"] = max(1e-6 * cfg["E0"], 1e-3)
    cfg["rmin"] = 2 * cfg["L"] / cfg["nelx"]

    opts = dict(doDiagnostic=True, diagnosticOnly=False, diagModes=5)
    paper = dict(init=68.7, opt=174.7)

    omega_best, xPhys_best, diag_out = topFreqOptimization_MMA(cfg, opts)

    print(
        "SS case: omega1 initial={:.1f} (paper {:.1f}) | optimized={:.1f} (paper {:.1f})".format(
            diag_out["initial"]["omega"][0], paper["init"], omega_best, paper["opt"]
        )
    )

    fig, ax = plt.subplots(figsize=(10, 2))
    fig.canvas.manager.set_window_title("Olhoff SS topology")
    ax.imshow(
        1 - xPhys_best.reshape(cfg["nely"], cfg["nelx"]),
        cmap="gray",
        aspect="equal",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.axis("off")
    ax.set_title(f"SS: omega1={omega_best:.1f} (paper: {paper['opt']:.1f})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
