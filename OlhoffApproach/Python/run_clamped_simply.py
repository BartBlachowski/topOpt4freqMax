"""
Olhoff & Du (2014) - Clamped-Simply beam (CS).
Natural frequency maximization with SIMP + MMA.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Use non-interactive matplotlib backend for better performance
import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OlhoffApproach.Python.solver import OlhoffConfig, run_optimization


def main() -> None:
    import matplotlib.pyplot as plt

    cfg = OlhoffConfig()
    cfg.nelx = 240
    cfg.nely = 30
    cfg.support_type = "CS"
    cfg.maxiter = 300
    cfg.rmin = 2 * cfg.L / cfg.nelx
    cfg.beta_schedule = (1, 2, 4, 8, 16, 32, 64)
    cfg.beta_interval = 40
    cfg.eigs_strategy = "shift-invert"
    cfg.finalize()

    paper_init = 104.1
    paper_opt = 288.7

    res = run_optimization(
        cfg,
        verbose=True,
        do_diagnostic=True,
        diag_modes=5,
        plot_iterations=False,  # Disabled for better performance
    )

    print(
        "CS case: omega1 initial={:.1f} (paper {:.1f}) | optimized={:.1f} (paper {:.1f})".format(
            float(res.diagnostics["initial"]["omega"][0]),
            paper_init,
            res.omega_best,
            paper_opt,
        )
    )

    fig, ax = plt.subplots(figsize=(10, 2.4))
    ax.imshow(1 - res.xPhys_best.reshape((cfg.nely, cfg.nelx), order="F"), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"CS: omega1={res.omega_best:.1f} (paper: {paper_opt:.1f})")
    fig.tight_layout()
    # Save to file instead of showing (Agg backend doesn't support interactive display)
    output_file = Path(__file__).parent / "output" / "CS_final_topology.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Final topology saved to: {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    main()
