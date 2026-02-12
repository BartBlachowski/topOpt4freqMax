"""
plot_dynamic_frequency_convergence.py - Plot dynamic-code omega history (Figure 6 style).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_dynamic_frequency_convergence(omegaHist, caseLabel="Yuksel dynamic benchmark"):
    """
    Plot omega1, omega2, omega3 vs iteration for the dynamic code.

    Parameters
    ----------
    omegaHist : ndarray, shape (nIter, nModes)
        Natural circular frequencies (rad/s) per iteration.
    caseLabel : str
        Title prefix.
    """
    if omegaHist is None or omegaHist.size == 0:
        print("Warning: Empty omega history provided.")
        return

    nIter = omegaHist.shape[0]
    it = np.arange(nIter)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.manager.set_window_title(f"{caseLabel} - Figure 6 history")

    ax.plot(it, omegaHist[:, 0], 'b-', linewidth=2.0, label=r'$\omega_1$')
    if omegaHist.shape[1] >= 2:
        ax.plot(it, omegaHist[:, 1], 'r:', linewidth=2.0, label=r'$\omega_2$')
    if omegaHist.shape[1] >= 3:
        ax.plot(it, omegaHist[:, 2], 'k--', linewidth=2.0, label=r'$\omega_3$')

    ax.set_xlabel('iteration')
    ax.set_ylabel('natural frequency (rad/s)')
    ax.grid(True)
    ax.set_xlim(0, max(200, nIter - 1))
    ax.legend(loc='upper right')
    ax.set_title(f"{caseLabel}: Convergence history (dynamic code)")
    fig.tight_layout()
    return fig
