"""
plot_frequency_convergence.py - Plot Figure-6-style omega histories for the two-stage method.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_frequency_convergence(info, caseLabel="Yuksel benchmark"):
    """
    Plot omega1, omega2, omega3 vs iteration for the two-stage inertial method.

    Parameters
    ----------
    info : dict
        Returned by top99neo_inertial_freq with nHistModes >= 3.
    caseLabel : str
        Title prefix.
    """
    s1 = info.get('stage1', {})
    s2 = info.get('stage2', {})
    oh1 = s1.get('omegaHist')
    oh2 = s2.get('omegaHist')

    if oh1 is None or oh2 is None or len(oh1) == 0 or len(oh2) == 0:
        print("Warning: No omega history found. "
              "Call top99neo_inertial_freq with nHistModes >= 3.")
        return

    if isinstance(oh1, list):
        oh1 = np.array(oh1)
    if isinstance(oh2, list):
        oh2 = np.array(oh2)

    om = np.vstack([oh1, oh2])
    nIt = om.shape[0]
    it = np.arange(1, nIt + 1)
    iSplit = oh1.shape[0] + 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.canvas.manager.set_window_title(f"{caseLabel} - Frequency Convergence")

    lines = []
    lines.append(ax.plot(it, om[:, 0], 'b-', linewidth=2.0, label=r'$\omega_1$')[0])
    if om.shape[1] >= 2:
        lines.append(ax.plot(it, om[:, 1], 'r:', linewidth=2.0, label=r'$\omega_2$')[0])
    if om.shape[1] >= 3:
        lines.append(ax.plot(it, om[:, 2], 'k--', linewidth=2.0, label=r'$\omega_3$')[0])

    ax.grid(True)
    ax.set_xlabel('iteration')
    ax.set_ylabel('natural frequency (rad/s)')
    ax.legend(loc='upper right')
    ax.set_title(f"{caseLabel}: Convergence history")

    yBase = om[:, 0]
    yBase = yBase[np.isfinite(yBase)]
    if len(yBase) == 0:
        yPos = 0
    else:
        yPos = yBase.min() + 0.05 * max(1, yBase.max() - yBase.min())

    yl = ax.get_ylim()
    ax.plot([iSplit, iSplit], yl, '--', color=(0.4, 0.4, 0.4), linewidth=1.0)
    ax.text(max(2, round(0.15*nIt)), yPos, 'Stage 1',
            color=(0.2, 0.2, 0.2), fontweight='bold')
    ax.text(min(nIt-5, round(0.65*nIt)), yPos, 'Stage 2',
            color=(0.2, 0.2, 0.2), fontweight='bold')

    fig.tight_layout()
    return fig
