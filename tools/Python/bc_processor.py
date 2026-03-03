"""
bc_processor.py — Convert bc.supports JSON entries to fixed DOF indices.

Mirrors tools/Matlab/supportsToFixedDofs.m.

Node numbering (0-based, column-major):
    node n at grid position (i=col, j=row):
        n = i*(nely+1) + j,  i = 0..nelx,  j = 0..nely
        x = i*(L/nelx),      y = j*(H/nely)
    DOFs (0-based):  ux = 2*n,  uy = 2*n+1

Processed types : vertical_line, horizontal_line, closest_point.
Ignored types   : hinge, clamp  (handled by solver's own BC builder).
"""

from __future__ import annotations
import numpy as np


def supports_to_fixed_dofs(
    supports: list[dict],
    nelx: int,
    nely: int,
    L: float,
    H: float,
) -> np.ndarray:
    """Convert bc.supports entries to an array of 0-based fixed DOF indices.

    Parameters
    ----------
    supports : list of dicts
        Each dict must have at least {"type": ..., "dofs": [...]} and
        type-specific fields (x, y, or location).
    nelx, nely : int
        Number of elements in x and y.
    L, H : float
        Physical domain dimensions.

    Returns
    -------
    np.ndarray, dtype int64
        Sorted unique 0-based fixed DOF indices.
    """
    dx = L / nelx
    dy = H / nely
    n_nodes = (nelx + 1) * (nely + 1)

    # Build coordinate arrays for all nodes (column-major).
    node_idx = np.arange(n_nodes, dtype=np.int64)
    i_arr = node_idx // (nely + 1)   # 0-based column (x-direction)
    j_arr = node_idx % (nely + 1)    # 0-based row    (y-direction)
    x_arr = i_arr * dx
    y_arr = j_arr * dy

    fixed_dofs = []

    for k, s in enumerate(supports):
        t = s.get("type", "").strip().lower()

        if t in {"hinge", "clamp"}:
            continue  # handled by solver's own BC builder

        if t == "vertical_line":
            x0 = float(s["x"])
            tol = float(s.get("tol", 1e-9))
            mask = np.abs(x_arr - x0) <= tol

        elif t == "horizontal_line":
            y0 = float(s["y"])
            tol = float(s.get("tol", 1e-9))
            mask = np.abs(y_arr - y0) <= tol

        elif t == "closest_point":
            loc = np.asarray(s["location"], dtype=float).ravel()
            dist2 = (x_arr - loc[0]) ** 2 + (y_arr - loc[1]) ** 2
            min_d2 = dist2.min()
            # Tie-break: smallest node index.
            sel_node = int(node_idx[dist2 == min_d2].min())
            mask = node_idx == sel_node

        else:
            continue  # unknown type — skip silently

        sel_nodes = node_idx[mask]
        entry_dofs = _parse_dof_names(s.get("dofs", []), sel_nodes)
        fixed_dofs.append(entry_dofs)
        print(f"  BC[{k}] {t}: {sel_nodes.size} nodes, {entry_dofs.size} dofs fixed")

    if not fixed_dofs:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(fixed_dofs).astype(np.int64))


# ---------------------------------------------------------------------------
def _parse_dof_names(dofs_field, nodes: np.ndarray) -> np.ndarray:
    """Convert dofs list (e.g. ["ux","uy"]) to 0-based DOF indices."""
    if isinstance(dofs_field, str):
        names = [dofs_field]
    elif isinstance(dofs_field, list):
        names = [str(d) for d in dofs_field]
    else:
        names = []

    nodes = nodes.astype(np.int64)
    dofs = []
    for nm in names:
        nm = nm.strip().lower()
        if nm == "ux":
            dofs.append(2 * nodes)
        elif nm == "uy":
            dofs.append(2 * nodes + 1)

    if not dofs:
        return np.array([], dtype=np.int64)
    return np.concatenate(dofs)
