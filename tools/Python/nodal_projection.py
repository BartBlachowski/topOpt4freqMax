"""
nodal_projection.py — Average Q4 element densities onto mesh nodes.

Mirrors tools/Matlab/projectQ4ElementDensityToNodes.m.

Canonical mapping (column-major in x, 0-based):
    node n at (col i, row j): n = i*(nely+1) + j
    element corners: n1(LL), n2(LR), n3(UR), n4(UL)
        n1 = (nely+1)*elx + ely
        n2 = (nely+1)*(elx+1) + ely
        n3 = n2 + 1
        n4 = n1 + 1
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csc_array


def build_nodal_projection_cache(nelx: int, nely: int) -> dict:
    """Build (and cache) the Q4 element-to-node averaging operator.

    Returns
    -------
    dict with keys:
        "nelx", "nely", "n_el", "n_nodes",
        "elem_nodes"   : (n_el, 4) int array — node indices per element [LL,LR,UR,UL]
        "node_counts"  : (n_nodes,) int array — number of adjacent elements
        "Pavg"         : (n_nodes, n_el) sparse matrix — averaging operator
    """
    n_el = nelx * nely
    n_nodes = (nelx + 1) * (nely + 1)

    elx = np.arange(n_el) // nely   # 0-based column
    ely = np.arange(n_el) % nely    # 0-based row

    n1 = (nely + 1) * elx + ely          # LL
    n2 = (nely + 1) * (elx + 1) + ely   # LR
    n3 = n2 + 1                           # UR
    n4 = n1 + 1                           # UL
    elem_nodes = np.column_stack([n1, n2, n3, n4]).astype(np.int64)  # (n_el, 4)

    # Build accumulation arrays.
    elem_node_flat = elem_nodes.ravel()               # (4*n_el,)
    elem_idx_flat = np.repeat(np.arange(n_el), 4)    # (4*n_el,)

    node_counts = np.bincount(elem_node_flat, minlength=n_nodes)
    node_counts_safe = np.maximum(node_counts, 1)

    # Averaging operator: Pavg[n, e] = 1/node_counts[n] if element e touches node n
    data = np.ones(4 * n_el, dtype=float)
    Pavg = csc_array(
        (data, (elem_node_flat, elem_idx_flat)),
        shape=(n_nodes, n_el),
    )
    # Divide rows by node counts.
    inv_counts = 1.0 / node_counts_safe
    Pavg = Pavg.multiply(inv_counts[:, np.newaxis])

    return {
        "nelx": nelx,
        "nely": nely,
        "n_el": n_el,
        "n_nodes": n_nodes,
        "elem_nodes": elem_nodes,
        "elem_node_flat": elem_node_flat,
        "node_counts": node_counts,
        "node_counts_safe": node_counts_safe,
        "Pavg": Pavg,
    }


def project_q4_element_density_to_nodes(
    rho_elem: np.ndarray,
    nelx: int,
    nely: int,
    cache: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Average element densities onto Q4 mesh nodes.

    Parameters
    ----------
    rho_elem : (nelx*nely,) float array
        Element density values.
    nelx, nely : int
        Mesh dimensions.
    cache : dict or None
        Previously built cache; rebuilt if None or mismatched.

    Returns
    -------
    rho_nodal : (n_nodes,) float array
        Node-averaged density values.
    cache : dict
        Projection cache (pass back to avoid rebuilding).
    """
    n_el = nelx * nely
    n_nodes = (nelx + 1) * (nely + 1)
    rho_elem = np.asarray(rho_elem, dtype=float).ravel()
    if rho_elem.size != n_el:
        raise ValueError(f"rho_elem must have {n_el} entries (got {rho_elem.size}).")

    if (
        cache is None
        or cache.get("nelx") != nelx
        or cache.get("nely") != nely
    ):
        cache = build_nodal_projection_cache(nelx, nely)

    # elem_node_flat is C-order: [e1n1,e1n2,e1n3,e1n4,e2n1,...].
    # Match that layout by repeating each element density 4 times:
    # [rho1,rho1,rho1,rho1,rho2,...]. Using tile() here would mismatch
    # weights and node indices, producing spurious nodal patterns.
    vals = np.repeat(rho_elem, 4)
    sum_vals = np.bincount(cache["elem_node_flat"], weights=vals, minlength=n_nodes)
    rho_nodal = sum_vals / cache["node_counts_safe"]
    return rho_nodal, cache
