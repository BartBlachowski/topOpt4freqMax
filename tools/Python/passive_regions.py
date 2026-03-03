"""
passive_regions.py — Parse domain.passive_regions JSON block to element index sets.

Mirrors tools/Matlab/parsePassiveRegions.m.

Element numbering (0-based, column-major):
    el = elx*nely + ely,  elx = 0..nelx-1,  ely = 0..nely-1
    centroid:  xc = (elx + 0.5)*dx,  yc = (ely + 0.5)*dy

Overlap policy: solid wins over void.
"""

from __future__ import annotations
import numpy as np


def parse_passive_regions(
    cfg: dict,
    nelx: int,
    nely: int,
    L: float,
    H: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert domain.passive_regions to solid/void element index arrays.

    Parameters
    ----------
    cfg : dict
        Full decoded JSON config.
    nelx, nely : int
        Number of elements in x and y.
    L, H : float
        Physical domain dimensions.

    Returns
    -------
    pas_s : np.ndarray, dtype int64
        Sorted 0-based element indices forced solid (density = 1).
    pas_v : np.ndarray, dtype int64
        Sorted 0-based element indices forced void  (density = 0).
    """
    empty = np.array([], dtype=np.int64)

    pr_block = cfg.get("domain", {}).get("passive_regions", None)
    if not pr_block:
        return empty.copy(), empty.copy()

    dx = L / nelx
    dy = H / nely
    n_el = nelx * nely

    # Precompute element centroids (column-major).
    el_idx = np.arange(n_el, dtype=np.int64)
    elx = el_idx // nely   # 0-based column (x-direction)
    ely = el_idx % nely    # 0-based row    (y-direction)
    xc = (elx + 0.5) * dx
    yc = (ely + 0.5) * dy

    solid_mask = np.zeros(n_el, dtype=bool)
    void_mask = np.zeros(n_el, dtype=bool)

    # ---- flanges --------------------------------------------------------
    flanges = pr_block.get("flanges", None)
    if flanges:
        bot = flanges.get("bottom", None)
        if bot:
            rh = _parse_relative_height(bot, "domain.passive_regions.flanges.bottom")
            is_sol = _parse_is_solid(bot)
            mask = yc <= H * rh
            if is_sol:
                solid_mask |= mask
            else:
                void_mask |= mask

        top = flanges.get("top", None)
        if top:
            rh = _parse_relative_height(top, "domain.passive_regions.flanges.top")
            is_sol = _parse_is_solid(top)
            mask = yc >= H * (1.0 - rh)
            if is_sol:
                solid_mask |= mask
            else:
                void_mask |= mask

    # ---- rect -----------------------------------------------------------
    rect_raw = pr_block.get("rect", None)
    if rect_raw is not None:
        if isinstance(rect_raw, dict):
            rect_list = [rect_raw]
        elif isinstance(rect_raw, list):
            rect_list = rect_raw
        else:
            raise ValueError("domain.passive_regions.rect must be an object or array of objects.")

        for k, r in enumerate(rect_list):
            label = f"domain.passive_regions.rect[{k}]"
            x0 = np.asarray(r["x0"], dtype=float).ravel()
            sz = np.asarray(r["size"], dtype=float).ravel()
            if len(x0) != 2 or len(sz) != 2:
                raise ValueError(f"{label}.x0 and .size must be [x, y] pairs.")
            if np.any(sz <= 0):
                raise ValueError(f"{label}.size components must be > 0.")
            is_sol = _parse_is_solid(r)

            x_start = max(0.0, x0[0])
            y_start = max(0.0, x0[1])
            x_end = min(L, x0[0] + sz[0])
            y_end = min(H, x0[1] + sz[1])

            if x_end <= x_start or y_end <= y_start:
                import warnings
                warnings.warn(f"{label} clips to an empty region — skipped.", stacklevel=2)
                continue

            mask = (xc >= x_start) & (xc <= x_end) & (yc >= y_start) & (yc <= y_end)
            if is_sol:
                solid_mask |= mask
            else:
                void_mask |= mask

    # Solid wins over void.
    void_mask &= ~solid_mask

    pas_s = el_idx[solid_mask]
    pas_v = el_idx[void_mask]
    return pas_s.astype(np.int64), pas_v.astype(np.int64)


# ---------------------------------------------------------------------------
def _parse_relative_height(s: dict, label: str) -> float:
    rh = s.get("relative_height", None)
    if rh is None:
        raise ValueError(f"{label}: missing required field 'relative_height'.")
    rh = float(rh)
    if not (0 < rh <= 1):
        raise ValueError(f"{label}.relative_height must be in (0, 1].")
    return rh


def _parse_is_solid(s: dict) -> bool:
    v = s.get("is_solid", True)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, str):
        lv = v.strip().lower()
        if lv in {"true", "yes", "1", "on"}:
            return True
        if lv in {"false", "no", "0", "off"}:
            return False
    raise ValueError(f"is_solid must be boolean-like (got {v!r}).")
