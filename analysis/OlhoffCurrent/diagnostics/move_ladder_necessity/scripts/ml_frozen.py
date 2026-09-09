#!/usr/bin/env python3
"""ml_frozen -- the FROZEN two-branch rule, recomputed offline and independently.

This is a re-implementation of two_branch_maturity_240/scripts/tb_branches.m
(sections 2-7 of that study's PREREGISTRATION.md), written here so that the
exhaustion events used by this audit are recovered from the RAW trajectory
(RHO and hist.dxNorm2) rather than read back from the controller's own log.

Nothing is tuned.  W = 20, P = 20, W_np = 10, tol = 0.05*sqrt(NE/3200).
"""
import numpy as np

W, P, WNP = 20, 20, 10


def tol_for(NE):
    return 0.05 * np.sqrt(NE / 3200.0)


def quantities(RHO, amp, rho0, NE, stage_start=1):
    """Stage-local frozen quantities.  RHO is (NE, nOuter); amp is hist.dxNorm2.

    stage_start = 1 reproduces tb_branches exactly (whole trajectory, one stage).
    """
    n = RHO.shape[1]
    X = np.column_stack([np.full(NE, rho0), RHO])      # X[:,k] = rho_{k-1}
    D = np.diff(X, axis=1)                              # D[:,k-1] = drho_k
    n2 = lambda v: np.linalg.norm(v) / np.sqrt(NE)

    cos = np.full(n, np.nan)
    net = np.full(n, np.nan)
    s = stage_start
    for k in range(1, n + 1):
        if k - 1 >= s:                                  # cos needs d_{k-1} in-stage
            a, b = D[:, k - 1], D[:, k - 2]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                cos[k - 1] = float(a @ b) / (na * nb)
        if k - WNP + 1 >= s and k >= WNP:               # net needs 10 in-stage steps
            pw = sum(n2(D[:, j - 1]) for j in range(k - WNP + 1, k + 1))
            if pw > 0:
                net[k - 1] = n2(X[:, k] - X[:, k - WNP]) / pw

    medcos = np.full(n, np.nan)
    mednet = np.full(n, np.nan)
    for k in range(1, n + 1):
        if k >= s + W - 1:                              # full window inside the stage
            w = slice(k - W, k)
            with np.errstate(all='ignore'):
                medcos[k - 1] = np.nanmedian(cos[w]) if np.any(~np.isnan(cos[w])) else np.nan
                mednet[k - 1] = np.nanmedian(net[w]) if np.any(~np.isnan(net[w])) else np.nan
    return dict(cos=cos, net=net, medcos=medcos, mednet=mednet)


def branches(q, amp, NE):
    tol = tol_for(NE)
    mc, mn = q['medcos'], q['mednet']
    A = (~np.isnan(mc)) & (~np.isnan(mn)) & (mc < 0) & (mn < 0.5) & (amp >= tol)
    B = (~np.isnan(mc)) & (mc > 0) & (amp < tol)
    return A, B, tol


def declare(A, B, P=P):
    """First iteration at which either branch has held for P consecutive steps.

    Returns (declaration_iteration, branch, window_start) 1-indexed, or (None,...).
    A and B are mutually exclusive per iteration (amp >= tol vs amp < tol), so no
    tie is possible.
    """
    cA = cB = 0
    for k in range(len(A)):
        cA = cA + 1 if A[k] else 0
        cB = cB + 1 if B[k] else 0
        if cA >= P:
            return k + 1, 'A', k + 2 - P
        if cB >= P:
            return k + 1, 'B', k + 2 - P
    return None, None, None
