#!/usr/bin/env python3
"""tr_frozen -- the FROZEN two-branch rule with STAGE-LOCAL windows, offline.

A re-use of move_ladder_necessity/scripts/ml_frozen.py, extended to the exact
stage-reset semantics of +impl/architecture/+olh/+move/exhaustion.m so that the
rule can be replayed inside stage 2 as well as stage 1.

NOTHING IS TUNED.  W = 20, P = 20, W_np = 10, tol = 0.05*sqrt(NE/3200).
Stage locality, the net-path anchor rho_{s-1}, and the declaration rule are
transcribed from exhaustion.m, not re-invented.
"""
import numpy as np

W, P, WNP = 20, 20, 10


def tol_for(NE):
    return 0.05 * np.sqrt(NE / 3200.0)


def replay_stage(RHO, amp, rho0, NE, s, kmax):
    """Replay the detector over one stage beginning at outer iteration s (1-indexed)
    and running to kmax inclusive.  RHO is (NE, nOuter) with RHO[:,k-1] = rho_k.

    Mirrors exhaustion.m exactly:
      d_k    = rho_k - rho_{k-1}                      (rho_0 = rho0)
      cos(k) defined for k-1 >= s
      net(k) defined for k-WNP+1 >= s, anchor rho_{k-WNP} (pre-stage only at k=s+9)
      medians defined for k >= s + W - 1
    Returns dict of per-iteration arrays indexed 0..kmax-1 (global 1-indexed k).
    """
    n = kmax
    X = np.column_stack([np.full(NE, rho0), RHO[:, :n]])   # X[:, k] = rho_k
    D = np.diff(X, axis=1)                                  # D[:, k-1] = d_k
    n2 = lambda v: float(np.linalg.norm(v)) / np.sqrt(NE)

    cos = np.full(n, np.nan)
    net = np.full(n, np.nan)
    for k in range(s, n + 1):
        if (k - 1) >= s:
            a, b = D[:, k - 1], D[:, k - 2]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                cos[k - 1] = float(a @ b) / (na * nb)
        if (k - WNP + 1) >= s:
            pw = sum(n2(D[:, j - 1]) for j in range(k - WNP + 1, k + 1))
            if pw > 0:
                net[k - 1] = n2(X[:, k] - X[:, k - WNP]) / pw

    medcos = np.full(n, np.nan)
    mednet = np.full(n, np.nan)
    for k in range(s, n + 1):
        if k >= s + W - 1:
            w = slice(k - W, k)
            cc, nn = cos[w], net[w]
            medcos[k - 1] = np.nanmedian(cc) if np.any(~np.isnan(cc)) else np.nan
            mednet[k - 1] = np.nanmedian(nn) if np.any(~np.isnan(nn)) else np.nan

    tol = tol_for(NE)
    A = np.zeros(n, bool); B = np.zeros(n, bool)
    nA = np.zeros(n, int); nB = np.zeros(n, int)
    cA = cB = 0
    decl = None; branch = None; begin = None
    for k in range(s, n + 1):
        i = k - 1
        mc, mn = medcos[i], mednet[i]
        a = (not np.isnan(mc)) and (not np.isnan(mn)) and mc < 0 and mn < 0.5 and amp[i] >= tol
        b = (not np.isnan(mc)) and mc > 0 and amp[i] < tol
        A[i], B[i] = a, b
        cA = cA + 1 if a else 0
        cB = cB + 1 if b else 0
        nA[i], nB[i] = cA, cB
        if decl is None:
            if cA >= P:
                decl, branch, begin = k, 'A', k - P + 1
            elif cB >= P:
                decl, branch, begin = k, 'B', k - P + 1
    return dict(cos=cos, net=net, medcos=medcos, mednet=mednet, A=A, B=B,
                E=(A | B), nA=nA, nB=nB, tol=tol,
                decl=decl, branch=branch, begin=begin, stageStart=s)
