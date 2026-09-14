"""Independent Python re-implementation of the frozen Phase-2A decision engines,
written from +ie2a/reference_phase.m, scan_persistence.m and measurement_budget.m.
Used only to measure DECISION MARGINS on already-stored quality sequences.
READ-ONLY.  No optimizer, no methodology change."""
import numpy as np


def reference_phase(Q, H0, P=100, L_ref=500, epsilon_ref=1e-3, B_ref=3200):
    Q = np.asarray(Q, float); H0 = np.asarray(H0, bool)
    n = min(Q.shape[0], B_ref)
    Q = Q[:n]; H0 = H0[:n]
    F = np.full((n, 3), np.nan)
    best = np.full(3, np.nan)
    valid = np.zeros(n, bool)
    for b in range(n):                      # 0-based; MATLAB b = b+1
        b1 = b + 1
        if b1 >= P and H0[b1-P:b1].all() and np.isfinite(Q[b1-P:b1]).all():
            fl = Q[b1-P:b1].min(axis=0); valid[b] = True
            best = fl if np.isnan(best).any() else np.maximum(best, fl)
        F[b] = best
    gain = np.full((n, 3), np.nan)
    cand = np.zeros(n, bool)
    b_ref = None
    for b1 in range(P, n+1, P):
        b = b1 - 1; bl = b1 - L_ref - 1
        if bl >= 0 and np.isfinite(F[b]).all() and np.isfinite(F[bl]).all() and (F[b] > 0).all():
            gain[b] = (F[b] - F[bl]) / F[b]
            cand[b] = bool((gain[b] <= epsilon_ref).all())
            if b_ref is None and cand[b]:
                b_ref = b1
    out = dict(F=F, gain=gain, valid=valid, candidate=cand, b_ref=b_ref,
               Q_ref=(F[b_ref-1].copy() if b_ref else np.full(3, np.nan)),
               status=('PASS' if b_ref else 'REFERENCE_NOT_ESTABLISHED'))
    return out


def scan_persistence(passM, P=100):
    """passM: (n, m) boolean. Returns k_enter, k_cert (1-based) per column."""
    passM = np.asarray(passM, bool)
    n, m = passM.shape
    ke = np.full(m, np.nan); kc = np.full(m, np.nan); inst = np.full(m, np.nan)
    for j in range(m):
        w = np.flatnonzero(passM[:, j])
        if w.size: inst[j] = w[0] + 1
        run = 0
        for k in range(n):
            run = run + 1 if passM[k, j] else 0
            if run == P:
                kc[j] = k + 1; ke[j] = k + 1 - P + 1
                break
    return dict(k_enter=ke, k_cert=kc, instantaneous_first=inst)


def measurement_budget(B0, b_ref, P=100, B_ref=3200):
    req = b_ref + P - 1
    return dict(B0=B0, b_ref=b_ref, P=P, B_ref=B_ref, requested_end=req,
                B_meas=min(max(B0, req), B_ref),
                certification_tail_truncated=req > B_ref,
                tail_truncation=max(0, req - B_ref))


def acceptance(Q, Q_ref, H0, levels):
    ratio = np.asarray(Q, float) / np.asarray(Q_ref, float)
    rob = ratio.min(axis=1)
    H0 = np.asarray(H0, bool)
    passM = np.column_stack([H0 & (rob >= q) for q in levels])
    return passM, rob
