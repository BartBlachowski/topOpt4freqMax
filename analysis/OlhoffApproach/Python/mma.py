"""
mma.py — Python port of Svanberg's MMA subproblem solver (September 2007).

Translated from mmasub.m + subsolv.m (Krister Svanberg, KTH).
Interface matches the Matlab version exactly so topopt_freq.py can import it
without modification.
"""

import numpy as np


def mmasub(m, n, iter_, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, fval, dfdx, low, upp,
           a0, a, c, d):
    """One MMA iteration.

    Solves:
        min  f_0(x) + a0*z + sum(c_i*y_i + 0.5*d_i*y_i^2)
        s.t. f_i(x) - a_i*z - y_i <= 0,  i=1..m
             xmin_j <= x_j <= xmax_j
             z >= 0, y_i >= 0

    Parameters (all column vectors unless noted)
    -------------------------------------------
    m       : int — number of constraints
    n       : int — number of design variables
    iter_   : int — current iteration (1-based)
    xval    : (n,1) current design
    xmin    : (n,1) variable lower bounds
    xmax    : (n,1) variable upper bounds
    xold1   : (n,1) design one iteration ago
    xold2   : (n,1) design two iterations ago
    f0val   : float — objective value
    df0dx   : (n,1) objective gradient
    fval    : (m,)  or (m,1) constraint values
    dfdx    : (m,n) constraint Jacobian
    low     : (n,) or (n,1) lower asymptotes (previous iter; ignored if iter_<2)
    upp     : (n,) or (n,1) upper asymptotes (previous iter; ignored if iter_<2)
    a0      : float scalar
    a       : (m,1) or (m,) constants in a_i*z term
    c       : (m,1) or (m,) constants in c_i*y_i term
    d       : (m,1) or (m,) constants in 0.5*d_i*y_i^2 term

    Returns
    -------
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
    """
    # ---- flatten everything to 1-D for internal computation ----
    xval  = np.asarray(xval,  dtype=float).ravel()
    xmin  = np.asarray(xmin,  dtype=float).ravel()
    xmax  = np.asarray(xmax,  dtype=float).ravel()
    xold1 = np.asarray(xold1, dtype=float).ravel()
    xold2 = np.asarray(xold2, dtype=float).ravel()
    df0dx = np.asarray(df0dx, dtype=float).ravel()
    fval  = np.asarray(fval,  dtype=float).ravel()
    dfdx  = np.asarray(dfdx,  dtype=float).reshape(m, n)
    low   = np.asarray(low,   dtype=float).ravel()
    upp   = np.asarray(upp,   dtype=float).ravel()
    a_vec = np.asarray(a,     dtype=float).ravel()   # length m
    c_vec = np.asarray(c,     dtype=float).ravel()   # length m
    d_vec = np.asarray(d,     dtype=float).ravel()   # length m

    epsimin = 1e-7
    raa0    = 1e-5
    move    = 1.0
    albefa  = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.7

    eeen = np.ones(n)
    xmami    = np.maximum(xmax - xmin, 1e-5 * eeen)
    xmamiinv = 1.0 / xmami

    # ---- asymptote update ----
    if iter_ < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz    = (xval - xold1) * (xold1 - xold2)
        factor = np.where(zzz > 0, asyincr, np.where(zzz < 0, asydecr, 1.0))
        low    = xval - factor * (xold1 - low)
        upp    = xval + factor * (upp - xold1)
        lowmin = xval - 0.2 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 0.2 * (xmax - xmin)
        low = np.clip(low, lowmin, lowmax)
        upp = np.clip(upp, uppmin, uppmax)

    # ---- bounds alfa and beta ----
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    alfa  = np.maximum(np.maximum(zzz1, zzz2), xmin)

    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    beta  = np.minimum(np.minimum(zzz1, zzz2), xmax)

    # ---- p0, q0, P, Q, b ----
    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    uxinv = 1.0 / ux1
    xlinv = 1.0 / xl1

    p0 = np.maximum(df0dx, 0.0)
    q0 = np.maximum(-df0dx, 0.0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = (p0 + pq0) * ux2
    q0 = (q0 + pq0) * xl2

    P = np.maximum(dfdx, 0.0)         # (m, n)
    Q = np.maximum(-dfdx, 0.0)        # (m, n)
    PQ = 0.001 * (P + Q) + raa0 * np.outer(np.ones(m), xmamiinv)
    P = (P + PQ) * ux2[np.newaxis, :]   # (m, n) * broadcast
    Q = (Q + PQ) * xl2[np.newaxis, :]   # (m, n) * broadcast
    b = P @ uxinv + Q @ xlinv - fval    # (m,)

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = _subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q,
        float(a0), a_vec, b, c_vec, d_vec,
    )

    return (
        xmma.reshape(-1, 1),   # (n,1) to match Matlab convention
        ymma,
        zmma,
        lam,
        xsi,
        eta,
        mu,
        zet,
        s,
        low,
        upp,
    )


def _subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q,
             a0, a, b, c, d):
    """Primal-dual Newton solve of the MMA subproblem.

    Transliterated from subsolv.m (Svanberg 2006).
    All vectors are 1-D numpy arrays.
    """
    een  = np.ones(n)
    eem  = np.ones(m)
    epsi = 1.0

    x   = 0.5 * (alfa + beta)
    y   = eem.copy()
    z   = 1.0
    lam = eem.copy()
    xsi = np.maximum(1.0 / (x - alfa), een)
    eta = np.maximum(1.0 / (beta - x), een)
    mu  = np.maximum(0.5 * c, eem)
    zet = 1.0
    s   = eem.copy()

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem

        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = 1.0 / ux1
        xlinv1 = 1.0 / xl1

        plam  = p0 + P.T @ lam       # (n,)
        qlam  = q0 + Q.T @ lam       # (n,)
        gvec  = P @ uxinv1 + Q @ xlinv1  # (m,)
        dpsidx = plam / ux2 - qlam / xl2  # (n,)

        rex   = dpsidx - xsi + eta
        rey   = c + d * y - mu - lam
        rez   = a0 - zet - np.dot(a, lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu  = mu * y - epsvecm
        rezet = zet * z - epsi
        res   = lam * s - epsvecm

        residu = np.concatenate([rex, rey, [rez], relam, rexsi, reeta, remu, [rezet], res])
        residunorm = np.linalg.norm(residu)
        residumax  = np.max(np.abs(residu))

        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1

            ux1  = upp - x
            xl1  = x - low
            ux2  = ux1 * ux1
            xl2  = xl1 * xl1
            ux3  = ux1 * ux2
            xl3  = xl1 * xl2
            uxinv1 = 1.0 / ux1
            xlinv1 = 1.0 / xl1
            uxinv2 = 1.0 / ux2
            xlinv2 = 1.0 / xl2

            plam  = p0 + P.T @ lam
            qlam  = q0 + Q.T @ lam
            gvec  = P @ uxinv1 + Q @ xlinv1
            GG    = P * uxinv2[np.newaxis, :] - Q * xlinv2[np.newaxis, :]  # (m,n)
            dpsidx = plam / ux2 - qlam / xl2

            delx  = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely  = c + d * y - lam - epsvecm / y
            delz  = a0 - np.dot(a, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam

            diagx   = plam / ux3 + qlam / xl3
            diagx   = 2.0 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = 1.0 / diagx
            diagy   = d + mu / y
            diagyinv = 1.0 / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                # Solve an (m+1)×(m+1) system (cheaper when m < n)
                blam  = dellam + dely / diagy - GG @ (delx / diagx)
                bb    = np.append(blam, delz)
                Alam  = np.diag(diaglamyi) + GG @ (diagxinv[:, np.newaxis] * GG.T)
                azz_s = -zet / z
                # AA is (m+1)×(m+1)
                AA           = np.zeros((m + 1, m + 1))
                AA[:m, :m]   = Alam
                AA[:m, m]    = a
                AA[m, :m]    = a
                AA[m, m]     = azz_s
                solut = np.linalg.solve(AA, bb)
                dlam  = solut[:m]
                dz    = solut[m]
                dx    = -(delx / diagx) - (GG.T @ dlam) / diagx
            else:
                # Solve an (n+1)×(n+1) system (cheaper when m >= n)
                diaglamyiinv = 1.0 / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx   = np.diag(diagx) + GG.T @ (diaglamyiinv[:, np.newaxis] * GG)
                azz_s = zet / z + float(np.dot(a, a / diaglamyi))
                axz_s = -(GG.T @ (a / diaglamyi))
                bx    = delx + GG.T @ (dellamyi / diaglamyi)
                bz_s  = delz - float(np.dot(a, dellamyi / diaglamyi))
                # AA is (n+1)×(n+1)
                AA           = np.zeros((n + 1, n + 1))
                AA[:n, :n]   = Axx
                AA[:n, n]    = axz_s
                AA[n, :n]    = axz_s
                AA[n, n]     = azz_s
                bb    = np.append(-bx, -bz_s)
                solut = np.linalg.solve(AA, bb)
                dx    = solut[:n]
                dz    = float(solut[n])
                dlam  = (GG @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy   = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu  = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds   = -s + epsvecm / lam - (s * dlam) / lam

            xx  = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])

            stepxx  = -1.01 * dxx / xx
            stmxx   = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa  = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta  = np.max(stepbeta)
            stmalbe  = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv   = max(stmalbexx, 1.0)
            steg     = 1.0 / stminv

            xold   = x.copy();  yold = y.copy();  zold = z
            lamold = lam.copy(); xsiold = xsi.copy(); etaold = eta.copy()
            muold  = mu.copy();  zetold = zet;  sold = s.copy()

            itto    = 0
            resinew = 2.0 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x   = xold   + steg * dx
                y   = yold   + steg * dy
                z   = zold   + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu  = muold  + steg * dmu
                zet = zetold + steg * dzet
                s   = sold   + steg * ds

                ux1  = upp - x
                xl1  = x - low
                ux2  = ux1 * ux1
                xl2  = xl1 * xl1
                uxinv1 = 1.0 / ux1
                xlinv1 = 1.0 / xl1

                plam  = p0 + P.T @ lam
                qlam  = q0 + Q.T @ lam
                gvec  = P @ uxinv1 + Q @ xlinv1
                dpsidx = plam / ux2 - qlam / xl2
                rex   = dpsidx - xsi + eta
                rey   = c + d * y - mu - lam
                rez   = a0 - zet - np.dot(a, lam)
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu  = mu * y - epsvecm
                rezet = zet * z - epsi
                res   = lam * s - epsvecm
                residu = np.concatenate([rex, rey, [rez], relam, rexsi, reeta, remu, [rezet], res])
                resinew = np.linalg.norm(residu)
                steg   /= 2.0

            residunorm = resinew
            residumax  = np.max(np.abs(residu))
            steg *= 2.0   # undo last halving (matches Matlab: steg = 2*steg)

        epsi *= 0.1

    return x, y, z, lam, xsi, eta, mu, zet, s
