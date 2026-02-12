"""
subsolv.py - MMA subproblem solver.

Version Dec 2006.
Krister Svanberg <krille@math.kth.se>
Department of Mathematics, KTH, SE-10044 Stockholm, Sweden.

Translated from Matlab to Python.
"""

import numpy as np
from scipy.sparse import diags as spdiags


def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    """
    Solve the MMA subproblem.

    minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
             + SUM[ ci*yi + 0.5*di*(yi)^2 ]

    subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi
               alfaj <= xj <= betaj,  yi >= 0,  z >= 0.

    All vectors are 1-D numpy arrays.  P, Q are scipy sparse (m x n).
    """
    low = low.ravel();  upp = upp.ravel()
    alfa = alfa.ravel();  beta = beta.ravel()
    p0 = p0.ravel();  q0 = q0.ravel()
    a = a.ravel();  b = b.ravel();  c = c.ravel();  d = d.ravel()

    een = np.ones(n)
    eem = np.ones(m)
    epsi = 1.0

    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = 1.0
    lam = eem.copy()
    xsi = een / (x - alfa);  xsi = np.maximum(xsi, een)
    eta = een / (beta - x);   eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1.0
    s = eem.copy()

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem

        ux1 = upp - x;  xl1 = x - low
        ux2 = ux1 * ux1;  xl2 = xl1 * xl1
        uxinv1 = een / ux1;  xlinv1 = een / xl1

        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ uxinv1 + Q @ xlinv1
        dpsidx = plam / ux2 - qlam / xl2

        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - float(a @ lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm

        residu = np.concatenate([rex, rey, [rez], relam, rexsi, reeta,
                                 remu, [rezet], res])
        residunorm = float(np.sqrt(residu @ residu))
        residumax = float(np.max(np.abs(residu)))

        ittt = 0
        while residumax > 0.9 * epsi and ittt < 200:
            ittt += 1

            ux1 = upp - x;  xl1 = x - low
            ux2 = ux1 * ux1;  xl2 = xl1 * xl1
            ux3 = ux1 * ux2;  xl3 = xl1 * xl2
            uxinv1 = een / ux1;  xlinv1 = een / xl1
            uxinv2 = een / ux2;  xlinv2 = een / xl2

            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ uxinv1 + Q @ xlinv1

            GG = (P @ spdiags(uxinv2, offsets=0, shape=(n, n))
                  - Q @ spdiags(xlinv2, offsets=0, shape=(n, n)))

            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - float(a @ lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam

            diagx = plam / ux3 + qlam / xl3
            diagx = 2.0 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = np.concatenate([blam, [delz]])
                Alam = (spdiags(diaglamyi, offsets=0, shape=(m, m))
                        + GG @ spdiags(diagxinv, offsets=0, shape=(n, n)) @ GG.T)
                AA = np.zeros((m + 1, m + 1))
                AA[:m, :m] = Alam.toarray() if hasattr(Alam, 'toarray') else Alam
                AA[:m, m] = a
                AA[m, :m] = a
                AA[m, m] = -zet / z
                solut = np.linalg.solve(AA, bb)
                dlam = solut[:m]
                dz = solut[m]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = (spdiags(diagx, offsets=0, shape=(n, n))
                       + GG.T @ spdiags(diaglamyiinv, offsets=0, shape=(m, m)) @ GG)
                azz = zet / z + float(a @ (a / diaglamyi))
                axz = -(GG.T @ (a / diaglamyi))
                bx = delx + GG.T @ (dellamyi / diaglamyi)
                bz = delz - float(a @ (dellamyi / diaglamyi))
                AA = np.zeros((n + 1, n + 1))
                Axx_d = Axx.toarray() if hasattr(Axx, 'toarray') else Axx
                AA[:n, :n] = Axx_d
                AA[:n, n] = axz
                AA[n, :n] = axz
                AA[n, n] = azz
                bb = np.concatenate([-bx, [-bz]])
                solut = np.linalg.solve(AA, bb)
                dx = solut[:n]
                dz = solut[n]
                dlam = (GG @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam

            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])

            stepxx = -1.01 * dxx / xx
            stmxx = float(np.max(stepxx))
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = float(np.max(stepalfa))
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = float(np.max(stepbeta))
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv

            xold = x.copy();  yold = y.copy();  zold = z
            lamold = lam.copy();  xsiold = xsi.copy();  etaold = eta.copy()
            muold = mu.copy();  zetold = zet;  sold = s.copy()

            itto = 0
            resinew = 2.0 * residunorm
            while resinew > residunorm and itto < 50:
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds

                ux1 = upp - x;  xl1 = x - low
                ux2 = ux1 * ux1;  xl2 = xl1 * xl1
                uxinv1 = een / ux1;  xlinv1 = een / xl1
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ uxinv1 + Q @ xlinv1
                dpsidx = plam / ux2 - qlam / xl2

                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - float(a @ lam)
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm

                residu = np.concatenate([rex, rey, [rez], relam, rexsi,
                                         reeta, remu, [rezet], res])
                resinew = float(np.sqrt(residu @ residu))
                steg = steg / 2.0

            residunorm = resinew
            residumax = float(np.max(np.abs(residu)))
            steg = 2.0 * steg

        epsi = 0.1 * epsi

    return x, y, z, lam, xsi, eta, mu, zet, s
