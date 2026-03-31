import numpy as np
from scipy.sparse import diags, issparse


def _to_dense(A):
    """Convert sparse or matrix to dense array."""
    if issparse(A):
        return A.toarray()
    return np.asarray(A)


def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d):
    """
    This is Ai based translation of the code:
    Version September 2007 (and a small change August 2008)

    Krister Svanberg <krille@math.kth.se>
    Department of Mathematics, SE-10044 Stockholm, Sweden.

    This function mmasub performs one MMA-iteration, aimed at
    solving the nonlinear programming problem:

      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                xmin_j <= x_j <= xmax_j,    j = 1,...,n
                z >= 0,   y_i >= 0,         i = 1,...,m

    INPUT:
      m     : The number of general constraints.
      n     : The number of variables x_j.
      iter  : Current iteration number (=1 the first time mmasub is called).
      xval  : Column vector with the current values of the variables x_j.
      xmin  : Column vector with the lower bounds for the variables x_j.
      xmax  : Column vector with the upper bounds for the variables x_j.
      xold1 : xval, one iteration ago (provided that iter>1).
      xold2 : xval, two iterations ago (provided that iter>2).
      f0val : The value of the objective function f_0 at xval.
      df0dx : Column vector with the derivatives of the objective function
              f_0 with respect to the variables x_j, calculated at xval.
      fval  : Column vector with the values of the constraint functions f_i,
              calculated at xval.
      dfdx  : (m x n)-matrix with the derivatives of the constraint functions
              f_i with respect to the variables x_j, calculated at xval.
              dfdx[i,j] = the derivative of f_i with respect to x_j.
      low   : Column vector with the lower asymptotes from the previous
              iteration (provided that iter>1).
      upp   : Column vector with the upper asymptotes from the previous
              iteration (provided that iter>1).
      a0    : The constants a_0 in the term a_0*z.
      a     : Column vector with the constants a_i in the terms a_i*z.
      c     : Column vector with the constants c_i in the terms c_i*y_i.
      d     : Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.

    OUTPUT:
      xmma  : Column vector with the optimal values of the variables x_j
              in the current MMA subproblem.
      ymma  : Column vector with the optimal values of the variables y_i
              in the current MMA subproblem.
      zmma  : Scalar with the optimal value of the variable z
              in the current MMA subproblem.
      lam   : Lagrange multipliers for the m general MMA constraints.
      xsi   : Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
      eta   : Lagrange multipliers for the n constraints x_j - beta_j <= 0.
      mu    : Lagrange multipliers for the m constraints -y_i <= 0.
      zet   : Lagrange multiplier for the single constraint -z <= 0.
      s     : Slack variables for the m general MMA constraints.
      low   : Column vector with the lower asymptotes, calculated and used
              in the current MMA subproblem.
      upp   : Column vector with the upper asymptotes, calculated and used
              in the current MMA subproblem.

    Version September 2007 (and a small change August 2008)
    Original MATLAB code by Krister Svanberg <krille@math.kth.se>
    Department of Mathematics, SE-10044 Stockholm, Sweden.
    Converted to Python.
    """

    # epsimin = np.sqrt(m + n) * 1e-9
    epsimin = 1e-7
    raa0 = 0.00001
    move = 1.0
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.7
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))

    # Calculation of the asymptotes low and upp
    if iter < 2.5:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 0.2 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 0.2 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0, q0, P, Q and b
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = eeen / ux1
    xlinv = eeen / xl1

    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2

    P = np.maximum(dfdx, 0)
    Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * np.dot(eeem, xmamiinv.T)
    P = P + PQ
    Q = Q + PQ
    P = P @ diags(ux2.flatten(), 0, shape=(n, n))
    Q = Q @ diags(xl2.flatten(), 0, shape=(n, n))
    b = P @ uxinv + Q @ xlinv - fval.reshape(-1, 1)

    # Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = \
        subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    """
    This function subsolv solves the MMA subproblem:

    minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
             + SUM[ ci*yi + 0.5*di*(yi)^2 ],

    subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
               alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.

    Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    Output: xmma, ymma, zmma, slack variables and Lagrange multipliers.

    Version Dec 2006.
    Original MATLAB code by Krister Svanberg <krille@math.kth.se>
    Department of Mathematics, KTH, SE-10044 Stockholm, Sweden.
    Converted to Python.
    """
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = 1
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = 1
    s = eem.copy()
    itera = 0

    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        plam = p0 + P.T @ lam
        qlam = q0 + Q.T @ lam
        gvec = P @ uxinv1 + Q @ xlinv1
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.vstack([rex, rey, rez])
        residu2 = np.vstack([relam, rexsi, reeta, remu, rezet, res])
        residu = np.vstack([residu1, residu2])
        residunorm = np.sqrt(residu.T @ residu)[0, 0]
        residumax = np.max(np.abs(residu))
        ittt = 0

        while residumax > 0.9 * epsi and ittt < 200:
            ittt = ittt + 1
            itera = itera + 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ uxinv1 + Q @ xlinv1
            GG = P @ diags(uxinv2.flatten(), 0, shape=(n, n)) - Q @ diags(xlinv2.flatten(), 0, shape=(n, n))
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a.T @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            if m < n:
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = np.vstack([blam, delz])
                Alam = diags(diaglamyi.flatten(), 0, shape=(m, m)) + GG @ diags(diagxinv.flatten(), 0,
                                                                                shape=(n, n)) @ GG.T
                AA = np.vstack([
                    np.hstack([_to_dense(Alam), a]),
                    np.hstack([a.T, np.atleast_2d(-zet / z)])
                ])
                solut = np.linalg.solve(AA, bb)
                dlam = solut[:m]
                dz = solut[m:m + 1]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = diags(diagx.flatten(), 0, shape=(n, n)) + GG.T @ diags(diaglamyiinv.flatten(), 0,
                                                                             shape=(m, m)) @ GG
                azz = zet / z + a.T @ (a / diaglamyi)
                axz = -GG.T @ (a / diaglamyi)
                bx = delx + GG.T @ (dellamyi / diaglamyi)
                bz = delz - a.T @ (dellamyi / diaglamyi)
                AA = np.vstack([
                    np.hstack([_to_dense(Axx), axz]),
                    np.hstack([axz.T, np.atleast_2d(azz)])
                ])
                bb = np.vstack([-bx, -bz])
                solut = np.linalg.solve(AA, bb)
                dx = solut[:n]
                dz = solut[n:n + 1]
                dlam = (GG @ dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = np.vstack([y, z, lam, xsi, eta, mu, zet, s])
            dxx = np.vstack([dy, dz, dlam, dxsi, deta, dmu, dzet, ds])

            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1)
            steg = 1 / stminv

            xold = x.copy()
            yold = y.copy()
            zold = z
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet
            sold = s.copy()

            itto = 0
            resinew = 2 * residunorm
            while resinew > residunorm and itto < 50:
                itto = itto + 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ uxinv1 + Q @ xlinv1
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a.T @ lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = np.vstack([rex, rey, rez])
                residu2 = np.vstack([relam, rexsi, reeta, remu, rezet, res])
                residu = np.vstack([residu1, residu2])
                resinew = np.sqrt(residu.T @ residu)[0, 0]
                steg = steg / 2

            residunorm = resinew
            residumax = np.max(np.abs(residu))
            steg = 2 * steg

        if ittt > 198:
            pass

        epsi = 0.1 * epsi

    xmma = x
    ymma = y
    zmma = z
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma
