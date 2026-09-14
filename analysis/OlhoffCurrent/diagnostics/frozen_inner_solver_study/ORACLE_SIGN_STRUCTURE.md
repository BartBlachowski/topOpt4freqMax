# Exact sign condition and first-mode approximation

For the oracle's exact nonlinear multipliers, let

    g_e = v_min' F_e v_min / lamref,
    q_e = -mu1*g_e - mu2*g_second,e - mu3*fJJ_e/lamref + mu4/Vtot.

The gradient includes both off-diagonal terms and the eigenvector induced by
the affine SOC coupling. Exact KKT gives q_e - xi_e + eta_e=0, xi,eta>=0.
Consequently q_e>0 requires the lower available box bound; q_e<0 requires the
upper available bound; an interior coordinate requires q_e=0. A zero cost can
also occur at a bound. The available bounds may be density-limited, not ±move.
This is necessary and, together with primal/dual feasibility and complementarity
on the convex problem, sufficient. No nonlinear coupling was discarded.

The oracle multipliers are approximately mu=[1,0,0,0.729113644725], with
volume threshold 5.06328919948e-05. Its SOC direction is
[-0.999999999615,-2.77583885322e-5]. Effective sensitivity differs from F11/lamref
by 0.00032005 relative L2. All classified lower
and upper oracle bounds have the correct reduced-cost sign. There are 14,314
positive and 14,486 negative reduced costs; only 22 have magnitude below
1e-6 times the maximum. Small nonzero numerical costs at interior-like points
are handled by the certified complementarity tolerance, not asserted exact zeros.

A greedy solution of the first-mode linear functional with the same volume/box
was evaluated using the full production spectral equations. It loses only
0.000599722103 of oracle gain (about 0.06%), reproducing
the preceding ~6e-4 observation. Yet d2=0.073718 and
dinf=2.000000; small objective error can conceal a few
full sign reversals. This is an interpretation aid, not a certified replacement
for the SOC. Fields and maps are retained in structure.mat.
