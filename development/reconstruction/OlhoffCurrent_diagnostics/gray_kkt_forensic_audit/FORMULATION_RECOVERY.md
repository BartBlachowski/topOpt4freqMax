# Exact formulation recovery

The physical target is max lambda1(rho), equivalently max omega1=sqrt(lambda1), subject to mean(rho)<=0.5 and 0.001<=rho<=1. K(rho) phi=lambda M(rho) phi. The code implements a sequence of filtered, nonlinear spectral subproblems for that target. It does **not** define a separate scalar physical objective whose exact derivative equals its sensitivity-filtered vector. This distinction is essential to interpreting stationarity.

## Variables, model and interpolation

rho is both the design density and the FE density on this path. No distinct filtered physical field is formed. MMA's density variables are increments d=Delta rho, plus b=beta/lambda1_ref. The outer update adds the returned increment to rho; **this audit never calls that update or its enclosing solver**.

Domain 8 by 1, thickness 1, Q4 plane-stress elements, consistent mass, E=1e7, nu=0.3, solid density 1. Mid-height simply-supported ends with axialRestraint=bothEnds. Full saved configurations are `evaluations/config_{400,480,800}.json`.

K=sum_e rho_e^3 K0_e (no additive Emin interpolation). M=sum_e m(rho_e) M0_e, where m(r)=r for r>0.1 and m(r)=6e5*r^6-5e6*r^7 for r<=0.1. Thus m'=1 above 0.1 and 3.6e6*r^5-3.5e7*r^6 below. This is eq4b, C1 at the cutoff; p=3, q=1. Neither exponent continues. The low-density correction does not apply inside the gray class rho>0.1. Volume is density volume, not the interpolated inertial mass.

Sources: `+impl/fem/assemble2D.m`, `fem/model2D.m`, `fem/elemMats2D.m`, `architecture/+olh/+material/massInterpolation.m`. Du & Olhoff (2007), local `references/Du2007_Topological.pdf`, pp.92–93, equations (1)–(4b), and pp.94–98, equations (7), (19), (24), (25). The original page 98 was rendered and visually checked: its printed (25d) lacks the increment symbol. The implementation explicitly uses the corrected increment form; bibliographic confirmation: [authors' correction](https://vbn.aau.dk/en/publications/topological-design-of-freely-vibrating-continuum-structures-for-m-3/).

## Spectral and subspace equations

Five lowest eigenpairs are calculated with eigs, fixed deterministic starting vector, tolerance 1e-12, maxit=5000, Krylov dimension max(20,4*5)=20. Each mode is mass-normalized; orthogonality and residuals are measured again here. Fixed subspace size N=2, J=3. N=2 remains fixed even when omega1 and omega2 are widely separated; the configured 0.05 multiplicity tolerance is not a classifier on this path. It is relevant to the next-mode warning.

Raw generalized blocks are

F_sk,e = 3*rho_e^2 phi_s,e' K0 phi_k,e - lambda_tilde m'(rho_e) phi_s,e' M0 phi_k,e.

Off-diagonal blocks use lambda_tilde=lambda1. Diagonal blocks are explicitly rebuilt using their own lambda_j. The next-mode block fJJ uses lambda3. All N×N stiffness and signed mass blocks, raw and filtered spectral blocks, and fJJ are retained in `evaluations/spectral_*.mat`.

Let Fhat be the filtered tensor and D=diag(lambda_j-lambda1). The predicted eigenvalues are the ordered eigenvalues of B(d)=diag(lambda1,lambda2)+sum_e Fhat_e*d_e. `deltaLambda` returns eig(D+A(d)) minus the old per-mode offsets. This is a class-C reconstruction for separated modes, not a formula attributed to the paper. Its derivative is v_j' Fhat_e v_j with v_j the subspace eigenvector. The audit calls the **native** deltaLambda at d=0 and for analysis directions. At all three saved states the lowest eigenvalue is resolved and simple; D's lowest eigenvector is the first basis vector. Consequently the exact derivative at d=0 reduces to Fhat_11, as a result of the implemented subspace problem, not an assumed replacement of it. At exact equality a PSD trace-one cluster dual would be required.

## Filter and derivative supplied to MMA

H_ei=max(0,R/h-distance_in_elements(e,i)), Hs_e=sum_i H_ei, R=0.06. Element radii are 3.0, 3.6 and 6.0. The operation is exactly

Fhat_sk,e = sum_i H_ei*rho_i*F_sk,i / (Hs_e*max(0.001,rho_e)).

It is applied separately to every diagonal and off-diagonal block and to fJJ. The volume gradient stays constant. rho is not filtered. Therefore d(lambda)/d rho is the **raw** derivative; Fhat is the derivative of a frozen local predicted spectral model with respect to its increment d. There is no density-map Jacobian to apply on this path, and FD of the physical eigenvalue should not reproduce Fhat. This follows directly from `filter/applyFilter.m` and `architecture/olhoffSolve.m:253`; it is consistent with the distinction between sensitivity and density filtering in [Andreassen et al., section 2.3](https://www.topopt.mek.dtu.dk/-/media/subsites/topopt/apps/dokumenter-og-filer-til-apps/topopt88.pdf?hash=E80FAB2808804A29FFB181CA05D2EEFECAA86686&la=da). We do not assert or test global non-integrability of the filtered vector field; no corresponding scalar map is implemented.

## MMA subproblem, signs and scaling

Native `algo/innerLoop.m` minimizes -b over x=[d;b], with constraints b-predicted_lambda_j(d)/lambda1_ref<=0 for j=1,2, b-(lambda3+fhatJJ'd)/lambda1_ref<=0, and (sum(rho+d)-0.5*NE)/(0.5*NE)<=0. lambda1_ref is fixed within this subproblem. Hence df0/db=-1; spectral density rows are -d(predicted_lambda)/dd /lambda1_ref; spectral b entries are 1; volume density entries are 1/(0.5*NE).

Increment bounds are max(0.001-rho,-move)<=d<=min(1-rho,move), and 0<=b<=5. Each outer subproblem resets x=[0;1], xold1=xold2=x, low=xmin, upp=xmax. Published MMA uses a0=1, a=0, c=1000, dMMA=0, m=N+2=4. Its auxiliary feasibility variables are internal MMA approximation variables; no global physical constraints beyond those above are inferred from them. Inner minimum/maximum 5/500; termination uses max|inner density step|/max(max|accumulated d|,1e-12)<0.05. This is a relative sub-iterate stopping rule, **not a KKT tolerance**. Returned dual variables are discarded; neither exact duals nor final inner iterates/asymptotes are retained. hist.beta alone is not a dual. This audit does not rerun the inner problem.

## Controller and temporal indexing

Three-rung stageExhaustion policy is frozen. For 400 only the exact S3 endpoint of its four-rung parent is used. Nothing here reopens controller selection. hist.omega(k) is evaluated at rho before update k; RHO(:,k) is after it. For interior columns, hist.omega(k+1) matches RHO(:,k). Terminal FE is separately reevaluated. For 400 that yields omega1=166.452298433..., not the pre-update 166.4427... quoted by the older architecture table. This is indexing, not a changed design.
