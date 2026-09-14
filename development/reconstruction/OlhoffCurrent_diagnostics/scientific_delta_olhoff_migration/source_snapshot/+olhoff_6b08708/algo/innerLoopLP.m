function [drho, st] = innerLoopLP(ctx)
%INNERLOOPLP  Step 3 solved as a LINEAR PROGRAM -- the route of the final
%   paragraph of Du & Olhoff (2007) sec. 3.5.3, after Krog & Olhoff (1999).
%
%   Imposing the additional constraints  f_sk' drho = 0,  s ~= k  (eq. 22) makes
%   the eigenvalue increments linear,  dlam_j = f_jj' drho  (eq. 23), so (25)
%   becomes an LP in [drho ; beta]:
%
%       max  beta
%       s.t. beta - (lam_j + f_jj' drho) <= 0,   j = n .. n+N-1     (25c -> 23)
%            beta - (lam_J + f_JJ' drho) <= 0,   j = J = n+N        (25b)
%            f_sk' drho = 0,                     s < k              (22)
%            sum(rho + drho) <= volfrac*NE                          (25e)
%            box on drho                                            (25f)
%
%   This is why the LP route exists: the equalities remove the eigenvector
%   rotation that makes the (25d) gradients non-differentiable.  MMA cannot take
%   these equalities -- it is an interior-point method and they leave the
%   feasible set with empty interior (observed: subsolv RCOND ~ 9e-18 and the
%   design freezes).  An LP solver handles them natively, in ONE solve rather
%   than ~100 MMA sub-iterates.

NE = numel(ctx.rho);
N  = numel(ctx.lam);
lamref = ctx.lam(1);
Vtot   = ctx.volfrac*NE;
nvar   = NE + 1;                              % [drho ; beta_scaled]

lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1          - ctx.rho,  ctx.move);
lb = [lo; 0];
ub = [hi; 5];

% ---- objective: maximise beta ------------------------------------------
f = zeros(nvar,1); f(end) = -1;

% ---- inequality constraints ---------------------------------------------
nIneq = N + 1 + 1;
A = zeros(nIneq, nvar);
b = zeros(nIneq, 1);
for j = 1:N                                   % (25c) with dlam_j = f_jj'drho
    A(j,1:NE) = -ctx.F(:,j,j).'/lamref;
    A(j,nvar) = 1;
    b(j)      = ctx.lam(j)/lamref;
end
A(N+1,1:NE) = -ctx.fJJ.'/lamref;              % (25b)
A(N+1,nvar) = 1;
b(N+1)      = ctx.lamJ/lamref;
A(N+2,1:NE) = 1;                              % (25e)
b(N+2)      = Vtot - sum(ctx.rho);

% ---- equality constraints: vanishing off-diagonals, eq. (22) -------------
npair = N*(N-1)/2;
Aeq = zeros(npair, nvar);
beq = zeros(npair, 1);
r = 0;
for s = 1:N
    for k = s+1:N
        r = r+1;
        Aeq(r,1:NE) = ctx.F(:,s,k).'/lamref;
    end
end

opts = optimoptions('linprog','Display','none','Algorithm','dual-simplex-highs');
[x, ~, flag] = linprog(f, A, b, Aeq, beq, lb, ub, opts);

st = struct('nInner',1,'degenHits',0,'conv',flag==1,'dxHist',[],'relHist',[], ...
            'lpFlag',flag);
if flag ~= 1 || isempty(x)
    % infeasible / failed: take no step and let the caller log it
    drho = zeros(NE,1);
    st.beta = ctx.lam(1);
    return
end
drho = x(1:NE);
st.beta = x(end)*lamref;
end
