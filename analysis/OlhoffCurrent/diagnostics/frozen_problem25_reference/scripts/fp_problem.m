function P = fp_problem(ctx)
%FP_PROBLEM  The exact frozen production subproblem, as data and functions.
%
%   P.evalProd(x)  -- mirror of innerLoop.m's constraint arithmetic (character
%                     for character), returning fval (m x 1), dfdx (m x nvar),
%                     dlam, ddlam, and the sub-eigenvalues.
%   P.coneResid(x) -- the SOC residual  ||Ac x - bc|| - (dc'x - gammac), in the
%                     SAME production-scaled units as fval(1).
%   P.e1closed / P.ge1closed / P.hessmult -- closed-form lambda_min and its
%                     derivatives for the 2x2 offset matrix.
%   Nothing here touches a density.
NE = numel(ctx.rho); N = numel(ctx.lam);
assert(N == 2, 'fp_problem:N', 'closed forms assume N = 2, got %d', N);
lamref = ctx.lam(1); Vtot = ctx.volfrac*NE; nvar = NE+1; m = N+2;
P = struct('NE',NE,'N',N,'nvar',nvar,'m',m,'lamref',lamref,'Vtot',Vtot);
P.lam = ctx.lam(:); P.lamJ = ctx.lamJ; P.rho = ctx.rho(:); P.move = ctx.move;
P.rhomin = ctx.rhomin; P.fJJ = ctx.fJJ(:);
P.dOff = ctx.dOff(:); P.d2 = ctx.dOff(2);
P.F11 = ctx.F(:,1,1); P.F12 = ctx.F(:,1,2); P.F22 = ctx.F(:,2,2);
P.F21 = ctx.F(:,2,1);
lo = max(ctx.rhomin - ctx.rho, -ctx.move);
hi = min(1          - ctx.rho,  ctx.move);
P.xmin = [lo; 0];  P.xmax = [hi; 5];
P.loMoveLimited = (-ctx.move) > (ctx.rhomin - ctx.rho);   % true: -move binds
P.hiMoveLimited = ( ctx.move) < (1 - ctx.rho);            % true: +move binds
P.f = [zeros(NE,1); -1];
% linear rows (production rows N+1, N+2 rewritten as Alin*x <= blin, exactly)
P.Alin = [ -ctx.fJJ(:).'/lamref, 1 ; ones(1,NE)/Vtot, 0 ];
P.blin = [ ctx.lamJ/lamref ; (Vtot - sum(ctx.rho))/Vtot ];
% cone, production-scaled (all terms divided by lamref)
P.Ac = [ 0.5*(P.F11-P.F22).'/lamref, 0 ; P.F12.'/lamref, 0 ];
P.bc = [ P.d2/(2*lamref) ; 0 ];
P.dc = [ 0.5*(P.F11+P.F22)/lamref ; -1 ];
P.gammac = -(P.d2/(2*lamref) + 1);
P.abc = @(drho) [P.F11.'*drho; P.F12.'*drho; P.F22.'*drho];
P.uvr = @(drho) local_uvr(P, drho);
P.e1closed = @(drho) local_e1(P, drho);
P.ge1closed = @(drho) local_ge1(P, drho);
P.coneResid = @(x) norm(P.Ac*x - P.bc) - (P.dc.'*x - P.gammac);
P.coneGrad  = @(x) local_coneGrad(P, x);
P.evalProd  = @(x) local_evalProd(ctx, x);
P.hessmult  = @(x, lamStruct, v) local_hessmult(P, x, lamStruct, v);
P.smoothMargin = local_margin(P);
end

function [u,v,r,a,b,c] = local_uvr(P, drho)
a = P.F11.'*drho; b = P.F12.'*drho; c = P.F22.'*drho;
u = (a - c - P.d2)/2; v = b; r = sqrt(u^2 + v^2);
end
function e1 = local_e1(P, drho)
[u,v,r,a,~,c] = local_uvr(P, drho); %#ok<ASGLU>
e1 = (a + c + P.d2)/2 - r;
end
function g = local_ge1(P, drho)
[u,v,r] = local_uvr(P, drho);
g = 0.5*(P.F11+P.F22) - (u*0.5*(P.F11-P.F22) + v*P.F12)/r;
end
function g = local_coneGrad(P, x)
s = P.Ac*x - P.bc; ns = norm(s);
g = (P.Ac.'*s)/ns - P.dc;
end
function [fval, dfdx, dlam, ddlam, ev, degen] = local_evalProd(ctx, x)
% character-for-character mirror of innerLoop.m's constraint block
NE = numel(ctx.rho); N = numel(ctx.lam); lamref = ctx.lam(1); Vtot = ctx.volfrac*NE;
nvar = NE+1; m = N+2;
drho = x(1:NE); bs = x(end);
dOff = ctx.dOff;
[dlam, ddlam, ~, degen] = deltaLambda(ctx.F, drho, dOff);
ev = dlam(:) + dOff(:);                       % eigenvalues of diag(dOff)+A
fval = zeros(m,1); dfdx = zeros(m,nvar);
for j = 1:N
    fval(j)      = bs - (ctx.lam(j) + dlam(j))/lamref;
    dfdx(j,1:NE) = -ddlam(:,j).'/lamref;
    dfdx(j,nvar) = 1;
end
fval(N+1)      = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;
dfdx(N+1,1:NE) = -ctx.fJJ.'/lamref;
dfdx(N+1,nvar) = 1;
fval(N+2)      = (sum(ctx.rho + drho) - Vtot)/Vtot;
dfdx(N+2,1:NE) = 1/Vtot;
end
function W = local_hessmult(P, x, lamStruct, v)
% Hessian of the Lagrangian  -bs + sum lam_i c_i(x)  times v (v may have columns)
% Only rows 1 and 2 are nonlinear:  c1 = bs - (lam1 + e1)/lamref,
% c2 = bs - (lam1 + e2)/lamref,  e1 + e2 = trace (linear)  =>  H(c2) = -H(c1).
% H(e1) = -J' Hr J,  J = [0.5(F11-F22)'; F12'],  Hr = [v^2 -uv; -uv u^2]/r^3.
NE = P.NE;
lam = lamStruct.ineqnonlin(:);
drho = x(1:NE);
[u,vv,r] = local_uvr(P, drho);
Hr = [vv^2, -u*vv; -u*vv, u^2]/r^3;
coef = (lam(1) - lam(2))/P.lamref;       % H(c1)*lam1 + H(c2)*lam2 = (lam1-lam2) J'HrJ/lamref
W = zeros(size(v));
if coef ~= 0
    Jv = [0.5*((P.F11-P.F22).'*v(1:NE,:)); P.F12.'*v(1:NE,:)];   % 2 x k
    HJv = Hr*Jv;
    W(1:NE,:) = coef*(0.5*(P.F11-P.F22)*HJv(1,:) + P.F12*HJv(2,:));
end
end
function M = local_margin(P)
NE = P.NE;
w = max(abs(P.xmin(1:NE)), abs(P.xmax(1:NE)));
M = struct();
M.max_box_abs_a_minus_c = sum(w.*abs(P.F11 - P.F22));
M.max_box_abs_b = sum(w.*abs(P.F12));
M.d2 = P.d2;
M.lower_bound_u2_plus_v2 = ((P.d2 - M.max_box_abs_a_minus_c)/2)^2 * (P.d2 > M.max_box_abs_a_minus_c);
M.apex_reachable_in_box = ~(P.d2 > M.max_box_abs_a_minus_c);
M.min_separation_e2_minus_e1_lower_bound = 2*sqrt(M.lower_bound_u2_plus_v2);
end
