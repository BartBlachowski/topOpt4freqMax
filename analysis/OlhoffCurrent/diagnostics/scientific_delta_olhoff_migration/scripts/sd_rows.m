function [fval, dfdx] = sd_rows(ctx)
%SD_ROWS  Problem (25) constraint values and gradients at drho = 0, beta = 1,
%   formed with the statements of innerLoop.m (identical in source and target)
%   for the offDiag = true, linear-volume path.  Post hoc; feeds nothing back.
NE = numel(ctx.rho); N = numel(ctx.lam); lamref = ctx.lam(1);
Vtot = ctx.volfrac*NE; nvar = NE + 1;
assert(ctx.offDiag, 'sd:rows', 'only the full-coupling path is evaluated');
m = N + 2;
drho = zeros(NE,1); bs = 1;
[dlam, ddlam] = deltaLambda(ctx.F, drho, ctx.dOff);
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
