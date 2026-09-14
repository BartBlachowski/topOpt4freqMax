function out = fi_inner_converge()
%FI_INNER_CONVERGE  FROZEN-SUBPROBLEM CERTIFICATION, extended.
%
%   Does the inner MMA iteration for the final 480 subproblem converge to a
%   FIXED POINT at all?  Only the iteration cap and stopping tolerance differ
%   from production; the subproblem -- objective, constraints, bounds, F, lam,
%   move, MMA constants -- is identical.  Every drho produced is DISCARDED.

S = fi_setup();
study = fileparts(fileparts(mfilename('fullpath')));
L = load(fullfile(study,'evaluations','inner_kkt_state.mat'));
ctx = L.ctx;

ctxLong = ctx; ctxLong.maxInner = 5000;
t0 = tic;
[drhoL, stL, recL] = fi_innerloop_audit(ctxLong, 1e-12, true);
wall = toc(t0);

rs = stL.relStepHist(:).';
NE = numel(ctx.rho);
dxmax = stL.maxAbsDrhoHist(:);

out = struct();
out.label = 'FROZEN-SUBPROBLEM CERTIFICATION -- drho discarded, never applied';
out.maxInner = ctxLong.maxInner;
out.tolInner = 1e-12;
out.nInner = stL.nInner;
out.converged = stL.conv;
out.wall_s = wall;
out.relStep = struct('first',rs(1),'at19',rs(19),'at100',rs(100), ...
    'at500',rs(500),'at1000',rs(min(1000,end)),'final',rs(end), ...
    'min',min(rs),'last100_mean',mean(rs(end-99:end)), ...
    'last100_min',min(rs(end-99:end)),'last100_max',max(rs(end-99:end)), ...
    'monotone',all(diff(rs)<=0));
out.maxAbsDrho = struct('at19',dxmax(19),'at100',dxmax(100),'at500',dxmax(500), ...
    'final',max(abs(drhoL)),'moveLimit',ctx.move, ...
    'final_over_move',max(abs(drhoL))/ctx.move, ...
    'final_over_production',max(abs(drhoL))/max(abs(L.stP.xFinal(1:NE))));
out.production = struct('nInner',L.stP.nInner,'tolInner',ctx.tolInner, ...
    'maxAbsDrho',max(abs(L.stP.xFinal(1:NE))));

% subsampled histories for the figure
idx = unique(round(logspace(0, log10(numel(rs)), 400)));
out.hist_iter = idx;
out.hist_relStep = rs(idx);
out.hist_maxAbsDrho = dxmax(idx).';

% exact MMA stationarity at the extended endpoint
r = recL(1); x = stL.xFinal; N = numel(ctx.lam); nvar = NE+1;
lamref = ctx.lam(1); Vtot = ctx.volfrac*NE;
drho = x(1:NE); bs = x(end);
[dlam, ddlam] = deltaLambda(ctx.F, drho, ctx.dOff);
fval = zeros(N+2,1); dfdx = zeros(N+2,nvar);
for j = 1:N
    fval(j) = bs - (ctx.lam(j)+dlam(j))/lamref;
    dfdx(j,1:NE) = -ddlam(:,j).'/lamref; dfdx(j,nvar) = 1;
end
fval(N+1) = bs - (ctx.lamJ + ctx.fJJ.'*drho)/lamref;
dfdx(N+1,1:NE) = -ctx.fJJ.'/lamref; dfdx(N+1,nvar) = 1;
fval(N+2) = (sum(ctx.rho+drho)-Vtot)/Vtot; dfdx(N+2,1:NE) = 1/Vtot;
df0dx = zeros(nvar,1); df0dx(nvar) = -1;
gL = df0dx + dfdx.'*r.lam(:);
gLfull = gL - r.xsi(:) + r.eta(:);
sRow = sqrt(mean((ddlam(:,1)/lamref).^2));
out.kkt_extended = struct('lam',r.lam(:).','fval',fval(:).', ...
    'exact_norm_rms', sqrt(mean(gLfull(1:NE).^2))/sRow, ...
    'exact_norm_max', max(abs(gLfull(1:NE)))/sRow, ...
    'nobox_norm_rms', sqrt(mean(gL(1:NE).^2))/sRow, ...
    'max_fval', max(fval), 'max_lam_f', max(abs(r.lam(:).*fval)));
clear drhoL

f = fullfile(study,'evaluations','inner_convergence.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('\n[fi_inner_converge] nInner=%d converged=%d wall=%.1fs\n', out.nInner, out.converged, wall);
fprintf('  relStep: 1 -> %.4g(19) -> %.4g(100) -> %.4g(500) -> %.4g(%d)  min=%.4g monotone=%d\n', ...
    rs(19), rs(100), rs(500), rs(end), numel(rs), min(rs), out.relStep.monotone);
fprintf('  last100 relStep: min=%.4g mean=%.4g max=%.4g\n', ...
    out.relStep.last100_min, out.relStep.last100_mean, out.relStep.last100_max);
fprintf('  max|drho|: prod(19)=%.4e  ext(%d)=%.4e  move=%.3g  ext/move=%.3f  ext/prod=%.1f\n', ...
    out.production.maxAbsDrho, out.nInner, out.maxAbsDrho.final, ctx.move, ...
    out.maxAbsDrho.final_over_move, out.maxAbsDrho.final_over_production);
fprintf('  extended KKT: exact normRMS=%.4e  nobox normRMS=%.4f  maxfval=%.3e\n', ...
    out.kkt_extended.exact_norm_rms, out.kkt_extended.nobox_norm_rms, out.kkt_extended.max_fval);
end
