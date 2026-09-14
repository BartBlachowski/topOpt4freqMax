function out = fi_subproblem()
%FI_SUBPROBLEM  Parts 2 and 3: recover the exact final inner subproblem,
%   reproduce it bitwise, capture its duals, and certify its KKT.
%
%   NOTHING HERE UPDATES ANY DENSITY.  Every drho computed is discarded.

S = fi_setup();
study = fileparts(fileparts(mfilename('fullpath')));

% =====================================================================
% PART 2 -- rebuild the subproblem the solver actually posed at outer 386
% =====================================================================
% olhoffSolve evaluates the FE problem at the density BEFORE the update, so the
% final subproblem was built at rho385, not at the frozen endpoint rho386.
E = fi_eval(S, S.rho385, 'full');

ctx = struct('F',E.F,'fJJ',E.fJJ,'lam',E.lam(E.idx),'lamJ',E.lam(S.n+E.N), ...
             'rho',S.rho385,'rhomin',S.rhomin,'volfrac',S.volfrac, ...
             'move',S.move, ...
             'maxInner',S.g('optimizer.inner.maxIterations'), ...
             'tolInner',S.g('optimizer.inner.tolerance'), ...
             'minInner',S.g('optimizer.inner.minIterations'), ...
             'offDiag',S.g('multiplicity.offDiagonal'), ...
             'dOff',E.dOff);

o = struct();
o.built_at = 'rho385';
o.N = E.N; o.J = S.n + E.N;
o.lamref = ctx.lam(1);
o.omega_at_rho385 = E.omega(:).';
o.gap12_at_rho385 = E.gap12;
o.move = S.move;
o.Vtot = S.Vtot;
o.nvar = S.NE + 1;
o.m = E.N + 2;
o.mma_constants = struct('a0',1,'a',0,'c',1000,'d',0);
o.stopping = struct('tolInner',ctx.tolInner,'minInner',ctx.minInner, ...
                    'maxInner',ctx.maxInner,'criterion','relative max|dx| step');

% =====================================================================
% PART 3a -- PRODUCTION REPRODUCTION (bitwise)
% =====================================================================
[drhoP, stP, recP] = fi_innerloop_audit(ctx);
o.production = struct('nInner',stP.nInner,'conv',stP.conv,'beta',stP.beta, ...
    'relStep_final',stP.relHist(end),'degenHits',stP.degenHits);
o.reproduction = struct( ...
    'drho_bitwise_equal', isequal(drhoP, S.drho386), ...
    'max_abs_diff', max(abs(drhoP - S.drho386)), ...
    'nInner_recorded', S.hist.nInner(end), ...
    'nInner_reproduced', stP.nInner, ...
    'nInner_match', stP.nInner == S.hist.nInner(end), ...
    'beta_recorded', S.hist.beta(end), ...
    'beta_reproduced', stP.beta, ...
    'beta_abs_diff', abs(stP.beta - S.hist.beta(end)));

o.retained_exact_dual = false;   % innerLoop discards mmasub's duals with ~
o.dual_class = 'reconstructed (identical mmasub inputs, duals captured)';

% KKT at the production terminal iterate
o.kkt_production = local_kkt(recP(end), stP.xFinal, ctx, S, E);

% =====================================================================
% PART 3b -- FROZEN-SUBPROBLEM CERTIFICATION (tight tolerance)
% =====================================================================
% Identical subproblem; only the stopping point differs.  drho DISCARDED.
[drhoC, stC, recC] = fi_innerloop_audit(ctx, 1e-10);
o.certification = struct('tolInner',1e-10,'nInner',stC.nInner,'conv',stC.conv, ...
    'beta',stC.beta,'relStep_final',stC.relHist(end), ...
    'drho_vs_production_maxabs', max(abs(drhoC - drhoP)), ...
    'drho_vs_production_relative', max(abs(drhoC - drhoP))/max(max(abs(drhoP)),eps), ...
    'label','FROZEN-SUBPROBLEM CERTIFICATION -- drho discarded, never applied');
o.kkt_certified = local_kkt(recC(end), stC.xFinal, ctx, S, E);
clear drhoC drhoP

% inner-iterate trace, for the residual-history figure
o.trace = struct('iter',num2cell(1:numel(recP)), ...
                 'relStep',num2cell([recP.relStep]), ...
                 'dx',num2cell([recP.dx]));
o.trace_certified_relStep = [recC.relStep];

% =====================================================================
% verdict
% =====================================================================
K = o.kkt_production;
pass = o.reproduction.drho_bitwise_equal && ...
       K.primal.max_fval <= 1e-8 && K.dual.min_lam >= -1e-12 && ...
       K.complementarity.max_abs_lam_f <= 1e-6 && ...
       K.stationarity.drho_norm_rms <= 1e-3;
failv = K.stationarity.drho_norm_rms >= 1e-1 || K.primal.max_fval >= 1e-4 || ...
        K.dual.min_lam <= -1e-8;
if pass,       o.verdict = 'FINAL_INNER_MMA_KKT_PASS';
elseif failv,  o.verdict = 'FINAL_INNER_MMA_KKT_FAIL';
else,          o.verdict = 'FINAL_INNER_MMA_KKT_INCONCLUSIVE';
end

f = fullfile(study,'evaluations','inner_kkt.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(o,'PrettyPrint',true)); fclose(fid);
save(fullfile(study,'evaluations','inner_kkt_state.mat'), ...
     'ctx','recP','recC','stP','stC','-v7.3');

fprintf('\n[fi_subproblem] %s\n', o.verdict);
fprintf('  reproduction: bitwise=%d  maxdiff=%.3e  nInner %d vs %d\n', ...
    o.reproduction.drho_bitwise_equal, o.reproduction.max_abs_diff, ...
    o.reproduction.nInner_reproduced, o.reproduction.nInner_recorded);
fprintf('  PRODUCTION  : maxfval=%.3e  minlam=%.3e  maxcomp=%.3e  statRMS=%.4e  statMax=%.4e\n', ...
    K.primal.max_fval, K.dual.min_lam, K.complementarity.max_abs_lam_f, ...
    K.stationarity.drho_norm_rms, K.stationarity.drho_norm_max);
KC = o.kkt_certified;
fprintf('  CERTIFIED   : nInner=%d  maxfval=%.3e  statRMS=%.4e  statMax=%.4e\n', ...
    stC.nInner, KC.primal.max_fval, KC.stationarity.drho_norm_rms, KC.stationarity.drho_norm_max);
fprintf('  lam(production) = %s\n', mat2str(K.dual.lam(:).',6));
fprintf('  fval(production)= %s\n', mat2str(K.primal.fval(:).',6));
out = o;
end

% =========================================================================
function K = local_kkt(r, xFinal, ctx, S, E)
%LOCAL_KKT  KKT of the TRUE nonlinear inner problem (25) at the returned x.
%   r is the last captured mmasub record; xFinal is the x it returned.
NE = numel(ctx.rho); nvar = NE+1; N = numel(ctx.lam);
lamref = ctx.lam(1); Vtot = ctx.volfrac*NE;
drho = xFinal(1:NE); bs = xFinal(end);

% --- true constraint values and gradients AT the returned point ----------
[dlam, ddlam] = deltaLambda(ctx.F, drho, ctx.dOff);
m = N+2;
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
df0dx = zeros(nvar,1); df0dx(nvar) = -1;

lam = r.lam(:);                       % reconstructed dual (exact for the MMA subproblem)
xmin = r.xmin; xmax = r.xmax;

% --- primal feasibility --------------------------------------------------
K.primal = struct('fval',fval, 'max_fval',max(fval), ...
    'bound_lower_violation', max(max(xmin - xFinal),0), ...
    'bound_upper_violation', max(max(xFinal - xmax),0), ...
    'move_limit', ctx.move, ...
    'volume_slack', fval(N+2));

% --- dual feasibility ----------------------------------------------------
K.dual = struct('lam',lam,'min_lam',min(lam), ...
    'min_xsi',min(r.xsi),'min_eta',min(r.eta), ...
    'ymma',r.ymma,'zmma',r.zmma,'zet',r.zet);

% --- complementarity -----------------------------------------------------
K.complementarity = struct('lam_times_f', lam.*fval, ...
    'max_abs_lam_f', max(abs(lam.*fval)), ...
    'active', abs(fval) < 1e-8, ...
    'bound_lower_comp_max', max(r.xsi.*(xFinal - xmin)), ...
    'bound_upper_comp_max', max(r.eta.*(xmax - xFinal)));

% --- stationarity of L = f0 + sum lam_i f_i, projected on the box --------
gL = df0dx + dfdx.'*lam;
tolB = 1e-12*max(xmax - xmin, eps);
atLo = xFinal <= xmin + tolB;
atHi = xFinal >= xmax - tolB;
rproj = gL;
rproj(atLo) = min(gL(atLo), 0);
rproj(atHi) = max(gL(atHi), 0);

sRow = sqrt(mean((ddlam(:,1)/lamref).^2));     % preregistered normalizer
K.stationarity = struct( ...
    'normalizer_sRow', sRow, ...
    'drho_raw_rms', sqrt(mean(rproj(1:NE).^2)), ...
    'drho_raw_max', max(abs(rproj(1:NE))), ...
    'drho_norm_rms', sqrt(mean(rproj(1:NE).^2))/sRow, ...
    'drho_norm_max', max(abs(rproj(1:NE)))/sRow, ...
    'bs_residual', gL(nvar), ...
    'n_at_lower', sum(atLo(1:NE)), 'n_at_upper', sum(atHi(1:NE)), ...
    'n_interior', sum(~atLo(1:NE) & ~atHi(1:NE)));
K.stationarity.rproj_drho = [];   % full field saved separately in the .mat
end
