function out = fi_kkt_refine()
%FI_KKT_REFINE  Part 3, corrected and robustness-swept.
%
%   The preregistered projected residual classifies a variable as "at a bound"
%   only within 1e-12 of the box width.  mmasub's solver is an INTERIOR POINT
%   method: it never places a variable exactly on a bound, and instead returns
%   box multipliers xsi, eta that carry the bound activity in smoothed form.
%   With the preregistered tolerance ZERO variables are classified active, so
%   the preregistered statistic omits -xsi + eta from the Lagrangian gradient.
%
%   Both are reported.  The preregistered number is NOT altered.  The exact MMA
%   KKT residual (which includes xsi, eta) is added, plus a bound-tolerance
%   sweep, because that is what actually answers "did MMA solve its own
%   subproblem".  Nothing here updates any density.

S = fi_setup();
study = fileparts(fileparts(mfilename('fullpath')));
L = load(fullfile(study,'evaluations','inner_kkt_state.mat'));
ctx = L.ctx;

out = struct();
for which = {'production','certified'}
    w = which{1};
    if strcmp(w,'production'), r = L.recP(end); st = L.stP; rr = L.recP;
    else,                      r = L.recC(end); st = L.stC; rr = L.recC; end

    NE = numel(ctx.rho); nvar = NE+1; N = numel(ctx.lam);
    lamref = ctx.lam(1); Vtot = ctx.volfrac*NE;
    x = st.xFinal; drho = x(1:NE); bs = x(end);

    [dlam, ddlam] = deltaLambda(ctx.F, drho, ctx.dOff);
    fval = zeros(N+2,1); dfdx = zeros(N+2,nvar);
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

    lam = r.lam(:); xsi = r.xsi(:); eta = r.eta(:);
    xmin = r.xmin; xmax = r.xmax;
    width = xmax - xmin;

    gL = df0dx + dfdx.'*lam;              % WITHOUT box multipliers
    gLfull = gL - xsi + eta;              % the EXACT MMA stationarity residual
    sRow = sqrt(mean((ddlam(:,1)/lamref).^2));

    A = struct();
    A.sRow = sRow;
    A.exact_mma_stationarity = struct( ...
        'drho_rms', sqrt(mean(gLfull(1:NE).^2)), ...
        'drho_max', max(abs(gLfull(1:NE))), ...
        'drho_norm_rms', sqrt(mean(gLfull(1:NE).^2))/sRow, ...
        'drho_norm_max', max(abs(gLfull(1:NE)))/sRow, ...
        'bs_residual', gLfull(nvar));
    A.without_box_multipliers = struct( ...
        'drho_norm_rms', sqrt(mean(gL(1:NE).^2))/sRow, ...
        'drho_norm_max', max(abs(gL(1:NE)))/sRow);

    % ---- where does drho actually sit inside its box? -------------------
    pos = (drho - xmin(1:NE))./max(width(1:NE), eps);
    A.box_position = struct('min',min(pos),'p01',quantile(pos,0.01), ...
        'median',median(pos),'p99',quantile(pos,0.99),'max',max(pos), ...
        'frac_below_1e6', mean(pos < 1e-6), 'frac_above_1m1e6', mean(pos > 1-1e-6), ...
        'frac_below_1e3', mean(pos < 1e-3), 'frac_above_1m1e3', mean(pos > 1-1e-3), ...
        'frac_interior_1e3', mean(pos >= 1e-3 & pos <= 1-1e-3));
    A.box_width = struct('min',min(width(1:NE)),'median',median(width(1:NE)), ...
        'max',max(width(1:NE)), 'frac_zero', mean(width(1:NE) <= 0));

    % ---- preregistered projected residual, bound-tolerance SWEEP --------
    tols = [1e-12 1e-8 1e-6 1e-4 1e-3 1e-2];
    sw = zeros(numel(tols),4);
    for t = 1:numel(tols)
        tb = tols(t)*max(width, eps);
        atLo = x <= xmin + tb;  atHi = x >= xmax - tb;
        rp = gL; rp(atLo) = min(gL(atLo),0); rp(atHi) = max(gL(atHi),0);
        sw(t,:) = [tols(t), sqrt(mean(rp(1:NE).^2))/sRow, ...
                   sum(atLo(1:NE)), sum(atHi(1:NE))];
    end
    A.projected_sweep_cols = {'boundTol','normRMS','nAtLower','nAtUpper'};
    A.projected_sweep = sw;

    % ---- complementarity, exact -----------------------------------------
    A.complementarity = struct( ...
        'max_lam_f', max(abs(lam.*fval)), ...
        'max_xsi_gap', max(xsi.*(x - xmin)), ...
        'max_eta_gap', max(eta.*(xmax - x)), ...
        'min_xsi', min(xsi), 'max_xsi', max(xsi), ...
        'min_eta', min(eta), 'max_eta', max(eta));
    A.primal = struct('max_fval',max(fval),'fval',fval, ...
        'max_box_violation', max(max(x-xmax), max(xmin-x)));
    A.dual_lam = lam;
    A.nInner = st.nInner; A.conv = st.conv; A.tolUsed = st.tolUsed;
    A.relStep_history = [rr.relStep];
    A.max_abs_drho = max(abs(drho));

    out.(w) = A;
end

% ---- did the inner iteration converge at all? -------------------------
rs = out.certified.relStep_history;
out.convergence = struct( ...
    'certified_nInner', numel(rs), ...
    'relStep_first', rs(1), 'relStep_at_19', rs(min(19,end)), ...
    'relStep_at_100', rs(min(100,end)), 'relStep_final', rs(end), ...
    'relStep_min', min(rs), 'relStep_last50_mean', mean(rs(max(1,end-49):end)), ...
    'monotone_decreasing', all(diff(rs) <= 0), ...
    'production_stopped_at', out.production.nInner, ...
    'production_tol', 0.05);

f = fullfile(study,'evaluations','inner_kkt_refined.json');
fid = fopen(f,'w'); fprintf(fid,'%s',jsonencode(out,'PrettyPrint',true)); fclose(fid);

fprintf('\n=== EXACT MMA KKT (includes box multipliers xsi, eta) ===\n');
for which = {'production','certified'}
    w = which{1}; A = out.(w);
    fprintf('%-12s nInner=%-4d exact statRMS=%.3e (norm %.3e)  max=%.3e (norm %.3e)\n', ...
        w, A.nInner, A.exact_mma_stationarity.drho_rms, A.exact_mma_stationarity.drho_norm_rms, ...
        A.exact_mma_stationarity.drho_max, A.exact_mma_stationarity.drho_norm_max);
    fprintf('%-12s without box mult: norm statRMS=%.4f\n', '', A.without_box_multipliers.drho_norm_rms);
    fprintf('%-12s box position: frac<1e-6=%.4f frac>1-1e-6=%.4f interior(1e-3)=%.4f\n', '', ...
        A.box_position.frac_below_1e6, A.box_position.frac_above_1m1e6, A.box_position.frac_interior_1e3);
    fprintf('%-12s comp: max|lam*f|=%.2e  max xsi*gap=%.2e  max eta*gap=%.2e\n', '', ...
        A.complementarity.max_lam_f, A.complementarity.max_xsi_gap, A.complementarity.max_eta_gap);
    fprintf('%-12s projected sweep (tol, normRMS, nLo, nHi):\n', '');
    disp(A.projected_sweep);
end
fprintf('\n=== inner convergence ===\n'); disp(out.convergence);
end
