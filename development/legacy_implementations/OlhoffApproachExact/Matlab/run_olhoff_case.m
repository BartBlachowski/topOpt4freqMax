function res = run_olhoff_case(name, overrides, outdir)
% RUN_OLHOFF_CASE  Run one Olhoff & Du (2014) example and report against the paper.
%
%   res = run_olhoff_case(name)
%   res = run_olhoff_case(name, overrides)
%   res = run_olhoff_case(name, overrides, outdir)
%
%   name is any case of OLHOFF2014_CASE.  Prints the exactness contract actually
%   used, the paper target next to the computed value, and the Phase 7.1
%   decision-rule verdict.  Saves the density field, the iteration history and
%   plots under outdir (default: ../experiments/paper_examples/<name>).
%
%   DECISION RULE (declared in PLAN_Olhoff2014_exact.md section 7.1, before the
%   runs, so that whatever happens is reportable):
%     PASS    |omega - target|/target <= 3 %
%             AND N = target multiplicity sustained over the final >= 20 iters
%             AND a single spanning 8-connected structural component
%             AND stop_reason = kkt_converged
%     PARTIAL any strict subset
%     FAIL    omega_n collapse (> 50 % below the initial value) or no convergence

if nargin < 2, overrides = struct(); end
here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', '..', 'tools', 'Matlab'));

[cfg, tgt] = olhoff2014_case(name, overrides);

if nargin < 3 || isempty(outdir)
    outdir = fullfile(here, '..', 'experiments', 'paper_examples', name);
end
if ~exist(outdir, 'dir'), mkdir(outdir); end

fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('  Olhoff & Du (2014)  case %s   (%s)\n', upper(name), tgt.figure);
fprintf('%s\n', repmat('=', 1, 78));
fprintf('  contract:  p = %g [D1] | mass = %s [D3] | filter = %d, rmin = %g [D4]\n', ...
    cfg.penal, cfg.mass_mode, cfg.sensitivity_filter, cfg.rmin_elem);
fprintf('             mesh = %dx%d [R2] | move = %g [R1] | solver = %s [R9]\n', ...
    cfg.nelx, cfg.nely, cfg.move, cfg.subproblem_solver);
fprintf('             mult tol join/leave = %g/%g on lambda [R3] | cluster model %s [R4]\n', ...
    cfg.mult_tol_join, cfg.mult_tol_leave, cfg.cluster_model);
if isfield(tgt,'omega_initial')
    fprintf('  paper:     omega_%d initial = %.1f, optimum = %.1f, multiplicity %d\n', ...
        cfg.n_target, tgt.omega_initial, tgt.omega_opt, tgt.multiplicity);
elseif isfield(tgt,'omega_opt')
    fprintf('  paper:     omega_%d optimum = %.1f, multiplicity %d\n', ...
        cfg.n_target, tgt.omega_opt, tgt.multiplicity);
else
    fprintf('  paper:     gap omega_%d - omega_%d = %.0f (+%.0f %%), multiplicity %d\n', ...
        cfg.n_target, cfg.n_target-1, tgt.gap_opt, tgt.increase_pct, tgt.multiplicity);
end
fprintf('%s\n', repmat('-', 1, 78));

t0 = tic;
[rho, h] = topopt_freq_exact(cfg);
res.elapsed_s = toc(t0);

%% ------------------------------------------------------------- evaluate
n  = cfg.n_target;
om = h.final_omega;
res.name         = name;
res.cfg          = cfg;
res.target       = tgt;
res.hist         = h;
res.rho          = rho;
res.omega_final  = om;
res.stop_reason  = h.stop_reason;
res.outer_iters  = h.outer_iters;
res.volume       = h.final_volume;
res.components   = h.final_components;
res.final_N      = h.final_N;

is_gap = strcmpi(cfg.objective, 'gap');
if is_gap
    res.value  = om(n) - om(n-1);
    res.value0 = h.omega(1,n) - h.omega(1,n-1);
    res.target_value = tgt.gap_opt;
else
    res.value  = om(n);
    res.value0 = h.omega(1,n);
    res.target_value = tgt.omega_opt;
end
res.err_pct = 100*(res.value - res.target_value)/res.target_value;

% Multiplicity sustained over the final 20 iterations.
tail = max(1, h.outer_iters-19) : h.outer_iters;
res.N_tail_min = min(h.N(tail));
res.N_sustained = res.N_tail_min >= tgt.multiplicity;
if h.final_N >= 2
    res.cluster_spread_omega = (om(n+h.final_N-1) - om(n))/om(n);
else
    res.cluster_spread_omega = NaN;
end

% Decision rule.
c_freq = abs(res.err_pct) <= 3;
c_mult = res.N_sustained && h.final_N >= tgt.multiplicity;
c_conn = h.final_components == 1;
c_conv = strcmp(h.stop_reason, 'kkt_converged');
collapsed = res.value < 0.5*res.value0;
if collapsed || strcmp(h.stop_reason,'eigensolver_failure')
    res.verdict = 'FAIL';
elseif c_freq && c_mult && c_conn && c_conv
    res.verdict = 'PASS';
else
    res.verdict = 'PARTIAL';
end
res.criteria = struct('frequency', c_freq, 'multiplicity', c_mult, ...
                      'connected', c_conn, 'converged', c_conv);

%% --------------------------------------------------------------- report
fprintf('\n  RESULT  %s\n', res.verdict);
fprintf('   computed / paper        : %.4f / %.1f   (%+.2f %%)\n', ...
    res.value, res.target_value, res.err_pct);
fprintf('   initial value           : %.4f', res.value0);
if isfield(tgt,'omega_initial')
    fprintf('   (paper %.1f, %+.2f %%)', tgt.omega_initial, ...
        100*(res.value0-tgt.omega_initial)/tgt.omega_initial);
end
fprintf('\n');
fprintf('   increase                : %+.1f %%', 100*(res.value/res.value0 - 1));
if isfield(tgt,'increase_pct'), fprintf('   (paper %+d %%)', tgt.increase_pct); end
fprintf('\n');
fprintf('   omega(1:%d)              : %s\n', min(6,numel(om)), ...
    mat2str(round(om(1:min(6,numel(om)))',2)));
fprintf('   multiplicity N          : final %g, min over last 20 iters %g (paper %d)\n', ...
    h.final_N, res.N_tail_min, tgt.multiplicity);
if ~isnan(res.cluster_spread_omega)
    fprintf('   cluster spread on omega : %.3f %%\n', 100*res.cluster_spread_omega);
end
if isfield(tgt,'extra') && isfield(tgt.extra,'omega3_opt')
    fprintf('   omega_3                 : %.2f (paper %.1f, %+.2f %%)\n', om(3), ...
        tgt.extra.omega3_opt, 100*(om(3)-tgt.extra.omega3_opt)/tgt.extra.omega3_opt);
end
fprintf('   volume / structural comp: %.4f / %d\n', res.volume, res.components);
fprintf('   stop_reason             : %s after %d outer iterations (%.1f s)\n', ...
    h.stop_reason, h.outer_iters, res.elapsed_s);
fprintf('   criteria  freq %s | mult %s | connected %s | converged %s\n', ...
    yn(c_freq), yn(c_mult), yn(c_conn), yn(c_conv));
fd = h.fd_audit(~isnan(h.fd_audit));
if ~isempty(fd)
    fprintf('   FD audit (predicted vs realised dlambda): median %.2e, max %.2e\n', ...
        median(fd), max(fd));
end
ig = h.inner_gap_rel(~isnan(h.inner_gap_rel));
if ~isempty(ig)
    fprintf('   inner-loop optimality gap vs exact LP   : median %.2e, max %.2e\n', ...
        median(ig), max(ig));
end
fprintf('%s\n\n', repmat('=', 1, 78));

%% ----------------------------------------------------------------- save
writematrix(rho, fullfile(outdir, 'rho_final.csv'));
write_history(fullfile(outdir, 'history.csv'), h);
save(fullfile(outdir, 'result.mat'), 'res', '-v7.3');
write_summary(fullfile(outdir, 'summary.txt'), res);
make_plots(outdir, name, rho, h, cfg, tgt, res);
fprintf('  saved to %s\n\n', outdir);
end

%% =======================================================================
function write_history(fn, h)
    ni = h.outer_iters;
    T = table((1:ni)', 'VariableNames', {'iter'});
    for j = 1:size(h.omega,2)
        T.(sprintf('omega%d', j)) = h.omega(1:ni, j);
    end
    T.beta_sqrt    = sqrt(max(h.beta(1:ni),0));
    T.N            = h.N(1:ni);
    T.R            = h.R(1:ni);
    T.volume       = h.volume(1:ni);
    T.drho_inf     = h.drho_inf(1:ni);
    T.frac_bound   = h.frac_at_bound(1:ni);
    T.inner_iters  = h.inner_iters(1:ni);
    T.inner_cuts   = h.inner_cuts(1:ni);
    T.inner_gap    = h.inner_gap_rel(1:ni);
    T.fd_audit     = h.fd_audit(1:ni);
    T.cluster_spread = h.cluster_spread(1:ni);
    T.eigengap     = h.eigengap(1:ni);
    T.components   = h.n_components(1:ni);
    T.grey_frac    = h.grey_frac(1:ni);
    T.mac_mode_n   = h.mac_mode_n(1:ni);
    T.subspace_ang = h.subspace_ang(1:ni);
    T.inner_stop   = h.inner_stop(1:ni);
    writetable(T, fn);
end

function write_summary(fn, res)
    fid = fopen(fn, 'w');
    fprintf(fid, 'Olhoff & Du (2014) case %s  (%s)\n', res.name, res.target.figure);
    fprintf(fid, 'verdict          : %s\n', res.verdict);
    fprintf(fid, 'computed         : %.6f\n', res.value);
    fprintf(fid, 'paper target     : %.6f\n', res.target_value);
    fprintf(fid, 'relative error   : %+.4f %%\n', res.err_pct);
    fprintf(fid, 'initial value    : %.6f\n', res.value0);
    fprintf(fid, 'omega final      : %s\n', mat2str(res.omega_final, 8));
    fprintf(fid, 'final N          : %g (paper %d)\n', res.final_N, res.target.multiplicity);
    fprintf(fid, 'N min last 20    : %g\n', res.N_tail_min);
    fprintf(fid, 'volume           : %.6f\n', res.volume);
    fprintf(fid, 'components       : %d\n', res.components);
    fprintf(fid, 'stop_reason      : %s\n', res.stop_reason);
    fprintf(fid, 'outer iterations : %d\n', res.outer_iters);
    fprintf(fid, 'elapsed          : %.1f s\n', res.elapsed_s);
    fprintf(fid, '\n--- contract ---\n');
    c = res.cfg;
    fprintf(fid, 'penal            : %g   [D1]\n', c.penal);
    fprintf(fid, 'mass_mode        : %s   [D3]\n', c.mass_mode);
    fprintf(fid, 'sens filter/rmin : %d / %g   [D4]\n', c.sensitivity_filter, c.rmin_elem);
    fprintf(fid, 'mesh             : %dx%d   [R2]\n', c.nelx, c.nely);
    fprintf(fid, 'move limit       : %g   [R1]\n', c.move);
    fprintf(fid, 'mult tol j/l     : %g / %g on lambda   [R3]\n', ...
        c.mult_tol_join, c.mult_tol_leave);
    fprintf(fid, 'cluster model    : %s / %s   [R4]\n', c.cluster_model, c.lam_ref_rule);
    fprintf(fid, 'solver           : %s   [R9]\n', c.subproblem_solver);
    fprintf(fid, 'outer tol / kkt  : %g / %g   [R8]\n', c.outer_tol, c.kkt_tol);
    fclose(fid);
end

function make_plots(outdir, name, rho, h, cfg, tgt, res)
    ni = h.outer_iters;

    f1 = figure('Visible','off','Position',[100 100 900 220]);
    imagesc(1 - reshape(rho, cfg.nely, cfg.nelx));
    colormap(gray); axis equal tight off; clim([0 1]);
    title(sprintf('%s: computed %.1f, paper %.1f (%s, %s)', ...
        upper(name), res.value, res.target_value, tgt.figure, res.verdict));
    exportgraphics_compat(f1, fullfile(outdir, 'topology.png'));  close(f1);

    f2 = figure('Visible','off','Position',[100 100 760 520]);
    nm = min(3, size(h.omega,2));
    styles = {'k-o','b-+','r-^'};
    hold on
    for j = 1:nm
        plot(1:ni, h.omega(1:ni,j), styles{j}, 'MarkerSize', 3, 'LineWidth', 1);
    end
    if isfield(tgt,'omega_opt')
        yline(tgt.omega_opt, 'k--', sprintf('paper %.1f', tgt.omega_opt));
    end
    xlabel('Iteration number'); ylabel('Eigenfrequencies');
    legend(arrayfun(@(j) sprintf('\\omega_%d', j), 1:nm, 'UniformOutput', false), ...
        'Location','best');
    title(sprintf('%s -- compare with %s', upper(name), tgt.figure));
    grid on; box on
    exportgraphics_compat(f2, fullfile(outdir, 'frequency_history.png'));  close(f2);

    f3 = figure('Visible','off','Position',[100 100 760 620]);
    subplot(3,1,1); plot(1:ni, h.N(1:ni), 'b-', 'LineWidth', 1.2);
    ylabel('N'); ylim([0 max(3,max(h.N(1:ni))+0.5)]); grid on
    title('multiplicity, step size, linearization error');
    subplot(3,1,2); semilogy(1:ni, max(h.drho_inf(1:ni),1e-16), 'k-'); hold on
    yline(cfg.move, 'r--', 'move limit'); ylabel('||\Delta\rho||_\infty'); grid on
    subplot(3,1,3); semilogy(1:ni, max(h.fd_audit(1:ni),1e-16), 'm-');
    ylabel('FD audit'); xlabel('Iteration'); grid on
    exportgraphics_compat(f3, fullfile(outdir, 'diagnostics.png'));  close(f3);
end

function exportgraphics_compat(fh, fn)
    try
        exportgraphics(fh, fn, 'Resolution', 150);
    catch
        print(fh, fn, '-dpng', '-r150');
    end
end

function s = yn(tf)
    if tf, s = 'yes'; else, s = 'NO '; end
end
