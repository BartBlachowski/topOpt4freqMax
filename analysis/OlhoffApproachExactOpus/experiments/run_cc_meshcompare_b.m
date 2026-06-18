% Exp-1 (regime B): CC at 80x20 vs 40x5 using the EXACT audit-baseline cfg
% that produced the analyzed disconnected two-block 462 design
% (optimizer_audit/baseline_trace.mat 'base'). ONLY nelx,nely change.
% No new fixes/heuristics/continuation/projection/optimizer changes.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results');
if ~exist(outdir,'dir'), mkdir(outdir); end

% --- exact baseline cfg (copied verbatim from baseline_trace.mat 'base') ---
base = struct();
base.support_type     = 'CC';
base.volfrac          = 0.5;
base.mass_mode        = 'du2007_c1';
base.sensitivity_filter = true;
base.rmin_elem        = 2.5;
base.n_target         = 1;
base.n_modes          = 4;
base.mult_tol         = 1e-3;
base.outer_max_iter   = 80;
base.outer_tol        = 1e-6;
base.inner_max_iter   = 30;
base.inner_tol        = 1e-4;
base.move_lim         = 0.2;
base.outer_move       = 0.2;
base.alpha            = 0.5;
base.acceptance_check = false;
base.verbose          = false;

meshes = {[40 5],[80 20]};
for mi = 1:numel(meshes)
    cfg = base;
    cfg.nelx = meshes{mi}(1);
    cfg.nely = meshes{mi}(2);
    tag = sprintf('b_%dx%d', cfg.nelx, cfg.nely);
    fprintf('\n===== CC(B) run mesh %dx%d =====\n', cfg.nelx, cfg.nely);
    t0 = tic;
    [rho_final, hist] = topopt_freq_exact(cfg);
    el = toc(t0);

    ot = hist.omega_trial; if all(isnan(ot(:))), ot = hist.omega; end
    [bestw1, bi] = max(ot(:,1));
    ni = hist.outer_iters;
    fprintf('--- mesh %dx%d done in %.1fs, outer_iters=%d ---\n', cfg.nelx,cfg.nely, el, ni);
    fprintf('FINAL : omega1=%.3f omega2=%.3f N=%d vol=%.4f\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_N, hist.final_volume);
    fprintf('BEST  : iter=%d omega1=%.3f omega2=%.3f Ntrial=%d vol=%.4f\n', ...
        bi, ot(bi,1), ot(bi,2), hist.N_trial(bi), hist.volume(bi));

    save(fullfile(outdir, sprintf('cc_%s.mat', tag)), 'rho_final','hist','cfg','el','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_%s.csv', tag)));
    % save the BEST-iter design too (the near-paper-frequency design)
    if isfield(hist,'omega_trial')
        writematrix([cfg.nelx cfg.nely bi ot(bi,1) ot(bi,2)], ...
            fullfile(outdir, sprintf('bestinfo_%s.csv', tag)));
    end
end
fprintf('\nAll regime-B runs complete.\n');
