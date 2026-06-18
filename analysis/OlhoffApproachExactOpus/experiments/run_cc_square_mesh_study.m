% CC square-element mesh resolution study for OlhoffApproachExactOpus.
%
% Purpose: isolate mesh resolution only.  The baseline configuration below is
% copied from experiments/run_cc_meshcompare_b.m; only cfg.nelx/cfg.nely vary.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_square_mesh');
if ~exist(outdir,'dir'), mkdir(outdir); end

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

meshes = [80 10; 160 20; 320 40];

summary = struct([]);
for mi = 1:size(meshes,1)
    cfg = base;
    cfg.nelx = meshes(mi,1);
    cfg.nely = meshes(mi,2);
    tag = sprintf('%dx%d', cfg.nelx, cfg.nely);
    fprintf('\n===== CC square-mesh run %s =====\n', tag);

    t0 = tic;
    [rho_final, hist] = topopt_freq_exact(cfg);
    el = toc(t0);

    ot = hist.omega_trial;
    if all(isnan(ot(:))), ot = hist.omega; end
    [bestw1, bi] = max(ot(:,1));

    fprintf('--- mesh %s done in %.1fs, outer_iters=%d ---\n', tag, el, hist.outer_iters);
    fprintf('INIT  : omega1=%.3f omega2=%.3f omega3=%.3f\n', ...
        hist.omega(1,1), hist.omega(1,2), hist.omega(1,3));
    fprintf('BEST  : iter=%d omega1=%.3f omega2=%.3f Ntrial=%d vol=%.4f\n', ...
        bi, ot(bi,1), ot(bi,2), hist.N_trial(bi), hist.volume(bi));
    fprintf('FINAL : omega1=%.3f omega2=%.3f N=%d vol=%.4f\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_N, hist.final_volume);

    save(fullfile(outdir, sprintf('cc_square_%s.mat', tag)), ...
        'rho_final','hist','cfg','el','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_square_%s.csv', tag)));

    summary(mi).mesh = tag; %#ok<SAGROW>
    summary(mi).elapsed = el;
    summary(mi).outer_iters = hist.outer_iters;
    summary(mi).initial_omega = hist.omega(1,1:min(3,size(hist.omega,2)));
    summary(mi).best_iter = bi;
    summary(mi).best_omega = ot(bi,1:min(3,size(ot,2)));
    summary(mi).best_N = hist.N_trial(bi);
    summary(mi).best_volume = hist.volume(bi);
    summary(mi).final_omega = hist.final_omega(1:min(3,numel(hist.final_omega)));
    summary(mi).final_N = hist.final_N;
    summary(mi).final_volume = hist.final_volume;
end

save(fullfile(outdir, 'cc_square_mesh_summary.mat'), 'summary', 'base', 'meshes', '-v7');
fprintf('\nAll square-mesh runs complete. Results saved to %s\n', outdir);
