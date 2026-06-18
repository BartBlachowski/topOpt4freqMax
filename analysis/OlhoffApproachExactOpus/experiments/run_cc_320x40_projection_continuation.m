function run_cc_320x40_projection_continuation()
%RUN_CC_320X40_PROJECTION_CONTINUATION  Heaviside beta continuation diagnostic.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','projection_diagnostic'));
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
src = fullfile(indir, 'cc_local_B_move005.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_projection_continuation');
if ~exist(outdir,'dir'), mkdir(outdir); end

S = load(src, 'rho_final', 'cfg');
rho_start = S.rho_final(:);

base = S.cfg;
base.initial_rho = rho_start;
base.outer_max_iter = 100;
base.rho_snapshot_interval = 1;
base.verbose = false;
base.move_lim = 0.05;
base.outer_move = 0.05;
base.mult_tol = 0.05;
base.projection_eta = 0.5;
base.design_filter = 'heaviside_projection_only';

cases = struct( ...
    'name', {'PC0_beta3_fixed', 'PC1_beta123_i25', 'PC2_beta1234_i25', 'PC3_beta12346_i20'}, ...
    'projection_beta', {3.0, 1.0, 1.0, 1.0}, ...
    'projection_beta_schedule', {[], [1 2 3], [1 2 3 4], [1 2 3 4 6]}, ...
    'projection_beta_interval', {25, 25, 25, 20});

for ci = 1:numel(cases)
    cfg = base;
    cfg.projection_beta = cases(ci).projection_beta;
    cfg.projection_beta_schedule = cases(ci).projection_beta_schedule;
    cfg.projection_beta_interval = cases(ci).projection_beta_interval;
    tag = cases(ci).name;

    fprintf('\n===== CC projection continuation %s =====\n', tag);
    if isempty(cfg.projection_beta_schedule)
        sched_str = sprintf('fixed %.3g', cfg.projection_beta);
    else
        sched_str = sprintf('%s every %d', mat2str(cfg.projection_beta_schedule), cfg.projection_beta_interval);
    end
    fprintf('start: Case B move=0.05 final design; mult_tol=%.4g; beta=%s eta=%.3g\n', ...
        cfg.mult_tol, sched_str, cfg.projection_eta);
    t0 = tic;
    [rho_final, x_final, hist] = topopt_freq_projection_diag(cfg);
    el = toc(t0);

    ot = hist.omega_trial;
    if all(isnan(ot(:))), ot = hist.omega; end
    [bestw1, bi] = max(ot(:,1));
    gap_best = (ot(bi,2) - ot(bi,1)) / ot(bi,1);

    fprintf('--- %s done in %.1fs, outer_iters=%d ---\n', tag, el, hist.outer_iters);
    fprintf('BEST  : local_iter=%d omega1=%.3f omega2=%.3f omega3=%.3f gap12=%.4f Ntrial=%d vol=%.4f beta=%.3g\n', ...
        bi, ot(bi,1), ot(bi,2), ot(bi,3), gap_best, hist.N_trial(bi), hist.volume(bi), hist.projection_beta_iter(bi));
    fprintf('FINAL : omega1=%.3f omega2=%.3f omega3=%.3f N=%d vol=%.4f designVol=%.4f beta=%.3g\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_omega(3), hist.final_N, ...
        hist.final_volume, hist.final_design_volume, hist.final_projection_beta);
    fprintf('LAST  : dx_norm=%.3e dx_max=%.3e\n', hist.drho_norm(end), hist.drho_max(end));

    save(fullfile(outdir, sprintf('cc_projcont_%s.mat', tag)), ...
        'rho_start','rho_final','x_final','hist','cfg','el','bestw1','bi','src','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_projcont_%s.csv', tag)));
    writematrix(x_final(:), fullfile(outdir, sprintf('x_projcont_%s.csv', tag)));
end

fprintf('\nProjection-continuation diagnostic complete. Results saved to %s\n', outdir);
end
