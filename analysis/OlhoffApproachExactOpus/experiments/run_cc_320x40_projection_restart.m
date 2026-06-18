function run_cc_320x40_projection_restart()
%RUN_CC_320X40_PROJECTION_RESTART  Heaviside projection-only diagnostic.
%
% P1/P2 start from the saved connected 320x40 Case-B local move-control
% design.  The design variables are mapped to physical densities by an
% elementwise Heaviside projection.  No density filter is used in the
% design-to-physical map.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','projection_diagnostic'));
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
src = fullfile(indir, 'cc_local_B_move005.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_projection');
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
    'name', {'P1_beta1', 'P2_beta3'}, ...
    'projection_beta', {1.0, 3.0});

for ci = 1:numel(cases)
    cfg = base;
    cfg.projection_beta = cases(ci).projection_beta;
    tag = cases(ci).name;

    fprintf('\n===== CC projection restart %s =====\n', tag);
    fprintf('start: Case B move=0.05 final design; mult_tol=%.4g; beta=%.3g eta=%.3g\n', ...
        cfg.mult_tol, cfg.projection_beta, cfg.projection_eta);
    t0 = tic;
    [rho_final, x_final, hist] = topopt_freq_projection_diag(cfg);
    el = toc(t0);

    ot = hist.omega_trial;
    if all(isnan(ot(:))), ot = hist.omega; end
    [bestw1, bi] = max(ot(:,1));
    gap_best = (ot(bi,2) - ot(bi,1)) / ot(bi,1);

    fprintf('--- %s done in %.1fs, outer_iters=%d ---\n', tag, el, hist.outer_iters);
    fprintf('BEST  : local_iter=%d omega1=%.3f omega2=%.3f omega3=%.3f gap12=%.4f Ntrial=%d vol=%.4f\n', ...
        bi, ot(bi,1), ot(bi,2), ot(bi,3), gap_best, hist.N_trial(bi), hist.volume(bi));
    fprintf('FINAL : omega1=%.3f omega2=%.3f omega3=%.3f N=%d vol=%.4f designVol=%.4f\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_omega(3), hist.final_N, ...
        hist.final_volume, hist.final_design_volume);
    fprintf('LAST  : dx_norm=%.3e dx_max=%.3e\n', hist.drho_norm(end), hist.drho_max(end));

    save(fullfile(outdir, sprintf('cc_projection_%s.mat', tag)), ...
        'rho_start','rho_final','x_final','hist','cfg','el','bestw1','bi','src','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_projection_%s.csv', tag)));
    writematrix(x_final(:), fullfile(outdir, sprintf('x_projection_%s.csv', tag)));
end

fprintf('\nProjection diagnostic complete. Results saved to %s\n', outdir);
end
