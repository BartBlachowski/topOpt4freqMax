function run_cc_320x40_rmin_sensitivity()
%RUN_CC_320X40_RMIN_SENSITIVITY  Sensitivity-filter radius diagnostic.
%
% Uses the current best stabilized branch settings:
% Heaviside beta continuation 1->2->3 every 25 iterations,
% move_lim = outer_move = 0.05, mult_tol = 0.05.
% Only rmin_elem changes across cases.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','projection_diagnostic'));
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
src = fullfile(indir, 'cc_local_B_move005.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_rmin_sensitivity');
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
base.projection_beta = 1.0;
base.projection_beta_schedule = [1 2 3];
base.projection_beta_interval = 25;
base.design_filter = 'heaviside_projection_only';

cases = struct( ...
    'name', {'rmin20', 'rmin25', 'rmin30'}, ...
    'rmin_elem', {2.0, 2.5, 3.0});

for ci = 1:numel(cases)
    cfg = base;
    cfg.rmin_elem = cases(ci).rmin_elem;
    tag = cases(ci).name;

    fprintf('\n===== CC rmin sensitivity %s =====\n', tag);
    fprintf('start: Case B; rmin=%.3g; move=0.05; mult_tol=%.4g; beta schedule=%s/%d\n', ...
        cfg.rmin_elem, cfg.mult_tol, mat2str(cfg.projection_beta_schedule), cfg.projection_beta_interval);
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

    save(fullfile(outdir, sprintf('cc_rmin_%s.mat', tag)), ...
        'rho_start','rho_final','x_final','hist','cfg','el','bestw1','bi','src','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_rmin_%s.csv', tag)));
    writematrix(x_final(:), fullfile(outdir, sprintf('x_rmin_%s.csv', tag)));
end

fprintf('\nRmin sensitivity diagnostic complete. Results saved to %s\n', outdir);
end
