function run_cc_320x40_density_filter_restart()
%RUN_CC_320X40_DENSITY_FILTER_RESTART  Paper-inspired density-filter diagnostic.
%
% S1 starts from the saved connected 320x40 Case-B local move-control design.
% The design variables are filtered to physical densities before assembly.
% No Heaviside projection is used in this diagnostic.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','density_filter_diagnostic'));
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
src = fullfile(indir, 'cc_local_B_move005.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_density_filter');
if ~exist(outdir,'dir'), mkdir(outdir); end

S = load(src, 'rho_final', 'cfg');
rho_start = S.rho_final(:);

cfg = S.cfg;
cfg.initial_rho = rho_start;
cfg.outer_max_iter = 100;
cfg.rho_snapshot_interval = 1;
cfg.verbose = false;
cfg.move_lim = 0.05;
cfg.outer_move = 0.05;
cfg.mult_tol = 0.05;
cfg.diagnostic_classification = 'paper-inspired stabilized reproduction';
cfg.design_filter = 'density_filter_only';

fprintf('\n===== CC density-filter restart S1_density_filter =====\n');
fprintf('start: Case B move=0.05 final design; mult_tol=%.4g; design filter=density only\n', cfg.mult_tol);
t0 = tic;
[rho_final, x_final, hist] = topopt_freq_density_filter_diag(cfg);
el = toc(t0);

ot = hist.omega_trial;
if all(isnan(ot(:))), ot = hist.omega; end
[bestw1, bi] = max(ot(:,1));
gap_best = (ot(bi,2) - ot(bi,1)) / ot(bi,1);

fprintf('--- S1 done in %.1fs, outer_iters=%d ---\n', el, hist.outer_iters);
fprintf('BEST  : local_iter=%d omega1=%.3f omega2=%.3f omega3=%.3f gap12=%.4f Ntrial=%d vol=%.4f\n', ...
    bi, ot(bi,1), ot(bi,2), ot(bi,3), gap_best, hist.N_trial(bi), hist.volume(bi));
fprintf('FINAL : omega1=%.3f omega2=%.3f omega3=%.3f N=%d vol=%.4f designVol=%.4f\n', ...
    hist.final_omega(1), hist.final_omega(2), hist.final_omega(3), hist.final_N, ...
    hist.final_volume, hist.final_design_volume);
fprintf('LAST  : dx_norm=%.3e dx_max=%.3e\n', hist.drho_norm(end), hist.drho_max(end));

save(fullfile(outdir, 'cc_density_filter_S1.mat'), ...
    'rho_start','rho_final','x_final','hist','cfg','el','bestw1','bi','src','-v7');
writematrix(rho_final(:), fullfile(outdir, 'rho_density_filter_S1.csv'));
writematrix(x_final(:), fullfile(outdir, 'x_density_filter_S1.csv'));

fprintf('\nDensity-filter diagnostic complete. Results saved to %s\n', outdir);
end
