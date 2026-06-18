function run_cc_320x40_long_horizon(horizon)
%RUN_CC_320X40_LONG_HORIZON  Long-horizon CC convergence run.
%
% Uses the exact same optimizer settings that produced the 320x40 connected
% topology in run_cc_square_mesh_study.m.  The only optimization-setting change
% is the requested run horizon (outer_max_iter).  rho_snapshot_interval is
% diagnostic-only and does not affect the optimizer.

if nargin < 1 || isempty(horizon)
    horizon = 200;
end

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_long_horizon');
if ~exist(outdir,'dir'), mkdir(outdir); end

cfg = struct();
cfg.support_type     = 'CC';
cfg.volfrac          = 0.5;
cfg.mass_mode        = 'du2007_c1';
cfg.sensitivity_filter = true;
cfg.rmin_elem        = 2.5;
cfg.n_target         = 1;
cfg.n_modes          = 4;
cfg.mult_tol         = 1e-3;
cfg.outer_max_iter   = horizon;
cfg.outer_tol        = 1e-6;
cfg.inner_max_iter   = 30;
cfg.inner_tol        = 1e-4;
cfg.move_lim         = 0.2;
cfg.outer_move       = 0.2;
cfg.alpha            = 0.5;
cfg.acceptance_check = false;
cfg.verbose          = true;
cfg.nelx             = 320;
cfg.nely             = 40;
cfg.rho_snapshot_interval = 5;

tag = sprintf('320x40_%diter', horizon);
fprintf('\n===== CC long-horizon run %s =====\n', tag);
t0 = tic;
[rho_final, hist] = topopt_freq_exact(cfg);
el = toc(t0);

ot = hist.omega_trial;
if all(isnan(ot(:))), ot = hist.omega; end
[bestw1, bi] = max(ot(:,1));

fprintf('--- %s done in %.1fs, outer_iters=%d ---\n', tag, el, hist.outer_iters);
fprintf('BEST  : iter=%d omega1=%.3f omega2=%.3f omega3=%.3f Ntrial=%d vol=%.4f\n', ...
    bi, ot(bi,1), ot(bi,2), ot(bi,3), hist.N_trial(bi), hist.volume(bi));
fprintf('FINAL : omega1=%.3f omega2=%.3f omega3=%.3f N=%d vol=%.4f\n', ...
    hist.final_omega(1), hist.final_omega(2), hist.final_omega(3), hist.final_N, hist.final_volume);
fprintf('LAST  : drho_norm=%.3e drho_max=%.3e\n', hist.drho_norm(end), hist.drho_max(end));

save(fullfile(outdir, sprintf('cc_long_%s.mat', tag)), ...
    'rho_final','hist','cfg','el','bestw1','bi','-v7');
writematrix(rho_final(:), fullfile(outdir, sprintf('rho_long_%s.csv', tag)));
fprintf('Saved long-horizon result to %s\n', outdir);
end
