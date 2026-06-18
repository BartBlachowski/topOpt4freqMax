function run_cc_320x40_local_move_restart()
%RUN_CC_320X40_LOCAL_MOVE_RESTART  Local move-limit restart around connected branch.
%
% Starts from the saved 320x40 long-horizon iteration-150 density and runs
% three local move-control cases for 100 additional outer iterations.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_long_horizon');
src = fullfile(indir, 'cc_long_320x40_200iter.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
if ~exist(outdir,'dir'), mkdir(outdir); end

S = load(src, 'hist', 'cfg');
snap_iters = S.hist.rho_snapshot_iters(1:S.hist.rho_snapshot_count);
snap_idx = find(snap_iters == 150, 1, 'first');
if isempty(snap_idx)
    error('Could not find rho snapshot at iteration 150 in %s.', src);
end
rho_start = S.hist.rho_snapshots(:, snap_idx);

base = S.cfg;
base.initial_rho = rho_start;
base.outer_max_iter = 100;
base.rho_snapshot_interval = 5;
base.verbose = false;

cases = struct( ...
    'name', {'A_baseline_move020', 'B_move005', 'C_move0025'}, ...
    'move', {0.20, 0.05, 0.025});

for ci = 1:numel(cases)
    cfg = base;
    cfg.move_lim = cases(ci).move;
    cfg.outer_move = cases(ci).move;
    tag = cases(ci).name;

    fprintf('\n===== CC local restart %s =====\n', tag);
    fprintf('start: 320x40 long-horizon snapshot iter 150; move_lim=outer_move=%.4g\n', cfg.move_lim);
    t0 = tic;
    [rho_final, hist] = topopt_freq_exact(cfg);
    el = toc(t0);

    ot = hist.omega_trial;
    if all(isnan(ot(:))), ot = hist.omega; end
    [bestw1, bi] = max(ot(:,1));

    fprintf('--- %s done in %.1fs, outer_iters=%d ---\n', tag, el, hist.outer_iters);
    fprintf('BEST  : local_iter=%d omega1=%.3f omega2=%.3f omega3=%.3f Ntrial=%d vol=%.4f\n', ...
        bi, ot(bi,1), ot(bi,2), ot(bi,3), hist.N_trial(bi), hist.volume(bi));
    fprintf('FINAL : omega1=%.3f omega2=%.3f omega3=%.3f N=%d vol=%.4f\n', ...
        hist.final_omega(1), hist.final_omega(2), hist.final_omega(3), hist.final_N, hist.final_volume);
    fprintf('LAST  : drho_norm=%.3e drho_max=%.3e\n', hist.drho_norm(end), hist.drho_max(end));

    save(fullfile(outdir, sprintf('cc_local_%s.mat', tag)), ...
        'rho_start','rho_final','hist','cfg','el','bestw1','bi','src','-v7');
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_local_%s.csv', tag)));
end

fprintf('\nAll local move-control restarts complete. Results saved to %s\n', outdir);
end
