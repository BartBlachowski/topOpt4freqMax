function run_cc_320x40_basin_initialization()
%RUN_CC_320X40_BASIN_INITIALIZATION  Initial-state basin diagnostic.
%
% Starts from the saved connected 320x40 Case-B design and creates four
% volume-consistent alternatives.  The optimizer settings are the current
% best stabilized reproduction settings: Heaviside continuation beta
% 1->2->3 every 25 restart iterations, move limits 0.05, mult_tol 0.05.

root = '/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','projection_diagnostic'));
addpath(fullfile(root,'analysis','OlhoffApproachExact','Matlab'));
addpath(fullfile(root,'tools','Matlab'));

indir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_local_move');
src = fullfile(indir, 'cc_local_B_move005.mat');
outdir = fullfile(root,'analysis','OlhoffApproachExactOpus','experiments','results_basin_initialization');
if ~exist(outdir,'dir'), mkdir(outdir); end

S = load(src, 'rho_final', 'cfg');
x_raw = S.rho_final(:);

base = S.cfg;
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

nelx = base.nelx;
nely = base.nely;
if ~isfield(base, 'volfrac') || isempty(base.volfrac), base.volfrac = 0.5; end
if ~isfield(base, 'rho_min') || isempty(base.rho_min), base.rho_min = 1e-3; end
if ~isfield(base, 'rmin_elem') || isempty(base.rmin_elem), base.rmin_elem = 2.5; end
volfrac = base.volfrac;
rho_min = base.rho_min;
[h, Hs] = build_filter(nelx, nely, base.rmin_elem);

x_raw = match_volume(x_raw, volfrac, rho_min);
x_thresh = threshold_volume_preserving(x_raw, volfrac, rho_min);
x_smooth = match_volume(smooth_design(x_raw, h, Hs, nely, nelx), volfrac, rho_min);
x_thresh_smooth = match_volume(smooth_design(x_thresh, h, Hs, nely, nelx), volfrac, rho_min);

cases = struct( ...
    'name', {'raw', 'thresholded', 'smoothed', 'thresholded_smoothed'}, ...
    'x0', {x_raw, x_thresh, x_smooth, x_thresh_smooth});

for ci = 1:numel(cases)
    cfg = base;
    cfg.initial_rho = cases(ci).x0;
    tag = cases(ci).name;

    fprintf('\n===== CC basin initialization %s =====\n', tag);
    fprintf('initial mean=%.6f grey=%.4f; beta schedule=%s every %d\n', ...
        mean(cfg.initial_rho), mean(cfg.initial_rho > 0.1 & cfg.initial_rho < 0.9), ...
        mat2str(cfg.projection_beta_schedule), cfg.projection_beta_interval);
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

    x_initial = cfg.initial_rho;
    save(fullfile(outdir, sprintf('cc_basin_%s.mat', tag)), ...
        'x_initial','rho_final','x_final','hist','cfg','el','bestw1','bi','src','-v7');
    writematrix(x_initial(:), fullfile(outdir, sprintf('x_initial_%s.csv', tag)));
    writematrix(rho_final(:), fullfile(outdir, sprintf('rho_basin_%s.csv', tag)));
    writematrix(x_final(:), fullfile(outdir, sprintf('x_basin_%s.csv', tag)));
end

fprintf('\nBasin-initialization diagnostic complete. Results saved to %s\n', outdir);
end

function x = smooth_design(x, h, Hs, nely, nelx)
    x_mat = reshape(x, nely, nelx);
    x = reshape(conv2(x_mat, h, 'same') ./ Hs, [], 1);
end

function x = threshold_volume_preserving(x, volfrac, rho_min)
    n = numel(x);
    k = max(1, min(n, round(volfrac * n)));
    [~, idx] = sort(x, 'descend');
    x = rho_min * ones(n, 1);
    high = (volfrac * n - rho_min * (n - k)) / k;
    high = max(rho_min, min(1, high));
    x(idx(1:k)) = high;
end

function x = match_volume(x, volfrac, rho_min)
    x = max(rho_min, min(1, x(:)));
    for it = 1:30
        delta = volfrac - mean(x);
        if abs(delta) < 1e-10
            break
        end
        free = x > rho_min + 1e-12 & x < 1 - 1e-12;
        if ~any(free)
            break
        end
        x(free) = x(free) + delta * numel(x) / nnz(free);
        x = max(rho_min, min(1, x));
    end
end
