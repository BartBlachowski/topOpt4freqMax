function run_mesh_campaign(bc, regime, nelx, nely, seed)
% RUN_MESH_CAMPAIGN  Mesh-resolution verification campaign for Du & Olhoff (2007).
%
%   run_mesh_campaign(bc, regime, nelx, nely, seed)
%
% `seed` fixes the global RNG stream that eigs() draws its start vector from.
% It is a REPLICATE INDEX, not an algorithm parameter: it perturbs only the
% Lanczos start vector, i.e. round-off-level noise in the eigenpairs.  Because
% the trajectory turns out to be chaotic, several seeds per mesh are required
% before any mesh-to-mesh difference can be called a real effect.
%
% CONTROLLED EXPERIMENT.  This driver calls the PRODUCTION solver
% analysis/OlhoffApproachExact/Matlab/topopt_freq_exact.m VERBATIM.  Not one
% solver file is modified.  Across the runs of one regime the ONLY quantities
% that differ are nelx and nely.  Every optimization parameter -- MMA, inner
% loop, convergence criteria, continuation, line search, damping, move limits,
% filters, sensitivities, generalized gradients, eigenvalue tracking, mass and
% stiffness interpolation -- is copied verbatim from the on-disk configuration
% that defines each regime and is held identical at every mesh.
%
% Regimes (both taken verbatim from files already in the repository; no new
% parameter values are invented here):
%
%   'B'  audit baseline -- audit_optimizer_nochange.m, struct `base`, lines
%        12-31.  This is the configuration that produced the disconnected
%        40x5 design analysed in the previous campaign (on-record final
%        omega_1 = 413.869).  PRIMARY regime.
%
%   'A'  committed benchmark script -- run_clamped_clamped_exact.m, lines
%        11-24 (paper-literal: move_lim=Inf, alpha=1, no acceptance check).
%        Regime control.
%
% Diagnostic-only instrumentation used (documented in topopt_freq_exact.m as
% having no effect on the optimizer):
%   cfg.rho_snapshot_interval = 1   -- stores rho each outer iteration
%
% Outputs (results/<tag>/):
%   run.mat          full cfg + hist + timing + support geometry
%   history.csv      per-iteration objective / frequency / convergence history
%   rho_final.csv    final physical density, nely x nelx
%   rho_snapshots.csv  nEl x n_iter design history (topology snapshots)
%   log.txt          verbose solver log

this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));

if nargin < 5 || isempty(seed), seed = 0; end

tag = sprintf('%s_regime%s_%dx%d_s%d', upper(bc), upper(regime), nelx, nely, seed);
out_dir = fullfile(this_dir, 'results', tag);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

% ---- Deterministic eigs start vector -------------------------------------
% eigs() draws a random start vector from the global RNG stream.  Fixing the
% stream makes each run bit-reproducible.  This is a driver-level
% reproducibility control; it does not alter the algorithm.
rng(seed, 'twister');

% ---- Configuration -------------------------------------------------------
cfg = struct();

switch upper(regime)
    case 'B'
        % VERBATIM from audit_optimizer_nochange.m, struct `base`.
        cfg.support_type       = upper(bc);
        cfg.volfrac            = 0.5;
        cfg.mass_mode          = 'du2007_c1';
        cfg.sensitivity_filter = true;
        cfg.rmin_elem          = 2.5;
        cfg.n_target           = 1;
        cfg.n_modes            = 4;
        cfg.mult_tol           = 1e-3;
        cfg.outer_max_iter     = 80;
        cfg.outer_tol          = 1e-6;
        cfg.inner_max_iter     = 30;
        cfg.inner_tol          = 1e-4;
        cfg.move_lim           = 0.2;
        cfg.outer_move         = 0.2;
        cfg.alpha              = 0.5;
        cfg.acceptance_check   = false;
    case 'A'
        % VERBATIM from run_clamped_clamped_exact.m.
        cfg.support_type       = upper(bc);
        cfg.volfrac            = 0.5;
        cfg.mass_mode          = 'du2007_c1';
        cfg.sensitivity_filter = true;
        cfg.rmin_elem          = 2.5;
        cfg.outer_max_iter     = 300;
        cfg.inner_max_iter     = 30;
        cfg.outer_tol          = 1e-4;
        cfg.move_lim           = Inf;
        cfg.acceptance_check   = false;
    otherwise
        error('run_mesh_campaign:BadRegime', 'regime must be A or B');
end

% ---- THE ONLY QUANTITIES THAT VARY ACROSS THE CAMPAIGN --------------------
cfg.nelx = nelx;
cfg.nely = nely;
% --------------------------------------------------------------------------

cfg.verbose               = true;
cfg.rho_snapshot_interval = 1;   % diagnostic-only (topopt_freq_exact.m:52-54)

% ---- Record the support geometry actually used ---------------------------
L = 8.0; H = 1.0;                        % topopt_freq_exact.m set_defaults
nNode   = (nelx + 1) * (nely + 1);
nodeNrs = reshape(1:nNode, nely + 1, nelx + 1);
fixed   = build_supports_exact(cfg.support_type, nodeNrs);
fnodes  = unique(ceil(fixed(:) / 2));
iy      = mod(fnodes - 1, nely + 1) + 1;
ix      = floor((fnodes - 1) / (nely + 1)) + 1;
support = struct();
support.nodes    = fnodes;
support.x        = (ix - 1) * (L / nelx);
support.y        = (iy - 1) * (H / nely);
support.y_over_H = support.y / H;
support.mid_idx  = round(nely / 2) + 1;
support.mid_y    = (support.mid_idx - 1) * (H / nely);
support.mid_offset_over_H = (support.mid_y - H / 2) / H;
support.exact_midheight   = abs(support.mid_offset_over_H) < 1e-12;
support.nFixedDof = numel(fixed);
support.nDof      = 2 * nNode;
support.nEl       = nelx * nely;

diary_file = fullfile(out_dir, 'log.txt');
if exist(diary_file, 'file'), delete(diary_file); end
diary(diary_file);
fprintf('\n================================================================\n');
fprintf(' MESH-RESOLUTION CAMPAIGN  --  %s\n', tag);
fprintf(' BC=%s  regime=%s  mesh=%dx%d  nEl=%d  nDof=%d\n', ...
    cfg.support_type, upper(regime), nelx, nely, support.nEl, support.nDof);
fprintf(' L/H=%.4f  dx=%.6f  dy=%.6f  elem aspect=%.6f\n', ...
    L / H, L / nelx, H / nely, (L / nelx) / (H / nely));
fprintf(' mid-height node row=%d at y/H=%.6f (offset %+.6f H, exact=%d)\n', ...
    support.mid_idx, support.mid_y / H, support.mid_offset_over_H, support.exact_midheight);
fprintf(' nFixedDof=%d\n', support.nFixedDof);
fprintf('================================================================\n');
disp(cfg);

% ---- Run -----------------------------------------------------------------
t_wall = tic;
t_cpu0 = cputime;
[rho_final, hist] = topopt_freq_exact(cfg);
wall_time = toc(t_wall);
cpu_time  = cputime - t_cpu0;

fprintf('\n---- RESULT %s ----\n', tag);
fprintf(' outer iterations : %d\n', hist.outer_iters);
fprintf(' final omega      : %s\n', mat2str(hist.final_omega, 6));
fprintf(' final N          : %g\n', hist.final_N);
fprintf(' final volume     : %.6f\n', hist.final_volume);
fprintf(' wall time        : %.2f s\n', wall_time);
fprintf(' cpu time         : %.2f s\n', cpu_time);
diary off;

% ---- Persist -------------------------------------------------------------
meta = struct('tag', tag, 'bc', upper(bc), 'regime', upper(regime), ...
    'nelx', nelx, 'nely', nely, 'L', L, 'H', H, ...
    'wall_time', wall_time, 'cpu_time', cpu_time, ...
    'matlab_version', version, 'timestamp', datestr(now, 31)); %#ok<TNOW1,DATST>

save(fullfile(out_dir, 'run.mat'), 'cfg', 'hist', 'rho_final', 'support', 'meta', '-v7.3');

ni = hist.outer_iters;
% Regime A does not set n_modes; the solver default is max(n_target+3,4)=4.
if isfield(cfg, 'n_modes') && ~isempty(cfg.n_modes)
    nm = cfg.n_modes;
else
    nm = size(hist.omega, 2);
end

fid = fopen(fullfile(out_dir, 'history.csv'), 'w');
hdr = 'iter,beta,beta_omega,volume,N_pre,N_trial,drho_norm,drho_max,step_alpha,inner_iters,inner_cpu_time,inner_converged';
for j = 1:nm, hdr = [hdr sprintf(',omega_pre_%d', j)]; end
for j = 1:nm, hdr = [hdr sprintf(',omega_post_%d', j)]; end
fprintf(fid, '%s\n', hdr);
for k = 1:ni
    fprintf(fid, '%d,%.10g,%.10g,%.10g,%g,%g,%.10g,%.10g,%.10g,%g,%.6g,%d', ...
        k, hist.beta(k), sqrt(max(hist.beta(k), 0)), hist.volume(k), ...
        hist.N(k), hist.N_trial(k), hist.drho_norm(k), hist.drho_max(k), ...
        hist.step_alpha(k), hist.inner_iters(k), hist.inner_cpu_time(k), ...
        hist.inner_converged(k));
    for j = 1:nm, fprintf(fid, ',%.10g', hist.omega(k, j)); end
    for j = 1:nm, fprintf(fid, ',%.10g', hist.omega_trial(k, j)); end
    fprintf(fid, '\n');
end
fclose(fid);

writematrix(reshape(rho_final, nely, nelx), fullfile(out_dir, 'rho_final.csv'));
if isfield(hist, 'rho_snapshots')
    ns = hist.rho_snapshot_count;
    writematrix(hist.rho_snapshots(:, 1:ns), fullfile(out_dir, 'rho_snapshots.csv'));
    writematrix(hist.rho_snapshot_iters(1:ns), fullfile(out_dir, 'rho_snapshot_iters.csv'));
end

fid = fopen(fullfile(out_dir, 'summary.csv'), 'w');
fprintf(fid, 'tag,bc,regime,nelx,nely,nEl,nDof,nFixedDof,mid_y_over_H,exact_midheight,');
fprintf(fid, 'outer_iters,final_omega1,final_omega2,final_omega3,final_omega4,final_N,final_volume,wall_time,cpu_time\n');
fo = hist.final_omega; fo(end+1:4) = NaN;
fprintf(fid, '%s,%s,%s,%d,%d,%d,%d,%d,%.10f,%d,%d,%.8f,%.8f,%.8f,%.8f,%g,%.8f,%.4f,%.4f\n', ...
    tag, upper(bc), upper(regime), nelx, nely, support.nEl, support.nDof, ...
    support.nFixedDof, support.mid_y / H, support.exact_midheight, ...
    hist.outer_iters, fo(1), fo(2), fo(3), fo(4), hist.final_N, ...
    hist.final_volume, wall_time, cpu_time);
fclose(fid);

fprintf('Saved to %s\n', out_dir);
end
