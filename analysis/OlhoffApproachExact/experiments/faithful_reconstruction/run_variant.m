function run_variant(variant, nelx, nely, bc, opts)
% RUN_VARIANT  Execute one reconstruction variant and persist every artefact.
%
%   run_variant(variant, nelx, nely, bc, opts)
%
%   variant : 'V0'..'V5'  (Phase-5 ablation matrix), or
%             'V1a','V1b' (Phase-3 continuation-schedule sensitivity)
%
%   Ablation matrix (Phase 5)
%   -------------------------
%     V   continuation   fail-closed   step controls
%     V0      no             no        paper-literal
%     V1      yes            no        paper-literal
%     V2      no             yes       paper-literal
%     V3      yes            yes       paper-literal
%     V4      no             yes       existing Regime-B step controls
%     V5      yes            yes       existing Regime-B step controls
%
%   "paper-literal"  = move_lim Inf, outer_move Inf, alpha 1, outer_tol 1e-4
%                      (run_clamped_clamped_exact.m lines 11-24)
%   "Regime-B"       = move_lim 0.2, outer_move 0.2, alpha 0.5, outer_tol 1e-6
%                      (audit_optimizer_nochange.m struct `base`, lines 12-31)
%   Neither set is retuned.  Only the outer ITERATION BUDGET is unified at 300
%   across all variants so the ablation is controlled; this is declared in the
%   report and never used to select a favourable run.
%
%   Continuation reconstruction (Phase 3, primary): p = 1, 1.5, 2, 2.5, 3 in
%   five equal fixed-length stages.  Paper Section 2.1 states p is "normally
%   assigned values increasing from 1 to 3 during the optimization process";
%   the schedule itself is not given and equal stages are the simplest
%   defensible reconstruction.

this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);
addpath(fullfile(this_dir, '..', '..', 'Matlab'));
addpath(fullfile(this_dir, '..', '..', '..', '..', 'tools', 'Matlab'));

if nargin < 4 || isempty(bc),   bc   = 'CC'; end
if nargin < 5, opts = struct(); end
if ~isfield(opts,'seed'),            opts.seed = 0; end
if ~isfield(opts,'outer_max_iter'),  opts.outer_max_iter = 300; end
if ~isfield(opts,'save_inner_full'), opts.save_inner_full = false; end
if ~isfield(opts,'tag_suffix'),      opts.tag_suffix = ''; end
% inner_max_iter is a RECONSTRUCTION choice, not a paper value: Du & Olhoff
% specify only "inner loop until the increments have converged" (Fig. 1).
% 30 is the value recorded in both on-disk regimes; larger budgets are used
% in the Phase-2 acceptance audit to ask what a genuinely converged inner
% solve does, and are always reported as such.
if ~isfield(opts,'inner_max_iter'),  opts.inner_max_iter = 30; end
if ~isfield(opts,'cont_stage_len'),  opts.cont_stage_len = 25; end

cfg = struct();
cfg.support_type = bc;
cfg.nelx = nelx;   cfg.nely = nely;
cfg.volfrac = 0.5; cfg.rho_min = 1e-3;
cfg.mass_mode = 'du2007_c1';
cfg.rmin_elem = 2.5;
cfg.n_target = 1;  cfg.n_modes = 4;  cfg.mult_tol = 1e-3;
cfg.inner_max_iter = opts.inner_max_iter;  cfg.inner_tol = 1e-4;
cfg.penal = 3.0;
cfg.outer_max_iter = opts.outer_max_iter;
cfg.verbose = true;
cfg.save_inner_full = opts.save_inner_full;

% ---- step-control regime -------------------------------------------------
% VR is the REFERENCE control: exactly the configuration of the preceding
% mesh-resolution campaign (Regime-B step controls, inner budget 30, no
% continuation, no fail-closed gate).  It quantifies what accepting
% non-converged inner solutions actually buys.
paper_literal = ismember(variant, {'V0','V1','V2','V3'});
if paper_literal
    cfg.move_lim = Inf;  cfg.outer_move = Inf;  cfg.alpha = 1.0;
    cfg.outer_tol = 1e-4;   step_label = 'paper-literal';
else
    cfg.move_lim = 0.2;  cfg.outer_move = 0.2;  cfg.alpha = 0.5;
    cfg.outer_tol = 1e-6;   step_label = 'regimeB';
end

% ---- continuation --------------------------------------------------------
switch variant
    case {'V1','V3','V5'}
        cfg.cont = struct('enabled',true,'p_values',[1 1.5 2 2.5 3], ...
                          'mode','fixed','stage_len',opts.cont_stage_len);
    case 'V5a'   % schedule sensitivity on the only viable variant: coarser ladder
        cfg.cont = struct('enabled',true,'p_values',[1 2 3], ...
                          'mode','fixed','stage_len',opts.cont_stage_len);
    case 'V5b'   % schedule sensitivity: shorter stages, same ladder
        cfg.cont = struct('enabled',true,'p_values',[1 1.5 2 2.5 3], ...
                          'mode','fixed','stage_len',max(5,round(0.6*opts.cont_stage_len)));
    otherwise
        cfg.cont = struct('enabled',false);
end

% ---- fail-closed ---------------------------------------------------------
cfg.fail_closed = ismember(variant, {'V2','V3','V4','V5','V5a','V5b'});
cfg.fc_vol_tol  = 1e-4;
cfg.fc_bound_tol = 1e-9;

tag = sprintf('%s_%s_%dx%d%s', variant, upper(bc), nelx, nely, opts.tag_suffix);
cfg.tag = tag;
out_dir = fullfile(this_dir, 'results', tag);
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

diary_file = fullfile(out_dir, 'log.txt');
if exist(diary_file,'file'), delete(diary_file); end
diary(diary_file);
cleanup = onCleanup(@() diary('off'));

fprintf('\n==================================================================\n');
fprintf(' VARIANT %s   %s   %dx%d   continuation=%d  fail_closed=%d  steps=%s\n', ...
    variant, upper(bc), nelx, nely, cfg.cont.enabled, cfg.fail_closed, step_label);
if cfg.cont.enabled
    fprintf(' continuation p = [%s]  mode=%s  stage_len=%d\n', ...
        num2str(cfg.cont.p_values,'%g '), cfg.cont.mode, cfg.cont.stage_len);
end
fprintf(' move_lim=%g outer_move=%g alpha=%g outer_tol=%g outer_max=%d\n', ...
    cfg.move_lim, cfg.outer_move, cfg.alpha, cfg.outer_tol, cfg.outer_max_iter);
fprintf('==================================================================\n');

rng(opts.seed, 'twister');
t0 = tic;
out = recon_solve(cfg);
out.wall_time = toc(t0);
out.variant = variant;
out.step_label = step_label;

fprintf('\n STOP STATUS : %s  after %d outer iterations (%.1f s)\n', ...
    out.stop_status, out.outer_iters, out.wall_time);
fprintf(' final omega  : %s\n', num2str(out.final_omega, '%.4f  '));
fprintf(' final N      : %g   g12 = %.4e   vol = %.4f\n', ...
    out.final_N, out.final_g12, out.final_volume);
fprintf(' final @ p=3  : omega = %s   N = %g\n', ...
    num2str(out.final_omega_p3, '%.4f  '), out.final_N_p3);

persist_run(out_dir, out);
save(fullfile(out_dir,'run.mat'), 'out', '-v7.3');
fprintf(' artefacts written to %s\n', out_dir);
diary('off');
end

%% =====================================================================
function persist_run(d, out)
h = out.hist;  n = out.outer_iters;
w = @(f) fullfile(d, f);

% ---------- outer history ----------
T = table(h.iter(1:n), h.penal, h.cont_stage, h.N, h.N_trial, h.J_idx, ...
    h.lambda_bar, h.lambda_J, h.beta, h.pred_dlambda, h.real_dlambda, ...
    h.pred_ratio, h.inner_iters, double(h.inner_converged), ...
    h.inner_last_change, h.inner_singular_warn, h.inner_cpu, ...
    h.drho_inf, h.drho_l2n, h.drho_min, h.drho_max_signed, ...
    h.frac_at_bound, h.frac_near_bound, h.n_at_lb, h.n_at_ub, ...
    h.pred_vol, h.vol_before, h.vol, h.dvol, h.grayness, ...
    h.step_alpha, double(h.accepted), h.eig_flag, ...
    h.d1_inf, h.d1_rms, h.d2_inf, h.d2_rms, ...
    h.rel_domega1, h.rel_dbeta, h.n_top_elems_90, ...
    h.asym_width_min, h.asym_width_mean, h.asym_width_max, ...
    h.asym_beta_low, h.asym_beta_upp, ...
    h.mma_lam_max, h.mma_ymma_max, h.mma_zmma, ...
    h.fsk_norm, h.fsk_absmax, ...
    'VariableNames', {'iter','penal','stage','N','N_trial','J_idx', ...
    'lambda_bar','lambda_J','beta','pred_dlambda','real_dlambda','pred_ratio', ...
    'inner_iters','inner_converged','inner_last_change','inner_singular_warn', ...
    'inner_cpu','drho_inf','drho_l2n','drho_min','drho_max', ...
    'frac_at_bound','frac_near_bound','n_at_lb','n_at_ub','pred_vol','vol_before','vol','dvol', ...
    'grayness','step_alpha','accepted','eig_flag','d1_inf','d1_rms', ...
    'd2_inf','d2_rms','rel_domega1','rel_dbeta','n_top_elems_90', ...
    'asym_w_min','asym_w_mean','asym_w_max','asym_beta_low','asym_beta_upp', ...
    'mma_lam_max','mma_y_max','mma_z','fsk_norm','fsk_absmax'});
writetable(T, w('outer_history.csv'));

% ---------- reject reasons / inner termination reasons ----------
R = table(h.iter(1:n), h.reject_reason(1:n), h.inner_reason(1:n), ...
    'VariableNames', {'iter','accept_reason','inner_termination'});
writetable(R, w('accept_history.csv'));

% ---------- eigen / gap history ----------
nm = size(h.omega, 2);
E = array2table([h.iter(1:n), h.omega, h.omega_trial, h.lambda, h.lambda_trial, ...
                 h.g12, h.g12_trial, h.g23], ...
    'VariableNames', [{'iter'}, ...
      arrayfun(@(k) sprintf('omega%d',k), 1:nm, 'uni', 0), ...
      arrayfun(@(k) sprintf('omega%d_trial',k), 1:nm, 'uni', 0), ...
      arrayfun(@(k) sprintf('lambda%d',k), 1:nm, 'uni', 0), ...
      arrayfun(@(k) sprintf('lambda%d_trial',k), 1:nm, 'uni', 0), ...
      {'g12','g12_trial','g23'}]);
writetable(E, w('eigen_history.csv'));

% ---------- multiplicity audit ----------
Mu = table(h.iter(1:n), h.N, h.N_trial, h.N_rec_1em2, h.N_rec_5em3, ...
    h.N_rec_1em3, h.N_rec_1em4, h.g12, h.g12_trial, h.omega(:,1), ...
    h.omega(:,2), h.omega(:,min(3,nm)), ...
    'VariableNames', {'iter','N_solver','N_trial','N_tol1e2','N_tol5e3', ...
    'N_tol1e3','N_tol1e4','g12','g12_trial','omega1','omega2','omega3'});
writetable(Mu, w('multiplicity_history.csv'));

% ---------- MAC history ----------
MA = array2table([h.iter(1:n), h.mac11, h.mac_best_1, h.mac_best_idx_1, h.macmat], ...
    'VariableNames', [{'iter','mac11','mac_best_mode1','mac_best_idx_mode1'}, ...
      arrayfun(@(k) sprintf('mac_%d_%d', 1+mod(k-1,nm), 1+floor((k-1)/nm)), ...
               1:nm*nm, 'uni', 0)]);
writetable(MA, w('mac_history.csv'));

% ---------- inner MMA history (every inner iteration) ----------
rows = [];
for k = 1:numel(out.inner_summary)
    s = out.inner_summary(k);
    ni = numel(s.drho_change);
    fj = s.fval_J;  if numel(fj) < ni, fj = nan(1,ni); end
    rows = [rows; [repmat(k,ni,1), (1:ni)', s.drho_change(:), s.beta(:), ...
            s.fval_cluster(:), fj(:), s.fval_vol(:), s.frac_at_bound(:), ...
            s.frac_near_bound(:), s.drho_inf(:), s.pred_vol(:), s.asym_width_mean(:), ...
            s.mma_lam_max(:), s.mma_ymma_max(:), ...
            repmat(double(s.converged),ni,1), repmat(s.n_singular_warn,ni,1)]]; %#ok<AGROW>
end
I = array2table(rows, 'VariableNames', {'outer_iter','inner_iter', ...
    'drho_change','beta','fval_cluster','fval_J','fval_vol','frac_at_bound', ...
    'frac_near_bound','drho_inf','pred_vol','asym_width_mean','mma_lam_max','mma_y_max', ...
    'inner_converged','n_singular_warn'});
writetable(I, w('inner_history.csv'));

% ---------- topology ----------
writematrix(reshape(out.rho_final, out.cfg.nely, out.cfg.nelx), w('rho_final.csv'));
snap = out.rho_snapshots;
keep = unique([1:5:size(snap,2), max(1,size(snap,2)-19):size(snap,2)]);
writematrix([keep; snap(:,keep)], w('rho_snapshots.csv'));

% ---------- terminal-cycle diagnostics (Q3) ----------
tail = max(1, n-59):n;
if size(snap,2) >= 3
    t3 = max(3, n-59):n;
    d3inf = nan(numel(t3),1);  d3rms = nan(numel(t3),1);
    for i = 1:numel(t3)
        k = t3(i);
        if k-3 >= 1
            dd = snap(:,k) - snap(:,k-3);
            d3inf(i) = max(abs(dd));  d3rms(i) = norm(dd)/sqrt(size(snap,1));
        end
    end
else
    t3 = tail;  d3inf = nan(numel(tail),1);  d3rms = d3inf;
end
C = table(h.iter(tail), h.d1_inf(tail), h.d1_rms(tail), h.d2_inf(tail), ...
    h.d2_rms(tail), h.rel_domega1(tail), h.rel_dbeta(tail), h.dvol(tail), ...
    h.grayness(tail), h.n_top_elems_90(tail), h.omega_trial(tail,1), ...
    'VariableNames', {'iter','d1_inf','d1_rms','d2_inf','d2_rms', ...
    'rel_domega1','rel_dbeta','dvol','grayness','n_top_elems_90','omega1'});
writetable(C, w('convergence_cycle.csv'));
writetable(table(t3(:), d3inf, d3rms, 'VariableNames', {'iter','d3_inf','d3_rms'}), ...
    w('convergence_cycle_lag3.csv'));

% ---------- tail deltas at full spatial resolution ----------
nt = min(10, size(snap,2)-1);
if nt >= 1
    D = zeros(size(snap,1), nt);
    for i = 1:nt, D(:,i) = snap(:,end-nt+i) - snap(:,end-nt+i-1); end
    writematrix(D, w('tail_deltas.csv'));
end

% ---------- machine-readable summary ----------
s = struct();
s.tag = out.cfg.tag;           s.variant = out.variant;
s.step_controls = out.step_label;
s.bc = out.cfg.support_type;   s.nelx = out.cfg.nelx;  s.nely = out.cfg.nely;
s.nEl = out.nEl;               s.nDof = out.nDof;
s.continuation_enabled = out.cfg.cont.enabled;
if out.cfg.cont.enabled
    s.continuation_p = out.cfg.cont.p_values;
    s.continuation_mode = out.cfg.cont.mode;
    s.continuation_stage_len = out.cfg.cont.stage_len;
end
s.fail_closed = out.cfg.fail_closed;
s.move_lim = out.cfg.move_lim;  s.outer_move = out.cfg.outer_move;
s.alpha = out.cfg.alpha;        s.outer_tol = out.cfg.outer_tol;
s.outer_max_iter = out.cfg.outer_max_iter;
s.inner_max_iter = out.cfg.inner_max_iter;  s.inner_tol = out.cfg.inner_tol;
s.stop_status = out.stop_status;
s.outer_iters = out.outer_iters;
s.wall_time_s = out.wall_time;
s.run_started_ok = out.outer_iters >= 1 && all(isfinite(h.omega(1,:)));
s.omega1_init = h.omega(1,1);
s.omega1_final = out.final_omega(1);
s.omega2_final = out.final_omega(2);
s.omega1_final_p3 = out.final_omega_p3(1);
s.omega2_final_p3 = out.final_omega_p3(2);
ot = h.omega_trial(:,1);  ot = ot(isfinite(ot));
if isempty(ot), ot = NaN; end
[s.omega1_best, bi] = max(ot);   s.omega1_best_iter = bi;
s.final_N = out.final_N;  s.final_N_p3 = out.final_N_p3;
s.final_g12 = out.final_g12;  s.final_g12_p3 = out.final_g12_p3;
gg = h.g12_trial(isfinite(h.g12_trial));
if isempty(gg), s.min_g12_trial = NaN; else, s.min_g12_trial = min(gg); end
gg2 = h.g12(isfinite(h.g12));
if isempty(gg2), s.min_g12 = NaN; else, s.min_g12 = min(gg2); end
i2 = find(h.N >= 2, 1, 'first');
if isempty(i2), s.first_N2_iter = -1; else, s.first_N2_iter = i2; end
i2t = find(h.N_trial >= 2, 1, 'first');
if isempty(i2t), s.first_N2_trial_iter = -1; else, s.first_N2_trial_iter = i2t; end
s.n_N2_iters = sum(h.N >= 2);
s.n_N2_trial_iters = sum(h.N_trial >= 2);
s.n_N2_tol1e2 = sum(h.N_rec_1em2 >= 2);
s.final_volume = out.final_volume;
s.final_grayness = mean(4*out.rho_final.*(1-out.rho_final));
s.n_inner_converged = sum(h.inner_converged);
s.n_inner_nonconverged = sum(~h.inner_converged);
s.frac_inner_converged = mean(double(h.inner_converged));
s.n_rejected_outer = sum(~h.accepted);
s.n_singular_warn_total = nansum(h.inner_singular_warn);
s.mechanism_collapse = any(ot < 0.05 * h.omega(1,1));
s.min_omega1_trial = min(ot);
s.final_d1_rms = h.d1_rms(end);  s.final_d1_inf = h.d1_inf(end);
if n >= 2, s.final_d2_rms = h.d2_rms(end); else, s.final_d2_rms = NaN; end
s.median_pred_ratio = median(h.pred_ratio(isfinite(h.pred_ratio)));
s.n_pred_overshoot = sum(h.pred_ratio < 0.1 & isfinite(h.pred_ratio));
s.paper_omega1 = paper_target(out.cfg.support_type);
s.pct_of_paper = 100 * out.final_omega_p3(1) / s.paper_omega1;
fid = fopen(w('summary.json'), 'w');
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
fclose(fid);
end

%% =====================================================================
function v = paper_target(bc)
switch upper(bc)
    case 'SS', v = 174.7;
    case 'CS', v = 288.7;
    otherwise, v = 456.4;
end
end
