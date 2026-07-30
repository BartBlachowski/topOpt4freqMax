function T = run_step_calibration(outdir)
% RUN_STEP_CALIBRATION  Phase 5: calibrate the move limit against Fig. 4.
%
%   T = run_step_calibration()
%   T = run_step_calibration(outdir)
%
%   The move limit m is the ONE reconstructed parameter of this solver ([R1] in
%   PLAN_Olhoff2014_exact.md).  Olhoff & Du (2014) Fig. 1 does not show a trust
%   region, but for N = 1 subproblem (19) is a linear program (section 2.5), so
%   its exact optimum over the full box of (19f) is a vertex; sequential LINEAR
%   programming is not defined without a move limit, and the paper's own LP
%   reduction cites Krog & Olhoff (1999).
%
%   Fig. 4 gives a complete 80-iteration history for the SS beam, which is a far
%   stronger calibration target than the endpoint alone.  Values read off the
%   figure at 300 dpi (+-3 % on frequency, +-2 on iteration index):
%
%       omega_1(1)            68.7        omega_1 monotone non-decreasing
%       omega_2(1)            ~255        omega_2 peak ~327 at iteration ~7
%       omega_3(1)            ~432        omega_3 peak ~528 at iteration ~9
%       coalescence           iteration ~20, at omega ~160
%       omega_1(80)           174.7       omega_3(80) ~288 (Fig. 5c: 284.9)
%
%   The monotone omega_1 is the decisive one: it says the step actually taken in
%   the paper was small enough never to overshoot.  Selection rule: the LARGEST m
%   that keeps omega_1 monotone, then best fingerprint score.
%
%   Writes a CSV of per-iteration histories and a summary table.

if nargin < 1 || isempty(outdir)
    outdir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if ~exist(outdir, 'dir'), mkdir(outdir); end

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', '..', 'Matlab'));
addpath(fullfile(here, '..', '..', '..', '..', 'tools', 'Matlab'));

moves   = [0.01 0.02 0.05 0.10 0.20 Inf];
solvers = {'lp', 'mma'};
NIT     = 80;                       % Fig. 4 horizon

% Fig. 4 reference fingerprint.
ref = struct('w1_0', 68.7, 'w2_0', 255, 'w3_0', 432, ...
             'w2_peak', 327, 'w2_peak_it', 7, ...
             'w3_peak', 528, 'w3_peak_it', 9, ...
             'coal_it', 20, 'w1_end', 174.7, 'w3_end', 284.9);

rows = {};
fprintf('\n=== Phase 5: move-limit calibration, SS 160x20, %d iterations ===\n', NIT);
fprintf('%-6s %-6s %-9s %-9s %-8s %-8s %-8s %-8s %-7s %-9s %-6s %s\n', ...
    'solver','m','w1_end','w1_err%','w2pk','w2pk@','w3pk','w3pk@','coal@','w1 mono','score','stop');
fprintf('%s\n', repmat('-', 1, 118));

for si = 1:numel(solvers)
  for mi = 1:numel(moves)
    m = moves(mi);
    cfg = struct();
    cfg.support_type      = 'SS';
    cfg.nelx = 160; cfg.nely = 20;
    cfg.move              = m;
    cfg.subproblem_solver = solvers{si};
    cfg.outer_max_iter    = NIT;
    cfg.outer_tol         = 0;          % run the full horizon for comparability
    cfg.n_modes           = 6;
    cfg.verbose           = false;
    if strcmp(solvers{si},'mma')
        cfg.inner       = struct('max_iter', 300);
        cfg.inner_audit = true;
    end

    t0 = tic;
    [rho, h] = topopt_freq_exact(cfg);
    el = toc(t0);

    s = score_run(h, ref);
    tag = sprintf('%s_m%s', solvers{si}, num2str(m));
    tag = strrep(strrep(tag, '.', 'p'), 'Inf', 'inf');

    write_hist(fullfile(outdir, ['hist_' tag '.csv']), h);
    writematrix(rho, fullfile(outdir, ['rho_' tag '.csv']));

    fprintf('%-6s %-6g %-9.3f %+8.2f %-8.1f %-8d %-8.1f %-8d %-7s %-9s %-6.3f %s\n', ...
        solvers{si}, m, s.w1_end, s.w1_err_pct, s.w2_peak, s.w2_peak_it, ...
        s.w3_peak, s.w3_peak_it, num2str(s.coal_it), yn(s.w1_monotone), ...
        s.score, h.stop_reason);

    rows{end+1} = struct('solver', solvers{si}, 'move', m, 'elapsed_s', el, ...
        'w1_end', s.w1_end, 'w1_err_pct', s.w1_err_pct, ...
        'w2_peak', s.w2_peak, 'w2_peak_it', s.w2_peak_it, ...
        'w3_peak', s.w3_peak, 'w3_peak_it', s.w3_peak_it, ...
        'w3_end', s.w3_end, 'coal_it', s.coal_it, ...
        'w1_monotone', s.w1_monotone, 'w1_min_drop_pct', s.w1_min_drop_pct, ...
        'score', s.score, 'final_N', h.final_N, 'volume', h.final_volume, ...
        'components', h.final_components, ...
        'max_fd_audit', max(h.fd_audit(~isnan(h.fd_audit))), ...
        'max_inner_gap', maxnan(h.inner_gap_rel), ...
        'stop_reason', h.stop_reason); %#ok<AGROW>
  end
end

T = struct2table_compat(rows);
writetable(T, fullfile(outdir, 'step_calibration_summary.csv'));

% ---- Selection rule ----------------------------------------------------
lp = T(strcmp(T.solver,'lp') & T.w1_monotone == 1, :);
fprintf('\n--- selection (LP solver, omega_1 monotone required) ---\n');
if isempty(lp)
    fprintf(' NO m keeps omega_1 monotone.  Report as inconclusive.\n');
else
    [~, k] = max(lp.score);
    kk = find(lp.move == max(lp.move(lp.score >= lp.score(k) - 0.05)), 1, 'last');
    if isempty(kk), kk = k; end
    fprintf(' best score      : m = %g (score %.3f)\n', lp.move(k), lp.score(k));
    fprintf(' SELECTED m      : %g  (largest m within 0.05 of the best score)\n', lp.move(kk));
end
fprintf('\nwritten to %s\n\n', outdir);
end

%% =======================================================================
function s = score_run(h, ref)
    w   = h.omega;                       % iteration x mode, pre-update values
    ni  = size(w,1);
    w1  = w(:,1);  w2 = w(:,2);  w3 = w(:,3);

    s.w1_end     = h.final_omega(1);
    s.w1_err_pct = 100*(s.w1_end - ref.w1_end)/ref.w1_end;
    s.w3_end     = h.final_omega(3);

    [s.w2_peak, s.w2_peak_it] = max(w2);
    [s.w3_peak, s.w3_peak_it] = max(w3);

    d = diff(w1);
    s.w1_monotone      = all(d >= -1e-9*max(w1));
    s.w1_min_drop_pct  = 100*min([0; d./w1(1:end-1)]);

    coal = find((w2 - w1)./w1 < 0.01, 1, 'first');
    if isempty(coal), s.coal_it = NaN; else, s.coal_it = coal; end

    % Fingerprint score in [0,1]: 6 equally weighted terms.
    t = zeros(6,1);
    t(1) = double(s.w1_monotone);
    t(2) = tol_score(s.w2_peak,    ref.w2_peak,    0.10) * it_score(s.w2_peak_it, ref.w2_peak_it, 3);
    t(3) = tol_score(s.w3_peak,    ref.w3_peak,    0.10) * it_score(s.w3_peak_it, ref.w3_peak_it, 3);
    if isnan(s.coal_it), t(4) = 0; else, t(4) = it_score(s.coal_it, ref.coal_it, 5); end
    t(5) = tol_score(s.w1_end,     ref.w1_end,     0.03);
    t(6) = tol_score(s.w3_end,     ref.w3_end,     0.05);
    s.score = mean(t);
    s.terms = t;
end

function v = tol_score(x, target, rel)
    e = abs(x - target)/abs(target);
    v = max(0, 1 - e/rel);
end

function v = it_score(x, target, tol)
    v = max(0, 1 - abs(x - target)/tol);
end

function v = maxnan(x)
    x = x(~isnan(x));
    if isempty(x), v = NaN; else, v = max(x); end
end

function s = yn(tf)
    if tf, s = 'yes'; else, s = 'NO'; end
end

function write_hist(fn, h)
    ni = h.outer_iters;
    nm = size(h.omega,2);
    Tb = table((1:ni)', 'VariableNames', {'iter'});
    for j = 1:nm
        Tb.(sprintf('omega%d', j)) = h.omega(1:ni, j);
    end
    Tb.beta_sqrt   = sqrt(max(h.beta(1:ni),0));
    Tb.N           = h.N(1:ni);
    Tb.volume      = h.volume(1:ni);
    Tb.drho_inf    = h.drho_inf(1:ni);
    Tb.frac_bound  = h.frac_at_bound(1:ni);
    Tb.inner_iters = h.inner_iters(1:ni);
    Tb.inner_gap   = h.inner_gap_rel(1:ni);
    Tb.fd_audit    = h.fd_audit(1:ni);
    Tb.components  = h.n_components(1:ni);
    Tb.grey_frac   = h.grey_frac(1:ni);
    Tb.min_mac     = h.mac_mode_n(1:ni);
    writetable(Tb, fn);
end

function T = struct2table_compat(rows)
    fn = fieldnames(rows{1});
    C  = cell(numel(rows), numel(fn));
    for i = 1:numel(rows)
        for j = 1:numel(fn)
            v = rows{i}.(fn{j});
            if islogical(v), v = double(v); end
            C{i,j} = v;
        end
    end
    T = cell2table(C, 'VariableNames', fn);
end
