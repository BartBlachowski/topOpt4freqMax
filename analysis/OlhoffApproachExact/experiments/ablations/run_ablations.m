function T = run_ablations(nelx, nely, nit, arm_sel, outdir)
% RUN_ABLATIONS  Phase 7.3: one-factor-at-a-time ablations, SS beam.
%
%   T = run_ablations()                 % 80x10, 200 iterations
%   T = run_ablations(160, 20, 300)
%
%   Each arm changes exactly ONE thing relative to the reference configuration.
%   A1 and A2 measure how much of any success is owed to the Du & Olhoff (2007)
%   imports [D3]/[D4] rather than to anything stated in Olhoff & Du (2014); A3
%   and A4 test the two remaining reconstructed choices.
%
%     ref            du2007_c1 mass [D3], sensitivity filter [D4],
%                    cluster model CA [R4], LP subproblem [R9], trust region [R1]
%     A1_pow         mass = olhoff2014_pow q = 1  -- Eq. (5) literal, [E6]
%     A1_step        mass = du2007_step          -- Eq. (4) literal
%     A2_nofilter    no sensitivity filter        -- Olhoff2014 mentions none
%     A2_rmin1p5     rmin = 1.5 instead of 2.5
%     A3_CC          cluster model CC (degenerate perturbation)
%     A4_mma         MMA subproblem solver instead of the exact LP
%     S_fixed        fixed move limit 0.02 instead of the trust region
%     S_tr_big       trust region starting at move_max = 0.2

%   COST WARNING.  Every arm except A4_mma costs a few seconds even at 160x20.
%   A4_mma is 300-1400x slower (80 inner MMA iterations on nelx*nely+1 variables
%   per outer iteration: 1442 s at 80x10, ~7.6 s/outer at 160x20) and floods
%   subsolv with RCOND ~ 6e-17 warnings.  Pass arm_sel to exclude it, e.g.
%       run_ablations(160, 20, 300, {'ref','A1_pow','A2_nofilter'})
%   Arms are individually guarded: a failing arm is recorded and the sweep
%   continues.

if nargin < 1 || isempty(nelx), nelx = 80;  end
if nargin < 2 || isempty(nely), nely = 10;  end
if nargin < 3 || isempty(nit),  nit  = 200; end
if nargin < 4, arm_sel = []; end
here = fileparts(mfilename('fullpath'));
if nargin < 5 || isempty(outdir), outdir = fullfile(here, 'results'); end
if ~exist(outdir,'dir'), mkdir(outdir); end

addpath(fullfile(here, '..', '..', 'Matlab'));
addpath(fullfile(here, '..', '..', '..', '..', 'tools', 'Matlab'));

base = struct('support_type','SS','nelx',nelx,'nely',nely, ...
              'outer_max_iter',nit,'verbose',false,'n_modes',6);

arms = {
  'ref',         struct()
  'A1_pow',      struct('mass_mode','olhoff2014_pow','mass_q',1)
  'A1_step',     struct('mass_mode','du2007_step')
  'A2_nofilter', struct('sensitivity_filter',false)
  'A2_rmin1p5',  struct('rmin_elem',1.5)
  'A3_CC',       struct('cluster_model','CC')
  'A4_mma',      struct('subproblem_solver','mma','inner',struct('max_iter',80), ...
                        'inner_audit',false)
  'S_fixed',     struct('step_control','fixed','move',0.02)
  'S_tr_big',    struct('move',0.2,'move_max',0.4)
};

if ~isempty(arm_sel)
    keep = ismember(arms(:,1), arm_sel);
    if ~any(keep)
        error('run_ablations:NoSuchArm', 'None of the requested arms exist: %s', ...
            strjoin(cellstr(string(arm_sel)), ', '));
    end
    arms = arms(keep, :);
end

fprintf('\n=== Phase 7.3 ablations: SS %dx%d, %d iterations, target omega_1 = 174.7 ===\n\n', ...
    nelx, nely, nit);
fprintf('%-13s %9s %8s %9s %9s %5s %6s %6s %6s %-22s %s\n', ...
    'arm','omega_1','err %','omega_2','gap12 %','N','vol','comp','iters','stop_reason','s');
fprintf('%s\n', repmat('-', 1, 118));

rows = {};
for k = 1:size(arms,1)
    cfg = base;
    ov  = arms{k,2};
    fn  = fieldnames(ov);
    for j = 1:numel(fn), cfg.(fn{j}) = ov.(fn{j}); end

    % Each arm is guarded so that one failure does not lose the whole sweep.
    lastwarn('');
    t0 = tic;
    try
        [rho, h] = topopt_freq_exact(cfg);
        el  = toc(t0);
        err = '';
    catch ME
        el  = toc(t0);
        err = sprintf('%s: %s', ME.identifier, ME.message);
        fprintf('%-13s %s\n', arms{k,1}, ['ERROR  ' err]);
        for s = 1:min(4, numel(ME.stack))
            fprintf('              at %s line %d\n', ME.stack(s).name, ME.stack(s).line);
        end
        rows{end+1} = failed_row(arms{k,1}, el, err); %#ok<AGROW>
        continue
    end
    [wmsg, wid] = lastwarn;
    if ~isempty(wid), wnote = wid; else, wnote = ''; end

    w1 = h.final_omega(1);  w2 = h.final_omega(2);
    gap = 100*(w2-w1)/w1;
    fprintf('%-13s %9.3f %+8.2f %9.3f %9.3f %5g %6.3f %6d %6d %-22s %.0f%s\n', ...
        arms{k,1}, w1, 100*(w1-174.7)/174.7, w2, gap, h.final_N, ...
        h.final_volume, h.final_components, h.outer_iters, h.stop_reason, el, ...
        warn_tag(wnote));

    writematrix(rho, fullfile(outdir, sprintf('rho_%s.csv', arms{k,1})));
    rows{end+1} = struct('arm', string(arms{k,1}), 'omega1', w1, ...
        'err_pct', 100*(w1-174.7)/174.7, 'omega2', w2, 'gap12_pct', gap, ...
        'N', h.final_N, 'volume', h.final_volume, 'components', h.final_components, ...
        'iters', h.outer_iters, 'stop_reason', string(h.stop_reason), ...
        'accepted_frac', mean(h.accepted(~isnan(h.accepted))), ...
        'final_move', h.move(h.outer_iters), ...
        'median_ratio', median(h.ratio(~isnan(h.ratio))), ...
        'median_fd', median(h.fd_audit(~isnan(h.fd_audit))), ...
        'last_warning', string(wnote), 'error', string(err), ...
        'seconds', el); %#ok<AGROW>
end

T = struct2table(cell2mat(rows));
writetable(T, fullfile(outdir, sprintf('ablations_%dx%d.csv', nelx, nely)));
nfail = sum(T.error ~= "");
if nfail > 0
    fprintf('\n %d of %d arms FAILED:\n', nfail, height(T));
    for k = 1:height(T)
        if T.error(k) ~= "", fprintf('   %-13s %s\n', T.arm(k), T.error(k)); end
    end
end
fprintf('\nwritten to %s\n\n', outdir);
end

%% =======================================================================
function r = failed_row(arm, el, err)
    r = struct('arm', string(arm), 'omega1', NaN, 'err_pct', NaN, 'omega2', NaN, ...
        'gap12_pct', NaN, 'N', NaN, 'volume', NaN, 'components', NaN, ...
        'iters', NaN, 'stop_reason', "error", 'accepted_frac', NaN, ...
        'final_move', NaN, 'median_ratio', NaN, 'median_fd', NaN, ...
        'last_warning', "", 'error', string(err), 'seconds', el);
end

function s = warn_tag(w)
    if isempty(w), s = ''; else, s = sprintf('  [warn: %s]', w); end
end
