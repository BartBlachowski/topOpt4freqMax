function bg_extract_new_runs()
%BG_EXTRACT_NEW_RUNS  Metrics, per-iteration CSVs and identity checks for every
%   docs/bimodality_gap/runs/BG_*.mat produced by bg_run_arm.
%   Identity checks against the recorded Pedersen campaign (read-only):
%     filterEl3 @ 400x50  : rminEl = 3 = 0.06/(1/50)  -> final rho must be bitwise equal
%     budget400 @ any mesh: the prefix 1..k_stop of hist.omega must be bitwise equal
%                           (stop.tolerance enters nothing but the stop test)
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(here);
dataDir = fullfile(fileparts(here), 'data'); runDir = fullfile(fileparts(here), 'runs');
a = 8; b = 1;
pdir = fullfile(root, 'examples/Performance/conference_benchmark/nine_mesh_pedersen_b21483b/runs');
files = dir(fullfile(runDir, 'BG_*.mat'));
rows = {}; checks = {};
for i = 1:numel(files)
    D = load(fullfile(runDir, files(i).name));
    nelx = D.mesh(1); nely = D.mesh(2);
    rho = double(D.rho(:));
    m = bg_metrics(rho, nelx, nely, a, b, D.rminEl);
    m.formulation = D.arm; m.arm = D.arm; m.nelx = nelx; m.nely = nely;
    m.nOuter = D.nOuter; m.status = D.status; m.cfgHash = D.cfgHash;
    m.omega1 = D.omega(1); m.omega2 = D.omega(2); m.gap12 = (D.omega(2)-D.omega(1))/D.omega(1);
    m.eps = D.eps; m.final_l2 = D.hist.dxNorm2(end); m.final_maxabs = D.hist.dxOuter(end);
    m.final_move_max = D.hist.move(end);
    if isfield(D.aux, 'moveMean') && ~isempty(D.aux.moveMean), m.final_move_mean = D.aux.moveMean(end); else, m.final_move_mean = NaN; end
    m.final_stage = D.hist.stage(end);
    m.frac_iters_N2 = mean(D.hist.N >= 2);
    m.last_iter_gap_lt_005 = max([0, find(D.hist.gap12 < 0.05, 1, 'last')]);
    m.Mnd_min_hist = min(D.aux.Mnd); m.Mnd_argmin = find(D.aux.Mnd == min(D.aux.Mnd), 1);
    m.Mnd_slope_last20 = (D.aux.Mnd(end) - D.aux.Mnd(max(1,end-20))) / min(20, max(D.nOuter-1,1));
    m.inner_total = sum(D.hist.nInner); m.n_inner_not_conv = sum(~logical(D.hist.innerConv));
    m.wall_s = D.wall_s; m.repo_head = D.repo_head; m.impl_tree_sha256 = D.impl_tree_sha256;
    m.preset = D.upstreamPreset; m.overrides = jsonencode(D.overrides);
    m.source = files(i).name;
    % descents of a ladder (move decreases)
    m.n_move_descents = sum(diff(D.hist.move(:)) < 0);
    % localized-mode spikes: omega_1 falling by more than 30 % from one outer iteration
    % to the next (the eigenvalue is evaluated at the START of each iteration), and the
    % final post-update analysis vs the last in-loop value
    w1 = D.hist.omega(1,:);
    m.n_spikes_30pct = sum(w1(2:end) < 0.7*w1(1:end-1));
    m.omega1_min_hist = min(w1);
    m.omega1_last_inloop = w1(end);
    m.omega1_final_over_last = D.omega(1)/w1(end);
    m.final_has_localized_mode = D.omega(1) < 0.7*max(w1);
    rows{end+1} = m; %#ok<AGROW>
    T = table((1:D.nOuter)', D.hist.omega(1,:)', D.hist.omega(2,:)', D.hist.N(:), D.hist.gap12(:), ...
        D.hist.dxNorm2(:), D.hist.dxOuter(:), D.hist.move(:), D.hist.nInner(:), D.hist.vol(:), D.aux.Mnd(:), D.hist.stage(:), D.hist.beta(:), ...
        'VariableNames', {'k','omega1','omega2','N','gap12','dxNorm2','dxMax','moveMax','nInner','vol','Mnd','stage','beta'});
    writetable(T, fullfile(dataDir, sprintf('iter_%s_%dx%d.csv', D.arm, nelx, nely)));
    save(fullfile(dataDir, sprintf('rho_%s_%dx%d.mat', D.arm, nelx, nely)), 'rho', 'nelx', 'nely', '-v7');
    fprintf('%-16s %4dx%-3d %-16s k=%3d Mnd=%.4f gray=%.4f w=%.4f (%.2f el) w1=%.4f gap=%.3f\n', ...
        D.arm, nelx, nely, D.status, D.nOuter, m.Mnd, m.gray_01_09, m.w_gray_phys, m.w_gray_el, m.omega1, m.gap12);
    % ---- identity checks vs the recorded campaign ---------------------
    ref = fullfile(pdir, sprintf('%dx%d', nelx, nely), 'SOLVER_RESULT.mat');
    if exist(ref, 'file')
        Sref = load(ref); R = Sref.resTap;
        if strcmp(D.arm, 'filterEl3') && nelx == 400
            c = struct('check', 'filterEl3_400x50_rho_bitwise_vs_campaign', 'pass', isequal(rho, double(R.rho(:))), ...
                'detail', sprintf('max|drho| = %.3e, rminEl(new) = %.17g', max(abs(rho - double(R.rho(:)))), D.rminEl));
            checks{end+1} = c; fprintf('  CHECK %s: %d (%s)\n', c.check, c.pass, c.detail); %#ok<AGROW>
        end
        if strcmp(D.arm, 'budget400')
            ks = numel(R.hist.N);
            same = isequal(D.hist.omega(:,1:ks), R.hist.omega) && isequal(D.aux.Mnd(1:ks), R.aux.Mnd) && isequal(D.hist.nInner(1:ks), R.hist.nInner);
            c = struct('check', sprintf('budget400_%dx%d_prefix_bitwise_vs_campaign', nelx, nely), 'pass', same, ...
                'detail', sprintf('prefix length %d (campaign stop); Mnd at k_stop: new %.6f, campaign %.6f', ks, D.aux.Mnd(ks), R.aux.Mnd(end)));
            checks{end+1} = c; fprintf('  CHECK %s: %d (%s)\n', c.check, c.pass, c.detail); %#ok<AGROW>
        end
    end
end
if ~isempty(rows), bg_write_rows(rows, fullfile(dataDir, 'new_run_metrics.csv')); end
fid = fopen(fullfile(dataDir, 'identity_checks.json'), 'w'); fwrite(fid, jsonencode(checks, 'PrettyPrint', true)); fclose(fid);
end
