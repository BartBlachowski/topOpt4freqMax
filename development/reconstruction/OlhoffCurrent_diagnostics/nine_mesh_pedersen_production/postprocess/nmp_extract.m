function X = nmp_extract(lockName, outName)
%NMP_EXTRACT  Post-processing ONLY (Parts 11, 13-16, 18).  Read-only on every raw
%   production file.  Run after the campaign MATLAB process has exited.
%
%   Inputs (raw, never modified):
%     <output_root>/benchmark_records.mat          runner records (authoritative row data)
%     <output_root>/runs/<mesh>/SOLVER_RESULT.mat  tapped solver result (history)
%     <output_root>/runs/<mesh>/RUN_OUT.mat        olhoffcurrent_run result
%     <output_root>/runs/<mesh>/{PRE,POST}CHECK.json, TAP.json
%     upstream sweep snapshot res.mat files and sweep_verification.json (comparison only)
%   Outputs:
%     evidence/EXTRACT.json, evidence/history/<mesh>.csv, evidence/topology_grid/<mesh>.bin
%
%   Additional computation, all on the FINAL designs only (no optimization):
%     - native final analysis repeated with the solver's own final-analysis
%       arithmetic, to obtain mode shapes for modal kinetic-energy location;
%     - eq. (4) re-evaluation with the arithmetic of sd_verify_sweeps.m.
here = fileparts(mfilename('fullpath'));
D = fileparts(here);
oc = fileparts(fileparts(D));
repo = fileparts(fileparts(oc));
restoredefaultpath;
addpath(fullfile(repo, 'examples', 'Performance'));
addpath(fullfile(repo, 'examples', 'Performance', 'conference_bench'));
addpath(fullfile(repo, 'tools', 'Matlab'));
addpath(fullfile(repo, 'analysis', 'three_method_parametric_study'));
addpath(oc); addpath(here); addpath(fullfile(D, 'scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);
if nargin < 1 || isempty(lockName); lockName = 'CAMPAIGN_LOCK.json'; end
if nargin < 2 || isempty(outName); outName = 'EXTRACT.json'; end

L = jsondecode(fileread(fullfile(D, lockName)));
outRoot = L.output_root_abs;
R = load(fullfile(outRoot, 'benchmark_records.mat'));
meshes = arrayfun(@(r) sprintf('%dx%d', r.mesh(1), r.mesh(2)), R.records, 'UniformOutput', false);
if strcmp(L.mode, 'campaign')
    assert(isequal(meshes(:), L.meshes(:)), 'runner records are not exactly the nine locked meshes in order');
end
isCampaign = strcmp(L.mode, 'campaign');
snap = fullfile(oc, 'diagnostics', 'scientific_delta_olhoff_migration', 'source_snapshot', ...
    '+olhoff_6b08708', 'repro', 'results');
sweepV = jsondecode(fileread(fullfile(oc, 'diagnostics', 'scientific_delta_olhoff_migration', ...
    'evaluations', 'sweep_verification.json')));
if iscell(sweepV); sweepV = [sweepV{:}]; end

if isCampaign
    histDir = fullfile(D, 'evidence', 'history');
    gridDir = fullfile(D, 'evidence', 'topology_grid');
else
    histDir = fullfile(tempname(), 'history'); gridDir = fullfile(fileparts(histDir), 'topology_grid');
end
if exist(histDir, 'dir') ~= 7; mkdir(histDir); end
if exist(gridDir, 'dir') ~= 7; mkdir(gridDir); end

X = struct('schema', 'nmp_extract/1', 'when', nmp_now_local(), 'matlab', version(), ...
    'output_root', strrep(outRoot, [repo filesep], ''), 'runs', struct([]));
GRID = [800 100];
grids = cell(numel(meshes), 1);

for i = 1:numel(meshes)
    tag = meshes{i};
    rd = fullfile(outRoot, 'runs', tag);
    S = load(fullfile(rd, 'SOLVER_RESULT.mat'));
    O = load(fullfile(rd, 'RUN_OUT.mat'));
    res = S.resTap; out = O.out; h = res.hist; cfg = res.cfg;
    pre = jsondecode(fileread(fullfile(rd, 'PRECHECK.json')));
    post = jsondecode(fileread(fullfile(rd, 'POSTCHECK.json')));
    tap = jsondecode(fileread(fullfile(rd, 'TAP.json')));
    nelx = cfg.domain.mesh.nelx; nely = cfg.domain.mesh.nely; NE = nelx*nely;
    rec = R.records(i);
    assert(isequal(rec.mesh, [nelx nely]), 'record order differs from lock at %s', tag);

    rho = double(res.rho(:));
    w = double(res.omega(:));
    nOut = numel(h.N);
    W = double(h.omega);                       % J x nOut
    gap = double(h.gap12(:));                  % (w2-w1)/w1 at each outer iterate
    w1 = W(1, :).';
    tOuter = double(h.tOuter(:)); tEig = double(h.tEig(:)); tInner = double(h.tInner(:));
    tGrad = double(h.tGrad(:)); nIn = double(h.nInner(:));
    tolMult = cfg.multiplicity.tolerance;      % existing configuration row (0.05)
    tolCoal = 0.02;                            % upstream repro/run_repro.m coalescenceIter
    spikeK = find(w1(2:end) < 0.7*w1(1:end-1)) + 1;   % upstream sweep audit spike_events

    % touch-then-separate episodes: gap < 0.02 followed later by gap > 0.05
    episodes = 0; inTouch = false;
    for k = 1:nOut
        if gap(k) < tolCoal; inTouch = true; end
        if inTouch && gap(k) > tolMult; episodes = episodes + 1; inTouch = false; end
    end
    spikeRec = struct('iter', {}, 'omega1_before', {}, 'omega1_after', {}, 'recovered_to_099', {}, 'recovery_iter', {});
    for s = spikeK(:).'
        after = find(w1(s:end) >= 0.99*w1(s-1), 1);
        spikeRec(end+1) = struct('iter', s, 'omega1_before', w1(s-1), 'omega1_after', w1(s), ...
            'recovered_to_099', ~isempty(after), 'recovery_iter', ifEmpty(after, NaN, s-1+after)); %#ok<AGROW>
    end

    % ---- native final analysis repeated (mode shapes for energy location) ----
    flat = olh.config.toLegacy(cfg);
    mdl = model2D(flat);
    massFinal = cfg.material.mass;
    stiffFinal = struct('model', cfg.material.stiffness.model, 'p', cfg.material.stiffness.p, ...
        'linearBelow', cfg.material.stiffness.linearBelow);
    eigOpts = struct('tol', cfg.eigen.tolerance, 'maxit', cfg.eigen.maxIterations, 'pFactor', cfg.eigen.krylovFactor);
    Jcalc = cfg.eigen.targetMode + cfg.eigen.maxCluster;
    [K, M] = assemble2D(mdl, rho, stiffFinal, massFinal);
    [wRe, Phi] = eigSolve(K, M, Jcalc, cfg.eigen.solver, [], eigOpts);
    gm = massScale(rho, massFinal);
    M0 = reshape(mdl.M0, 8, 8);
    energyShare = NaN(2, 1); energyShareSolid = NaN(2, 1);
    for j = 1:2
        full = zeros(mdl.ndof, 1); full(mdl.free) = Phi(:, j);
        U = full(mdl.edofMat);                              % nele x 8
        Ee = gm(:) .* sum((U*M0) .* U, 2);
        energyShare(j) = sum(Ee(rho <= 0.1))/sum(Ee);
        energyShareSolid(j) = sum(Ee(rho >= 0.9))/sum(Ee);
    end

    % ---- eq. (4) re-evaluation (arithmetic of sd_verify_sweeps.m) -----------
    m4 = struct('model', 'eq4', 'q', 1, 'lowDensityExponent', 6, 'cutoff', 0.1);
    [K4, M4] = assemble2D(mdl, rho, cfg.material.stiffness.p, m4);
    w4 = eigSolve(K4, M4, 3, 'eigs');

    % ---- common grid for topology comparisons ---------------------------------
    Z = reshape(rho, nely, nelx);
    iy = min(nely, max(1, ceil((1:GRID(2))*nely/GRID(2))));
    ix = min(nelx, max(1, ceil((1:GRID(1))*nelx/GRID(1))));
    G = Z(iy, ix);
    grids{i} = G;
    fid = fopen(fullfile(gridDir, [tag '.bin']), 'w', 'l'); fwrite(fid, G(:), 'double'); fclose(fid);

    % ---- history CSV ----------------------------------------------------------
    T = table((1:nOut).', 'VariableNames', {'outer'});
    for j = 1:size(W, 1); T.(sprintf('omega%d', j)) = W(j, :).'; end
    T.gap12 = gap; T.N = double(h.N(:)); T.nInner = nIn; T.cumInner = double(h.cumInner(:));
    T.innerConv = double(h.innerConv(:)); T.dxNorm2 = double(h.dxNorm2(:)); T.dxMax = double(h.dxOuter(:));
    T.moveMax = double(h.move(:)); T.moveMean = double(res.aux.moveMean(:)); T.vol = double(h.vol(:));
    T.volErr = double(h.volErr(:)); T.Mnd = double(res.aux.Mnd(:)); T.beta = double(h.beta(:));
    T.tOuter = tOuter; T.tEig = tEig; T.tGrad = tGrad; T.tInner = tInner;
    writetable(T, fullfile(histDir, [tag '.csv']));

    % ---- upstream sweep S (comparison only; read-only) ------------------------
    oldFile = fullfile(snap, ['S' tag], 'res.mat');
    old = struct('available', exist(oldFile, 'file') == 2);
    if old.available
        Lo = load(oldFile);
        ro = double(Lo.res.rho(:)); ho = Lo.res.hist;
        sv = sweepV(strcmp({sweepV.run}, ['S' tag]));
        old.status = Lo.res.status; old.outer = Lo.res.nOuter; old.inner = sum(ho.nInner);
        old.omega = double(Lo.res.omega(1:3)).'; old.w_eq4 = [sv.w1_eq4 sv.w2_eq4 sv.w3_eq4];
        old.gap_native_pct = 100*(old.omega(2)-old.omega(1))/old.omega(1);
        old.gap_eq4_pct = sv.gap_pct; old.Mnd = 4*mean(ro.*(1-ro)); old.gray_fraction = mean(ro > 0.1 & ro < 0.9);
        old.wall_s = sv.wall_s; old.rho_sha256 = sv.rho_sha256;
        old.rho_sha256_recomputed = nmp_util('sha256double', ro);
        old.rho_identical = isequal(ro, rho);
        old.rho_L1_mean = mean(abs(ro - rho)); old.rho_Linf = max(abs(ro - rho));
        old.omega_rel_diff = (w(1:3).' - old.omega)./old.omega;
        old.w_eq4_rel_diff = (w4(:).' - old.w_eq4)./old.w_eq4;
        old.hist_omega_identical = isequal(double(ho.omega), W);
        old.hist_nInner_identical = isequal(double(ho.nInner(:)), nIn);
        old.cfg_stop_tolerance = Lo.cfg.stop.tolerance;
    end

    a = out.accounting;
    r = struct();
    r.mesh = tag; r.nelx = nelx; r.nely = nely; r.NE = NE;
    r.ndof = S.mdlInfo.ndof; r.nfree = S.mdlInfo.nfree; r.nnode = S.mdlInfo.nnode;
    r.config_hash = pre.config_hash_of_cfg_about_to_be_solved;
    r.frozen_config_hash = pre.frozen_config_hash;
    r.runner = struct('status', rec.status, 'status_note', rec.status_note, 'ok', rec.ok, ...
        'method_key', rec.method_key, 'production_preset', rec.production_preset, ...
        'effective_config_hash', rec.effective_config_hash, 'scientific_observation', rec.scientific_observation, ...
        'times', rec.times, 'counts', rec.counts, 'stopping', rec.stopping, 'accounting', rec.accounting, ...
        'omega1_native', rec.omega1_native, 'x_sha256', nmp_util('sha256double', rec.x), ...
        'label', rec.label, 'profile_id', rec.profile_id);
    r.solver_status = res.status;
    r.log = res.log;
    r.log_converged_line = any(contains(res.log, 'converged at outer iteration'));
    r.outer = nOut; r.inner_total = sum(nIn); r.inner_mean = mean(nIn); r.inner_max = max(nIn);
    r.inner_min = min(nIn); r.inner_max_at = find(nIn == max(nIn), 1); r.n_inner_not_converged = sum(~logical(h.innerConv));
    r.omega = w(:).'; r.omega1 = w(1); r.omega2 = w(2); r.omega3 = w(3);
    r.lambda = double(res.lambda(:)).';
    r.omega_final_analysis_repeated = wRe(:).'; r.final_analysis_repeat_bitwise = isequal(wRe(:), w(:));
    r.gap_terminal = (w(2)-w(1))/w(1);
    r.gap_last_iterate = gap(end);
    [r.gap_min, r.gap_min_iter] = min(gap);
    r.gap_max = max(gap);
    r.n_iter_gap_lt_multiplicity_tol = sum(gap < tolMult); r.multiplicity_tolerance = tolMult;
    r.n_iter_gap_lt_coalescence = sum(gap < tolCoal); r.coalescence_threshold = tolCoal;
    r.first_coalescence_iter = ifEmpty(find(gap < tolCoal, 1), NaN, find(gap < tolCoal, 1));
    r.last_iter_gap_lt_multiplicity_tol = ifEmpty(find(gap < tolMult, 1, 'last'), NaN, find(gap < tolMult, 1, 'last'));
    r.touch_then_separate_episodes = episodes;
    r.terminal_effectively_bimodal = r.gap_terminal < tolMult;
    r.spikes = spikeRec; r.n_spikes = numel(spikeK); r.n_spikes_last10 = sum(spikeK > nOut - 10);
    [r.omega1_hist_peak, r.omega1_hist_peak_iter] = max(w1);
    r.omega1_initial = w1(1);
    r.iter99 = find(w1 >= 0.99*max(w1), 1);
    r.omega1_max_drop_rel = max([0; (w1(1:end-1) - w1(2:end))./w1(1:end-1)]);
    r.mode_table = res.modeTable;
    r.mode1_kinetic_energy_share_rho_le_0p1 = energyShare(1);
    r.mode2_kinetic_energy_share_rho_le_0p1 = energyShare(2);
    r.mode1_kinetic_energy_share_rho_ge_0p9 = energyShareSolid(1);
    r.mode2_kinetic_energy_share_rho_ge_0p9 = energyShareSolid(2);
    r.w_eq4 = w4(:).'; r.gap_eq4 = (w4(2)-w4(1))/w4(1);
    r.volume = mean(rho); r.volume_error = mean(rho) - cfg.design.volumeFraction;
    r.Mnd = 4*mean(rho.*(1-rho)); r.Mnd_hist_final = res.aux.Mnd(end);
    r.gray_fraction = mean(rho > 0.1 & rho < 0.9);
    r.void_fraction_le_0p1 = mean(rho <= 0.1); r.solid_fraction_ge_0p9 = mean(rho >= 0.9);
    r.rho_min = min(rho); r.rho_max = max(rho);
    r.frac_at_rho_min = mean(rho <= cfg.design.minimum*(1+1e-12)); r.frac_at_one = mean(rho >= 1-1e-12);
    r.eps = cfg.stop.tolerance; r.final_dxNorm2 = h.dxNorm2(end); r.final_dx_over_eps = h.dxNorm2(end)/cfg.stop.tolerance;
    r.final_dxMax = h.dxOuter(end); r.final_dx_rms = h.dxNorm2(end)/sqrt(NE);
    r.final_move_max = h.move(end); r.final_move_mean = res.aux.moveMean(end);
    r.move_floor = cfg.move.minimum; r.move_ceiling = cfg.move.initial;
    r.final_move_max_at_floor = h.move(end) <= cfg.move.minimum*(1+1e-12);
    r.n_iter_move_max_at_floor = sum(h.move <= cfg.move.minimum*(1+1e-12));
    r.final_dxMax_over_move_max = h.dxOuter(end)/h.move(end);
    r.dxNorm2_below_eps_iters = find(h.dxNorm2 < cfg.stop.tolerance);
    r.maxOuter = cfg.runtime.maxOuter;
    r.wall_s = S.callWall; r.sum_tOuter = sum(tOuter); r.overhead_s = S.callWall - sum(tOuter);
    r.tOuter_mean = mean(tOuter); r.tOuter_median = median(tOuter); r.tOuter_std = std(tOuter);
    r.tOuter_max = max(tOuter); r.tOuter_min = min(tOuter);
    nw = min(20, nOut);
    r.tOuter_first20_mean = mean(tOuter(1:nw)); r.tOuter_first20_median = median(tOuter(1:nw));
    r.tOuter_last20_mean = mean(tOuter(end-nw+1:end)); r.tOuter_last20_median = median(tOuter(end-nw+1:end));
    r.nInner_first20_mean = mean(nIn(1:nw)); r.nInner_last20_mean = mean(nIn(end-nw+1:end));
    r.wall_per_outer = S.callWall/nOut;
    r.eig_count = nOut + 1; r.eig_count_note = 'one assemble+eigs per outer iterate plus the final analysis (olhoffSolve.m:229, :615)';
    r.sum_tEig = sum(tEig); r.tEig_per_outer = sum(tEig)/nOut; r.tEig_per_call_in_loop = mean(tEig);
    r.sum_tGrad = sum(tGrad); r.tGrad_per_outer = sum(tGrad)/nOut;
    r.sum_tInner = sum(tInner); r.tInner_per_outer = sum(tInner)/nOut; r.tInner_per_inner_iter = sum(tInner)/sum(nIn);
    r.outer_excl_inner_per_outer = (sum(tOuter) - sum(tInner))/nOut;
    r.inner_share_of_wall = sum(tInner)/S.callWall;
    r.runner_accounting_equal_to_tap = struct( ...
        'wall', a.total_wall_time_s == S.callWall, 'sum_tOuter', a.outer_time_total_s == sum(tOuter), ...
        'median_tOuter', a.outer_time_per_outer_median_s == median(tOuter), 'outer', a.outer_iterations == nOut, ...
        'inner', a.inner_iterations_total == sum(nIn));
    r.evaluator = evalSummary(rec.evaluator);
    r.rho_sha256 = nmp_util('sha256double', rho);
    r.files = struct( ...
        'solver_result_sha256', olhoffcurrent_sha256_file(fullfile(rd, 'SOLVER_RESULT.mat')), ...
        'run_out_sha256', olhoffcurrent_sha256_file(fullfile(rd, 'RUN_OUT.mat')), ...
        'precheck_sha256', olhoffcurrent_sha256_file(fullfile(rd, 'PRECHECK.json')), ...
        'tap_sha256', olhoffcurrent_sha256_file(fullfile(rd, 'TAP.json')), ...
        'postcheck_sha256', olhoffcurrent_sha256_file(fullfile(rd, 'POSTCHECK.json')), ...
        'topology_png_sha256', shaOr(fullfile(outRoot, 'topologies', ['topology_olhoff_' tag '.png'])));
    r.precheck = struct('pass', pre.pass, 'when', pre.when, 'checks', pre.checks, 'identity', pre.identity);
    r.postcheck = struct('pass', post.pass, 'when', post.when, 'checks', post.checks, 'identity', post.identity, ...
        'tap_crosscheck', post.tap_crosscheck);
    r.tap = struct('when', tap.when, 'file_sha256', tap.file_sha256, 'rho_sha256', tap.rho_sha256);
    r.upstream_sweep = old;
    X.runs = [X.runs, r];
    fprintf('%-8s %-16s outer %3d inner %5d  w %.6f/%.6f  gap %.3f%%  Mnd %.4f  gray %.4f  wall %.0f s  bitwise-old %d\n', ...
        tag, rec.status, nOut, sum(nIn), w(1), w(2), 100*r.gap_terminal, r.Mnd, r.gray_fraction, S.callWall, ...
        old.available && old.rho_identical);
end

% ---- neighbour and reference topology comparisons on the common grid --------
tc = struct('pair', {}, 'L1_mean', {}, 'RMS', {}, 'IoU_05', {});
for i = 1:numel(meshes)-1
    A = grids{i}; B = grids{i+1};
    tc(end+1) = struct('pair', [meshes{i} ' -> ' meshes{i+1}], 'L1_mean', mean(abs(A(:)-B(:))), ...
        'RMS', sqrt(mean((A(:)-B(:)).^2)), 'IoU_05', iou(A, B)); %#ok<AGROW>
end
for i = 1:numel(meshes)-1
    A = grids{i}; B = grids{end};
    tc(end+1) = struct('pair', [meshes{i} ' vs ' meshes{end}], 'L1_mean', mean(abs(A(:)-B(:))), ...
        'RMS', sqrt(mean((A(:)-B(:)).^2)), 'IoU_05', iou(A, B)); %#ok<AGROW>
end
X.topology_comparisons = tc;
X.topology_grid = struct('nelx', GRID(1), 'nely', GRID(2), 'method', 'nearest-neighbour upsampling, column order, float64 LE');

% ---- runner-level artifacts ----------------------------------------------------
X.runner = struct('cfg_methods', R.cfg.methods, 'cfg_resolutions', R.cfg.resolutions, ...
    'scientific_evidence', R.cfg.scientificEvidence, 'performance_campaign', R.cfg.performanceCampaign, ...
    'run_label', R.cfg.runLabel, 'n_records', numel(R.records), 'method_keys', {{R.records.method_key}}, ...
    'preflight_pass', R.manifest.preflight.pass, 'n_preflight_checks', numel(R.manifest.preflight.checks), ...
    'cap_summary', R.manifest.cap_summary, 'scaling', R.scaling, 'warmup', R.manifest.warmup);
if isfield(R.manifest, 'git'); X.runner.manifest_git = R.manifest.git; end

if isCampaign; xPath = fullfile(D, 'evidence', outName); else; xPath = fullfile(tempdir, outName); end
X.written_to = xPath;
fid = fopen(xPath, 'w');
fprintf(fid, '%s\n', jsonencode(X, 'PrettyPrint', true)); fclose(fid);
fprintf('EXTRACT written: %d runs\n', numel(X.runs));
end

% =============================================================================
function v = ifEmpty(x, dflt, val)
if isempty(x); v = dflt; else; v = val; end
end

function s = shaOr(p)
if exist(p, 'file') == 2; s = olhoffcurrent_sha256_file(p); else; s = ''; end
end

function v = iou(A, B)
a = A(:) >= 0.5; b = B(:) >= 0.5;
v = sum(a & b)/max(sum(a | b), 1);
end

function e = evalSummary(ev)
e = struct('present', ~isempty(ev));
if isempty(ev); return; end
keep = {'status', 'volume', 'volume_residual', 'grayness', 'gray_fraction_01_09', 'minimum_density', ...
    'maximum_density', 'binary_volume'};
for k = keep
    if isfield(ev, k{1}); e.(k{1}) = ev.(k{1}); end
end
for id = {'E1', 'E2', 'E3'}
    f = id{1};
    for nm = {'status_raw_', 'selected_ordinal_raw_', 'selected_omega_raw_', 'omega_raw_', 'omega_binary_'}
        fld = [nm{1} f];
        if isfield(ev, fld); e.(fld) = ev.(fld); end
    end
end
for nm = {'connectivity_raw_05', 'connectivity_binary'}
    if isfield(ev, nm{1}); e.(nm{1}) = ev.(nm{1}); end
end
end

function s = nmp_now_local()
s = char(datetime('now', 'TimeZone', 'local', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
end
