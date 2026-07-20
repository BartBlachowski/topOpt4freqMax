function screenResult = a4_preflight_spectral_screen(outDir, opts)
%A4_PREFLIGHT_SPECTRAL_SCREEN  Phase-2 Gate A4-Pre on common grid G.
%
%   screenResult = A4_PREFLIGHT_SPECTRAL_SCREEN(outDir)
%   screenResult = A4_PREFLIGHT_SPECTRAL_SCREEN(outDir, opts)
%
%   PROTOCOL
%     Reproduce the frozen trajectory on G and apply the exact Phase-2 adaptive
%     search. REFERENCE_UNAVAILABLE is recorded, not treated as a gate failure;
%     only SOLVER_FAILURE fails the pre-screen (§§3.7, 4.1, 5.4).
%
%   IMPLEMENTATION NOTE (recorded in A4_IMPLEMENTATION_REPORT.md)
%   The solver exposes no intermediate-design snapshot, and adding one would be
%   an unauthorized numerical change.  The frozen arm is DETERMINISTIC, so a run
%   capped at iteration c reproduces the full run's design at iteration c
%   bit-for-bit.  Each checkpoint is therefore obtained by a short frozen run
%   with max_iters = c.  No approximation is involved.

if nargin < 2, opts = struct(); end
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'scripts', 'revision_v1'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

c = a4_phase2_constants();
checkpoints = localOpt(opts, 'checkpoints', c.diagnostic_grid);
macThresh   = c.tau_mac;
baseConfig  = localOpt(opts, 'base_config', fullfile(scriptDir, 'a4_ss_400x50_base.json'));
finalIters  = localOpt(opts, 'final_iters', []);   % [] -> config max_iters

if ~exist(outDir, 'dir'), mkdir(outDir); end

cfg = jsondecode(fileread(baseConfig));
if isempty(finalIters)
    finalIters = cfg.optimization.max_iters;
end
allCheckpoints = unique([checkpoints(checkpoints <= finalIters)'; finalIters], 'stable');
allCheckpoints = allCheckpoints(:);

fprintf('\n=== GATE A4-Pre: spectral admissibility screen ===\n');
fprintf('  base config : %s\n', baseConfig);
fprintf('  checkpoints : %s (frozen arm, deterministic re-runs)\n', mat2str(allCheckpoints'));
fprintf('  screen      : A4_SPECIFICATION_V3 §4.3.1 (support-connectivity)\n\n');

screenResult = struct();
screenResult.gate = 'A4-Pre';
screenResult.base_config = baseConfig;
screenResult.checkpoints = allCheckpoints(:)';
screenResult.mac_threshold = macThresh;
screenResult.window_ladder = c.window_ladder;
screenResult.created_utc = localUtcNow();
screenResult.entries = repmat(localBlankEntry(), 0, 1);

pass = true;
failReasons = {};

for c = 1:numel(allCheckpoints)
    it = allCheckpoints(c);

    runCfg = cfg;
    runCfg.optimization.max_iters = it;
    runCfg.optimization.convergence_tol = 1e-16;  % never stop early: land exactly on `it`
    %  (must be > 0: run_topopt_from_json asserts a positive tolerance)
    % Frozen arm: update_after = 0.
    runCfg.domain.load_cases(1).loads(1).update_after = 0;
    runCfg.optimization.a4_endpoint_export = true;

    fprintf('  [A4-Pre] checkpoint iteration %d ... ', it);
    t0 = tic;
    % info is the SIXTH output of run_topopt_from_json (fifth is memory).
    [~, ~, ~, nIt, ~, info] = run_topopt_from_json(runCfg);
    el = toc(t0);

    endpoint = info.a4_endpoint;
    Kf = endpoint.K_final(endpoint.free,endpoint.free);
    Mf = endpoint.M_final(endpoint.free,endpoint.free);
    ctx = struct('nelx',endpoint.nelx,'nely',endpoint.nely, ...
        'edofMat',endpoint.edofMat,'KE',endpoint.KE,'ME',endpoint.ME, ...
        'M',endpoint.M_final,'free',endpoint.free,'Emax',endpoint.Emax, ...
        'Emin',endpoint.Emin,'rho0',endpoint.rho0,'rho_min',endpoint.rho_min, ...
        'penal',endpoint.penal,'massInterp',endpoint.massInterp);
    search = a4_adaptive_mode_search(Kf,Mf,endpoint.free,endpoint.ndof, ...
        endpoint.xPhys,ctx,endpoint.phi0_solid,endpoint.phi0_solid);

    entry = localBlankEntry();
    entry.iteration_requested = it;
    entry.iteration_reached = nIt;
    entry.elapsed_s = el;
    if isstruct(search.screen) && isfield(search.screen,'nComponents')
        entry.n_components = search.screen.nComponents;
    end
    if ~isempty(search.omegas), entry.omega1_min = search.omegas(1); end
    entry.omega1_tracked = NaN;
    entry.mode_index_jstar = search.selected_index;
    entry.mac_to_phi0 = NaN;
    entry.n_admissible = search.n_admissible;
    entry.selected = search.selected_index;
    entry.reason = search.search_outcome;
    entry.window_rungs_solved = search.window_rungs_solved;
    entry.m_final = search.m_final;
    entry.search_outcome = search.search_outcome;
    if search.selected_index > 0
        entry.omega1_tracked = search.candidates(search.selected_index).omega;
        entry.mac_to_phi0 = search.candidates(search.selected_index).mac_phi0;
    end

    % Admissible Phi1-type mode identifiable at this checkpoint?
    admissibleTracked = false;
    if search.selected_index >= 1 && search.selected_index <= numel(search.candidates)
        admissibleTracked = search.candidates(search.selected_index).admissible && ...
            isfinite(entry.mac_to_phi0) && entry.mac_to_phi0 >= macThresh;
    end
    entry.admissible_tracked = admissibleTracked;

    if strcmp(search.search_outcome,'SOLVER_FAILURE')
        pass = false;
        failReasons{end+1} = sprintf('iteration %d: solver failure: %s', ...
            it,search.failure_message); %#ok<AGROW>
    elseif ~admissibleTracked
        failReasons{end+1} = sprintf( ...
            'iteration %d: REFERENCE_UNAVAILABLE (j*=%d, MAC=%.4f, admissible=%d, components=%d)', ...
            it, entry.mode_index_jstar, entry.mac_to_phi0, entry.n_admissible, entry.n_components); %#ok<AGROW>
    end

    screenResult.entries(end+1, 1) = entry; %#ok<AGROW>
    fprintf('%s  (j*=%d, MAC=%.4f, admissible=%d/%d, comps=%d, %.1fs)\n', ...
        localTag(admissibleTracked), entry.mode_index_jstar, entry.mac_to_phi0, ...
        entry.n_admissible, entry.m_final, entry.n_components, el);
end

screenResult.pass = pass;
if pass
    screenResult.verdict = ['PASS: the adaptive common-grid pre-screen completed without ' ...
        'solver failure. REFERENCE_UNAVAILABLE events, if any, are measurements and do not halt A4.'];
else
    screenResult.verdict = 'FAIL: at least one adaptive pre-screen eigensolve failed (E-5).';
end
screenResult.fail_reasons = failReasons;

jsonPath = fullfile(outDir, 'a4_pre_screen.json');
localWriteJson(jsonPath, screenResult);
screenResult.artifact = jsonPath;

fprintf('\n  GATE A4-Pre: %s\n', localTag(pass));
fprintf('  %s\n', screenResult.verdict);
fprintf('  artifact: %s\n\n', jsonPath);
end

% =========================================================================

function e = localBlankEntry()
e = struct('iteration_requested', 0, 'iteration_reached', 0, 'elapsed_s', NaN, ...
    'n_components', 0, 'omega1_min', NaN, 'omega1_tracked', NaN, ...
    'mode_index_jstar', 0, 'mac_to_phi0', NaN, 'n_admissible', 0, ...
    'selected', 0, 'admissible_tracked', false, 'reason', '', ...
    'window_rungs_solved', [], 'm_final', 0, 'search_outcome', '');
end

function s = localTag(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end

function v = localOpt(s, name, default)
v = default;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
end
end

function t = localUtcNow()
t = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
end

function localWriteJson(path, data)
txt = jsonencode(data, PrettyPrint=true);
fid = fopen(path, 'w');
if fid < 0
    error('a4_preflight_spectral_screen:WriteFailed', 'Cannot write %s', path);
end
fprintf(fid, '%s\n', txt);
fclose(fid);
end
