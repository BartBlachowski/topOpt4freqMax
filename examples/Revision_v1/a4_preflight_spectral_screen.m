function screenResult = a4_preflight_spectral_screen(outDir, opts)
%A4_PREFLIGHT_SPECTRAL_SCREEN  Gate A4-Pre (A4_SPECIFICATION_V3 §6.1).
%
%   screenResult = A4_PREFLIGHT_SPECTRAL_SCREEN(outDir)
%   screenResult = A4_PREFLIGHT_SPECTRAL_SCREEN(outDir, opts)
%
%   WHY THIS GATE EXISTS
%   --------------------
%   No mass setting removes the spurious-mode contamination (linear, pmass=6 and
%   Du-Olhoff Eq.4b were all tested; none removes the family -- see
%   MASS_INTERPOLATION_DECISION.md).  A4 therefore cannot configure its way to a
%   clean spectrum.  All of the disconnected-component evidence comes from the
%   CLAMPED beam; whether the SS beam's INTERMEDIATE designs are affected is
%   NOT KNOWN.  This gate measures it, cheaply, before 16 h are committed.
%
%   It is the only thing standing between A4 and a repeat of EXP4.
%
%   PROTOCOL (spec §6.1)
%     Take the frozen (N=inf) arm's design at iterations {100, 300, 600, final}
%     and apply the §4.3.1 mode-admissibility screen to the first 10 modes,
%     recording the disconnected-component count.
%
%     PASS  an admissible Phi1-type mode (support-connected, MAC >= 0.8 to Phi0)
%           is identifiable at EVERY checkpoint  -> A4 proceeds.
%     FAIL  the low spectrum is dominated by disconnected-island modes at any
%           checkpoint -> A4 IS BLOCKED ON S1. This must be REPORTED, not worked
%           around (pre-registered decision-rule outcome 3, §5.3).
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

checkpoints = localOpt(opts, 'checkpoints', [100, 300, 600]);
nModes      = localOpt(opts, 'n_modes', 10);
macThresh   = localOpt(opts, 'mac_threshold', 0.8);
baseConfig  = localOpt(opts, 'base_config', fullfile(scriptDir, 'a4_ss_400x50_base.json'));
finalIters  = localOpt(opts, 'final_iters', []);   % [] -> config max_iters

if ~exist(outDir, 'dir'), mkdir(outDir); end

cfg = jsondecode(fileread(baseConfig));
if isempty(finalIters)
    finalIters = cfg.optimization.max_iters;
end
allCheckpoints = [checkpoints(:); finalIters];

fprintf('\n=== GATE A4-Pre: spectral admissibility screen ===\n');
fprintf('  base config : %s\n', baseConfig);
fprintf('  checkpoints : %s (frozen arm, deterministic re-runs)\n', mat2str(allCheckpoints'));
fprintf('  screen      : A4_SPECIFICATION_V3 §4.3.1 (support-connectivity)\n\n');

screenResult = struct();
screenResult.gate = 'A4-Pre';
screenResult.base_config = baseConfig;
screenResult.checkpoints = allCheckpoints(:)';
screenResult.mac_threshold = macThresh;
screenResult.n_modes = nModes;
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

    ep = a4_endpoint_eval(info.a4_endpoint, nModes);

    entry = localBlankEntry();
    entry.iteration_requested = it;
    entry.iteration_reached = nIt;
    entry.elapsed_s = el;
    entry.n_components = ep.n_components;
    entry.omega1_min = ep.omega1_min;
    entry.omega1_tracked = ep.omega1_tracked;
    entry.mode_index_jstar = ep.mode_index_jstar;
    entry.mac_to_phi0 = ep.mac_to_phi0;
    entry.n_admissible = sum([ep.screen.modes.admissible]);
    entry.selected = ep.screen.selected;
    entry.reason = ep.screen.reason;

    % Admissible Phi1-type mode identifiable at this checkpoint?
    admissibleTracked = false;
    if ep.mode_index_jstar >= 1 && ep.mode_index_jstar <= numel(ep.screen.modes)
        admissibleTracked = ep.screen.modes(ep.mode_index_jstar).admissible && ...
            isfinite(ep.mac_to_phi0) && ep.mac_to_phi0 >= macThresh;
    end
    entry.admissible_tracked = admissibleTracked;

    if ~admissibleTracked
        pass = false;
        failReasons{end+1} = sprintf( ...
            'iteration %d: no admissible Phi1-type mode (j*=%d, MAC=%.4f, admissible=%d, components=%d)', ...
            it, ep.mode_index_jstar, ep.mac_to_phi0, entry.n_admissible, ep.n_components); %#ok<AGROW>
    end

    screenResult.entries(end+1, 1) = entry; %#ok<AGROW>
    fprintf('%s  (j*=%d, MAC=%.4f, admissible=%d/%d, comps=%d, %.1fs)\n', ...
        localTag(admissibleTracked), ep.mode_index_jstar, ep.mac_to_phi0, ...
        entry.n_admissible, nModes, ep.n_components, el);
end

screenResult.pass = pass;
if pass
    screenResult.verdict = ['PASS: an admissible Phi1-type mode is identifiable at every ' ...
        'checkpoint. The SS beam intermediate spectra are usable; the refreshed arms have ' ...
        'a clean spectrum to refresh into. A4 proceeds.'];
else
    screenResult.verdict = ['FAIL: the low spectrum is dominated by disconnected-island modes. ' ...
        'A4 IS BLOCKED ON S1. No mass setting will rescue this. Refreshing cannot be made ' ...
        'meaningful on this benchmark. Report, do not work around ' ...
        '(pre-registered decision-rule outcome 3).'];
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
    'selected', 0, 'admissible_tracked', false, 'reason', '');
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
