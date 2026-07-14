function res = a4_eigenpair_refresh(outDir, opts)
%A4_EIGENPAIR_REFRESH  A4 driver (A4_SPECIFICATION_V3).
%
%   res = A4_EIGENPAIR_REFRESH(outDir)
%   res = A4_EIGENPAIR_REFRESH(outDir, opts)
%
%   THE RESEARCH QUESTION (spec Part 1)
%     Does optimizing the quasi-static surrogate with a PERMANENTLY FROZEN
%     reference eigenpair produce a design whose TRUE fundamental frequency is
%     materially worse than one obtained by periodically refreshing it -- and if
%     so, at what refresh interval, and by what mechanism, does the frozen mode
%     cease to be a valid proxy?
%
%   DESIGN (spec Part 2)
%     ONE independent variable: the refresh interval N in {inf, 50, 10, 5, 1}.
%     EVERYTHING else is fixed and comes from ONE base config; N is the only
%     injected override.  Five sibling JSONs are rejected as a design -- that is
%     how ss_beam_harmonic_*.json drifted into four simultaneous factor changes.
%
%   PRIMARY ENDPOINT (spec Part 4)
%     TRUE omega1 of the tracked Phi1-type mode, from an INDEPENDENT exact
%     eigensolve of the final design.  The surrogate objective may NOT judge
%     itself and is never compared across arms (each refresh redefines it).
%
%   ACCEPTANCE (spec Part 5)
%     Three classes.  Approximation failure (mode loss, capping, contamination)
%     is a RESULT (Class C), not a rejection.  Only clean Class B arms may serve
%     as an accuracy reference.
%
%   Signature is (outDir) for runner compatibility.

if nargin < 2, opts = struct(); end
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(scriptDir));
addpath(fullfile(repoRoot, 'scripts', 'revision_v1'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));

if ~exist(outDir, 'dir'), mkdir(outDir); end

baseConfig = localOpt(opts, 'base_config', fullfile(scriptDir, 'a4_ss_400x50_base.json'));
Nlevels    = localOpt(opts, 'n_levels', [Inf, 50, 10, 5, 1]);   % spec Part 2
delta      = localOpt(opts, 'delta', 0.05);                     % spec §1.2: 5%
nModes     = localOpt(opts, 'n_modes', 20);                     % spec §4.1
runPreScreen = localOpt(opts, 'run_pre_screen', true);
preScreenOpts = localOpt(opts, 'pre_screen_opts', struct());

fprintf('\n');
fprintf('====================================================================\n');
fprintf(' A4 -- Eigenpair-refresh study (A4_SPECIFICATION_V3)\n');
fprintf('====================================================================\n');
fprintf('  base config : %s\n', baseConfig);
fprintf('  N levels    : %s  (the ONLY independent variable)\n', mat2str(Nlevels));
fprintf('  delta       : %.1f%% (pre-declared equivalence margin)\n', 100*delta);
fprintf('  endpoint    : TRUE omega1 (independent exact eigensolve)\n\n');

cfg = jsondecode(fileread(baseConfig));
localAssertPreconditions(cfg, baseConfig);

res = struct();
res.spec = 'A4_SPECIFICATION_V3';
res.base_config = baseConfig;
res.base_config_hash = localHashFile(baseConfig);
res.commit_sha = localCommitSha(repoRoot);
res.delta = delta;
res.n_levels = Nlevels(:)';
res.created_utc = localUtcNow();
res.arms = repmat(localBlankArm(), 0, 1);

% ---- Gate A4-Pre --------------------------------------------------------
res.pre_screen = struct('run', false, 'pass', false, 'verdict', 'not run');
if runPreScreen
    sr = a4_preflight_spectral_screen(outDir, preScreenOpts);
    res.pre_screen = struct('run', true, 'pass', sr.pass, 'verdict', sr.verdict, ...
        'artifact', sr.artifact);
    if ~sr.pass
        % Spec §7.4: abort with a specific identifier naming S1 as the blocker.
        error('run_all:A4SpectrumInadmissible', ...
            ['GATE A4-Pre FAILED. The SS beam intermediate spectra are dominated by ' ...
             'disconnected-island modes, so a refreshed reference cannot be made ' ...
             'meaningful. A4 IS BLOCKED ON S1. No mass setting will rescue this ' ...
             '(MASS_INTERPOLATION_DECISION.md). This is pre-registered decision-rule ' ...
             'outcome 3 and must be REPORTED, not worked around. Details: %s'], ...
            strjoin(sr.fail_reasons, ' | '));
    end
end

% ---- the sweep: N is the only thing that changes -------------------------
for i = 1:numel(Nlevels)
    N = Nlevels(i);
    tag = localNTag(N);
    fprintf('\n---- A4 arm N = %s ----\n', tag);

    runCfg = cfg;
    % THE SINGLE INJECTED OVERRIDE.  update_after = 0 means frozen (N = inf).
    if isinf(N)
        runCfg.domain.load_cases(1).loads(1).update_after = 0;
    else
        runCfg.domain.load_cases(1).loads(1).update_after = N;
    end
    runCfg.optimization.a4_endpoint_export = true;

    arm = localBlankArm();
    arm.N = N;
    arm.tag = tag;
    arm.base_config_hash = res.base_config_hash;
    arm.pmass = cfg.optimization.pmass;
    arm.baseline = cfg.optimization.semi_harmonic_baseline;
    arm.load_sensitivity = cfg.optimization.load_sensitivity;
    arm.cap = cfg.optimization.max_iters;
    arm.tol = cfg.optimization.convergence_tol;

    t0 = tic;
    ok = true;
    try
        % NOTE: run_topopt_from_json returns
        %   [x, omega, tIter, nIter, mem_usage, diagnostics]
        % `info` is the SIXTH output; the fifth is memory.
        [xFin, ~, ~, nIt, ~, info] = run_topopt_from_json(runCfg);
    catch ME
        ok = false;
        arm.success = false;
        arm.exception_id = ME.identifier;
        arm.exception_message = ME.message;
        % R-1 raises this when NO mode passes the §4.3.1 screen at a refresh.
        % Spec §7.1: it is a B3 event; it must be CLASSIFIED, not recovered.
        arm.refresh_inadmissible = strcmp(ME.identifier, 'topopt_freq:SemiHarmonicRefreshInadmissible');
        fprintf('  [%s] EXCEPTION: [%s] %s\n', tag, ME.identifier, ME.message);
    end
    arm.wall_clock_s = toc(t0);   % provenance only; never a performance claim

    if ok
        arm.success = true;
        arm.iterations = nIt;
        arm.final_design_change = localLastChange(info);
        ep = a4_endpoint_eval(info.a4_endpoint, nModes);

        arm.omega1_tracked     = ep.omega1_tracked;
        arm.mode_index_jstar   = ep.mode_index_jstar;
        arm.mac_to_phi0        = ep.mac_to_phi0;
        arm.omega1_min         = ep.omega1_min;
        arm.omega1_thresholded = ep.omega1_thresholded;
        arm.omega1_omega2_gap  = ep.omega1_omega2_gap;
        arm.grayness           = ep.grayness;
        arm.feasibility        = ep.feasibility;
        arm.n_components       = ep.n_components;

        ref = info.semi_harmonic_refresh;
        arm.n_refresh           = ref.n_refresh;
        arm.n_refresh_predicted = ref.n_refresh_predicted;
        arm.refresh_events      = ref.events;
        % Analytic operation count (spec §4.5): NOT a measurement.
        arm.eigensolves_analytic = 1 + arm.n_refresh + 1;   % init + refreshes + final verify
        arm.topology = xFin(:);

        fprintf(['  [%s] omega1_tracked=%.4f  j*=%d  MAC=%.4f  omega1_min=%.4f  ' ...
                 'refreshes=%d  iters=%d\n'], ...
            tag, arm.omega1_tracked, arm.mode_index_jstar, arm.mac_to_phi0, ...
            arm.omega1_min, arm.n_refresh, arm.iterations);
    end

    % ---- classify (spec Part 5) -----------------------------------------
    [cls, bd, why] = check_a4_run(arm);
    arm.class = cls;
    arm.breakdown = bd;
    arm.class_reason = why;
    fprintf('  [%s] CLASS: %s%s -- %s\n', tag, cls, localBd(bd), why);

    res.arms(end+1, 1) = arm; %#ok<AGROW>
end

% ---- H0/H1 decision (spec §5.3), computed from CLEAN arms only -----------
res.decision = localDecide(res.arms, delta);
fprintf('\n---- A4 DECISION ----\n  %s\n  %s\n', res.decision.outcome, res.decision.statement);

% ---- artifacts -----------------------------------------------------------
res.artifacts = localWriteArtifacts(outDir, res);
fprintf('\nA4 artifacts written to %s\n\n', outDir);
end

% =========================================================================

function localAssertPreconditions(cfg, path)
% Spec §7.4: the stage must fail loud if pmass ~= 1 or the baseline is not solid.
if ~isfield(cfg.optimization, 'pmass') || abs(double(cfg.optimization.pmass) - 1) > 0
    error('a4:BadMassModel', ...
        ['A4 requires pmass = 1 (LINEAR mass, the declared method -- see ' ...
         'MASS_INTERPOLATION_DECISION.md). Config %s declares pmass = %s.'], ...
        path, mat2str(localFieldOr(cfg.optimization, 'pmass', NaN)));
end
if ~isfield(cfg.optimization, 'semi_harmonic_baseline') || ...
        ~strcmpi(cfg.optimization.semi_harmonic_baseline, 'solid')
    error('a4:BadBaseline', ...
        'A4 requires semi_harmonic_baseline = "solid" (Gate A0-F1). Config %s declares "%s".', ...
        path, localFieldOr(cfg.optimization, 'semi_harmonic_baseline', '<missing>'));
end
end

function d = localDecide(arms, delta)
d = struct('outcome', '', 'statement', '', 'delta', delta, 'reference', NaN);

frozen = arms([arms.N] == Inf);
if isempty(frozen)
    d.outcome = 'INDETERMINATE';
    d.statement = 'no N=inf arm present';
    return;
end
frozen = frozen(1);

isClean = @(a) strcmp(a.class, 'ACCEPTED');
finite = arms(~isinf([arms.N]));

if ~isClean(frozen)
    d.outcome = 'INDETERMINATE';
    d.statement = sprintf(['the N=inf arm (the published method) is %s/%s -- it cannot serve ' ...
        'as the reference. Report as an observation about the published method.'], ...
        frozen.class, frozen.breakdown);
    return;
end
d.reference = frozen.omega1_tracked;

usable = finite(arrayfun(isClean, finite));
contaminated = finite(arrayfun(@(a) strcmp(a.breakdown, 'B3') || strcmp(a.breakdown, 'B4'), finite));

if isempty(usable) && ~isempty(contaminated)
    % Pre-registered outcome 3 -- and it is EXP4's outcome.
    d.outcome = 'OUTCOME_3_REFRESH_REFERENCE_UNAVAILABLE';
    d.statement = ['Neither hypothesis is supported: every refreshed arm is contaminated (B3) ' ...
        'or unstable (B4), so no refreshed reference could be constructed. This is EXP4''s ' ...
        'outcome. Report it as such; fall back to the MAC-threshold route and scope the ' ...
        'frozen-mode reliability claim explicitly. Do NOT read the frozen arm''s advantage ' ...
        'as evidence that refreshing hurts.'];
    return;
end

gains = arrayfun(@(a) (a.omega1_tracked - d.reference) / d.reference, usable);
if any(gains > delta)
    d.outcome = 'H1_FREEZING_COSTS_ACCURACY';
    d.statement = sprintf(['H0 REJECTED: a clean refreshed arm exceeds the frozen arm by ' ...
        '%.1f%% > delta=%.1f%%. Freezing costs accuracy; report the penalty and bound the ' ...
        'scope. main.tex:704 is evidenced.'], 100*max(gains), 100*delta);
elseif all(gains < -delta)
    d.outcome = 'FROZEN_EXCEEDS_CLEAN_REFRESH';
    d.statement = ['The frozen arm exceeds every CLEAN refreshed arm by more than delta. ' ...
        'Refreshing hurts, and the refreshed arms are not contaminated. Report it; do not ' ...
        'explain it away.'];
else
    d.outcome = 'H0_FREEZING_IS_BENIGN';
    d.statement = sprintf(['H0 RETAINED: every clean refreshed arm is within delta=%.1f%% of ' ...
        'the frozen arm (max deviation %.1f%%). Refreshing confers no measurable benefit on ' ...
        'this benchmark; main.tex:704''s directional claim must be softened.'], ...
        100*delta, 100*max(abs(gains)));
end
end

function a = localWriteArtifacts(outDir, res)
a = struct();

% ---- result .mat (runner requires a .mat artifact) ----------------------
a.mat_file = fullfile(outDir, 'a4_eigenpair_refresh_results.mat');
save(a.mat_file, 'res', '-v7.3');

% ---- result JSON (schema, spec §7.5) -----------------------------------
slim = res;
for i = 1:numel(slim.arms)
    slim.arms(i).topology = [];   % topology goes to CSV, not JSON
end
a.result_json = fullfile(outDir, 'a4_result.json');
localWriteJson(a.result_json, slim);

% ---- per-arm topology CSV ----------------------------------------------
a.topology_csv = {};
for i = 1:numel(res.arms)
    if isempty(res.arms(i).topology), continue; end
    p = fullfile(outDir, sprintf('a4_topology_N%s.csv', res.arms(i).tag));
    writematrix(res.arms(i).topology, p);
    a.topology_csv{end+1} = p; %#ok<AGROW>
end

% ---- Table A4-1 ---------------------------------------------------------
a.table_md = fullfile(outDir, 'a4_table.md');
localWriteTable(a.table_md, res);

% ---- figures ------------------------------------------------------------
a.figures = a4_plots(outDir, res);

% ---- manifest -----------------------------------------------------------
a.manifest = fullfile(outDir, 'a4_manifest.json');
man = struct('stage', 'A4', 'spec', 'A4_SPECIFICATION_V3', ...
    'created_utc', res.created_utc, 'commit_sha', res.commit_sha, ...
    'base_config', res.base_config, 'base_config_hash', res.base_config_hash, ...
    'n_levels', res.n_levels, 'decision', res.decision.outcome, ...
    'output_dir', outDir, 'files', {localListFiles(outDir)});
localWriteJson(a.manifest, man);
end

function localWriteTable(path, res)
L = {};
L{end+1} = '# Table A4-1 — Eigenpair-refresh study';
L{end+1} = '';
L{end+1} = sprintf('Spec: `A4_SPECIFICATION_V3`. Base config hash: `%s`. Commit: `%s`.', ...
    res.base_config_hash, res.commit_sha);
L{end+1} = sprintf('Pre-declared equivalence margin delta = %.1f%%.', 100*res.delta);
L{end+1} = '';
L{end+1} = '`Δω₁ vs N=∞` is populated **only for clean (Class B) arms**. It is left BLANK for';
L{end+1} = 'B3/B4 arms — a contaminated or unstable arm is disqualified as an accuracy reference';
L{end+1} = 'and its endpoint must not be read as one.';
L{end+1} = '';
L{end+1} = '| N | ω₁_tracked | ω₁_min | ω₁_thresh | MAC | j* | iters | conv | refreshes | eigensolves | grayness | comps | class | Δω₁ vs N=∞ |';
L{end+1} = '|---|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|---:|';

frozen = res.arms([res.arms.N] == Inf);
ref = NaN;
if ~isempty(frozen) && strcmp(frozen(1).class, 'ACCEPTED')
    ref = frozen(1).omega1_tracked;
end

for i = 1:numel(res.arms)
    a = res.arms(i);
    if strcmp(a.class, 'ACCEPTED') && isfinite(ref) && ref > 0
        dstr = sprintf('%+.2f%%', 100*(a.omega1_tracked - ref)/ref);
    else
        dstr = '';   % deliberately blank, not zero, not a dash
    end
    conv = 'no';
    if isfinite(a.final_design_change) && isfinite(a.tol) && ...
            a.final_design_change <= a.tol && a.iterations < a.cap
        conv = 'yes';
    end
    cls = a.class;
    if ~isempty(a.breakdown), cls = sprintf('%s/%s', a.class, a.breakdown); end
    L{end+1} = sprintf('| %s | %.4f | %.4f | %.4f | %.4f | %d | %d | %s | %d | %d | %.4f | %d | %s | %s |', ...
        a.tag, a.omega1_tracked, a.omega1_min, a.omega1_thresholded, a.mac_to_phi0, ...
        a.mode_index_jstar, a.iterations, conv, a.n_refresh, a.eigensolves_analytic, ...
        a.grayness, a.n_components, cls, dstr); %#ok<AGROW>
end
L{end+1} = '';
L{end+1} = sprintf('**Decision: %s**', res.decision.outcome);
L{end+1} = '';
L{end+1} = res.decision.statement;
L{end+1} = '';
L{end+1} = '_Wall-clock time is recorded for provenance only and may not appear in any';
L{end+1} = 'performance claim (spec §4.5)._';

fid = fopen(path, 'w');
fprintf(fid, '%s\n', strjoin(L, newline));
fclose(fid);
end

% ---- small helpers -------------------------------------------------------

function arm = localBlankArm()
arm = struct( ...
    'N', NaN, 'tag', '', ...
    'base_config_hash', '', 'pmass', NaN, 'baseline', '', 'load_sensitivity', '', ...
    'success', false, 'exception_id', '', 'exception_message', '', ...
    'refresh_inadmissible', false, ...
    'iterations', 0, 'cap', NaN, 'tol', NaN, 'final_design_change', NaN, ...
    'omega1_tracked', NaN, 'mode_index_jstar', 0, 'mac_to_phi0', NaN, ...
    'omega1_min', NaN, 'omega1_thresholded', NaN, 'omega1_omega2_gap', NaN, ...
    'grayness', NaN, 'feasibility', NaN, 'n_components', 0, ...
    'n_refresh', 0, 'n_refresh_predicted', 0, 'refresh_events', [], ...
    'eigensolves_analytic', 0, ...
    'limit_cycle', false, 'omitted_term_ratio', NaN, ...
    'wall_clock_s', NaN, 'topology', [], ...
    'class', '', 'breakdown', '', 'class_reason', '');
end

function t = localNTag(N)
if isinf(N), t = 'inf'; else, t = sprintf('%d', N); end
end

function s = localBd(bd)
if isempty(bd), s = ''; else, s = ['/' bd]; end
end

function c = localLastChange(info)
c = NaN;
if isstruct(info) && isfield(info, 'last_change') && ~isempty(info.last_change)
    c = info.last_change;
end
end

function v = localOpt(s, name, default)
v = default;
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
end
end

function v = localFieldOr(s, name, default)
v = default;
if isstruct(s) && isfield(s, name), v = s.(name); end
end

function h = localHashFile(path)
txt = fileread(path);
bytes = uint8(txt);
hash = uint32(2166136261);
prime = uint32(16777619);
for k = 1:numel(bytes)
    hash = bitxor(hash, uint32(bytes(k)));
    hash = hash * prime;
end
h = sprintf('fnv1a32_%08x', hash);
end

function sha = localCommitSha(repoRoot)
sha = 'unknown';
try
    [st, out] = system(sprintf('git -C "%s" rev-parse --short HEAD', repoRoot));
    if st == 0, sha = strtrim(out); end
catch
end
end

function t = localUtcNow()
t = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
end

function localWriteJson(path, data)
txt = jsonencode(data, PrettyPrint=true);
fid = fopen(path, 'w');
if fid < 0
    error('a4_eigenpair_refresh:WriteFailed', 'Cannot write %s', path);
end
fprintf(fid, '%s\n', txt);
fclose(fid);
end

function files = localListFiles(outDir)
files = {};
d = dir(outDir);
for k = 1:numel(d)
    if ~d(k).isdir
        files{end+1, 1} = d(k).name; %#ok<AGROW>
    end
end
end
