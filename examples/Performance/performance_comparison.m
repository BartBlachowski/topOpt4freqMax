%PERFORMANCE_COMPARISON  Conference performance benchmark: Proposed, Yuksel, Du-Olhoff.
%
%   Press Run.  Everything that decides WHAT is measured is in the USER
%   CONFIGURATION block immediately below, in literals you can edit.  To run a
%   smaller mesh subset, edit cfg.resolutions and nothing else: there is no
%   manifest to update, no mode to select, no environment variable to clear.
%
%   Reproducibility comes from RECORDING the configuration that ran
%   (benchmark_manifest.json), not from making the configuration hard to change.
%
%       user edits cfg  ->  cfg validated  ->  benchmark runs  ->  manifest records
%
%   The three methods are architecturally different and are NOT forced into one
%   iteration count.  Total wall time is the common performance quantity; the
%   counts and component times explain the architecture behind it.  See
%   conference_bench/confbench_timing_schema.m, written out as timing_schema.json.
%
%   Memory is deliberately absent.  Reliable, method-independent peak-memory
%   measurement was not available in the MATLAB environment, so it was omitted
%   rather than reported with inconsistent semantics.
%
%   The previous driver is preserved at legacy_r3/performance_comparison_r3.m.

clear; clc; close all;

%% ============================================================
%  USER CONFIGURATION  -- this block is the whole control surface
%  ============================================================
cfg = struct();

% ---- Which meshes.  EDIT THIS MATRIX AND NOTHING ELSE to change the run. ----
% Uncomment exactly one matrix.  160x20 is the documented mesh-resolution floor,
% so even the single-row matrix is scientific evidence: it proves the whole
% fixed pipeline end to end in minutes rather than hours.  Anything wider also
% needs cfg.confirmLongCampaign below set to true.
% cfg.resolutions = [
%     160   20
% ];

% The four-resolution partial campaign:
% cfg.resolutions = [
%     160   20
%     240   30
%     320   40
%     400   50
% ];

% The nine-resolution conference campaign:
cfg.resolutions = [
    160   20
    240   30
    320   40
    400   50
    480   60
    560   70
    640   80
    720   90
    800  100
];

% ---- Which methods -------------------------------------------------------
cfg.methods = struct('proposed', true, 'yuksel', true, 'olhoff', true);

% ---- Execution -----------------------------------------------------------
cfg.singleThread = true;    % pin maxNumCompThreads(1) for every measured run
cfg.runWarmup    = true;    % one throwaway solve per method, off-campaign mesh
cfg.runEvaluator = true;    % common E1/E2/E3 evaluator, OUTSIDE all timing
cfg.fitScaling   = true;    % T(Ne) = C*Ne^p; refused unless this is a full campaign

% ---- Outputs -------------------------------------------------------------
cfg.writeCSV   = true;
cfg.writeJSON  = true;
cfg.writeLaTeX = true;
cfg.outputDir  = '';        % '' = auto, under examples/Performance/conference_benchmark/
cfg.runLabel   = '';        % '' = auto ('smoke' / 'preflight_160x20' / 'campaign_9mesh')

% ---- Guards --------------------------------------------------------------
% Running anything above 160x20 costs minutes to hours per row.  Flip this to
% true when you actually intend to launch it; the mesh list above stays visible
% and editable either way.
% Set true on 2026-09-04 for the four-resolution partial campaign selected
% above (160x20, 240x30, 320x40, 400x50).
cfg.confirmLongCampaign = true;

% Truncated outer budget for MECHANICS-ONLY smoke tests.  [] = the methods'
% own frozen budgets.  Any value here marks the whole run non-scientific.
cfg.maxOuterOverride = [];

% Yuksel per-stage SAFETY budget.  [] = the frozen profile value (1000).
%
% Raised to 5000 on 2026-09-05.  In campaign_9mesh, Yuksel reached the frozen
% 1000 in stage 1 at 640x80, 720x90 and 800x100 and in stage 2 at 640x80 and
% 800x100, so those rows report a CAP_HIT lower bound on the iterations and the
% time the method actually needed -- they are censored, and a scaling exponent
% fitted through them is biased downwards.  Extrapolating the uncensored
% per-stage counts (n1 ~ 0.217*Ne^0.760, n2 ~ 0.193*Ne^0.780) predicts about
% 1161 and 1286 at 800x100, so 5000 per stage carries roughly a four-fold
% margin and every mesh should stop on Yuksel's own rule instead of the cap.
%
% RAISING a safety budget does NOT make a run non-scientific: the manifest
% records this value's role as "per-stage safety budget; CAP_HIT is not
% convergence", so a larger budget only lets the native stopping rule decide.
% LOWERING it below the frozen value is truncation, and is treated exactly like
% cfg.maxOuterOverride.
cfg.yukselMaxIters = 5000;

% ---- Timing-accounting tolerances (predeclared, recorded in the artifacts) --
cfg.timingTolAbs     = 1e-6;   % |T_total - (T1+T2+T_overhead)|, seconds
cfg.timingTolRel     = 1e-9;   % ... or this fraction of T_total, whichever larger
cfg.crosscheckTolRel = 0.05;   % caller-side total vs solver self-reported total

%% ============================================================
%  PATHS
%  ============================================================
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(scriptDir));
addpath(scriptDir);
addpath(fullfile(scriptDir, 'conference_bench'));
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(fullfile(repoRoot, 'analysis', 'three_method_parametric_study'));
% The conference Du-Olhoff reconstruction.  Its solver core lives under
% +frozen/ and is reachable ONLY through olhoffm4_paths(), which asserts the
% implementation's identity before anything runs.  No superseded analysis/Olhoff*
% or Matlab/reproduction2007 path is added here, by design.
addpath(fullfile(repoRoot, 'analysis', 'OlhoffM4Reconstruction'));

% Not adding a superseded implementation is not enough: MATLAB paths are
% session state, and other scripts in this repository (examples/Revision_v1/*.m,
% Matlab/reproduction2007/runner/repro2007_verify_isolation.m) call
% addpath(genpath(<repo>/analysis)), which leaves analysis/Olhoff* and
% Matlab/reproduction2007 on the path for the rest of the session.  olhoffOpt
% then resolves to whichever of the seven realizations came first -- a run that
% looks fine and is scientifically void.  This driver curates its own path, so
% it REMOVES what the session handed it rather than inheriting it.  The
% preflight below re-checks the result independently; the scrub is recorded in
% the benchmark manifest.
pathScrub = olhoffm4_scrub_forbidden_paths(repoRoot);
if ~isempty(pathScrub)
    fprintf('Removed %d superseded Olhoff path entr%s inherited from this MATLAB session.\n', ...
        numel(pathScrub), pluralIes(numel(pathScrub)));
end

if cfg.singleThread
    maxNumCompThreads(1);
end

%% ============================================================
%  DERIVED RUN CLASS  (derived from cfg, never declared alongside it)
%  ============================================================
% These two flags decide how the run may be cited.  They are computed from the
% configuration so they cannot contradict what actually ran.
CAMPAIGN_MESHES = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
elementCounts = cfg.resolutions(:,1) .* cfg.resolutions(:,2);

% 3200 elements = 160x20, this project's documented mesh-resolution floor:
% nothing below it is scientific evidence, however cleanly it runs.
% A raised Yuksel budget is still scientific evidence; a lowered one is not.
% The frozen value is READ from the freeze manifest rather than restated here,
% so this test cannot drift away from the number the run actually uses.
yukselFrozenBudget = confbench_frozen_budget('yuksel');
if isempty(cfg.yukselMaxIters)
    cfg.yukselMaxIters = yukselFrozenBudget;
end
validateattributes(cfg.yukselMaxIters, {'numeric'}, ...
    {'scalar','integer','positive','finite'}, mfilename, 'cfg.yukselMaxIters');
yukselBudgetTruncated = cfg.yukselMaxIters < yukselFrozenBudget;

cfg.scientificEvidence  = isempty(cfg.maxOuterOverride) && all(elementCounts >= 3200) ...
    && ~yukselBudgetTruncated;
cfg.performanceCampaign = cfg.scientificEvidence && isequal(cfg.resolutions, CAMPAIGN_MESHES);

if isempty(cfg.runLabel)
    if ~isempty(cfg.maxOuterOverride) || yukselBudgetTruncated
        cfg.runLabel = 'smoke';
    elseif cfg.performanceCampaign
        cfg.runLabel = 'campaign_9mesh';
    elseif size(cfg.resolutions,1) == 1
        cfg.runLabel = sprintf('preflight_%dx%d', cfg.resolutions(1,1), cfg.resolutions(1,2));
    else
        cfg.runLabel = sprintf('partial_%dmesh', size(cfg.resolutions,1));
    end
end
if isempty(cfg.outputDir)
    cfg.outputDir = fullfile(scriptDir, 'conference_benchmark', cfg.runLabel);
end
if exist(cfg.outputDir, 'dir') ~= 7; mkdir(cfg.outputDir); end

methodKeys = {'proposed', 'yuksel', 'olhoff'};
methodKeys = methodKeys(cellfun(@(k) cfg.methods.(k), methodKeys));

fprintf('==========================================================\n');
fprintf(' CONFERENCE PERFORMANCE BENCHMARK\n');
fprintf('==========================================================\n');
fprintf('  resolutions          : %s\n', meshListStr(cfg.resolutions));
fprintf('  methods              : %s\n', strjoin(cellfun(@confbench_display_name, ...
    methodKeys, 'UniformOutput', false), ', '));
fprintf('  single thread        : %d (maxNumCompThreads = %d)\n', cfg.singleThread, maxNumCompThreads());
fprintf('  warm-up              : %d\n', cfg.runWarmup);
fprintf('  common evaluator     : %d (run OUTSIDE every solver timer)\n', cfg.runEvaluator);
fprintf('  scaling fit          : %d\n', cfg.fitScaling);
fprintf('  tables CSV/JSON/TeX  : %d / %d / %d\n', cfg.writeCSV, cfg.writeJSON, cfg.writeLaTeX);
fprintf('  outer budget override: %s\n', mat2str(cfg.maxOuterOverride));
fprintf('  Yuksel stage budget  : %d (frozen %d)%s\n', cfg.yukselMaxIters, ...
    yukselFrozenBudget, budgetNote(cfg.yukselMaxIters, yukselFrozenBudget));
fprintf('  run label            : %s\n', cfg.runLabel);
fprintf('  output directory     : %s\n', cfg.outputDir);
fprintf('  DERIVED scientific_evidence  : %d\n', cfg.scientificEvidence);
fprintf('  DERIVED performance_campaign : %d\n', cfg.performanceCampaign);
fprintf('  memory               : NOT MEASURED, NOT REPORTED\n\n');

%% ============================================================
%  METHOD CONFIGURATIONS  (top-level cfg -> validated method configs)
%  ============================================================
% Built for every (method, mesh) BEFORE the first expensive solve, so a stale
% or drifted frozen profile stops the run here rather than eight meshes later.
nRes = size(cfg.resolutions, 1);
nMet = numel(methodKeys);
methodConfigs = cell(nRes, nMet);
profileIds = cell(1, nMet);
for m = 1:nMet
    for r = 1:nRes
        [mc, pid] = confbench_method_config(methodKeys{m}, ...
            cfg.resolutions(r,1), cfg.resolutions(r,2), cfg.outputDir);
        methodConfigs{r, m} = mc;
        if r == 1
            profileIds{m} = pid;
        else
            assert(strcmp(profileIds{m}, pid), 'performance_comparison:ProfileDrift', ...
                'Frozen profile id for %s changed between meshes.', methodKeys{m});
        end
    end
end

fprintf('Frozen scientific settings bound for this run:\n');
for m = 1:nMet
    fprintf('  %-30s %s\n', confbench_display_name(methodKeys{m}), profileIds{m});
end
printMethodSettings(methodKeys, methodConfigs);

%% ============================================================
%  VALIDATION / PREFLIGHT
%  ============================================================
firstConfigs = struct();
for m = 1:nMet; firstConfigs.(methodKeys{m}) = methodConfigs{1, m}; end

fprintf('\n---------------- PREFLIGHT ----------------\n');
pre = confbench_preflight(cfg, firstConfigs);
for i = 1:numel(pre.checks)
    c = pre.checks(i);
    fprintf('  [%s] %s\n', tick(c.pass), c.name);
    if ~isempty(c.detail)
        fprintf('         %s\n', c.detail);
    end
end
for i = 1:numel(pre.notes)
    fprintf('  (note) %s\n', pre.notes{i});
end
fprintf('  PREFLIGHT: %s\n\n', verdict(pre.pass));
if ~pre.pass
    writeJsonFile(fullfile(cfg.outputDir, 'preflight_FAILED.json'), pre);
    error('performance_comparison:PreflightFailed', ...
        'Preflight failed; nothing was solved.  See %s', ...
        fullfile(cfg.outputDir, 'preflight_FAILED.json'));
end

%% ============================================================
%  WARM-UP  (discarded; never an observation)
%  ============================================================
% One throwaway solve per method at a mesh that is NOT in the run, so JIT
% compilation, BLAS/LAPACK initialization and first-touch allocation are paid
% before the first measured row rather than by the smallest mesh.
warmup = struct('ran', false, 'mesh', [], 'notes', {{}});
if cfg.runWarmup
    wNelx = 48; wNely = 6; wOuter = 5;
    warmup.ran = true; warmup.mesh = [wNelx wNely];
    fprintf('Warm-up at %dx%d (discarded)...\n', wNelx, wNely);
    for m = 1:nMet
        try
            wc = confbench_method_config(methodKeys{m}, wNelx, wNely, ...
                fullfile(cfg.outputDir, 'warmup'));
            wr = confbench_run_case(methodKeys{m}, wc, struct( ...
                'max_outer_override', wOuter, 'warmup', true, 'label', 'warmup'));
            warmup.notes{end+1} = sprintf('%s: %s in %.2f s', ...
                confbench_display_name(methodKeys{m}), wr.status, ...
                fieldOr(wr.times, 'total_wall_time_s', NaN)); %#ok<SAGROW>
        catch wErr
            warmup.notes{end+1} = sprintf('%s: warm-up FAILED (%s)', ...
                confbench_display_name(methodKeys{m}), wErr.message); %#ok<SAGROW>
            warning('performance_comparison:WarmupFailed', '%s', warmup.notes{end});
        end
        fprintf('  %s\n', warmup.notes{end});
    end
    fprintf('\n');
end

%% ============================================================
%  MAIN LOOP OVER RESOLUTIONS
%  ============================================================
records = struct([]);
runOpts = struct('timing_tol_abs', cfg.timingTolAbs, ...
                 'timing_tol_rel', cfg.timingTolRel, ...
                 'crosscheck_tol_rel', cfg.crosscheckTolRel, ...
                 'label', cfg.runLabel);
runOpts.yuksel_max_iters = cfg.yukselMaxIters;
if ~isempty(cfg.maxOuterOverride)
    runOpts.max_outer_override = cfg.maxOuterOverride;
end

for r = 1:nRes
    nelx = cfg.resolutions(r,1); nely = cfg.resolutions(r,2);
    fprintf('=== mesh %dx%d (%d elements) ===\n', nelx, nely, nelx*nely);
    for m = 1:nMet
        key = methodKeys{m};
        fprintf('  %-30s ... ', confbench_display_name(key));

        % ---- RUN.  The timer that matters lives inside confbench_run_case,
        % immediately around the solve.  Nothing below is inside it.
        rec = confbench_run_case(key, methodConfigs{r,m}, runOpts);

        rec.mesh = [nelx nely];
        rec.n_elements = nelx*nely;
        rec.profile_id = profileIds{m};
        rec.scientific_observation = rec.ok && cfg.scientificEvidence;

        % ---- COMMON EVALUATION, deliberately OUTSIDE every solver timer ----
        % study_evaluate_design is the unchanged frozen E1/E2/E3 evaluator.  It
        % reports each design under three shared material models and is NOT the
        % native frequency the solver optimized; the two are never merged.
        rec.evaluator = [];
        if cfg.runEvaluator && ~isempty(rec.x) && numel(rec.x) == nelx*nely
            try
                rec.evaluator = study_evaluate_design(double(rec.x(:)), nelx, nely, 0.5);
            catch evErr
                warning('performance_comparison:EvaluatorFailed', ...
                    'Common evaluator failed for %s %dx%d: %s', key, nelx, nely, evErr.message);
            end
        end

        printRunLine(rec);
        rec = orderfields(rec);
        if isempty(records); records = rec; else; records(end+1) = rec; end %#ok<SAGROW>
    end
    fprintf('\n');
end

%% ============================================================
%  TIMING-ACCOUNTING ASSERTIONS
%  ============================================================
fprintf('---------------- TIMING ACCOUNTING ----------------\n');
fprintf('  identity: T_total = T1 + T2 + T_overhead   (tolerance %.1e s / %.1e rel)\n', ...
    cfg.timingTolAbs, cfg.timingTolRel);
anyFail = false;
for i = 1:numel(records)
    a = records(i).accounting;
    fprintf('  %-30s %6dx%-4d residual %+.3e s (%+.2e rel)  %s\n', ...
        records(i).method, records(i).mesh(1), records(i).mesh(2), ...
        a.timing_accounting_residual_s, a.timing_accounting_relative_residual, ...
        flagText(a.timing_accounting_fail, 'TIMING_ACCOUNTING_FAIL', 'ok'));
    fprintf('  %-30s %6s      independent cross-check %+.3e s  %s\n', '', '', ...
        a.independent_crosscheck_residual_s, ...
        flagText(a.independent_crosscheck_fail, 'TIMING_CROSSCHECK_FAIL', 'ok'));
    anyFail = anyFail || a.timing_accounting_fail || a.independent_crosscheck_fail;
end
fprintf('  %s\n\n', verdict(~anyFail));

%% ============================================================
%  OPTIONAL SCALING FIT
%  ============================================================
scaling = confbench_scaling_fit(cfg, records);
if scaling.fitted
    fprintf('---------------- SCALING  T(Ne) = C*Ne^p ----------------\n');
    for i = 1:numel(scaling.methods)
        s = scaling.methods(i);
        fprintf('  %-30s C = %.6e   p = %.4f   R^2 = %.4f   (%d points)\n', ...
            s.method, s.C, s.p, s.R2, s.n);
    end
    fprintf('\n');
else
    fprintf('Scaling fit NOT performed: %s\n\n', scaling.reason);
end

%% ============================================================
%  RESULT STORAGE AND EXPORT  (outside every solver timing boundary)
%  ============================================================
resolvedImpl = struct();
if cfg.methods.olhoff
    olh = records(strcmp({records.method_key}, 'olhoff'));
    if ~isempty(olh); resolvedImpl.olhoff = olh(1).resolved_implementation; end
end
resolvedImpl.run_topopt_from_json = which('run_topopt_from_json');
resolvedImpl.study_evaluate_design = which('study_evaluate_design');
resolvedImpl.study_base_config = which('study_base_config');

manifest = confbench_manifest(cfg, methodConfigs, resolvedImpl);
manifest.warmup = warmup;
manifest.preflight = pre;
% Assigned field by field: struct('removed_entries', {c}) collapses to a 0x0
% struct array when c is an empty cell, which is exactly the common case here.
manifest.path_scrub = struct();
manifest.path_scrub.removed_entries = pathScrub;
manifest.path_scrub.rationale = ['Superseded Olhoff implementations inherited ' ...
    'from this MATLAB session were removed from the path before preflight, so ' ...
    'dispatch is a property of this driver rather than of whatever ran earlier ' ...
    'in the session.'];

files = confbench_export(cfg, records, manifest, scaling);
save(fullfile(cfg.outputDir, 'benchmark_records.mat'), 'records', 'cfg', ...
    'manifest', 'scaling', '-v7.3');

% Complexity-fit figures.  ALWAYS produced, for every run, from the recorded
% results only -- they are a view of the data, not a second measurement, so
% they are not gated on cfg.fitScaling or on the campaign flags.  What IS gated
% is which rows the curve is fitted through: confbench_complexity_plots fits
% the ok rows, the same ones confbench_scaling_fit accepts, and draws the rest
% hollow.  A figure that cannot be written must not lose a completed campaign,
% so the failure is a warning, not an error.
try
    plotFiles = confbench_complexity_plots(cfg, records, scaling);
    pf = fieldnames(plotFiles);
    for i = 1:numel(pf)
        files.(pf{i}) = plotFiles.(pf{i});
    end
catch plotErr
    warning('performance_comparison:ComplexityPlotsFailed', ...
        'Complexity-fit figures were not produced (%s).', plotErr.message);
end

% Final topology image per run -- one per method per mesh, so a 9-mesh
% three-method campaign leaves 27.  Rendered from the recorded design vector
% records(i).x through the single shared renderer, so nothing is re-solved and
% the three methods stay visually comparable.  Same failure policy as above: a
% figure must not be able to lose a completed campaign.
try
    topoInfo = confbench_topology_images(cfg, records);
    files.topologies_dir = topoInfo.dir;
catch topoErr
    warning('performance_comparison:TopologyImagesFailed', ...
        'Final topology images were not produced (%s).', topoErr.message);
end

fprintf('---------------- ARTIFACTS ----------------\n');
fn = fieldnames(files);
for i = 1:numel(fn)
    fprintf('  %-16s %s\n', fn{i}, files.(fn{i}));
end
fprintf('  %-16s %s\n', 'records_mat', fullfile(cfg.outputDir, 'benchmark_records.mat'));
fprintf('\nscientific_evidence  = %d\nperformance_campaign = %d\n', ...
    cfg.scientificEvidence, cfg.performanceCampaign);
fprintf('%s\n', confbench_caveats().olhoff);

%% ============================================================
%  LOCAL HELPERS
%  ============================================================
function s = budgetNote(used, frozen)
if used > frozen
    s = '  RAISED -- still scientific evidence';
elseif used < frozen
    s = '  TRUNCATED -- run is NOT scientific evidence';
else
    s = '';
end
end

function s = pluralIes(n)
if n == 1; s = 'y'; else; s = 'ies'; end
end

function s = meshListStr(R)
parts = arrayfun(@(i) sprintf('%dx%d', R(i,1), R(i,2)), 1:size(R,1), 'UniformOutput', false);
s = strjoin(parts, ', ');
end

function s = tick(ok)
if ok; s = 'PASS'; else; s = 'FAIL'; end
end

function s = verdict(ok)
if ok; s = 'PASS'; else; s = 'FAIL'; end
end

function s = flagText(isFail, failText, okText)
if isFail; s = failText; else; s = okText; end
end

function v = fieldOr(S, name, dflt)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name)); v = S.(name); else; v = dflt; end
end

function printRunLine(rec)
T = rec.times; C = rec.counts;
fprintf('%-22s ', rec.status);
if isfield(T, 'total_wall_time_s')
    fprintf('total %8.2f s | %s %-8s %s %-8s | %s %7.3f s %s %7.3f s | omega1 %.4f', ...
        T.total_wall_time_s, ...
        shortName(C, 'count1_name'), numStr(C, 'count1'), ...
        shortName(C, 'count2_name'), numStr(C, 'count2'), ...
        shortName(T, 'time1_name'), fieldOr(T, 'time1', NaN), ...
        shortName(T, 'time2_name'), fieldOr(T, 'time2', NaN), ...
        rec.omega1_native);
end
fprintf('\n');
if ~isempty(rec.error)
    fprintf('      %s\n', rec.status_note);
end
end

function s = shortName(S, name)
% Console-only abbreviation.  The exported artifacts always carry the full
% field name; this is just to keep the progress line readable.
full = char(string(fieldOr(S, name, '?')));
map = { ...
    'eigenanalysis_solves',                     'eigSolves'; ...
    'simp_iterations',                          'simpIters'; ...
    'stage1_iterations',                        'stage1Iters'; ...
    'stage2_iterations',                        'stage2Iters'; ...
    'outer_iterations',                         'outerIters'; ...
    'inner_mma_iterations_total',               'innerMMA'; ...
    'stage1_eigenanalysis_and_preparation_s',   'T_prep+eig'; ...
    'stage2_simp_time_s',                       'T_simp'; ...
    'stage1_time_s',                            'T_stage1'; ...
    'stage2_time_s',                            'T_stage2'; ...
    'outer_time_excluding_inner_s',             'T_outer\inner'; ...
    'inner_mma_time_total_s',                   'T_innerMMA'};
idx = find(strcmp(map(:,1), full), 1);
if isempty(idx); s = full; else; s = map{idx, 2}; end
end

function s = numStr(S, name)
v = fieldOr(S, name, NaN);
if isnumeric(v) && isfinite(v); s = sprintf('%g', v); else; s = 'N/A'; end
end

function printMethodSettings(methodKeys, methodConfigs)
fprintf('\nMethod settings as they will be used (first mesh):\n');
for m = 1:numel(methodKeys)
    mc = methodConfigs{1, m};
    fprintf('  %s\n', confbench_display_name(methodKeys{m}));
    switch methodKeys{m}
        case 'olhoff'
            fprintf(['    nested MMA (innerSolver=%s innerVar=%s variant=%s offDiag=%d ' ...
                'tolInner=%g maxInner=%d)\n'], mc.innerSolver, mc.innerVar, ...
                mc.mmaVariant, mc.offDiag, mc.tolInner, mc.maxInner);
            fprintf('    M4 multiplicity: multRule=%s subN=%d (no threshold classifier)\n', ...
                mc.multRule, mc.subN);
            fprintf('    filter: physical R=%g -> rminEl=%g derived, mode=%s\n', ...
                mc.rminPhys, mc.rminPhys/(mc.b/mc.nely), mc.filterMode);
            fprintf(['    outer stop: %s norm < %.6g  (per-element RMS %.6e), ' ...
                'guard=%s, cap=%d\n'], mc.outerNorm, mc.tolOuter, ...
                mc.tolOuter/sqrt(mc.nelx*mc.nely), mc.outerGuard, mc.maxOuter);
            fprintf('    continuation: %s ladder %s, window %d, tol %g, signal beta (legacy)\n', ...
                mc.moveFamily, mat2str(mc.s2Levels), mc.s2Window, mc.s2Tol);
            fprintf('    threads=%d, diagnostics recorder=%d\n', mc.threads, mc.diag);
        otherwise
            o = mc.optimization;
            fprintf('    optimizer=%s move=%g rmin=%g el tol=%g maxIters=%d volfrac=%g p=%g\n', ...
                o.optimizer, o.move_limit, o.filter.radius, o.convergence_tol, ...
                o.max_iters, o.volume_fraction, o.penalization);
            if isfield(o, 'yuksel')
                fprintf('    stage1: tol=%g maxIters=%d | stage2: tol=%g\n', ...
                    o.yuksel.stage1_tol, o.yuksel.stage1_max_iters, o.yuksel.stage2_tol);
            end
            if isfield(o, 'semi_harmonic_baseline')
                fprintf('    semi-harmonic baseline=%s, load sensitivity=%d\n', ...
                    o.semi_harmonic_baseline, o.semi_harmonic_load_sensitivity);
            end
    end
end
end

function writeJsonFile(path, s)
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end
