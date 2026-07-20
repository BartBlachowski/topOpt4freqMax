function run_all_revision_experiments(mode, varargin)
%RUN_ALL_REVISION_EXPERIMENTS  Fail-loud master runner for paper revision experiments.
%
%   run_all_revision_experiments            % full mode (10 samples, 4 meshes)
%   run_all_revision_experiments('fast')    % fast mode (2 samples, 2 meshes)
%   run_all_revision_experiments('smoke')   % Gate I1 verification only
%   run_all_revision_experiments('full', 'resume', true)
%   run_all_revision_experiments('full', 'dry_run', true)
%   run_all_revision_experiments('stage', 'Exp2')
%   run_all_revision_experiments('stage', 'Exp3', 'force', true)
%
%   MODES
%   -----
%   full   Run all mandatory experiments.  Errors if any experiment fails.
%   fast   Same as full with reduced samples and meshes for quick checks.
%   smoke  Run only the intentionally failing exp_smoke_fail to verify that
%          the fail-loud infrastructure correctly detects capped runs and
%          reports the exact failed acceptance condition (Gate I1).
%          This mode ALWAYS ends with error() -- that is the expected result.
%
%   ACCEPTANCE CONDITIONS
%   ---------------------
%   The master runner rejects any mandatory experiment that:
%     (a) throws an exception
%     (b) returns an empty or non-struct result
%     (c) has required NaN values in key output fields
%     (d) reaches the iteration cap without meeting the convergence criterion
%         (detectable for schema-based results; partial check for legacy)
%     (e) loses the target mode (MAC < declared threshold)
%     (f) omits a required .mat artifact
%     (g) is a mandatory stage with no implementation (e.g. A4).  Such a stage
%         is registered as a visible placeholder and can never be silently
%         skipped or resume-validated.
%
%   PREFLIGHT (runs before ANY computation; see localPreflight)
%   ----------------------------------------------------------
%     P1  no active MATLAB path entry may reference OlhoffApproachExact
%     P2  no stage may invoke an archived or pre-authoritative runner
%     P3  no mandatory placeholder stage may be silently skipped
%     P4  output conflicts are reported for every stage up-front
%
%   Full stack traces are preserved and printed for all exceptions.
%
%   COMPARATOR SCOPE
%   ----------------
%   No stage performs a cross-code performance comparison. The former EXP1
%   (performance table) and EXP5 (scaling fit) are RETIRED as reviewer evidence:
%   the local Olhoff and Yuksel comparators are not faithful reference
%   implementations, so a timing or scaling comparison against them is
%   construct-invalid regardless of instrumentation. Both are preserved in
%   archive/obsolete_evidence/exp1_exp5/. See SCIENTIFIC_DECISION_EXP1_EXP5.md.
%   OlhoffApproachExact and its reconstruction pilots remain archived diagnostics.
%
%   OUTPUT DIRECTORIES
%   ------------------
%   Each experiment writes to its own subdirectory under output/:
%     examples/Revision_v1/output/s1/
%     examples/Revision_v1/output/exp2/
%     examples/Revision_v1/output/exp2b/
%     examples/Revision_v1/output/exp3/
%     examples/Revision_v1/output/a4/     (placeholder; not implemented)
%     examples/Revision_v1/output/smoke/
%
%   Unless resume=true or force=true, the runner refuses to overwrite an output
%   directory that already contains .mat, .csv, or .png files.
%
%   REVIEWER ISSUES ADDRESSED
%   -------------------------
%   S1     M7      Localized low-density mode mitigation. Gates Exp2b/Exp3.
%   Exp 2  V5/m5   Initial freqs. M5 mode shapes. M6 grayness. CR1/V4 MAC.
%   Exp 2b V3/MR5  Building tables. M7 spurious modes.
%   Exp 3  V3/MR5  Mesh convergence 200x25 vs 400x50.
%   A4     R1/V1C4 Eigenpair-refresh N={1,5,10,50,inf}.  NOT IMPLEMENTED.
%
%   RETIRED: the former Exp4 stage (exp4_sensitivity_ablation) used the
%   pre-authoritative load rho_nodal(x)*omega0*M_solid*Phi_solid and all four of
%   its variants terminated at the iteration cap.  It is superseded by the CR2
%   study and is NOT an A4 implementation.  It must not be relabelled as one.
%
%   EXPECTED RUNTIMES (Apple M-class, R2025b)
%   -----------------------------------------
%   smoke  mode: < 5 seconds
%   fast   mode (2 samples, 2 meshes):  ~15-30 min
%   full   mode (10 samples, 4 meshes): ~5-12 hours

if nargin < 1 || isempty(mode), mode = 'full'; end
rawMode = char(mode);
mode = lower(strtrim(rawMode));

scriptDir = fileparts(mfilename('fullpath'));
outRoot   = fullfile(scriptDir, 'output');
if ~exist(outRoot, 'dir'), mkdir(outRoot); end
prevDir   = pwd;
cleanupCd = onCleanup(@() cd(prevDir)); %#ok<NASGU>
cd(scriptDir);

localEnsurePaths(scriptDir);
opts = localParseRunnerOptions(mode, varargin{:});

fprintf('\n');
fprintf('+====================================================================+\n');
fprintf('|  REVISION EXPERIMENTS -- topOpt4freqMax  (fail-loud runner v3)     |\n');
fprintf('+====================================================================+\n\n');

localPrintHardwareInfo();

% ---- smoke mode: Gate I1 verification only ------------------------------
if strcmp(mode, 'smoke')
    localRunSmokeVerification(scriptDir, outRoot);
    % localRunSmokeVerification always calls error(); execution never reaches here
    return;
end

% ---- fast / full mode ---------------------------------------------------
switch mode
    case 'fast'
        nSamples  = 2;
        meshTable = [160, 20; 400, 50];
        alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0];
        fprintf('[MODE] fast -- %d samples, 2 meshes.\n\n', nSamples);
    case 'full'
        nSamples  = 10;
        meshTable = [160, 20; 240, 30; 320, 40; 400, 50];
        alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0];
        fprintf('[MODE] full -- %d samples, 4 meshes.\n\n', nSamples);
    case 'stage'
        nSamples  = 10;
        meshTable = [160, 20; 240, 30; 320, 40; 400, 50];
        alphaVals = [1.0, 0.75, 0.5, 0.25, 0.0];
        fprintf('[MODE] stage -- %s.\n\n', opts.stageName);
    otherwise
        error('run_all:BadMode', ...
            'Unknown mode "%s". Use "fast", "full", "stage", or "smoke".', mode);
end

% ---- per-experiment output directories ----------------------------------
% EXP4 is retired: exp4_sensitivity_ablation is a pre-authoritative runner
% (load rho_nodal(x)*omega0*M_solid*Phi_solid) and is superseded by CR2.
% It is NOT relabelled as A4.  A4 is a separate, not-yet-implemented study.
od = struct( ...
    's1',    fullfile(outRoot, 's1'),    ...
    'exp2',  fullfile(outRoot, 'exp2'),  ...
    'exp2b', fullfile(outRoot, 'exp2b'), ...
    'exp3',  fullfile(outRoot, 'exp3'),  ...
    'a4',    fullfile(outRoot, 'a4') );

campaignCfg = localCampaignConfig(mode, nSamples, meshTable, alphaVals);
stages = localBuildStages(od, nSamples, meshTable, alphaVals);
if strcmp(mode, 'stage')
    stages = localSelectStage(stages, opts.stageName);
end

% ---- PREFLIGHT: fail loud BEFORE any computation starts ------------------
localPreflight(stages, opts, mode);

progressPath = fullfile(outRoot, 'campaign_progress.json');
summaryPath  = fullfile(outRoot, 'campaign_summary.md');
progress = localInitProgress(progressPath, outRoot, mode, opts, stages);
progress.mode = mode;
progress.status = 'running';
progress.last_update_utc = localUtcNow();
localWriteProgress(progressPath, progress);

if opts.dryRun
    dryRows = localDryRun(stages, campaignCfg, opts);
    progress.status = 'completed';
    progress.current_stage = '';
    progress.last_update_utc = localUtcNow();
    progress.elapsed_seconds = localElapsedSince(progress.start_time_utc);
    localWriteProgress(progressPath, progress);
    localWriteCampaignSummary(summaryPath, stages, dryRows, progress);
    fprintf('\nDry run completed. No experiments executed.\n');
    fprintf('Summary: %s\n\n', summaryPath);
    return;
end

allResults = struct();
stageRecords = repmat(localBlankStageRecord(), 0, 1);

for k = 1:numel(stages)
    stage = stages(k);
    progress.current_stage = stage.tag;
    progress.last_update_utc = localUtcNow();
    localWriteProgress(progressPath, progress);

    [valid, vmsg] = localValidateStageArtifacts(stage, campaignCfg);
    if opts.resume && ~opts.force && valid
        fprintf('[%s] Resume validation passed; skipping existing stage.\n', stage.tag);
        rec = localStageRecord(stage, 'skipped', 0, vmsg, stage.outDir);
        stageRecords(end+1, 1) = rec; %#ok<AGROW>
        progress.skipped_stages{end+1} = stage.tag;
        progress.completed_stages{end+1} = stage.tag;
        progress.per_stage_elapsed.(stage.key) = 0;
        progress.output_directories.(stage.key) = stage.outDir;
        progress.last_update_utc = localUtcNow();
        progress.elapsed_seconds = localElapsedSince(progress.start_time_utc);
        localWriteProgress(progressPath, progress);
        continue;
    end

    stageWall = tic;
    try
        if ~opts.resume && ~opts.force
            localPrepareDirFail(stage.outDir);
        elseif ~exist(stage.outDir, 'dir')
            mkdir(stage.outDir);
        end


        [res, pass, cond, trace, elapsed] = localRunAndAccept( ...
            stage.tag, stage.desc, stage.runFn, stage.acceptFn);
    catch ME
        res = [];
        pass = false;
        cond = sprintf('runner exception before/during stage: [%s] %s', ME.identifier, ME.message);
        trace = ME.stack;
        elapsed = toc(stageWall);
    end

    progress.per_stage_elapsed.(stage.key) = elapsed;
    progress.output_directories.(stage.key) = stage.outDir;
    progress.elapsed_seconds = localElapsedSince(progress.start_time_utc);
    progress.last_update_utc = localUtcNow();

    if pass
        allResults.(stage.key) = res;
        localWriteStageMetadata(stage, campaignCfg, elapsed, 'accepted', cond);
        rec = localStageRecord(stage, 'run', elapsed, 'accepted', stage.outDir);
        stageRecords(end+1, 1) = rec; %#ok<AGROW>
        progress.completed_stages{end+1} = stage.tag;
        localWriteProgress(progressPath, progress);
    else
        f = localMakeFailure(stage.tag, cond, trace, elapsed);
        rec = localStageRecord(stage, 'failed', elapsed, cond, stage.outDir);
        stageRecords(end+1, 1) = rec; %#ok<AGROW>
        progress.failed_stages{end+1} = stage.tag;
        if localIsInterruptCondition(cond)
            progress.status = 'interrupted';
        else
            progress.status = 'failed';
        end
        progress.current_stage = stage.tag;
        localWriteProgress(progressPath, progress);
        localWriteCampaignSummary(summaryPath, stages, stageRecords, progress);
        localPrintFailureReport({f});
        error('run_all:MandatoryExperimentFailed', ...
            'Mandatory stage %s failed: %s', stage.tag, cond);
    end
end

progress.status = 'completed';
progress.current_stage = '';
progress.last_update_utc = localUtcNow();
progress.elapsed_seconds = localElapsedSince(progress.start_time_utc);
localWriteProgress(progressPath, progress);

fprintf('\n====================================================================\n');
fprintf(' EXPERIMENT SUMMARY\n');
fprintf('====================================================================\n');
localPrintStageSummary(stageRecords);

localWriteManifest(od, outRoot, allResults);
localWriteCampaignSummary(summaryPath, stages, stageRecords, progress);

save(fullfile(outRoot, 'all_revision_results.mat'), 'allResults');
fprintf('\nAll selected mandatory stages PASSED or were resume-skipped.\n');
fprintf('Progress: %s\n', progressPath);
fprintf('Summary : %s\n\n', summaryPath);
end

% =========================================================================
%  SMOKE VERIFICATION
% =========================================================================

function localRunSmokeVerification(scriptDir, outRoot) %#ok<DEFNU>
fprintf('====================================================================\n');
fprintf(' GATE I1 VERIFICATION: INTENTIONALLY FAILING SMOKE EXPERIMENT\n');
fprintf('====================================================================\n');
fprintf(' Purpose : verify that the fail-loud runner detects a capped run\n');
fprintf('           and reports the exact failed acceptance condition.\n');
fprintf(' Expected: runner ends with error() identifying the condition.\n');
fprintf('====================================================================\n\n');

smokeDir = fullfile(outRoot, 'smoke');

% Smoke mode always clears and recreates its output directory
if exist(smokeDir, 'dir') == 7
    files = [dir(fullfile(smokeDir,'*.mat')); ...
             dir(fullfile(smokeDir,'*.csv')); ...
             dir(fullfile(smokeDir,'*.png'))];
    for k = 1:numel(files)
        if ~files(k).isdir, delete(fullfile(smokeDir, files(k).name)); end
    end
else
    mkdir(smokeDir);
end

[~, pass, condition, trace, elapsed] = localRunAndAccept( ...
    'EXP_SMOKE', 'Intentionally failing smoke experiment', ...
    @() exp_smoke_fail(smokeDir), ...
    @(r) localAccept_Smoke(r, smokeDir));

fprintf('\n====================================================================\n');
fprintf(' GATE I1 RESULT\n');
fprintf('====================================================================\n\n');

if ~pass
    % Correct behaviour: the smoke experiment was rejected as expected
    fprintf('  Gate I1 PASSED -- master runner correctly identified failure.\n\n');
    fprintf('  Experiment: EXP_SMOKE\n');
    fprintf('  Elapsed   : %.2fs\n', elapsed);
    fprintf('  Detected condition:\n    %s\n', condition);
    if ~isempty(trace)
        fprintf('  Stack trace:\n');
        for j = 1:numel(trace)
            fprintf('    %2d.  %-40s  line %d\n', j, trace(j).name, trace(j).line);
        end
    end
    fprintf('\n');
    fprintf('  Fail-loud infrastructure is working correctly.\n');
    fprintf('  Use "fast" or "full" mode to run the real experiments.\n\n');
    error('run_all:GateI1Confirmed', ...
        'Gate I1 confirmed: master runner fails loud on mandatory experiment failure.\nDetected condition: %s', ...
        condition);
else
    % Wrong: smoke experiment unexpectedly passed
    error('run_all:GateI1NotTriggered', ...
        'Gate I1 NOT CONFIRMED: smoke experiment returned pass=true.\nCheck exp_smoke_fail.m -- it should always produce a failing result.');
end
end

% =========================================================================
%  CORE RUNNER
% =========================================================================

function [res, pass, condition, trace, elapsed] = localRunAndAccept(tag, desc, expFn, acceptFn)
%LOCALRUNANDACCEPT  Run one experiment and apply its acceptance check.
%
%   Returns res=[] on exception (full trace preserved in trace).
%   Acceptance failures populate condition; trace is empty for those.

res = []; pass = false; condition = '(not run)'; trace = []; elapsed = NaN;

fprintf('\n');
fprintf('---- %s: %s\n', tag, desc);

t0 = tic;
try
    res     = expFn();
    elapsed = toc(t0);
catch ME
    elapsed   = toc(t0);
    trace     = ME.stack;
    condition = sprintf('exception: [%s] %s', ME.identifier, ME.message);
    fprintf('[%s] EXCEPTION after %.1fs\n', tag, elapsed);
    fprintf('  Message   : %s\n', ME.message);
    fprintf('  Identifier: %s\n', ME.identifier);
    if ~isempty(ME.stack)
        fprintf('  Stack trace:\n');
        for j = 1:numel(ME.stack)
            fprintf('    %2d.  %-40s  line %d\n', ...
                j, ME.stack(j).name, ME.stack(j).line);
            fprintf('         %s\n', ME.stack(j).file);
        end
    end
    return;
end

fprintf('[%s] Ran in %.1fs. Checking acceptance...\n', tag, elapsed);

try
    [pass, condition] = acceptFn(res);
catch ME2
    pass      = false;
    trace     = ME2.stack;
    condition = sprintf('acceptance-check exception: [%s] %s', ...
        ME2.identifier, ME2.message);
    fprintf('[%s] Acceptance check threw: %s\n', tag, ME2.message);
    return;
end

if pass
    fprintf('[%s] ACCEPTED.\n', tag);
else
    fprintf('[%s] REJECTED: %s\n', tag, condition);
end
end

% =========================================================================
%  CAMPAIGN CONTROL, RESUME, AND PROGRESS
% =========================================================================

function opts = localParseRunnerOptions(mode, varargin)
opts = struct();
opts.resume = false;
opts.dryRun = false;
opts.force = false;
opts.stageName = '';

args = varargin;
if strcmp(mode, 'stage')
    if isempty(args)
        error('run_all:MissingStage', ...
            'Stage mode requires a stage name, e.g. run_all_revision_experiments(''stage'', ''Exp2'').');
    end
    opts.stageName = char(args{1});
    args = args(2:end);
end

if mod(numel(args), 2) ~= 0
    error('run_all:BadOptions', 'Options must be name/value pairs.');
end

for k = 1:2:numel(args)
    name = lower(strtrim(char(args{k})));
    value = args{k+1};
    switch name
        case 'resume'
            opts.resume = logical(value);
        case 'dry_run'
            opts.dryRun = logical(value);
        case 'force'
            opts.force = logical(value);
        case 'stage'
            opts.stageName = char(value);
        otherwise
            error('run_all:BadOptions', 'Unknown option "%s".', name);
    end
end

if opts.force && opts.dryRun
    fprintf('[DRY RUN] force=true: stages are reported as would-run, but not executed.\n');
end
end

function cfg = localCampaignConfig(mode, nSamples, meshTable, alphaVals)
cfg = struct();
cfg.runner = 'run_all_revision_experiments';
cfg.runner_version = 3;
cfg.mode = mode;
cfg.nSamples = nSamples;
cfg.meshTable = meshTable;
cfg.alphaVals = alphaVals;
cfg.created_utc = localUtcNow();
end

function stages = localBuildStages(od, nSamples, meshTable, alphaVals)
%LOCALBUILDSTAGES  Authoritative stage registry.
%
%   Execution order is the registry order.  S1 is FIRST because the localized
%   low-density mode mitigation is unresolved: EXP2b and EXP3 fail the MAC gate
%   at fine meshes until S1 is settled.
%
%   Thin adapters below resolve signature differences.  The numerical experiment
%   implementations are NOT modified to satisfy this interface.
%
%   ALPHA NOTE: exp2_authoritative_sweep fixes its own alpha list internally
%   ([1 .75 .5 .25 0]); the campaign alphaVals do NOT control it.  The stage
%   config records this honestly so the resume config-hash does not imply a
%   control that does not exist.

stages = repmat(localEmptyStage(), 0, 1);

% ---- S1: localized-mode mitigation (gates EXP2b and EXP3) ----------------
stages(end+1, 1) = localMakeStage( ...
    's1', 'S1', 'Localized low-density mode mitigation (gates EXP2b/EXP3)', od.s1, ...
    @() s1_mitigation_400x50_pilot(od.s1), ...              % adapter: (outDir)
    @(r) localAccept_S1(r, od.s1), ...
    {fullfile(od.s1, 's1_mitigation_400x50_result.mat'), ...
     fullfile(od.s1, 's1_mitigation_400x50_manifest.json')}, ...
    struct('mesh', '400x50', 'alpha', 1.0, 'mitigation', 'mass_interpolation'), ...
    struct('dependsOn', {{}}, 'estRuntimeSeconds', 10800));

% ---- EXP2: authoritative clamped-beam alpha sweep ------------------------
stages(end+1, 1) = localMakeStage( ...
    'exp2', 'EXP2', 'Clamped beam authoritative alpha sweep (F = omega0^2*M(x)*Phi0)', od.exp2, ...
    @() exp2_authoritative_sweep(od.exp2), ...              % adapter: drops alphaVals
    @(r) localAccept_Exp2Authoritative(r, od.exp2), ...
    {fullfile(od.exp2, 'exp2_authoritative_sweep_result.mat'), ...
     fullfile(od.exp2, 'exp2_authoritative_sweep_manifest.json')}, ...
    struct('alpha_source', 'script-internal fixed [1 0.75 0.5 0.25 0]', ...
           'mesh', '200x25', 'load', 'omega0^2*M(x)*Phi0'), ...
    struct('dependsOn', {{'S1'}}, 'estRuntimeSeconds', 10800));

% ---- EXP2b: building benchmark ------------------------------------------
stages(end+1, 1) = localMakeStage( ...
    'exp2b', 'EXP2b', 'Building benchmark', od.exp2b, ...
    @() exp2b_building(alphaVals, od.exp2b), ...
    @(r) localAccept_Exp2b(r, od.exp2b), ...
    {fullfile(od.exp2b, 'exp2b_building_results.mat')}, ...
    struct('alphaVals', alphaVals), ...
    struct('dependsOn', {{'S1'}}, 'estRuntimeSeconds', 7200));

% ---- EXP3: authoritative mesh convergence -------------------------------
stages(end+1, 1) = localMakeStage( ...
    'exp3', 'EXP3', 'Mesh convergence 200x25 vs 400x50 (authoritative)', od.exp3, ...
    @() exp3_authoritative_mesh_convergence(od.exp3), ...   % adapter: drops alphaVals
    @(r) localAccept_Exp3Authoritative(r, od.exp3), ...
    {fullfile(od.exp3, 'exp3_authoritative_mesh_convergence_result.mat'), ...
     fullfile(od.exp3, 'exp3_authoritative_mesh_convergence_manifest.json')}, ...
    struct('alpha', 1.0, 'meshes', '200x25,400x50', 'load', 'omega0^2*M(x)*Phi0'), ...
    struct('dependsOn', {{'S1', 'EXP2'}}, 'estRuntimeSeconds', 21600));

% ---- A4: eigenpair-refresh study (IMPLEMENTED per A4_SPECIFICATION_V3) ---
% Single independent variable: the refresh interval N = {inf,50,10,5,1}, taken
% over ONE base config (a4_ss_400x50_base.json).  Gate A4-Pre runs INSIDE this
% stage and aborts with run_all:A4SpectrumInadmissible (naming S1) if the SS
% beam's intermediate spectra are dominated by disconnected-island modes.
% exp4_sensitivity_ablation is a pre-authoritative sensitivity ablation and is
% NOT an A4 implementation; it remains denied by preflight P2.
stages(end+1, 1) = localMakeStage( ...
    'a4', 'A4', 'Recovery Phase 2: adaptive search + common diagnostics', od.a4, ...
    @() a4_eigenpair_refresh(od.a4), ...
    @(r) localAccept_A4(r, od.a4), ...
    {fullfile(od.a4, 'a4_eigenpair_refresh_results.mat'), ...
     fullfile(od.a4, 'a4_result.json'), ...
     fullfile(od.a4, 'a4_manifest.json'), ...
     fullfile(od.a4, 'a4_stage_manifest.json'), ...
     fullfile(od.a4, 'a4_screening_events.json'), ...
     fullfile(od.a4, 'a4_candidate_telemetry.csv'), ...
     fullfile(od.a4, 'a4_iteration_histories.csv'), ...
     fullfile(od.a4, 'a4_table.md'), fullfile(od.a4, 'a4_table2.md')}, ...
    struct('nLevels', [Inf 50 10 5 1], 'baseConfig', 'a4_ss_400x50_base.json', ...
           'pmass', 1, 'baseline', 'solid'), ...
    struct('dependsOn', {{}}, 'estRuntimeSeconds', 43200));

end

function st = localEmptyStage()
st = struct( ...
    'key', '', ...
    'tag', '', ...
    'desc', '', ...
    'outDir', '', ...
    'runFn', [], ...
    'acceptFn', [], ...
    'requiredArtifacts', {{}}, ...
    'resultJson', '', ...
    'manifestJson', '', ...
    'expectedDiagnostic', false, ...
    'config', struct(), ...
    'implemented', true, ...
    'notImplementedReason', '', ...
    'dependsOn', {{}}, ...
    'estRuntimeSeconds', NaN);
end

function st = localMakeStage(key, tag, desc, outDir, runFn, acceptFn, requiredArtifacts, cfg, meta)
if nargin < 9 || isempty(meta), meta = struct(); end
st = localEmptyStage();
st.key = key;
st.tag = tag;
st.desc = desc;
st.outDir = outDir;
st.runFn = runFn;
st.acceptFn = acceptFn;
st.requiredArtifacts = requiredArtifacts;
st.resultJson = fullfile(outDir, sprintf('%s_stage_result.json', key));
st.manifestJson = fullfile(outDir, sprintf('%s_stage_manifest.json', key));
st.expectedDiagnostic = false;
st.config = cfg;
st.implemented = true;
st.notImplementedReason = '';
if isfield(meta, 'dependsOn'), st.dependsOn = meta.dependsOn; end
if isfield(meta, 'estRuntimeSeconds'), st.estRuntimeSeconds = meta.estRuntimeSeconds; end
end

function st = localMakePlaceholderStage(key, tag, desc, outDir, reason, meta)
%LOCALMAKEPLACEHOLDERSTAGE  A mandatory stage that has no implementation yet.
%
%   The stage is registered so that it is VISIBLE in the registry, the dry run,
%   the progress file, and the campaign summary.  It can never be silently
%   skipped: preflight refuses to start a full/fast campaign while it is absent,
%   and if it is ever dispatched its acceptance check fails loud.
if nargin < 6 || isempty(meta), meta = struct(); end
st = localEmptyStage();
st.key = key;
st.tag = tag;
st.desc = desc;
st.outDir = outDir;
st.runFn = @() localRaiseNotImplemented(reason);
st.acceptFn = @(r) localAccept_NotImplemented(r, reason);
st.requiredArtifacts = {};
st.resultJson = fullfile(outDir, sprintf('%s_stage_result.json', key));
st.manifestJson = fullfile(outDir, sprintf('%s_stage_manifest.json', key));
st.expectedDiagnostic = false;
st.config = struct('state', 'A4_NOT_IMPLEMENTED');
st.implemented = false;
st.notImplementedReason = reason;
if isfield(meta, 'dependsOn'), st.dependsOn = meta.dependsOn; end
if isfield(meta, 'estRuntimeSeconds'), st.estRuntimeSeconds = meta.estRuntimeSeconds; end
end

function stages = localSelectStage(allStages, stageName)
needle = lower(strrep(strtrim(char(stageName)), '_', ''));
tags = lower(strrep({allStages.tag}, '_', ''));
keys = lower(strrep({allStages.key}, '_', ''));
idx = find(strcmp(needle, tags) | strcmp(needle, keys), 1);
if isempty(idx)
    error('run_all:BadStage', ...
        'Unknown stage "%s". Valid stages are: %s.', ...
        stageName, strjoin({allStages.tag}, ', '));
end
stages = allStages(idx);
end

function progress = localInitProgress(progressPath, outRoot, mode, opts, stages)
progress = struct();
if opts.resume && isfile(progressPath)
    oldProgress = localReadJsonSafe(progressPath);
    if isstruct(oldProgress)
        progress = oldProgress;
    end
end

if ~isfield(progress, 'campaign_id') || isempty(progress.campaign_id) || opts.force
    progress.campaign_id = sprintf('r1_%s_%s', mode, datestr(now, 'yyyymmddTHHMMSSFFF')); %#ok<TNOW1,DATST>
end
if ~isfield(progress, 'start_time_utc') || isempty(progress.start_time_utc) || opts.force
    progress.start_time_utc = localUtcNow();
end

progress.last_update_utc = localUtcNow();
progress.mode = mode;
progress.current_stage = '';
progress.status = 'running';
progress.elapsed_seconds = localElapsedSince(progress.start_time_utc);
progress.progress_file = progressPath;
progress.output_root = outRoot;
progress.resume = opts.resume;
progress.dry_run = opts.dryRun;
progress.force = opts.force;

progress.completed_stages = {};
progress.failed_stages = {};
progress.skipped_stages = {};
progress.per_stage_elapsed = struct();
if ~isfield(progress, 'output_directories') || ~isstruct(progress.output_directories)
    progress.output_directories = struct();
end
for k = 1:numel(stages)
    progress.output_directories.(stages(k).key) = stages(k).outDir;
end
end

function rows = localDryRun(stages, campaignCfg, opts)
fprintf('\n====================================================================\n');
fprintf(' DRY RUN -- no experiments will be executed\n');
fprintf('====================================================================\n\n');
fprintf('  %-7s  %-12s  %-18s  %s\n', 'Stage', 'Action', 'Validation', 'Major outputs');
fprintf('  %s\n', repmat('-', 1, 110));

rows = repmat(localBlankStageRecord(), 0, 1);
for k = 1:numel(stages)
    st = stages(k);
    [valid, msg] = localValidateStageArtifacts(st, campaignCfg);
    if opts.force
        action = 'would run';
        validation = 'bypassed by force';
    elseif valid
        action = 'would skip';
        validation = 'valid';
    else
        action = 'would run';
        validation = ['invalid: ' msg];
    end
    fprintf('  %-7s  %-12s  %-18s  %s\n', ...
        st.tag, action, localClip(validation, 18), strjoin(st.requiredArtifacts, ', '));
    rows(end+1, 1) = localStageRecord(st, action, 0, validation, st.outDir); %#ok<AGROW>
end
fprintf('\n');
end

function [valid, msg] = localValidateStageArtifacts(stage, campaignCfg) %#ok<INUSD>
valid = false;
msg = '';

% A not-implemented mandatory stage is NEVER resumable and never skippable.
if ~stage.implemented
    msg = sprintf('%s_NOT_IMPLEMENTED: %s', stage.tag, stage.notImplementedReason);
    return;
end

if ~localCheckArtifact(stage.resultJson)
    msg = sprintf('missing result JSON: %s', stage.resultJson);
    return;
end
if ~localCheckArtifact(stage.manifestJson)
    msg = sprintf('missing manifest: %s', stage.manifestJson);
    return;
end

result = localReadJsonSafe(stage.resultJson);
if ~isstruct(result)
    msg = sprintf('result JSON could not be decoded: %s', stage.resultJson);
    return;
end

status = '';
if isfield(result, 'status'), status = char(result.status); end
okStatus = strcmp(status, 'accepted') || (stage.expectedDiagnostic && strcmp(status, 'diagnostic'));
if ~okStatus
    msg = sprintf('result status "%s" is not resumable', status);
    return;
end

expectedHash = localHashStruct(stage.config);
if isfield(result, 'config_hash') && ~isempty(result.config_hash)
    if ~strcmp(char(result.config_hash), expectedHash)
        msg = sprintf('config hash mismatch: have %s, expected %s', ...
            char(result.config_hash), expectedHash);
        return;
    end
end

for k = 1:numel(stage.requiredArtifacts)
    if ~localCheckArtifact(stage.requiredArtifacts{k})
        msg = sprintf('missing required artifact: %s', stage.requiredArtifacts{k});
        return;
    end
end

manifest = localReadJsonSafe(stage.manifestJson);
if isstruct(manifest) && isfield(manifest, 'required_artifacts')
    req = manifest.required_artifacts;
    if ischar(req), req = {req}; end
    if isstring(req), req = cellstr(req); end
    if iscell(req)
        for k = 1:numel(req)
            if ~localCheckArtifact(req{k})
                msg = sprintf('manifest-listed artifact missing: %s', req{k});
                return;
            end
        end
    end
end

valid = true;
msg = 'valid';
end

function localWriteStageMetadata(stage, campaignCfg, elapsed, status, condition) %#ok<INUSD>
if ~exist(stage.outDir, 'dir'), mkdir(stage.outDir); end

result = struct();
result.stage = stage.tag;
result.key = stage.key;
result.description = stage.desc;
result.status = status;
result.condition = condition;
result.expected_diagnostic = stage.expectedDiagnostic;
result.elapsed_seconds = elapsed;
result.completed_utc = localUtcNow();
result.output_dir = stage.outDir;
result.config_hash = localHashStruct(stage.config);
result.config = stage.config;
result.required_artifacts = stage.requiredArtifacts;
result.result_json = stage.resultJson;
result.manifest = stage.manifestJson;
localWriteJsonAtomic(stage.resultJson, result);

manifest = struct();
if strcmp(stage.key,'a4') && isfile(fullfile(stage.outDir,'a4_manifest.json'))
    % Phase-2 §10.2: campaign and stage manifests must carry the identical
    % artifact set. Preserve the driver's authoritative set when the generic
    % stage wrapper adds its status metadata.
    manifest=localReadJsonSafe(fullfile(stage.outDir,'a4_manifest.json'));
    manifest.status=status;
    manifest.required_artifacts=stage.requiredArtifacts;
else
    manifest.stage = stage.tag;
    manifest.status = status;
    manifest.created_utc = localUtcNow();
    manifest.output_dir = stage.outDir;
    manifest.result_json = stage.resultJson;
    manifest.manifest_json = stage.manifestJson;
    manifest.required_artifacts = stage.requiredArtifacts;
    manifest.files = localListOutputFiles(stage.outDir);
end
localWriteJsonAtomic(stage.manifestJson, manifest);
end

function rec = localBlankStageRecord()
rec = struct( ...
    'stage', '', ...
    'key', '', ...
    'description', '', ...
    'status', '', ...
    'elapsed_seconds', 0, ...
    'message', '', ...
    'output_dir', '', ...
    'artifacts', {{}});
end

function rec = localStageRecord(stage, status, elapsed, message, outDir)
rec = localBlankStageRecord();
rec.stage = stage.tag;
rec.key = stage.key;
rec.description = stage.desc;
rec.status = status;
rec.elapsed_seconds = elapsed;
rec.message = message;
rec.output_dir = outDir;
rec.artifacts = stage.requiredArtifacts;
end

function localWriteProgress(progressPath, progress)
progress.last_update_utc = localUtcNow();
localWriteJsonAtomic(progressPath, progress);
end

function localWriteJsonAtomic(path, data)
txt = localJsonEncode(data);
localAtomicWriteText(path, txt);
end

function localAtomicWriteText(path, txt)
parent = fileparts(path);
if ~exist(parent, 'dir'), mkdir(parent); end
tmp = tempname(parent);
fid = fopen(tmp, 'w');
if fid < 0
    error('run_all:WriteFailed', 'Could not open temporary file for writing: %s', tmp);
end
clean = onCleanup(@() localDeleteIfExists(tmp));
fprintf(fid, '%s', txt);
fclose(fid);
movefile(tmp, path, 'f');
delete(clean);
end

function localDeleteIfExists(path)
if isfile(path)
    try, delete(path); catch, end
end
end

function txt = localJsonEncode(data)
try
    txt = jsonencode(data, PrettyPrint=true);
catch
    txt = jsonencode(data);
end
txt = [txt newline];
end

function data = localReadJsonSafe(path)
data = [];
try
    data = jsondecode(fileread(path));
catch
    data = [];
end
end

function t = localUtcNow()
t = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z'''));
end

function elapsedSeconds = localElapsedSince(startUtc)
elapsedSeconds = 0;
try
    t0 = datetime(startUtc, 'InputFormat', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''');
    t0.TimeZone = 'UTC';
    elapsedSeconds = seconds(datetime('now', 'TimeZone', 'UTC') - t0);
catch
    elapsedSeconds = NaN;
end
end

function h = localHashStruct(s)
h = fnv1a32_canonical_struct(s);
end

function files = localListOutputFiles(outDir)
files = {};
if exist(outDir, 'dir') ~= 7
    return;
end
listing = dir(outDir);
listing = listing(~[listing.isdir]);
for k = 1:numel(listing)
    files{end+1, 1} = fullfile(outDir, listing(k).name); %#ok<AGROW>
end
end

function localWriteCampaignSummary(summaryPath, stages, records, progress)
lines = {};
lines{end+1} = '# Revision R1 Campaign Summary';
lines{end+1} = '';
lines{end+1} = sprintf('- Campaign ID: `%s`', localStringField(progress, 'campaign_id'));
lines{end+1} = sprintf('- Mode: `%s`', localStringField(progress, 'mode'));
lines{end+1} = sprintf('- Status: `%s`', localStringField(progress, 'status'));
lines{end+1} = sprintf('- Started UTC: `%s`', localStringField(progress, 'start_time_utc'));
lines{end+1} = sprintf('- Last update UTC: `%s`', localStringField(progress, 'last_update_utc'));
lines{end+1} = sprintf('- Elapsed seconds: `%.2f`', localNumericField(progress, 'elapsed_seconds'));
lines{end+1} = '';
lines{end+1} = '| Stage | State | Elapsed s | Output directory | Message |';
lines{end+1} = '|---|---:|---:|---|---|';

for k = 1:numel(stages)
    idx = find(strcmp({records.stage}, stages(k).tag), 1, 'last');
    if isempty(idx)
        rec = localStageRecord(stages(k), 'not run', 0, '', stages(k).outDir);
    else
        rec = records(idx);
    end
    lines{end+1} = sprintf('| %s | %s | %.2f | `%s` | %s |', ...
        rec.stage, rec.status, rec.elapsed_seconds, rec.output_dir, localMdEscape(rec.message));
end

lines{end+1} = '';
lines{end+1} = '## Artifacts';
lines{end+1} = '';
for k = 1:numel(stages)
    lines{end+1} = sprintf('### %s', stages(k).tag);
    lines{end+1} = sprintf('- Output directory: `%s`', stages(k).outDir);
    lines{end+1} = sprintf('- Result JSON: `%s`', stages(k).resultJson);
    lines{end+1} = sprintf('- Manifest: `%s`', stages(k).manifestJson);
    for j = 1:numel(stages(k).requiredArtifacts)
        lines{end+1} = sprintf('- Required artifact: `%s`', stages(k).requiredArtifacts{j});
    end
    lines{end+1} = '';
end

localAtomicWriteText(summaryPath, strjoin(lines, newline));
end

function s = localStringField(st, fieldName)
s = '';
if isstruct(st) && isfield(st, fieldName) && ~isempty(st.(fieldName))
    s = char(st.(fieldName));
end
end

function x = localNumericField(st, fieldName)
x = NaN;
if isstruct(st) && isfield(st, fieldName) && isnumeric(st.(fieldName)) && ~isempty(st.(fieldName))
    x = st.(fieldName);
end
end

function s = localMdEscape(s)
if isempty(s), s = ''; return; end
s = char(s);
s = strrep(s, '|', '\|');
s = strrep(s, newline, ' ');
end

function s = localClip(s, n)
s = char(s);
if numel(s) > n
    s = [s(1:max(1,n-3)) '...'];
end
end

function tf = localIsInterruptCondition(condition)
tf = contains(lower(condition), 'interrupt') || contains(lower(condition), 'ctrl-c');
end

function localPrintFailureReport(failures)
fprintf('\n====================================================================\n');
fprintf(' MANDATORY STAGE FAILURE\n');
fprintf('====================================================================\n\n');
for k = 1:numel(failures)
    f = failures{k};
    fprintf('  %s failed after %s\n', f.tag, f.elapsedStr);
    fprintf('  Condition: %s\n', f.condition);
    if ~isempty(f.trace)
        fprintf('  Stack trace:\n');
        for j = 1:numel(f.trace)
            fprintf('    %2d.  %-40s  line %d\n', j, f.trace(j).name, f.trace(j).line);
            if isfield(f.trace(j), 'file')
                fprintf('         %s\n', f.trace(j).file);
            end
        end
    end
end
fprintf('\n');
end

function localPrintStageSummary(records)
fprintf('\n  %-8s  %-12s  %-10s  %s\n', 'Stage', 'Status', 'Elapsed', 'Output');
fprintf('  %s\n', repmat('-', 1, 80));
for k = 1:numel(records)
    fprintf('  %-8s  %-12s  %9.1fs  %s\n', ...
        records(k).stage, records(k).status, records(k).elapsed_seconds, records(k).output_dir);
end
fprintf('\n');
end

% =========================================================================
%  ACCEPTANCE FUNCTIONS (one per experiment)
% =========================================================================

function [pass, condition] = localAccept_Smoke(res, outDir)
%LOCALACCEPT_SMOKE  Schema-based check for the smoke experiment.
%
%   Uses check_experiment_result from scripts/revision_v1/.
%   Returns pass=false with a specific condition for every failure mode
%   the smoke test is designed to trigger.

pass = false; condition = '';

% Structural schema check
if ~isstruct(res)
    condition = 'returned non-struct result'; return;
end
[schOk, schIssues] = check_experiment_result(res);
if ~schOk
    condition = sprintf('schema invalid: %s', strjoin(schIssues, '; '));
    return;
end

% Required mat artifact
if ~localCheckArtifact(res.artifacts.mat_file)
    condition = sprintf('required artifact missing: %s', res.artifacts.mat_file);
    return;
end

% Detect termination condition
if ~res.success
    if res.termination.capped
        condition = sprintf( ...
            'reached iteration cap: %d/%d iterations without convergence, design change = %.2e', ...
            res.iterations.count, res.iterations.cap, ...
            res.convergence.final_design_change);
        return;
    end
    if res.termination.mode_lost
        condition = sprintf( ...
            'mode tracking lost: MAC = %.3f at final iteration (below threshold)', ...
            res.mode_tracking.mac_history(end));
        return;
    end
    if res.termination.exception
        condition = sprintf('exception during experiment: %s', res.termination.message);
        return;
    end
    condition = sprintf('experiment failed: reason=%s; %s', ...
        res.termination.reason, res.termination.message);
    return;
end

% Schema-valid and successful would be unexpected for the smoke test,
% but we still return pass=false to flag it as wrong
condition = 'smoke experiment unexpectedly returned success=true — check exp_smoke_fail.m';
end

% -------------------------------------------------------------------------

% =========================================================================
%  PREFLIGHT (fail-loud, before any computation)
% =========================================================================

function localPreflight(stages, opts, mode)
%LOCALPREFLIGHT  Refuse to start a campaign that cannot produce valid evidence.
%
%   P1  no active MATLAB path entry may reference OlhoffApproachExact
%   P2  no stage may invoke an archived or pre-authoritative runner
%   P3  no mandatory placeholder stage may be silently skipped
%   P4  output conflicts are reported for ALL stages before anything runs
%
%   In dry_run mode the findings are reported and the run continues; in
%   fast/full/stage mode any finding is fatal.

fprintf('---- PREFLIGHT\n');
problems = {};

% -- P1: archived reconstruction tree must not be on the path -------------
pathEntries = strsplit(path, pathsep);
badPath = pathEntries(contains(pathEntries, 'OlhoffApproachExact'));
if ~isempty(badPath)
    problems{end+1} = sprintf( ...
        'P1 path violation: OlhoffApproachExact is on the MATLAB path (%d entr(y/ies), e.g. %s). It is an archived diagnostic reconstruction and must not be reachable from a production run.', ...
        numel(badPath), badPath{1}); %#ok<AGROW>
end

% -- P2: no stage may dispatch to a retired / pre-authoritative runner -----
% Retired runners: pre-authoritative scripts, the archived Exact reconstruction,
% and the EXP1/EXP5 performance-benchmark pair (obsolete as reviewer evidence --
% the local comparators are not faithful reference implementations; see
% SCIENTIFIC_DECISION_EXP1_EXP5.md).
denied = {'exp2_clamped_beam', 'exp3_mesh_convergence', 'exp4_sensitivity_ablation', ...
          'exp1_perf_table', 'exp5_scaling', ...
          'pilot_olhoff_exact', 'phase1_olhoff_exact', 'phase2_olhoff_exact', ...
          'phase3_olhoff_exact', 'phase4_olhoff_exact', ...
          ... % A4_SPECIFICATION_V3 §7.4: the retired EXP4 configs must never
          ... % re-enter A4. Sweeping N over these was the original error --
          ... % they vary the load model, optimizer, sensitivity AND mesh.
          'ss_beam_harmonic_frozen', 'ss_beam_harmonic_periodic', ...
          'ablation_harmonic_frozen_solid', 'ablation_harmonic_periodic_solid', ...
          'ablation_semi_harmonic'};
for k = 1:numel(stages)
    if ~stages(k).implemented, continue; end
    fnText = func2str(stages(k).runFn);
    for d = 1:numel(denied)
        if contains(fnText, denied{d})
            problems{end+1} = sprintf( ...
                'P2 registry violation: stage %s dispatches to retired/pre-authoritative runner "%s".', ...
                stages(k).tag, denied{d}); %#ok<AGROW>
        end
    end
end
for k = 1:numel(stages)
    if ~stages(k).implemented, continue; end
    fnText = func2str(stages(k).runFn);
    tok = regexp(fnText, '([A-Za-z_]\w*)\s*\(', 'tokens');
    for t = 1:numel(tok)
        name = tok{t}{1};
        w = which(name);
        if ~isempty(w) && contains(w, [filesep 'archive' filesep])
            problems{end+1} = sprintf( ...
                'P2 registry violation: stage %s resolves "%s" to an ARCHIVED file: %s', ...
                stages(k).tag, name, w); %#ok<AGROW>
        end
    end
end

% -- P3: mandatory placeholder stages block a campaign --------------------
notImpl = stages(~[stages.implemented]);
if ~isempty(notImpl)
    for k = 1:numel(notImpl)
        fprintf('  [%s] state: %s_NOT_IMPLEMENTED\n', notImpl(k).tag, notImpl(k).tag);
    end
    if any(strcmp(mode, {'full', 'fast'}))
        names = strjoin({notImpl.tag}, ', ');
        problems{end+1} = sprintf( ...
            ['P3 mandatory stage not implemented: %s. A %s campaign cannot produce a ' ...
             'complete evidence chain and will not be started. Implement the stage, or ' ...
             'run individual stages with run_all_revision_experiments(''stage'', ''EXP2'').'], ...
            names, mode); %#ok<AGROW>
    end
end

% -- P4: output conflicts, reported up-front for every stage --------------
if ~opts.resume && ~opts.force
    conflicts = {};
    for k = 1:numel(stages)
        d = stages(k).outDir;
        if exist(d, 'dir') ~= 7, continue; end
        listing = [dir(fullfile(d,'*.mat')); dir(fullfile(d,'*.csv')); dir(fullfile(d,'*.png'))];
        listing = listing(~[listing.isdir]);
        if ~isempty(listing)
            conflicts{end+1} = sprintf('%s (%d file(s) in %s)', ...
                stages(k).tag, numel(listing), d); %#ok<AGROW>
        end
    end
    if ~isempty(conflicts)
        problems{end+1} = sprintf( ...
            ['P4 output conflict, detected BEFORE any computation: %s. ' ...
             'Archive or remove these directories, or pass force=true / resume=true.'], ...
            strjoin(conflicts, '; ')); %#ok<AGROW>
    end
end

if isempty(problems)
    fprintf('  All preflight checks passed (P1 path, P2 registry, P3 placeholders, P4 outputs).\n\n');
    return;
end

fprintf('\n  PREFLIGHT FAILED -- %d problem(s):\n', numel(problems));
for k = 1:numel(problems)
    fprintf('   %2d. %s\n', k, problems{k});
end
fprintf('\n');

if opts.dryRun
    fprintf('  [DRY RUN] Continuing so the full stage table can be reported.\n\n');
    return;
end

error('run_all:PreflightFailed', ...
    'Preflight failed with %d problem(s):\n  - %s', ...
    numel(problems), strjoin(problems, sprintf('\n  - ')));
end

% -------------------------------------------------------------------------

% =========================================================================
%  ACCEPTANCE GATES
% =========================================================================

function res = localRaiseNotImplemented(reason)
%LOCALRAISENOTIMPLEMENTED  Fail loud when a mandatory placeholder stage is
%   dispatched (only reachable via stage mode; preflight blocks full/fast).
%
%   Declared WITH an output argument on purpose.  localRunAndAccept calls
%   res = stage.runFn(), so an anonymous handle wrapping a bare error() raises
%   MATLAB:maxlhs ("Too many output arguments") and masks the real reason.
%   Routing through a named 1-output function makes the intended
%   run_all:A4NotImplemented identifier surface instead.
res = []; %#ok<NASGU>
error('run_all:A4NotImplemented', '%s', reason);
end

function [pass, condition] = localAccept_NotImplemented(~, reason)
%LOCALACCEPT_NOTIMPLEMENTED  A placeholder stage can never be accepted.
pass = false;
condition = sprintf('stage not implemented: %s', reason);
end

% -------------------------------------------------------------------------

function [pass, condition] = localAccept_A4(res, outDir)
%LOCALACCEPT_A4  Recovery Phase-2 run gate (§8.5).
%
%   *** A4's GATE IS DELIBERATELY DIFFERENT FROM EVERY OTHER STAGE. ***
%
%   The campaign rule "a capped run is a failure, not a result" is correct for
%   evidence runs.  It is WRONG for A4, whose PURPOSE is to characterize failure.
%   This gate therefore does NOT reject on MAC < 0.8 and does NOT reject on the
%   iteration cap: both are RESULTS (spec §5.1, §7.4).
%
%   A4 is gated on INTEGRITY OF MEASUREMENT, not on success of optimization:
%     - every declared N level attempted,
%     - every arm classified by check_a4_run (the single implementation),
%     - a pre-registered decision emitted,
%     - all required artifacts present.
%
%   An arm classified ACCEPTED_WITH_BREAKDOWN (B1..B4) is ACCEPTED evidence.
%   An arm classified REJECTED means the MACHINERY broke -> the stage fails.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end
for fn = {'arms', 'run_verdict', 'base_config_hash', 'n_levels'}
    if ~isfield(res, fn{1}) || isempty(res.(fn{1}))
        condition = sprintf('required field missing or empty: %s', fn{1}); return;
    end
end

% Every declared N level must have been attempted.
if numel(res.arms) ~= numel(res.n_levels)
    condition = sprintf('expected %d arms (one per N level), got %d', ...
        numel(res.n_levels), numel(res.arms));
    return;
end

% Single-factor integrity: all arms share the base-config hash (V-A4-2).
hashes = unique({res.arms.base_config_hash});
if numel(hashes) ~= 1 || ~strcmp(hashes{1}, res.base_config_hash)
    condition = 'factor drift: arms do not share one base-config hash';
    return;
end

% Preconditions (spec §7.4): pmass = 1 and baseline = solid.
if any(abs([res.arms.pmass] - 1) > 0)
    condition = 'A4 precondition violated: pmass ~= 1 (declared method is LINEAR mass)';
    return;
end
if ~all(strcmpi({res.arms.baseline}, 'solid'))
    condition = 'A4 precondition violated: semi_harmonic_baseline ~= "solid"';
    return;
end

% Every arm must carry a Phase-2 measurement-integrity status.
for k = 1:numel(res.arms)
    a = res.arms(k);
    if isempty(a.phase2_status)
        condition = sprintf('arm N=%s was not classified', a.tag);
        return;
    end
    if strcmp(a.phase2_status, 'REJECTED')
        condition = sprintf('arm N=%s REJECTED: %s', a.tag, ...
            strjoin(a.implementation_failures,' | '));
        return;
    end
end

if ~strcmp(res.run_verdict,'COMPLETE')
    condition = sprintf('Phase-2 run verdict is %s, not COMPLETE',res.run_verdict);
    return;
end

% Required artifacts.
req = {'a4_eigenpair_refresh_results.mat','a4_result.json','a4_screening_events.json', ...
       'a4_candidate_telemetry.csv','a4_iteration_histories.csv', ...
       'a4_manifest.json','a4_stage_manifest.json','a4_table.md','a4_table2.md'};
for k = 1:numel(req)
    p = fullfile(outDir, req{k});
    if ~localCheckArtifact(p)
        condition = sprintf('required artifact missing: %s', req{k});
        return;
    end
end

pass = true;
condition = sprintf('COMPLETE: %d/%d arms satisfy Phase-2 measurement integrity', ...
    numel(res.arms), numel(res.n_levels));
end

function [pass, condition] = localAccept_S1(res, outDir)
%LOCALACCEPT_S1  Acceptance gate for s1_mitigation_400x50_pilot.
%   The mitigation must be CONCLUSIVE. The saved pilot classified itself as
%   "inconclusive" (9/10 modes still localized), which is a failure here.
pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end
if ~isfield(res, 'classification') || isempty(res.classification)
    condition = 'required field missing: classification'; return;
end
cls = char(res.classification);
if ~strcmpi(cls, 'accepted')
    condition = sprintf( ...
        'S1 mitigation classification is "%s", not "accepted": the localized low-density mode mitigation is unresolved, so EXP2b and EXP3 cannot be trusted at fine meshes.', cls);
    return;
end
for a = {'s1_mitigation_400x50_result.mat', 's1_mitigation_400x50_manifest.json'}
    if ~localCheckArtifact(fullfile(outDir, a{1}))
        condition = sprintf('required artifact missing: %s in %s', a{1}, outDir); return;
    end
end
pass = true;
end

function [pass, condition] = localAccept_Exp2Authoritative(res, outDir)
%LOCALACCEPT_EXP2AUTHORITATIVE  Acceptance gate for exp2_authoritative_sweep.
%   The sweep classifies each alpha internally (accepted / capped / mode invalid
%   / implementation failure).  A capped run is a failure.  Every alpha must be
%   accepted for the stage to pass.
pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end
if ~isfield(res, 'cases') || isempty(res.cases)
    condition = 'required field missing or empty: cases'; return;
end
if ~isfield(res, 'all_accepted')
    condition = 'required field missing: all_accepted'; return;
end

bad = {};
for k = 1:numel(res.cases)
    c = res.cases(k);
    cls = '';
    if isfield(c, 'classification'), cls = char(c.classification); end
    if ~strcmpi(cls, 'accepted')
        alpha = NaN;
        if isfield(c, 'alpha'), alpha = c.alpha; end
        bad{end+1} = sprintf('alpha=%.2f -> %s', alpha, cls); %#ok<AGROW>
    end
end
if ~isempty(bad)
    condition = sprintf('non-accepted alpha case(s): %s', strjoin(bad, '; '));
    return;
end
if ~res.all_accepted
    condition = 'sweep reports all_accepted = false'; return;
end

for a = {'exp2_authoritative_sweep_result.mat', 'exp2_authoritative_sweep_manifest.json'}
    if ~localCheckArtifact(fullfile(outDir, a{1}))
        condition = sprintf('required artifact missing: %s in %s', a{1}, outDir); return;
    end
end
pass = true;
end

function [pass, condition] = localAccept_Exp3Authoritative(res, outDir)
%LOCALACCEPT_EXP3AUTHORITATIVE  Acceptance gate for
%   exp3_authoritative_mesh_convergence.  The study-level classification must be
%   "passed mesh convergence"; "failed mesh convergence" and
%   "inconclusive/capped/mode/topology invalid" are failures.
pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end
if ~isfield(res, 'classification') || isempty(res.classification)
    condition = 'required field missing: classification'; return;
end
cls = char(res.classification);
if ~strcmpi(cls, 'passed mesh convergence')
    condition = sprintf('Exp3 study classification is "%s"; mesh convergence is not demonstrated.', cls);
    return;
end
if isfield(res, 'cases')
    for k = 1:numel(res.cases)
        c = res.cases(k);
        if isfield(c, 'classification') && ~strcmpi(char(c.classification), 'accepted')
            lbl = '';
            if isfield(c, 'label'), lbl = char(c.label); end
            condition = sprintf('mesh %s not accepted: %s', lbl, char(c.classification));
            return;
        end
    end
end
for a = {'exp3_authoritative_mesh_convergence_result.mat', ...
         'exp3_authoritative_mesh_convergence_manifest.json'}
    if ~localCheckArtifact(fullfile(outDir, a{1}))
        condition = sprintf('required artifact missing: %s in %s', a{1}, outDir); return;
    end
end
pass = true;
end

% -------------------------------------------------------------------------

% -------------------------------------------------------------------------

function [pass, condition] = localAccept_Exp2b(res, outDir)
%LOCALACCEPT_EXP2B  Acceptance checks for exp2b_building.
%
%   Every alpha must satisfy the declared revision acceptance rule via
%   CHECK_REVISION_RUN, and must carry MAC evidence.  Previously nIter was
%   required to exist but its value was never compared with the cap, so the
%   capped alpha=1.00 and alpha=0.75 runs were accepted; and an alpha with no
%   MAC data was skipped rather than rejected.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end

for fn = {'omega1', 'omega2', 'grayness', 'nIter', 'macData', ...
          'max_iters', 'design_change_tol', 'success', 'design_change'}
    if ~isfield(res, fn{1})
        condition = sprintf('invalid result schema: required field missing: %s', fn{1});
        return;
    end
end

macThresh = 0.8;
if isfield(res, 'MAC_threshold') && ~isnan(res.MAC_threshold)
    macThresh = res.MAC_threshold;
end

% Required NaN: omega1, omega2, grayness
[nanOk, msg] = localCheckNoNaN(res.omega1, 'omega1');
if ~nanOk, condition = msg; return; end

[nanOk, msg] = localCheckNoNaN(res.omega2, 'omega2');
if ~nanOk, condition = msg; return; end

[nanOk, msg] = localCheckNoNaN(res.grayness, 'grayness');
if ~nanOk, condition = msg; return; end

nAlpha = numel(res.alphaVals);
for i = 1:nAlpha
    label = sprintf('EXP2b alpha=%.2f', res.alphaVals(i));

    % Declared acceptance rule: completed, not capped, converged.
    run = struct( ...
        'success',       logical(res.success(i)), ...
        'iterations',    res.nIter(i), ...
        'cap',           res.max_iters, ...
        'design_change', res.design_change(i), ...
        'tol',           res.design_change_tol);
    [ok, why] = check_revision_run(label, run);
    if ~ok, condition = why; return; end

    % Mode evidence must be PRESENT, not merely non-contradictory.
    macD = res.macData{i};
    if isempty(macD) || ~isfield(macD, 'best_mac') || isempty(macD.best_mac)
        condition = sprintf( ...
            '%s: missing convergence metadata (no MAC data); tracked-mode validity cannot be verified', ...
            label);
        return;
    end
    if macD.best_mac(1) < macThresh
        condition = sprintf('%s: tracked mode MAC = %.3f < threshold %.2f', ...
            label, macD.best_mac(1), macThresh);
        return;
    end
end

% Required artifact
if ~localCheckArtifact(fullfile(outDir, 'exp2b_building_results.mat'))
    condition = sprintf('required artifact missing: exp2b_building_results.mat in %s', outDir);
    return;
end

pass = true;
end

% -------------------------------------------------------------------------

% =========================================================================
%  UTILITIES
% =========================================================================

function [ok, msg] = localCheckNoNaN(v, fieldName)
%LOCALCHECKNONA  Return ok=false if v contains any NaN, with location info.
ok = true; msg = '';
nanIdx = find(isnan(v(:)));
if ~isempty(nanIdx)
    ok  = false;
    msg = sprintf('required NaN in %s at linear index [%s] — run(s) failed', ...
        fieldName, num2str(nanIdx(1:min(3,end))'));
end
end

function ok = localCheckArtifact(path)
%LOCALCHECKARTIFACT  True iff the file exists and is non-empty.
ok = ~isempty(path) && isfile(path) && (dir(path).bytes > 0);
end

function f = localMakeFailure(tag, condition, trace, elapsed)
%LOCALMAKEFAILURE  Pack a failure record.
f.tag       = tag;
f.condition = condition;
f.trace     = trace;
f.elapsed   = elapsed;
if isnan(elapsed)
    f.elapsedStr = 'N/A';
else
    f.elapsedStr = sprintf('%.1fs', elapsed);
end
end

function v = localGetField(s, fn)
v = [];
if isstruct(s) && isfield(s, fn), v = s.(fn); end
end

function localPrepareDirFail(d)
%LOCALPREPAREDIR  Create d if absent; fail if it already has result files.
if ~exist(d, 'dir')
    mkdir(d);
    return;
end
% Directory exists -- check for conflicting result files
listing = [dir(fullfile(d,'*.mat')); ...
           dir(fullfile(d,'*.csv')); ...
           dir(fullfile(d,'*.png'))];
listing  = listing(~[listing.isdir]);
if ~isempty(listing)
    error('run_all:OutputConflict', ...
        ['Output directory already contains %d file(s) that would be overwritten:\n' ...
         '  %s\n' ...
         'Delete or rename the directory before re-running:\n' ...
         '  rmdir(''%s'', ''s'')'], ...
        numel(listing), d, d);
end
end

% =========================================================================
%  REPORTING
% =========================================================================

function localPrintSummaryTable(od, failures)
experiments = {'s1','exp2','exp2b','exp3','a4'};
descs = { ...
    'S1: Localized low-density mode mitigation (gates EXP2b/EXP3)'; ...
    'Exp2: Clamped beam authoritative alpha sweep'; ...
    'Exp2b: Building (Tables 4/5, spurious modes)'; ...
    'Exp3: Mesh convergence (200x25 vs 400x50)'; ...
    'A4: Eigenpair-refresh study -- NOT IMPLEMENTED' };
failTags = cellfun(@(f) f.tag, failures, 'UniformOutput', false);
fprintf('\n  %-52s  %s\n', 'Experiment', 'Status');
fprintf('  %s\n', repmat('-', 1, 70));
for k = 1:numel(experiments)
    tag = upper(strrep(experiments{k}, 'exp', 'EXP'));
    tag = strrep(tag, 'EXP2B', 'EXP2b');
    if any(strcmp(failTags, tag))
        st = 'FAILED';
    else
        d = od.(experiments{k});
        mf = dir(fullfile(d,'*.mat'));
        if ~isempty(mf), st = 'PASSED'; else, st = 'PASSED (no artifacts written yet)'; end
    end
    fprintf('  %-52s  %s\n', descs{k}, st);
end
fprintf('\n');
end

function localWriteManifest(od, outRoot, allResults) %#ok<INUSD>
%LOCALWRITEMANIFEST  Write a manifest listing every output file.
manifestPath = fullfile(outRoot, 'manifest.txt');
try
    fid = fopen(manifestPath, 'w');
    if fid < 0, return; end
    fprintf(fid, 'Revision_v1 experiment manifest -- %s\n', datestr(now)); %#ok<TNOW1,DATST>
    fprintf(fid, 'Generated by run_all_revision_experiments\n\n');
    fields = fieldnames(od);
    for k = 1:numel(fields)
        d = od.(fields{k});
        fprintf(fid, '[%s]  %s\n', upper(fields{k}), d);
        files = dir(d);
        for j = 1:numel(files)
            if ~files(j).isdir
                fprintf(fid, '  %-40s  %d bytes\n', files(j).name, files(j).bytes);
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf('Manifest written: %s\n', manifestPath);
catch
end
end

% =========================================================================
%  HARDWARE INFO  (preserved from v1)
% =========================================================================

function localPrintHardwareInfo()
fprintf('--- Hardware and Software Specification ---\n\n');
fprintf('  MATLAB version : %s\n', version);
try, fprintf('  CPU cores      : %d logical\n', feature('numcores')); catch, end
try, fprintf('  MATLAB threads : %d\n', maxNumCompThreads); catch, end
try
    [~, cpuStr] = system('sysctl -n machdep.cpu.brand_string 2>/dev/null');
    cpuStr = strtrim(cpuStr);
    if ~isempty(cpuStr), fprintf('  CPU            : %s\n', cpuStr); end
catch, end
try
    [~, hw] = memory;
    fprintf('  RAM            : %.1f GB physical\n', hw.PhysicalMemory.Total / 1e9);
catch
    try
        [~, ms] = system('sysctl -n hw.memsize 2>/dev/null');
        mb = str2double(strtrim(ms));
        if ~isnan(mb), fprintf('  RAM            : %.1f GB\n', mb / 1e9); end
    catch, end
end
try
    [~, os] = system('sw_vers -productVersion 2>/dev/null || uname -r');
    fprintf('  OS             : %s\n', strtrim(os));
catch, end
fprintf('  BLAS/LAPACK    : Apple Accelerate (MATLAB macOS arm64)\n\n');
end

% =========================================================================
%  PATH SETUP
% =========================================================================

function localEnsurePaths(scriptDir)
repoRoot  = fileparts(fileparts(scriptDir));
toolsDir  = fullfile(repoRoot, 'tools', 'Matlab');
schemaDir = fullfile(repoRoot, 'scripts', 'revision_v1');
if exist(toolsDir,  'dir') == 7, addpath(toolsDir);  end
if exist(schemaDir, 'dir') == 7, addpath(schemaDir); end
localAddActiveAnalysisPaths(repoRoot);
addpath(scriptDir);
end

function localAddActiveAnalysisPaths(repoRoot)
% Production allowlist: archived reconstruction trees must not enter the path.
activeDirNames = {'ourApproach','OlhoffApproach','YukselApproach','elastic2D','LabandaApproach'};
for iDir = 1:numel(activeDirNames)
    activeDir = fullfile(repoRoot, 'analysis', activeDirNames{iDir});
    if exist(activeDir, 'dir') == 7, addpath(genpath(activeDir)); end
end
end
