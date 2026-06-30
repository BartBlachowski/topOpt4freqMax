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
%
%   Full stack traces are preserved and printed for all exceptions.
%
%   OUTPUT DIRECTORIES
%   ------------------
%   Each experiment writes to its own subdirectory under output/:
%     examples/Revision_v1/output/exp1/
%     examples/Revision_v1/output/exp2/
%     examples/Revision_v1/output/exp2b/
%     examples/Revision_v1/output/exp3/
%     examples/Revision_v1/output/exp4/
%     examples/Revision_v1/output/exp5/
%     examples/Revision_v1/output/smoke/
%
%   Unless resume=true or force=true, the runner refuses to overwrite an output
%   directory that already contains .mat, .csv, or .png files.
%
%   REVIEWER ISSUES ADDRESSED
%   -------------------------
%   Exp 1  V1/MR1  Timing std devs. V2/MR2 omega_1. R2/MR4 hardware spec.
%   Exp 2  V5/m5   Initial freqs. M5 mode shapes. M6 grayness. CR1/V4 MAC.
%   Exp 2b V3/MR5  Building tables. M7 spurious modes.
%   Exp 3  V3/MR5  Mesh convergence 200x25 vs 400x50.
%   Exp 4  CR2/C1  Sensitivity ablation + FD check.
%   Exp 5  M4/mn8  Log-log scaling fit.
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
od = struct( ...
    'exp1',  fullfile(outRoot, 'exp1'),  ...
    'exp2',  fullfile(outRoot, 'exp2'),  ...
    'exp2b', fullfile(outRoot, 'exp2b'), ...
    'exp3',  fullfile(outRoot, 'exp3'),  ...
    'exp4',  fullfile(outRoot, 'exp4'),  ...
    'exp5',  fullfile(outRoot, 'exp5') );

campaignCfg = localCampaignConfig(mode, nSamples, meshTable, alphaVals);
stages = localBuildStages(od, nSamples, meshTable, alphaVals);
if strcmp(mode, 'stage')
    stages = localSelectStage(stages, opts.stageName);
end

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

        if strcmp(stage.key, 'exp5')
            stage.runFn = @() exp5_scaling(localGetField(allResults, 'exp1'), stage.outDir);
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
stages = repmat(localEmptyStage(), 0, 1);

stages(end+1, 1) = localMakeStage( ...
    'exp1', 'EXP1', 'Performance table (timing + omega_1 statistics)', od.exp1, ...
    @() exp1_perf_table(nSamples, meshTable, od.exp1), ...
    @(r) localAccept_Exp1(r, od.exp1), ...
    {fullfile(od.exp1, 'exp1_perf_table_results.mat')}, ...
    struct('nSamples', nSamples, 'meshTable', meshTable));

stages(end+1, 1) = localMakeStage( ...
    'exp2', 'EXP2', 'Clamped beam benchmark', od.exp2, ...
    @() exp2_clamped_beam(alphaVals, od.exp2), ...
    @(r) localAccept_Exp2(r, od.exp2), ...
    {fullfile(od.exp2, 'exp2_clamped_beam_results.mat')}, ...
    struct('alphaVals', alphaVals));

stages(end+1, 1) = localMakeStage( ...
    'exp2b', 'EXP2b', 'Building benchmark', od.exp2b, ...
    @() exp2b_building(alphaVals, od.exp2b), ...
    @(r) localAccept_Exp2b(r, od.exp2b), ...
    {fullfile(od.exp2b, 'exp2b_building_results.mat')}, ...
    struct('alphaVals', alphaVals));

stages(end+1, 1) = localMakeStage( ...
    'exp3', 'EXP3', 'Mesh convergence study', od.exp3, ...
    @() exp3_mesh_convergence(alphaVals, od.exp3), ...
    @(r) localAccept_Exp3(r, od.exp3), ...
    {fullfile(od.exp3, 'exp3_mesh_convergence_results.mat')}, ...
    struct('alphaVals', alphaVals));

stages(end+1, 1) = localMakeStage( ...
    'exp4', 'EXP4', 'Sensitivity ablation + finite-difference check', od.exp4, ...
    @() exp4_sensitivity_ablation(od.exp4), ...
    @(r) localAccept_Exp4(r, od.exp4), ...
    {fullfile(od.exp4, 'exp4_sensitivity_ablation_results.mat'), ...
     fullfile(od.exp4, 'exp4_variant1_diary.txt')}, ...
    struct());

stages(end+1, 1) = localMakeStage( ...
    'exp5', 'EXP5', 'Scaling log-log analysis', od.exp5, ...
    @() exp5_scaling([], od.exp5), ...
    @(r) localAccept_Exp5(r, od.exp5), ...
    {fullfile(od.exp5, 'exp5_scaling_results.mat'), ...
     fullfile(od.exp5, 'exp5_scaling_loglog.png')}, ...
    struct('depends_on', 'EXP1'));
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
    'config', struct());
end

function st = localMakeStage(key, tag, desc, outDir, runFn, acceptFn, requiredArtifacts, cfg)
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
manifest.stage = stage.tag;
manifest.status = status;
manifest.created_utc = localUtcNow();
manifest.output_dir = stage.outDir;
manifest.result_json = stage.resultJson;
manifest.manifest_json = stage.manifestJson;
manifest.required_artifacts = stage.requiredArtifacts;
manifest.files = localListOutputFiles(stage.outDir);
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
txt = localJsonEncode(localOrderStruct(s));
bytes = uint8(txt);
hash = uint32(2166136261);
prime = uint32(16777619);
for k = 1:numel(bytes)
    hash = bitxor(hash, uint32(bytes(k)));
    hash = hash * prime;
end
h = sprintf('fnv1a32_%08x', hash);
end

function out = localOrderStruct(in)
if isstruct(in)
    out = in;
    if numel(in) == 1
        f = sort(fieldnames(in));
        out = struct();
        for k = 1:numel(f)
            out.(f{k}) = localOrderStruct(in.(f{k}));
        end
    else
        for k = 1:numel(in)
            out(k) = localOrderStruct(in(k)); %#ok<AGROW>
        end
    end
elseif iscell(in)
    out = in;
    for k = 1:numel(in)
        out{k} = localOrderStruct(in{k});
    end
else
    out = in;
end
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

function [pass, condition] = localAccept_Exp1(res, outDir)
%LOCALACCEPT_EXP1  Acceptance checks for exp1_perf_table.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end

% Required fields
for fn = {'omega_mean', 'tIter_mean', 'tTotal_mean'}
    if ~isfield(res, fn{1}) || isempty(res.(fn{1}))
        condition = sprintf('required field missing or empty: %s', fn{1}); return;
    end
end

% Required NaN check: omega_mean (all runs must have produced a frequency)
[nanOk, nanField] = localCheckNoNaN(res.omega_mean, 'omega_mean');
if ~nanOk, condition = nanField; return; end

% Required NaN check: tIter_mean
[nanOk, nanField] = localCheckNoNaN(res.tIter_mean, 'tIter_mean');
if ~nanOk, condition = nanField; return; end

% Required artifact
if ~localCheckArtifact(fullfile(outDir, 'exp1_perf_table_results.mat'))
    condition = sprintf('required artifact missing: exp1_perf_table_results.mat in %s', outDir);
    return;
end

pass = true;
end

% -------------------------------------------------------------------------

function [pass, condition] = localAccept_Exp2(res, outDir)
%LOCALACCEPT_EXP2  Acceptance checks for exp2_clamped_beam.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end
if ~isfield(res, 'allRes') || isempty(res.allRes)
    condition = 'required field allRes is missing or empty'; return;
end

macThresh = 0.8;
if isfield(res, 'MAC_threshold') && ~isnan(res.MAC_threshold)
    macThresh = res.MAC_threshold;
end

nForms = numel(res.allRes);
for f = 1:nForms
    r = res.allRes{f};

    % Required NaN: omega1 (each run must have produced a frequency)
    [nanOk, msg] = localCheckNoNaN(r.omega1, sprintf('allRes{%d}.omega1', f));
    if ~nanOk, condition = msg; return; end

    % Required NaN: grayness
    [nanOk, msg] = localCheckNoNaN(r.grayness, sprintf('allRes{%d}.grayness', f));
    if ~nanOk, condition = msg; return; end

    % MAC threshold: check primary tracked mode (mode 1) when data is available
    for i = 1:numel(r.macData)
        macD = r.macData{i};
        if isempty(macD) || ~isfield(macD, 'best_mac') || isempty(macD.best_mac)
            continue;
        end
        if macD.best_mac(1) < macThresh
            condition = sprintf( ...
                'allRes{%d}, alpha index %d: tracked mode MAC = %.3f < threshold %.2f', ...
                f, i, macD.best_mac(1), macThresh);
            return;
        end
    end
end

% Required artifact
if ~localCheckArtifact(fullfile(outDir, 'exp2_clamped_beam_results.mat'))
    condition = sprintf('required artifact missing: exp2_clamped_beam_results.mat in %s', outDir);
    return;
end

pass = true;
end

% -------------------------------------------------------------------------

function [pass, condition] = localAccept_Exp2b(res, outDir)
%LOCALACCEPT_EXP2B  Acceptance checks for exp2b_building.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end

for fn = {'omega1', 'omega2', 'grayness', 'nIter'}
    if ~isfield(res, fn{1})
        condition = sprintf('required field missing: %s', fn{1}); return;
    end
end

macThresh = 0.8;
if isfield(res, 'MAC_threshold') && ~isnan(res.MAC_threshold)
    macThresh = res.MAC_threshold;
end

% Required NaN: omega1, omega2
[nanOk, msg] = localCheckNoNaN(res.omega1, 'omega1');
if ~nanOk, condition = msg; return; end

[nanOk, msg] = localCheckNoNaN(res.omega2, 'omega2');
if ~nanOk, condition = msg; return; end

% MAC threshold
if isfield(res, 'macData')
    for i = 1:numel(res.macData)
        macD = res.macData{i};
        if isempty(macD) || ~isfield(macD, 'best_mac') || isempty(macD.best_mac)
            continue;
        end
        if macD.best_mac(1) < macThresh
            condition = sprintf( ...
                'alpha index %d: tracked mode MAC = %.3f < threshold %.2f', ...
                i, macD.best_mac(1), macThresh);
            return;
        end
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

function [pass, condition] = localAccept_Exp3(res, outDir)
%LOCALACCEPT_EXP3  Acceptance checks for exp3_mesh_convergence.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end
if ~isfield(res, 'res_all') || isempty(res.res_all)
    condition = 'required field res_all is missing or empty'; return;
end

macThresh = 0.8;
if isfield(res, 'MAC_threshold') && ~isnan(res.MAC_threshold)
    macThresh = res.MAC_threshold;
end

[nMeshes, nForms] = size(res.res_all);
for mi = 1:nMeshes
    for f = 1:nForms
        r = res.res_all{mi, f};
        if isempty(r), continue; end

        tag_mf = sprintf('res_all{%d,%d}', mi, f);

        % Required NaN: omega1
        [nanOk, msg] = localCheckNoNaN(r.omega1, [tag_mf '.omega1']);
        if ~nanOk, condition = msg; return; end

        % MAC check
        for i = 1:numel(r.macData)
            macD = r.macData{i};
            if isempty(macD) || ~isfield(macD, 'best_mac') || isempty(macD.best_mac)
                continue;
            end
            if macD.best_mac(1) < macThresh
                condition = sprintf( ...
                    '%s, alpha index %d: tracked mode MAC = %.3f < threshold %.2f', ...
                    tag_mf, i, macD.best_mac(1), macThresh);
                return;
            end
        end
    end
end

% Required artifact
if ~localCheckArtifact(fullfile(outDir, 'exp3_mesh_convergence_results.mat'))
    condition = sprintf('required artifact missing: exp3_mesh_convergence_results.mat in %s', outDir);
    return;
end

pass = true;
end

% -------------------------------------------------------------------------

function [pass, condition] = localAccept_Exp4(res, outDir)
%LOCALACCEPT_EXP4  Acceptance checks for exp4_sensitivity_ablation.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end

for fn = {'omega1', 'nIter', 'grayness'}
    if ~isfield(res, fn{1})
        condition = sprintf('required field missing: %s', fn{1}); return;
    end
end

% Required NaN: all four variants must have completed
[nanOk, msg] = localCheckNoNaN(res.omega1, 'omega1');
if ~nanOk, condition = msg; return; end

[nanOk, msg] = localCheckNoNaN(res.nIter, 'nIter');
if ~nanOk, condition = msg; return; end

% Required diary for Variant A (FD check output)
diaryA = fullfile(outDir, 'exp4_variant1_diary.txt');
if ~localCheckArtifact(diaryA)
    condition = sprintf('required artifact missing: exp4_variant1_diary.txt (FD check log) in %s', outDir);
    return;
end

% Required mat artifact
if ~localCheckArtifact(fullfile(outDir, 'exp4_sensitivity_ablation_results.mat'))
    condition = sprintf('required artifact missing: exp4_sensitivity_ablation_results.mat in %s', outDir);
    return;
end

pass = true;
end

% -------------------------------------------------------------------------

function [pass, condition] = localAccept_Exp5(res, outDir)
%LOCALACCEPT_EXP5  Acceptance checks for exp5_scaling.

pass = false; condition = '';

if isempty(res) || ~isstruct(res)
    condition = 'returned empty or non-struct result'; return;
end

for fn = {'beta', 'R2', 'tIter_data'}
    if ~isfield(res, fn{1})
        condition = sprintf('required field missing: %s', fn{1}); return;
    end
end

% At least one valid beta fit
if all(isnan(res.beta))
    condition = 'beta is all-NaN: no scaling fit completed (insufficient data?)';
    return;
end

% Required plot artifact
if ~localCheckArtifact(fullfile(outDir, 'exp5_scaling_loglog.png'))
    condition = sprintf('required artifact missing: exp5_scaling_loglog.png in %s', outDir);
    return;
end

% Required mat artifact
if ~localCheckArtifact(fullfile(outDir, 'exp5_scaling_results.mat'))
    condition = sprintf('required artifact missing: exp5_scaling_results.mat in %s', outDir);
    return;
end

pass = true;
end

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
experiments = {'exp1','exp2','exp2b','exp3','exp4','exp5'};
descs = { ...
    'Exp1: Performance table (omega_1, timing, init cost)'; ...
    'Exp2: Clamped beam (freqs, modes, MAC, semi_harm vs Eq.7)'; ...
    'Exp2b: Building (Tables 4/5, spurious modes)'; ...
    'Exp3: Mesh convergence (200x25 vs 400x50)'; ...
    'Exp4: Sensitivity ablation + FD gradient check'; ...
    'Exp5: Scaling log-log analysis' };
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
addpath(genpath(fullfile(repoRoot, 'analysis')));
addpath(scriptDir);
end
