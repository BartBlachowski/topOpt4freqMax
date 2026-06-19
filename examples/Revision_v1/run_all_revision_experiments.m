function run_all_revision_experiments(mode)
%RUN_ALL_REVISION_EXPERIMENTS  Fail-loud master runner for paper revision experiments.
%
%   run_all_revision_experiments            % full mode (10 samples, 4 meshes)
%   run_all_revision_experiments('fast')    % fast mode (2 samples, 2 meshes)
%   run_all_revision_experiments('smoke')   % Gate I1 verification only
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
%   The runner refuses to overwrite an output directory that already contains
%   .mat, .csv, or .png files.  Delete or rename the directory to re-run.
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
mode = lower(strtrim(char(mode)));

scriptDir = fileparts(mfilename('fullpath'));
outRoot   = fullfile(scriptDir, 'output');
prevDir   = pwd;
cleanupCd = onCleanup(@() cd(prevDir)); %#ok<NASGU>
cd(scriptDir);

localEnsurePaths(scriptDir);

fprintf('\n');
fprintf('+====================================================================+\n');
fprintf('|  REVISION EXPERIMENTS -- topOpt4freqMax  (fail-loud runner v2)     |\n');
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
    otherwise
        error('run_all:BadMode', ...
            'Unknown mode "%s". Use "fast", "full", or "smoke".', mode);
end

% ---- per-experiment output directories ----------------------------------
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

od = struct( ...
    'exp1',  fullfile(outRoot, 'exp1'),  ...
    'exp2',  fullfile(outRoot, 'exp2'),  ...
    'exp2b', fullfile(outRoot, 'exp2b'), ...
    'exp3',  fullfile(outRoot, 'exp3'),  ...
    'exp4',  fullfile(outRoot, 'exp4'),  ...
    'exp5',  fullfile(outRoot, 'exp5') );

odFields = fieldnames(od);
for k = 1:numel(odFields)
    localPrepareDirFail(od.(odFields{k}));
end

% ---- run experiments, collect failures ----------------------------------
failures   = {};
allResults = struct();

[allResults.exp1, pass, cond, trace, elapsed] = localRunAndAccept( ...
    'EXP1', 'Performance table + omega_1 + init timing', ...
    @() exp1_perf_table(nSamples, meshTable, od.exp1), ...
    @(r) localAccept_Exp1(r, od.exp1));
if ~pass, failures{end+1} = localMakeFailure('EXP1', cond, trace, elapsed); end

[allResults.exp2, pass, cond, trace, elapsed] = localRunAndAccept( ...
    'EXP2', 'Clamped beam: freqs, MAC, grayness, semi_harmonic vs Eq.7', ...
    @() exp2_clamped_beam(alphaVals, od.exp2), ...
    @(r) localAccept_Exp2(r, od.exp2));
if ~pass, failures{end+1} = localMakeFailure('EXP2', cond, trace, elapsed); end

[allResults.exp2b, pass, cond, trace, elapsed] = localRunAndAccept( ...
    'EXP2b', 'Building example (Tables 4/5), spurious mode check', ...
    @() exp2b_building(alphaVals, od.exp2b), ...
    @(r) localAccept_Exp2b(r, od.exp2b));
if ~pass, failures{end+1} = localMakeFailure('EXP2b', cond, trace, elapsed); end

[allResults.exp3, pass, cond, trace, elapsed] = localRunAndAccept( ...
    'EXP3', 'Mesh convergence 200x25 vs 400x50', ...
    @() exp3_mesh_convergence(alphaVals, od.exp3), ...
    @(r) localAccept_Exp3(r, od.exp3));
if ~pass, failures{end+1} = localMakeFailure('EXP3', cond, trace, elapsed); end

[allResults.exp4, pass, cond, trace, elapsed] = localRunAndAccept( ...
    'EXP4', 'Sensitivity ablation: Variant A/B + C/D + FD gradient check', ...
    @() exp4_sensitivity_ablation(od.exp4), ...
    @(r) localAccept_Exp4(r, od.exp4));
if ~pass, failures{end+1} = localMakeFailure('EXP4', cond, trace, elapsed); end

[allResults.exp5, pass, cond, trace, elapsed] = localRunAndAccept( ...
    'EXP5', 'Scaling log-log (corrects O(n_e^1.3) claim)', ...
    @() exp5_scaling(localGetField(allResults, 'exp1'), od.exp5), ...
    @(r) localAccept_Exp5(r, od.exp5));
if ~pass, failures{end+1} = localMakeFailure('EXP5', cond, trace, elapsed); end

% ---- summary table ------------------------------------------------------
fprintf('\n====================================================================\n');
fprintf(' EXPERIMENT SUMMARY\n');
fprintf('====================================================================\n');
localPrintSummaryTable(od, failures);

% ---- manifest -----------------------------------------------------------
localWriteManifest(od, outRoot, allResults);

% ---- fail-loud report ---------------------------------------------------
if ~isempty(failures)
    fprintf('\n====================================================================\n');
    fprintf(' FAILURE REPORT  (%d mandatory experiment(s) FAILED)\n', numel(failures));
    fprintf('====================================================================\n');
    for k = 1:numel(failures)
        f = failures{k};
        fprintf('\n  [FAILED] %s\n', f.tag);
        fprintf('  Condition : %s\n', f.condition);
        fprintf('  Elapsed   : %s\n', f.elapsedStr);
        if ~isempty(f.trace)
            fprintf('  Stack trace:\n');
            for j = 1:numel(f.trace)
                fprintf('    %2d.  %-40s  line %d\n', ...
                    j, f.trace(j).name, f.trace(j).line);
                fprintf('         %s\n', f.trace(j).file);
            end
        end
    end
    fprintf('\n');

    failIds = cellfun(@(f) f.tag, failures, 'UniformOutput', false);
    error('run_all:MandatoryExperimentFailed', ...
        '%d mandatory experiment(s) failed: %s.\nSee failure report above.', ...
        numel(failures), strjoin(failIds, ', '));
end

save(fullfile(outRoot, 'all_revision_results.mat'), 'allResults');
fprintf('\nAll mandatory experiments PASSED.\n');
fprintf('Results: %s\n\n', fullfile(outRoot, 'all_revision_results.mat'));
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
