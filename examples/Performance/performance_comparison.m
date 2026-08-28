clear; clc;
close all;

% This script now depends on helpers that live beside it (the task profile
% builder and the Olhoff admission preflight) as well as on tools/Matlab.  Put
% both on the path explicitly rather than relying on the current folder
% happening to be the right one.
scriptDir = fileparts(mfilename('fullpath'));
repoRoot  = fileparts(fileparts(scriptDir));
addpath(scriptDir);
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
% The final campaign's Olhoff runner and the common E1/E2/E3 evaluator do not
% live under tools/: they belong to the frozen stabilization audit and to the
% three-method parametric study.  Every path is derived from the SCRIPT's own
% location, so this file may be opened and Run from any current folder.
addpath(fullfile(repoRoot, 'analysis', 'olhoff_stabilization_audit'));
addpath(fullfile(repoRoot, 'analysis', 'three_method_parametric_study'));
addpath(fullfile(repoRoot, 'Matlab', 'reproduction2007', 'runner'));

% -------------------------------------------------------------------------
% BENCHMARK MODE
%
%   'final_campaign' DEFAULT.  The ultimate nine-resolution performance and
%                   scaling campaign, campaign_id
%                   ultimate_nine_resolution_performance_scaling_v1.  Every
%                   method runs the profile frozen for it, read from
%                   analysis/olhoff_stabilization_audit/final_campaign_profile.json:
%
%                     Olhoff    olhoff_fig3a_s1_native_bimodal_p100_move005_0025_fixed1600_v1
%                     Yuksel    yuksel_practical_move01_tol001
%                     Proposed  proposed_practical_move02_tol001
%
%                   The Olhoff column is NOT the legacy S0 / r_min = 2 profile
%                   this script used to dispatch.  It is the stabilized S1
%                   profile validated in analysis/olhoff_stabilization_audit/:
%                   r_min = 1.3 elements, move 0.005 with one causal reduction
%                   to 0.0025 after 100 consecutive native evaluations with
%                   N == 2 and gap12 <= 1%, and a FIXED 1600 outer-iteration
%                   work horizon that is explicitly NOT native convergence.
%                   It is produced by the runner the manifest names,
%                   analysis/olhoff_stabilization_audit/run_stabilization_case.m,
%                   not by run_topopt_from_json -> OlhoffDu2007Repro.
%
%                   Nothing runs until FINAL_CAMPAIGN_PREFLIGHT has passed, and
%                   every artifact goes to examples/Performance/final_campaign/
%                   so no earlier evidence is touched.
%
%   'r3'            SUPERSEDED.  The earlier R3 native-performance campaign,
%                   which dispatched the legacy Olhoff S0 profile at r_min = 2
%                   and wrote its artifacts into examples/Performance/ IN PLACE.
%                   Retained so the historical campaign stays reproducible; it
%                   is NOT the campaign of record any more.  See Table E of
%                   analysis/olhoff_stabilization_audit/OLHOFF_STABILIZATION_AUDIT.md.
%
%   'yuksel_table1' DIAGNOSTIC.  Reproduces the reporting interpretation of
%                   Yuksel & Yilmaz (2025) Table 1: the eigenvalue-per-iteration
%                   method -- the role our Olhoff column plays, what that paper
%                   calls the "dynamic code" -- is given a FIXED 200-iteration
%                   work budget, while the other methods keep their native
%                   stopping.  Section 6.2 states the rule directly, and Table 1
%                   is self-consistent with it: each dynamic-code total divided
%                   by its per-iteration time is 200 to three figures.
%
%                   Nothing about the Olhoff ALGORITHM changes.  Only
%                   optimization.repro2007.max_outer, a documented task-file
%                   key, which reaches OLHOFFOPT as cfg.maxOuter -- the bound of
%                   its outer for-loop and nothing else.
%
%                   Every output goes to a separate directory.  A diagnostic
%                   must not be able to overwrite the R3 artifacts, because the
%                   two answer different questions and their tables look alike.
benchmarkMode = 'final_campaign';

% Scripted runs can override the line above without editing this file:
%   setenv('TOPOPT_BENCHMARK_MODE','yuksel_table1'); performance_comparison
% Interactive use should just edit benchmarkMode directly -- the environment
% variable exists so that a diagnostic sweep does not have to modify, and risk
% leaving modified, the file that defines the campaign of record.
%
% setenv persists for the whole MATLAB SESSION, so a variable left over from an
% earlier diagnostic would otherwise silently redirect a campaign launched by
% pressing Run.  An override away from the campaign of record is therefore
% announced loudly rather than in passing.
envMode = getenv('TOPOPT_BENCHMARK_MODE');
if ~isempty(envMode)
    benchmarkMode = lower(strtrim(envMode));
    fprintf(2, ['\n*** TOPOPT_BENCHMARK_MODE is set in this MATLAB session.\n' ...
        '*** benchmarkMode = %s, NOT the default final_campaign.\n' ...
        '*** Clear it with  setenv(''TOPOPT_BENCHMARK_MODE'','''')  to run the campaign.\n\n'], ...
        benchmarkMode);
end
isFinalCampaign = strcmp(benchmarkMode, 'final_campaign');

% Load the benchmark task profile.  Every override the comparison applies on
% top of performance_comparison.json -- visualization off, the shared
% cross-resolution filter radius, and the Du-Olhoff 2007 block below -- lives in
% PERFORMANCE_BENCHMARK_PROFILE, which is the single definition of what this
% campaign runs.  The Olhoff benchmark-path equivalence harness reads the same
% function, so it cannot certify a profile this file no longer runs.
%
% -------------------------------------------------------------------------
% Du-Olhoff 2007 clean-room reproduction: settings that CANNOT come from the
% shared benchmark block.
%
% Four of the shared settings are not transferable to this solver, and using
% them produces numbers that are wrong rather than merely different.  Every
% override is scoped to this method alone; nothing there changes Yuksel or the
% Proposed method.
%
%  1. move limit.  The shared `optimization.move_limit` is 0.2, which is an
%     MMA/OC move limit.  In this solver the move limit is the trust region of
%     a sequential LINEAR program, and 0.2 destroys the design: measured at
%     160x20, r_min = 2 el, the run collapses to a disconnected island and
%     omega_1 ends at 2.9 rad/s instead of ~160 (NOTES.md section 8c documents
%     the same failure at move = 0.03).  The value used, 0.005, is the
%     documented `fig3a_best` reproduction value.
%
%  2. outer-iteration budget.  While the LP solves successfully this method is
%     move-saturated: the step always travels the full move limit, so max|drho|
%     stays at `move` and the native stop test does not fire.  With the shared
%     max_iters = 10000 every mesh would run 10000 outer iterations.  The
%     budget used, 1600, is the documented `fig3a_best` value and is what
%     produced the published reproduction.
%
%  3. void lower bound.  `void_material.rho_min` in the shared block is 1e-6,
%     a void MATERIAL DENSITY floor.  This solver's rho_min is a different
%     quantity: the DESIGN VARIABLE bound of Du & Olhoff (2007) eq. (7e), whose
%     value is 1e-3.  At 1e-6 the (K,M) pencil goes singular to working
%     precision (eigs reports RCOND = 1.6e-19); at 240x30 that produced
%     spurious omega_1 = 0 modes from outer iteration 101, melted the design to
%     volume 0.20, and ended the run on an infeasible LP that was misreported
%     as convergence.  See DIAGNOSTIC_REPRO2007_BENCHMARK.md.
%
%  4. outer tolerance.  The shared `convergence_tol` is 3e-3; this method's
%     documented outer tolerance is 1e-3.  Stated rather than inherited, so
%     that the stopping point is not set by a value chosen for other methods.
%
% All of them are listed explicitly in PERFORMANCE_BENCHMARK_PROFILE, so the
% task profile rather than the dispatcher is the record of what this method
% ran with.
%
% Reading the Olhoff column: while the LP succeeds, this method stops at its
% outer-iteration budget rather than on a convergence test, so its iteration
% count is a budget and t_iter and the scaling exponent are the meaningful
% entries -- not iter_total or wall time.  A stop reason other than
% `max_outer_iterations` means the LP failed and MUST be investigated, not
% read as convergence; `telemetry.stopping.status` now says SOLVER_FAILURE in
% that case, and the admission gate below refuses the row.
%
% Filter radius is deliberately NOT overridden per method: r_min = 2 elements
% is the benchmark's shared cross-resolution setting and the solver runs
% correctly at it (verified at 240x30: 1600 iterations, volume 0.5, no LP
% failures).  It is, however, not the radius that reproduces Fig. 3a (1.3
% elements), so the omega_1 reported here for Olhoff is a valid operating point
% of the method and NOT the paper-reproduction figure.
%
% NOTE ON THE FOUR ITEMS ABOVE.  They describe the LEGACY 'r3' Olhoff column,
% which this script no longer runs by default.  In 'final_campaign' mode none
% of PERFORMANCE_BENCHMARK_PROFILE is used for any method: each method's
% configuration is read from its frozen manifest by FINAL_CAMPAIGN_CONFIG, and
% the Olhoff column runs at r_min = 1.3 elements under the stabilized S1
% profile rather than at the shared r_min = 2.
if isFinalCampaign
    % Nothing here is a choice made in this file.  campaignManifest is the
    % authority; the per-method configurations are built below, once the mesh
    % is known, and are checked against the manifest before anything is solved.
    campaignManifest = jsondecode(fileread(fullfile(repoRoot, 'analysis', ...
        'olhoff_stabilization_audit', 'final_campaign_profile.json')));
    benchmarkProfileId = char(campaignManifest.campaign_id);
    benchmarkProfileMeta = struct('mode_note', ...
        ['Ultimate nine-resolution performance/scaling campaign under the frozen ' ...
         'per-method profiles; Olhoff uses the stabilized S1 profile at a FIXED ' ...
         '1600-outer work horizon, which is NOT native convergence.'], ...
        'campaign_manifest', campaignManifest);
    data = struct();   % populated per method below; see finalCampaignConfigs
    outputDir = fullfile(scriptDir, 'final_campaign');
else
    [data, benchmarkProfileId, benchmarkProfileMeta] = ...
        performance_benchmark_profile([], [], benchmarkMode);
end

% Where this run's artifacts go.  The final campaign writes into its own
% directory; 'r3' writes the superseded campaign artifacts in place; every
% other mode is a diagnostic confined to its own directory.  No mode can
% overwrite another mode's evidence.
if isFinalCampaign
    % outputDir already set above.
elseif strcmp(benchmarkMode, 'r3')
    outputDir = scriptDir;
else
    outputDir = fullfile(scriptDir, ['diagnostic_' benchmarkMode]);
end
if exist(outputDir, 'dir') ~= 7
    mkdir(outputDir);
end

% -------------------------------------------------------------------------
% TIMING PROTOCOL
%
% One computation thread per run, matching threads_per_run in the campaign
% manifest and the single-thread setting under which the stabilization audit
% was conducted.  Set here rather than assumed, and recorded in the artifacts.
%
% The audit's own cross-resolution timings were taken with several
% single-thread MATLAB processes running side by side, so they are descriptive
% only.  This campaign measures serially in one process; audit timings and
% campaign timings must not be compared directly.
threadsRequested = 1;
if isFinalCampaign
    threadsRequested = campaignManifest.threads_per_run;
end
previousThreads = maxNumCompThreads(threadsRequested);
threadsActive = maxNumCompThreads;
% Restore the user's thread setting when the script finishes or is interrupted.
threadGuard = onCleanup(@() maxNumCompThreads(previousThreads)); %#ok<NASGU>

fprintf('\n================================================================\n');
fprintf(' benchmark mode : %s\n', benchmarkMode);
fprintf(' profile        : %s\n', benchmarkProfileId);
fprintf(' interpretation : %s\n', benchmarkProfileMeta.mode_note);
fprintf(' artifacts      : %s\n', outputDir);
fprintf(' MATLAB         : %s on %s\n', version, computer);
fprintf(' comp threads   : %d requested, %d active (was %d)\n', ...
    threadsRequested, threadsActive, previousThreads);
if isFinalCampaign
    fprintf(' campaign       : %s\n', char(campaignManifest.campaign_id));
    fprintf(' *** FINAL CAMPAIGN -- frozen profiles, no tuning ***\n');
elseif strcmp(benchmarkMode, 'r3')
    fprintf(' *** SUPERSEDED R3 MODE -- writes in place, NOT the campaign of record ***\n');
else
    fprintf(' *** DIAGNOSTIC MODE -- these outputs are NOT a campaign ***\n');
end
fprintf('================================================================\n');

% -------------------------------------------------------------------------
% Resolutions: those from Table 1 in the paper (160x20, 240x30, 320x40)
% plus two additional ones (240x30 already in paper; 400x50 is new)
% -------------------------------------------------------------------------

if isFinalCampaign
    % READ from the manifest, not restated here.  The nine campaign meshes are
    % 160x20, 240x30, 320x40, 400x50, 480x60, 560x70, 640x80, 720x90, 800x100.
    % Taking them from mesh_sequence means the active set cannot drift away
    % from the set the campaign was authorized for.
    resolutions = double(campaignManifest.mesh_sequence);
else
    % The legacy development subset, used by 'r3' and the diagnostic modes.
    resolutions = [
        160,  20;
        240,  30;
        320,  40;
        400,  50;
    ];
end

% Mesh-list override, for smoke-testing the full reporting path cheaply.
%   setenv('TOPOPT_BENCHMARK_MESHES','40x5,60x8')
% A reporting bug that only appears after a 10-minute campaign is a bug you find
% the expensive way; this makes the whole script exercisable in seconds.  It is
% recorded in the artifacts so a smoke run can never be mistaken for a campaign.
%
% It is REFUSED in final-campaign mode.  setenv survives for the whole MATLAB
% session, and a leftover smoke-test variable that quietly shrank the campaign
% would produce a nine-mesh-looking artifact set from two toy meshes.
envMeshes = getenv('TOPOPT_BENCHMARK_MESHES');
if ~isempty(envMeshes)
    assert(~isFinalCampaign, 'performance_comparison:MeshOverrideInCampaign', ...
        ['TOPOPT_BENCHMARK_MESHES is set to "%s", but the final campaign runs the ' ...
         'nine meshes named in final_campaign_profile.json.  Clear it with ' ...
         'setenv(''TOPOPT_BENCHMARK_MESHES'','''') and run again.'], envMeshes);
    parts = strsplit(strtrim(envMeshes), ',');
    resolutions = zeros(numel(parts), 2);
    for q = 1:numel(parts)
        xy = sscanf(strtrim(parts{q}), '%dx%d');
        assert(numel(xy) == 2, 'performance_comparison:BadMeshOverride', ...
            'TOPOPT_BENCHMARK_MESHES entries must look like 160x20 (got "%s").', parts{q});
        resolutions(q, :) = xy(:).';
    end
    fprintf('[performance_comparison] resolutions overridden by TOPOPT_BENCHMARK_MESHES = %s\n', ...
        envMeshes);
end

nRes = size(resolutions, 1);

% Methods to compare.
%
% `approaches` is the method IDENTITY used for naming everywhere in the
% results -- console tables, CSV, JSON, LaTeX.  `solverApproaches` is the
% dispatch key handed to run_topopt_from_json.  The two are kept separate so
% that the implementation behind a column can be changed without renaming the
% column.
%
% The Olhoff column is now produced by the Du-Olhoff 2007 CLEAN-ROOM
% REPRODUCTION (Eq. 22 LP route) at Matlab/reproduction2007/, replacing the
% earlier analysis/OlhoffApproach/Matlab/topFreqOptimization_MMA.m call.  The
% reported name is unchanged; only the solver behind it moved.  See
% MIGRATION_REPRODUCTION2007_REPORT.md and Matlab/README.md.
approaches       = {'Olhoff',            'Yuksel',         'OurApproach'      };
solverApproaches = {'OlhoffDu2007Repro', 'Yuksel',         'OurApproach'      };
methodLabels     = {'OlhoffApproach',    'YukselApproach', 'ProposedApproach' };
% Short names used in every printed/exported table.  Defined here with the other
% method tables rather than next to the first table that happens to need it, so
% that a report block added anywhere below can rely on it existing.
displayNames     = {'Olhoff',            'Yuksel',         'Proposed'         };
nMethods         = numel(approaches);
assert(numel(solverApproaches) == nMethods, ...
    'performance_comparison:MethodTableMismatch', ...
    'approaches, solverApproaches and methodLabels must be the same length.');

nSamples = 1;

% Storage: rows = resolutions, columns = methods
omega_all  = NaN(nRes, nMethods);
tIter_all  = NaN(nRes, nMethods);
nIter_all  = NaN(nRes, nMethods);
mem_all    = NaN(nRes, nMethods);
nIterStage1_all = NaN(nRes, nMethods);
nIterStage2_all = NaN(nRes, nMethods);
nOuter_all = NaN(nRes, nMethods);
nInner_all = NaN(nRes, nMethods);
tInit_all  = NaN(nRes, nMethods);
tLoop_all  = NaN(nRes, nMethods);
tPost_all  = NaN(nRes, nMethods);
% In-loop eigensolve time.  Only methods that report it get a number; the rest
% stay NaN and print as "not separable" rather than as zero, which would read as
% "spends no time on eigenproblems".
eigLoop_all = NaN(nRes, nMethods);
tTotal_all = NaN(nRes, nMethods);
tReconstructed_all = NaN(nRes, nMethods);
stage1Share_all = NaN(nRes, nMethods);
stage2Share_all = NaN(nRes, nMethods);
stopReason_all = repmat({'N/A'}, nRes, nMethods);
finalMaxChange_all = NaN(nRes, nMethods);
finalRmsChange_all = NaN(nRes, nMethods);
finalRelObjectiveChange_all = NaN(nRes, nMethods);
finalGrayness_all = NaN(nRes, nMethods);
convergenceTolerance_all = NaN(nRes, nMethods);
% Precedence-ordered per-row verdict and the censoring mask derived from it.
% status_all keeps every row VISIBLE in the printed and exported tables;
% fitEligible_all decides, separately, which rows a scaling fit may treat as
% successful observations.  Keeping the two apart is the whole point: a failed
% or capped run must not vanish, and must not be counted either.
status_all = repmat({'NOT_RUN'}, nRes, nMethods);
statusNote_all = repmat({''}, nRes, nMethods);
fitEligible_all = false(nRes, nMethods);
evaluators_all = cell(nRes, nMethods);
runRecords = struct('method', {}, 'method_label', {}, 'mesh', {}, 'sample', {}, ...
    'iterations', {}, 'timing', {}, 'stopping', {}, 'configuration', {}, ...
    'results', {}, 'max_ram_mb', {});
runRecordIndex = 0;

% -------------------------------------------------------------------------
% BENCHMARK ADMISSION PREFLIGHT -- Olhoff path equivalence
%
% The Olhoff column is produced by a different implementation from the one the
% column is named after: performance_comparison dispatches through
% run_topopt_from_json -> OlhoffDu2007Repro -> run_repro2007 -> olhoffOpt, and
% every hop can rename, default or override a setting.  A mapping defect in
% that chain has already happened once and was silent -- both source JSONs were
% valid, no error was raised, and the trajectory diverged at outer iteration
% 101 (DIAGNOSTIC_REPRO2007_BENCHMARK.md).
%
% So a timing row is not admitted because the run completed.  It is admitted
% only when the dispatched path has been PROVED, for that mesh and this exact
% profile/commit/implementation/MATLAB, to reproduce a direct call to the
% clean-room reproduction bit for bit.  The proof is made by
% VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE and read back here by
% OLHOFF_EQUIVALENCE_GATE.
%
% A mesh that fails, or was never verified, is NOT run and NOT tabulated.  It
% is recorded with a row class saying why, and left out of every table, fit,
% ratio and speedup downstream.  The decision itself lives in OLHOFF_PREFLIGHT,
% so it can be tested without running a campaign.
%
% The equivalence gate above is specific to the LEGACY dispatched Olhoff path
% (run_topopt_from_json -> OlhoffDu2007Repro).  The final campaign does not use
% that path at all: it calls the runner named in final_campaign_profile.json
% directly, so there is no dispatcher hop left to certify.  Its admission gate
% is FINAL_CAMPAIGN_PREFLIGHT instead -- manifest hashes, frozen profile IDs,
% generated configurations, the optimizer-boundary settings and the output
% location, all with no optimization solve.
if isFinalCampaign
    campaignGate = final_campaign_preflight(resolutions, outputDir);
    olhoffGate = struct([]);
    olhoffAdmitted = true(nRes, 1);
    olhoffMethodIdx = [];

    % Per-mesh, per-method configurations, all built and checked BEFORE the
    % first expensive solve, so a stale or ambiguous profile stops the run
    % here rather than nine meshes later.
    finalCampaignConfigs = cell(nRes, nMethods);
    finalCampaignProfileIds = cell(1, nMethods);
    for r = 1:nRes
        for m = 1:nMethods
            [cfgRM, pidRM] = final_campaign_config(approaches{m}, ...
                resolutions(r,1), resolutions(r,2), outputDir);
            finalCampaignConfigs{r, m} = cfgRM;
            if r == 1
                finalCampaignProfileIds{m} = pidRM;
            else
                assert(strcmp(finalCampaignProfileIds{m}, pidRM), ...
                    'performance_comparison:ProfileIdDrift', ...
                    'Profile ID for %s changed between meshes.', approaches{m});
            end
        end
    end
    fprintf('\nFrozen profiles bound for this campaign:\n');
    for m = 1:nMethods
        fprintf('  %-16s %s\n', displayNames{m}, finalCampaignProfileIds{m});
    end
    fprintf('\n');
else
    campaignGate = struct([]);
    finalCampaignConfigs = {};
    finalCampaignProfileIds = {};
    [olhoffGate, olhoffAdmitted, olhoffMethodIdx] = ...
        olhoff_preflight(resolutions, solverApproaches, ...
                         struct('profile_mode', benchmarkMode));
end

% -------------------------------------------------------------------------
% TIMING WARM-UP
%
% One throwaway solve per method at a mesh that is NOT in the campaign, so that
% JIT compilation, BLAS/LAPACK initialization and first-touch allocation are
% paid before the first measured row.  Without it the smallest mesh -- the
% first row, and the left-hand anchor of every scaling fit -- carries startup
% cost the other rows do not, which tilts the fitted exponent.
%
% Warm-up results are discarded, never recorded as observations, and go to
% their own directory.  A warm-up failure is a warning, not a campaign stop:
% it degrades the measurement, it does not invalidate the profile.
warmupReport = struct('mesh', [], 'ran', false, 'methods', {{}}, 'notes', {{}});
if isFinalCampaign
    % Even nely: the Olhoff reproduction pins its supports at mid height and
    % rejects an odd nely.  48x6 also keeps elements square and is not one of
    % the nine campaign meshes.
    wNelx = 48; wNely = 6; wOuter = 5;
    warmupDir = fullfile(outputDir, 'warmup');
    fprintf('Timing warm-up at %dx%d (discarded, not a campaign observation)...\n', wNelx, wNely);
    warmupReport.mesh = [wNelx wNely];
    warmupReport.ran = true;
    for m = 1:nMethods
        try
            wCfg = final_campaign_config(approaches{m}, wNelx, wNely, warmupDir);
            wCfg.optimization.max_iters = wOuter;
            if isfield(wCfg.optimization, 'yuksel')
                wCfg.optimization.yuksel.stage1_max_iters = wOuter;
            end
            wOut = final_campaign_run_case(approaches{m}, wCfg, ...
                struct('warmup', true, 'max_outer_override', wOuter, 'label', 'warmup'));
            warmupReport.methods{end+1} = displayNames{m}; %#ok<SAGROW>
            warmupReport.notes{end+1} = sprintf('%s in %.2f s', ...
                wOut.status, wOut.driver_wall_time_s); %#ok<SAGROW>
            fprintf('  %-16s warm-up %s (%.2f s)\n', displayNames{m}, ...
                wOut.status, wOut.driver_wall_time_s);
        catch warmErr
            warmupReport.methods{end+1} = displayNames{m}; %#ok<SAGROW>
            warmupReport.notes{end+1} = sprintf('warm-up failed: %s', warmErr.message); %#ok<SAGROW>
            warning('performance_comparison:WarmupFailed', ...
                'Warm-up for %s failed (%s). Timing may include startup cost.', ...
                displayNames{m}, warmErr.message);
        end
    end
    fprintf('\n');
end

% -------------------------------------------------------------------------
% Run all (resolution × method) combinations, averaged over nSamples runs
% -------------------------------------------------------------------------
for r = 1:nRes
    if ~isFinalCampaign
        data.domain.mesh.nelx = resolutions(r, 1);
        data.domain.mesh.nely = resolutions(r, 2);
    end

    for m = 1:nMethods
        if ~isFinalCampaign
            data.optimization.approach = solverApproaches{m};
        end

        % Refused Olhoff rows are not executed.  Running one would produce a
        % number that looks like every other number in the table, and the only
        % thing standing between it and a scaling fit would be a reader
        % noticing a footnote.
        if ~isempty(olhoffMethodIdx) && m == olhoffMethodIdx && ~olhoffAdmitted(r)
            gi = find(strcmp({olhoffGate.mesh}, ...
                sprintf('%dx%d', resolutions(r,1), resolutions(r,2))), 1);
            fprintf('SKIPPING %-18s mesh %4dx%-3d -- %s (%s)\n', methodLabels{m}, ...
                resolutions(r,1), resolutions(r,2), olhoffGate(gi).row_class, ...
                olhoffGate(gi).status);
            stopReason_all{r,m} = olhoffGate(gi).row_class;
            continue
        end

        omega_s = NaN(1, nSamples);
        tIter_s = NaN(1, nSamples);
        nIter_s = NaN(1, nSamples);
        mem_s   = NaN(1, nSamples);
        nIterStage1_s = NaN(1, nSamples);
        nIterStage2_s = NaN(1, nSamples);
        nOuter_s = NaN(1, nSamples);
        nInner_s = NaN(1, nSamples);
        tInit_s = NaN(1, nSamples);
        tLoop_s = NaN(1, nSamples);
        tPost_s = NaN(1, nSamples);
        eigLoop_s = NaN(1, nSamples);
        tTotal_s = NaN(1, nSamples);
        stopReason_s = repmat({'N/A'}, 1, nSamples);
        finalMaxChange_s = NaN(1, nSamples);
        finalRmsChange_s = NaN(1, nSamples);
        finalRelObjectiveChange_s = NaN(1, nSamples);
        finalGrayness_s = NaN(1, nSamples);
        convergenceTolerance_s = NaN(1, nSamples);
        status_s = repmat({'NOT_RUN'}, 1, nSamples);
        statusNote_s = repmat({''}, 1, nSamples);
        ok_s = false(1, nSamples);

        for s = 1:nSamples
            fprintf('Running %-18s  mesh %4dx%-3d  sample %d/%d ...\n', ...
                methodLabels{m}, resolutions(r,1), resolutions(r,2), s, nSamples);

            if isFinalCampaign
                % Each method goes to the runner its frozen profile names.
                % Olhoff does NOT go through run_topopt_from_json here.
                caseOut = final_campaign_run_case(approaches{m}, ...
                    finalCampaignConfigs{r, m});
                x           = caseOut.x;
                omega       = caseOut.omega;
                tIter       = caseOut.tIter;
                nIter       = caseOut.nIter;
                mem         = caseOut.mem;
                nIterStage  = caseOut.nIterStage;
                telemetry   = caseOut.telemetry;
                totalWallTime = caseOut.total_wall_time_s;
                status_s{s} = caseOut.status;
                statusNote_s{s} = caseOut.status_note;
                ok_s(s)     = caseOut.ok;
                if strcmp(caseOut.status, 'RUN_ERROR')
                    % One case failing must not throw away the eight that
                    % already ran.  Record it and carry on; it is censored
                    % from every fit by fitEligible_all.
                    fprintf(2, '  RUN_ERROR %s %dx%d: %s\n', methodLabels{m}, ...
                        resolutions(r,1), resolutions(r,2), caseOut.status_note);
                    continue
                end
                fprintf('  status %s -- %s\n', caseOut.status, caseOut.status_note);
            else
                totalWallTic = tic;
                [x, omega, tIter, nIter, mem, nIterStage, telemetry] = run_topopt_from_json(data);
                totalWallTime = toc(totalWallTic);
                status_s{s} = telemetryStatusOr(telemetry, 'LEGACY_MODE_UNCLASSIFIED');
                ok_s(s) = true;
            end

            omega_s(s) = omega(1);
            tIter_s(s) = tIter;
            nIter_s(s) = nIter;
            mem_s(s)   = mem;
            nIterStage1_s(s) = nIterStage.stage1;
            nIterStage2_s(s) = nIterStage.stage2;
            if isfield(telemetry, 'iterations')
                nOuter_s(s) = telemetry.iterations.outer;
                nInner_s(s) = telemetry.iterations.inner;
            end
            tInit_s(s) = telemetry.timing.initialization_time;
            tLoop_s(s) = telemetry.timing.optimization_loop_time;
            tPost_s(s) = telemetry.timing.postprocessing_time;
            if isfield(telemetry.timing, 'eigensolve_time')
                eigLoop_s(s) = telemetry.timing.eigensolve_time;
            end
            tTotal_s(s) = totalWallTime;
            stopReason_s{s} = telemetry.stopping.stop_reason;
            finalMaxChange_s(s) = telemetry.stopping.final_max_density_change;
            finalRmsChange_s(s) = telemetry.stopping.final_rms_density_change;
            finalRelObjectiveChange_s(s) = telemetry.stopping.final_relative_objective_change;
            finalGrayness_s(s) = telemetry.stopping.final_grayness;
            convergenceTolerance_s(s) = telemetry.stopping.convergence_tolerance;

            if strcmpi(approaches{m}, 'Yuksel')
                assert(nIter == nIterStage.stage1 + nIterStage.stage2, ...
                    'performance_comparison:YukselIterationMismatch', ...
                    'Yuksel iter_total must equal iter_stage1 + iter_stage2.');
            end

            runRecordIndex = runRecordIndex + 1;
            runRecords(runRecordIndex) = make_run_record( ...
                display_method_name(approaches{m}), methodLabels{m}, ...
                resolutions(r,1), resolutions(r,2), s, ...
                x, omega, tIter, nIter, mem, nIterStage, telemetry, totalWallTime);
        end

        omega_all(r, m) = mean(omega_s);
        tIter_all(r, m) = mean(tIter_s);
        nIter_all(r, m) = round(mean(nIter_s));
        mem_all(r, m)   = mean(mem_s);
        nIterStage1_all(r, m) = round(mean(nIterStage1_s));
        nIterStage2_all(r, m) = round(mean(nIterStage2_s));
        nOuter_all(r, m) = round(mean(nOuter_s));
        nInner_all(r, m) = round(mean(nInner_s));
        tInit_all(r, m) = mean(tInit_s);
        tLoop_all(r, m) = mean(tLoop_s);
        tPost_all(r, m) = mean(tPost_s);
        eigLoop_all(r, m) = mean(eigLoop_s);
        tTotal_all(r, m) = mean(tTotal_s);
        tReconstructed_all(r, m) = tIter_all(r,m) * nIter_all(r,m);
        stopReason_all{r,m} = strjoin(unique(stopReason_s), '|');
        finalMaxChange_all(r,m) = mean(finalMaxChange_s);
        finalRmsChange_all(r,m) = mean(finalRmsChange_s);
        finalRelObjectiveChange_all(r,m) = mean(finalRelObjectiveChange_s);
        finalGrayness_all(r,m) = mean(finalGrayness_s);
        convergenceTolerance_all(r,m) = mean(convergenceTolerance_s);
        if strcmpi(approaches{m}, 'Yuksel')
            stage1Share_all(r,m) = 100 * nIterStage1_all(r,m) / nIter_all(r,m);
            stage2Share_all(r,m) = 100 * nIterStage2_all(r,m) / nIter_all(r,m);
        end
        status_all{r,m} = strjoin(unique(status_s), '|');
        statusNote_all{r,m} = strjoin(unique(statusNote_s(~cellfun(@isempty, statusNote_s))), '|');
        fitEligible_all(r,m) = all(ok_s);

        % ---- COMMON EVALUATORS, measured OUTSIDE every timing boundary -----
        % study_evaluate_design is the unchanged R3/study evaluator whose hash
        % the preflight checks.  It reports each method's final design under
        % three shared material models (E1/E2/E3), in both the raw-density and
        % the exact-count volume-preserving binary representation.  These are
        % NOT the native frequencies each solver optimizes -- omega_all keeps
        % those -- and the two must never be merged.
        if isFinalCampaign && ~isempty(x) && numel(x) == resolutions(r,1)*resolutions(r,2)
            try
                evaluators_all{r,m} = study_evaluate_design(double(x(:)), ...
                    resolutions(r,1), resolutions(r,2), 0.5);
            catch evalErr
                warning('performance_comparison:EvaluatorFailed', ...
                    'Common evaluator failed for %s %dx%d: %s', methodLabels{m}, ...
                    resolutions(r,1), resolutions(r,2), evalErr.message);
            end
        end
    end
end

% -------------------------------------------------------------------------
% CENSORING
%
% A row that failed, hit a cap, or stopped for an unrecognized reason stays in
% every printed and exported table -- with its status next to it -- and is
% removed from the scaling fits.  tTotal_fit is the ONLY array a fit may read.
tTotal_fit = tTotal_all;
tTotal_fit(~fitEligible_all) = NaN;
if isFinalCampaign
    nCensored = sum(~fitEligible_all(:));
    fprintf('\nRow admission for the scaling fits:\n');
    fprintf('%-20s %-9s %-38s %s\n', 'Method', 'Mesh', 'Status', 'In fit?');
    for r = 1:nRes
        for m = 1:nMethods
            if fitEligible_all(r,m); inFit = 'yes'; else; inFit = 'CENSORED'; end
            fprintf('%-20s %-9s %-38s %s\n', methodLabels{m}, ...
                sprintf('%dx%d', resolutions(r,1), resolutions(r,2)), ...
                status_all{r,m}, inFit);
        end
    end
    fprintf(['%d of %d rows censored.  Censored rows are visible above and in every ' ...
        'artifact, and enter no fit.\n\n'], nCensored, numel(fitEligible_all));
end

% Preserve the legacy reconstructed timing separately.  tTotal_all is now
% the measured wall-clock time around the complete top-level solver call.

% -------------------------------------------------------------------------
% Print performance table (mirrors Table 1 from Yuksel et al.)
% -------------------------------------------------------------------------
sepWidth = 210;
sep = repmat('-', 1, sepWidth);

fprintf('\n');
fprintf('Table 1. Run time comparison between methods for maximizing the first\n');
fprintf('natural frequency of a simply supported beam (8 m x 1 m, vf = 0.5).\n');
fprintf('Results averaged over %d runs.\n', nSamples);
fprintf('\n');
fprintf('%-20s  %-9s  %10s  %8s  %10s  %10s  %10s  %9s  %9s  %10s  %10s  %12s  %12s  %12s  %12s\n', ...
    'Method', 'Mesh', 'iter_total', 'outer', 'inner', 'stage1', 'stage2', ...
    'S1 share', 'S2 share', ...
    'init (s)', 'loop (s)', 'post (s)', 'wall (s)', 's/iter', 'Max RAM MB');
fprintf('%s\n', sep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));

    for m = 1:nMethods
        if isnan(tTotal_all(r,m))
            nIterStr  = 'N/A';
            initTimeStr = 'N/A';
            loopTimeStr = 'N/A';
            postTimeStr = 'N/A';
            wallTimeStr = 'N/A';
            iterStr   = 'N/A';
            ramStr    = 'N/A';
        else
            nIterStr = sprintf('%d',   nIter_all(r,m));
            initTimeStr = sprintf('%.3f', tInit_all(r,m));
            loopTimeStr = sprintf('%.3f', tLoop_all(r,m));
            postTimeStr = sprintf('%.3f', tPost_all(r,m));
            wallTimeStr = sprintf('%.3f', tTotal_all(r,m));
            iterStr  = sprintf('%.2f', tIter_all(r,m));
            ramStr   = sprintf('%.0f', mem_all(r,m));
        end
        if isnan(nIterStage1_all(r,m))
            stage1Str = 'N/A';
            stage1ShareStr = 'N/A';
        else
            stage1Str = sprintf('%d', nIterStage1_all(r,m));
            stage1ShareStr = sprintf('%.1f%%', stage1Share_all(r,m));
        end
        if isnan(nIterStage2_all(r,m))
            stage2Str = 'N/A';
            stage2ShareStr = 'N/A';
        else
            stage2Str = sprintf('%d', nIterStage2_all(r,m));
            stage2ShareStr = sprintf('%.1f%%', stage2Share_all(r,m));
        end
        % Outer/inner split: only methods with a genuine two-level loop report
        % it.  For the Olhoff column (Du-Olhoff 2007 reproduction) outer is the
        % Fig. 1 outer loop and inner the total subproblem solves.
        if isnan(nOuter_all(r,m))
            outerStr = 'N/A';
        else
            outerStr = sprintf('%d', nOuter_all(r,m));
        end
        if isnan(nInner_all(r,m))
            innerStr = 'N/A';
        else
            innerStr = sprintf('%d', nInner_all(r,m));
        end
        fprintf('%-20s  %-9s  %10s  %8s  %10s  %10s  %10s  %9s  %9s  %10s  %10s  %12s  %12s  %12s  %12s\n', ...
            methodLabels{m}, meshStr, nIterStr, outerStr, innerStr, ...
            stage1Str, stage2Str, ...
            stage1ShareStr, stage2ShareStr, initTimeStr, loopTimeStr, ...
            postTimeStr, wallTimeStr, iterStr, ramStr);
    end

    if r < nRes
        fprintf('%s\n', sep);
    end
end

fprintf('%s\n', sep);
fprintf('\n');
fprintf(['Timing definitions: init = configuration/solver setup; loop = timed optimization loops; ' ...
    'post = final modal analysis/reporting; wall = measured around run_topopt_from_json.\n']);
fprintf(['Iteration definitions: iter_total = all optimization iterations; outer/inner apply only to ' ...
    'methods with a two-level loop (Olhoff: Fig. 1 outer loop / subproblem (25) solves); ' ...
    'iter_stage1/stage2 apply only to Yuksel; shares are percentages of iter_total. ' ...
    'N/A means not meaningful.\n']);
if isFinalCampaign
    olhoffCfg = finalCampaignConfigs{1, 1}.optimization;
    fprintf(['Olhoff column: profile %s. Du-Olhoff 2007 clean-room reproduction ' ...
        '(Eq. 22 LP route) at r_min = %g elements, run by %s with the causal S1 ' ...
        'policy: move %g reduced once to %g after %d consecutive native evaluations ' ...
        'with N == 2 and gap12 <= %g. It ends at a FIXED %d-outer work horizon. ' ...
        'THAT ENDPOINT IS NOT NATIVE CONVERGENCE, and no convergence-speed claim ' ...
        'may be read from it; t_iter and the scaling exponent are the meaningful ' ...
        'entries.\n\n'], ...
        olhoffCfg.stabilization.profile_id, olhoffCfg.filter.radius, ...
        olhoffCfg.stabilization.runner, olhoffCfg.stabilization.move_initial, ...
        olhoffCfg.stabilization.move_stabilized, olhoffCfg.stabilization.persistence, ...
        olhoffCfg.stabilization.gap_threshold, olhoffCfg.stabilization.max_iters_expected);
    fprintf(['Yuksel and Proposed columns: profiles %s and %s, both stopped by their ' ...
        'own NATIVE tests. Olhoff fixed-work timing and native-stopped timing are ' ...
        'different quantities and are labelled separately throughout.\n\n'], ...
        finalCampaignProfileIds{2}, finalCampaignProfileIds{3});
else
    fprintf(['Olhoff column: produced by the Du-Olhoff 2007 clean-room reproduction (Eq. 22 LP route, ' ...
        'Matlab/reproduction2007). It is move-saturated by construction, so it always stops at its ' ...
        'outer-iteration budget (%d) rather than on a convergence test; read t_iter and scaling, not ' ...
        'iter_total or wall time. Its move limit is %g, scoped to this method.\n\n'], ...
        data.optimization.repro2007.max_outer, data.optimization.repro2007.move);
end

% -------------------------------------------------------------------------
% YUKSEL TABLE-1 VIEW (diagnostic modes only)
%
% Reproduces the *reporting* layout of Yuksel & Yilmaz (2025) Table 1:
%
%     Method | Mesh size | Run time (total) | Run time (iteration) | Max RAM
%
% Two conventions of that table are reproduced deliberately:
%
%  1. The eigenvalue-per-iteration method (their "dynamic code", our Olhoff
%     column) is reported over a FIXED 200-iteration budget.  Their per-iteration
%     figure is simply total/200, and dividing their printed totals by their
%     printed per-iteration times recovers 200 to three figures at all three of
%     their meshes.
%
%  2. For a method that eigensolves only at the END, the terminal cost is split
%     out into a parenthetical addendum -- their "104.4 s + (43.1 s)*" -- rather
%     than folded into the headline total.  That split is the paper's central
%     claim, so it is preserved rather than summed away.  Here the addendum is
%     the post-loop phase, which is where that terminal eigensolve lives.
%
% The in-loop eigensolve column is OURS, not theirs.  It is added because it
% measures directly what their Table 1 argues indirectly: how much of each
% method's loop time is spent solving eigenproblems.
% Diagnostic modes only.  The final campaign has its own reporting and must not
% emit a table laid out like the one whose 200-iteration convention it rejects.
if ~strcmp(benchmarkMode, 'r3') && ~isFinalCampaign
    fprintf('\n');
    fprintf('=================================================================================================\n');
    fprintf(' YUKSEL TABLE-1 VIEW  --  diagnostic, mode ''%s''\n', benchmarkMode);
    fprintf(' Layout follows Yuksel & Yilmaz (2025) Engineering Computations 42(9), Table 1.\n');
    fprintf(' Their hardware: 4 processors x 16 cores @ 2.29 GHz, 128 GB RAM.  Ours: %s.\n', computer);
    fprintf(' Absolute times are NOT comparable across machines; the ratios and the per-iteration\n');
    fprintf(' costs are what Table 1 is actually about.\n');
    fprintf('=================================================================================================\n');
    fprintf('%-12s %-12s %-26s %-16s %-12s %s\n', ...
        'Method', 'Mesh size', 'Run time (total)', 'Run time (iter)', 'Max RAM', 'In-loop eigensolve');
    fprintf('%s\n', repmat('-', 1, 97));
    for r = 1:nRes
        meshStr = sprintf('%d by %d', resolutions(r,1), resolutions(r,2));
        for m = 1:nMethods
            if isnan(tLoop_all(r,m))
                fprintf('%-12s %-12s %-26s %-16s %-12s %s\n', displayNames{m}, meshStr, ...
                    stopReason_all{r,m}, 'N/A', 'N/A', 'N/A');
                continue
            end
            % Headline total = the optimization loop.  Post-loop work is shown as
            % the paper's parenthetical addendum when it is a material share.
            if isfinite(tPost_all(r,m)) && tPost_all(r,m) > 0.05 * tLoop_all(r,m)
                totalStr = sprintf('%.1f s + (%.1f s)*', tLoop_all(r,m), tPost_all(r,m));
            else
                totalStr = sprintf('%.1f s', tLoop_all(r,m));
            end
            if isfinite(nIter_all(r,m)) && nIter_all(r,m) > 0
                iterStr = sprintf('%.3f s/iter', tLoop_all(r,m) / nIter_all(r,m));
            else
                iterStr = 'N/A';
            end
            if isfinite(mem_all(r,m))
                ramStr = sprintf('%.0f MB', mem_all(r,m));
            else
                ramStr = 'N/A';
            end
            eigStr = 'not separable';
            if isfinite(eigLoop_all(r,m))
                eigStr = sprintf('%.1f s (%.0f%% of loop)', eigLoop_all(r,m), ...
                    100 * eigLoop_all(r,m) / max(tLoop_all(r,m), eps));
            end
            fprintf('%-12s %-12s %-26s %-16s %-12s %s\n', ...
                displayNames{m}, meshStr, totalStr, iterStr, ramStr, eigStr);
        end
        if r < nRes
            fprintf('%s\n', repmat('-', 1, 97));
        end
    end
    fprintf('%s\n', repmat('=', 1, 97));
    fprintf(['Note: * marks the post-loop phase, reported separately exactly as Table 1 does for a\n' ...
             '      method whose eigenvalue calculation is only conducted at the end.\n']);
    fprintf(['Olhoff (their "dynamic code") ran a FIXED %d outer iterations at every mesh, per\n' ...
             '      Yuksel section 6.2 "the optimization process is terminated after 200 iterations".\n'], ...
             data.optimization.repro2007.max_outer);
    if isfield(benchmarkProfileMeta, 'yuksel_table1')
        dv = benchmarkProfileMeta.yuksel_table1.deviations_not_adopted;
        if ~isempty(dv)
            fprintf('Settings from Yuksel section 6.2 deliberately NOT adopted (outer budget only was changed):\n');
            for k = 1:numel(dv)
                fprintf('   - %s\n', dv{k});
            end
        end
    end
    fprintf('\n');
end

% -------------------------------------------------------------------------
% Print Table 1 in the paper's grouped-column layout (Mesh rows, one column
% group per method: t_iter, n_iter, T (s)), and export it as a LaTeX table.
% -------------------------------------------------------------------------
groupLabels = {'Olhoff--Du', 'Yuksel--Yilmaz', 'Proposed'};
paperTexPath = fullfile(outputDir, 'table1_paper_style.tex');
print_table1_paper_style(resolutions, groupLabels, tIter_all, nIter_all, tTotal_all, ...
    paperTexPath, nIterStage1_all, nIterStage2_all);

% -------------------------------------------------------------------------
% Stopping diagnostics and explicit benchmark convergence parameters
% -------------------------------------------------------------------------
fprintf('\nStopping diagnostics (N/A means the metric is not meaningful or unavailable):\n');
fprintf('%s\n', sep);
fprintf('%-20s %-9s %-24s %-38s %12s %12s %12s %12s %12s\n', ...
    'Method', 'Mesh', 'Stop reason', 'Status', 'max dx', 'RMS dx', 'rel obj/freq', ...
    'grayness', 'tol used');
fprintf('%s\n', sep);
for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        fprintf('%-20s %-9s %-24s %-38s %12s %12s %12s %12s %12s\n', ...
            methodLabels{m}, meshStr, stopReason_all{r,m}, status_all{r,m}, ...
            metric_string(finalMaxChange_all(r,m)), ...
            metric_string(finalRmsChange_all(r,m)), ...
            metric_string(finalRelObjectiveChange_all(r,m)), ...
            metric_string(finalGrayness_all(r,m)), ...
            metric_string(convergenceTolerance_all(r,m)));
    end
end
fprintf('%s\n', sep);
if isFinalCampaign
    yk = finalCampaignConfigs{1, 2}.optimization.yuksel;
    fprintf(['Yuksel Stage 1 configuration: stage1_tol=%.17g, stage1_max_iters=%d. ' ...
        'Stage 2 tolerance=%.17g.\n'], yk.stage1_tol, yk.stage1_max_iters, yk.stage2_tol);
    fprintf(['Status vocabulary: VALID_STABILIZED_STATE_AT_FIXED_WORK is the Olhoff ' ...
        'success state and is NOT convergence; NATIVE_CONVERGED is the Yuksel/Proposed ' ...
        'success state; CAP_HIT, SOLVER_FAILURE, UNRECOGNIZED_STOP and RUN_ERROR are ' ...
        'censored from the scaling fits and shown here regardless. The Olhoff ' ...
        '"tol used" column is N/A by construction: that profile has no native ' ...
        'convergence test.\n\n']);
else
    fprintf(['Yuksel Stage 1 configuration: stage1_tol=%.17g, stage1_max_iters=%d. ' ...
        'Stage 2 tolerance=%.17g.\n\n'], ...
        data.optimization.yuksel.stage1_tol, ...
        data.optimization.yuksel.stage1_max_iters, ...
        data.optimization.yuksel.stage2_tol);
end

% -------------------------------------------------------------------------
% Also print achieved natural frequencies for reference
% -------------------------------------------------------------------------
fprintf('Achieved first natural frequency omega_1 [rad/s]:\n');
fprintf('%s\n', sep);
fprintf('%-20s  %-9s  %16s\n', 'Method', 'Mesh size', 'omega_1 (rad/s)');
fprintf('%s\n', sep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        if isnan(omega_all(r,m))
            omStr = 'N/A';
        else
            omStr = sprintf('%.1f', omega_all(r,m));
        end
        fprintf('%-20s  %-9s  %16s\n', methodLabels{m}, meshStr, omStr);
    end
    if r < nRes
        fprintf('%s\n', sep);
    end
end
fprintf('%s\n', sep);

% -------------------------------------------------------------------------
% Save Table 1 as CSV
% -------------------------------------------------------------------------
csvPath = fullfile(outputDir, 'table1_performance.csv');
fid = fopen(csvPath, 'w');
assert(fid >= 0, 'performance_comparison:CsvOpenFailed', 'Cannot open %s for writing.', csvPath);
fprintf(fid, ['Method,Mesh,Iterations,IterStage1,IterStage2,RunTime_s,RunTimePerIter_s,MaxRAM_MB,' ...
    'iter_total,iter_stage1,iter_stage2,stage1_share_pct,stage2_share_pct,' ...
    'initialization_time_s,optimization_loop_time_s,postprocessing_time_s,total_wall_time_s,' ...
    'stop_reason,status,status_note,in_scaling_fit,profile_id,' ...
    'final_max_density_change,final_rms_density_change,final_relative_objective_change,' ...
    'final_grayness,convergence_tolerance_used,' ...
    'outer_iterations,inner_iterations\n']);
for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    for m = 1:nMethods
        if isnan(nIterStage1_all(r,m))
            stage1Csv = '';
        else
            stage1Csv = sprintf('%d', nIterStage1_all(r,m));
        end
        if isnan(nIterStage2_all(r,m))
            stage2Csv = '';
        else
            stage2Csv = sprintf('%d', nIterStage2_all(r,m));
        end
        if isnan(nOuter_all(r,m))
            outerCsv = '';
        else
            outerCsv = sprintf('%d', nOuter_all(r,m));
        end
        if isnan(nInner_all(r,m))
            innerCsv = '';
        else
            innerCsv = sprintf('%d', nInner_all(r,m));
        end
        if isFinalCampaign
            profileIdCsv = finalCampaignProfileIds{m};
        else
            profileIdCsv = benchmarkProfileId;
        end
        if fitEligible_all(r,m); inFitCsv = 'yes'; else; inFitCsv = 'no'; end
        fprintf(fid, ['%s,%s,%d,%s,%s,%.9g,%.9g,%.9g,%d,%s,%s,%s,%s,' ...
            '%.9g,%.9g,%.9g,%.9g,%s,%s,%s,%s,%s,' ...
            '%s,%s,%s,%s,%s,%s,%s\n'], ...
            displayNames{m}, meshStr, nIter_all(r,m), stage1Csv, stage2Csv, ...
            tReconstructed_all(r,m), tIter_all(r,m), mem_all(r,m), ...
            nIter_all(r,m), stage1Csv, stage2Csv, ...
            csv_metric(stage1Share_all(r,m)), csv_metric(stage2Share_all(r,m)), ...
            tInit_all(r,m), tLoop_all(r,m), tPost_all(r,m), tTotal_all(r,m), ...
            csv_text(stopReason_all{r,m}), csv_text(status_all{r,m}), ...
            csv_text(statusNote_all{r,m}), inFitCsv, csv_text(profileIdCsv), ...
            csv_metric(finalMaxChange_all(r,m)), csv_metric(finalRmsChange_all(r,m)), ...
            csv_metric(finalRelObjectiveChange_all(r,m)), csv_metric(finalGrayness_all(r,m)), ...
            csv_metric(convergenceTolerance_all(r,m)), outerCsv, innerCsv);
    end
end
fclose(fid);
fprintf('Table 1 saved to: %s\n', csvPath);

% -------------------------------------------------------------------------
% COMMON EVALUATORS -- exported to their OWN file
%
% Three families of frequency live in this campaign and must never be mixed:
%
%   native            what each solver optimizes, under its own material model
%                     and its own mesh conventions.  Reported as omega_1 above
%                     and in table1_performance.csv.
%   common raw        the final RAW density field re-evaluated under the shared
%                     E1/E2/E3 material models by study_evaluate_design.
%   common binary     the same three models applied to the exact-count,
%                     volume-preserving binary projection of that field.
%
% They answer different questions and disagree by construction, so they get
% separate columns in a separate file rather than a single "omega_1" that a
% reader has to interrogate.  study_evaluate_design is unchanged; the preflight
% checks its hash against the audit provenance.
if isFinalCampaign
    evalCsvPath = fullfile(outputDir, 'common_evaluators.csv');
    fid = fopen(evalCsvPath, 'w');
    assert(fid >= 0, 'performance_comparison:EvalCsvOpenFailed', ...
        'Cannot open %s for writing.', evalCsvPath);
    fprintf(fid, ['Method,Mesh,nelx,nely,status,in_scaling_fit,omega1_native,' ...
        'omega1_common_raw_E1,omega2_common_raw_E1,omega3_common_raw_E1,' ...
        'omega1_common_raw_E2,omega2_common_raw_E2,omega3_common_raw_E2,' ...
        'omega1_common_raw_E3,omega2_common_raw_E3,omega3_common_raw_E3,' ...
        'omega1_common_binary_E1,omega2_common_binary_E1,omega3_common_binary_E1,' ...
        'omega1_common_binary_E2,omega2_common_binary_E2,omega3_common_binary_E2,' ...
        'omega1_common_binary_E3,omega2_common_binary_E3,omega3_common_binary_E3,' ...
        'volume,volume_residual,grayness,gray_fraction_01_09,binary_volume,' ...
        'connected_raw,connected_binary,largest_component_fraction\n']);
    for r = 1:nRes
        meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
        for m = 1:nMethods
            ev = evaluators_all{r,m};
            if fitEligible_all(r,m); inFitCsv = 'yes'; else; inFitCsv = 'no'; end
            fprintf(fid, '%s,%s,%d,%d,%s,%s,%s', displayNames{m}, meshStr, ...
                resolutions(r,1), resolutions(r,2), csv_text(status_all{r,m}), ...
                inFitCsv, csv_metric(omega_all(r,m)));
            models = {'E1','E2','E3'};
            reps = {'raw','binary'};
            for rep = 1:2
                for mm = 1:3
                    for j = 1:3
                        v = NaN;
                        if ~isempty(ev)
                            w = ev.(['omega_' reps{rep} '_' models{mm}]);
                            if numel(w) >= j; v = w(j); end
                        end
                        fprintf(fid, ',%s', csv_metric(v));
                    end
                end
            end
            if isempty(ev)
                fprintf(fid, ',%s,%s,%s,%s,%s,%s,%s,%s\n', 'N/A','N/A','N/A','N/A', ...
                    'N/A','N/A','N/A','N/A');
            else
                fprintf(fid, ',%s,%s,%s,%s,%s,%d,%d,%s\n', ...
                    csv_metric(ev.volume), csv_metric(ev.volume_residual), ...
                    csv_metric(ev.grayness), csv_metric(ev.gray_fraction_01_09), ...
                    csv_metric(ev.binary_volume), ...
                    double(ev.connectivity_raw_05.left_right_connected), ...
                    double(ev.connectivity_binary.left_right_connected), ...
                    csv_metric(ev.connectivity_raw_05.largest_component_fraction));
            end
        end
    end
    fclose(fid);
    fprintf('Common E1/E2/E3 raw and binary evaluators saved to: %s\n', evalCsvPath);
end

% Save per-sample records and the complete benchmark configuration as JSON.
% In final-campaign mode there is no shared task file, so the three metadata
% values below come from the frozen per-method configurations instead.
if isFinalCampaign
    metaDiagnosticsEnabled = false;
    metaYukselStage1Tol = finalCampaignConfigs{1,2}.optimization.yuksel.stage1_tol;
    metaYukselStage1Cap = finalCampaignConfigs{1,2}.optimization.yuksel.stage1_max_iters;
else
    metaDiagnosticsEnabled = data.benchmark.enable_diagnostics;
    metaYukselStage1Tol = data.optimization.yuksel.stage1_tol;
    metaYukselStage1Cap = data.optimization.yuksel.stage1_max_iters;
end
jsonResultsPath = fullfile(outputDir, 'benchmark_results.json');
benchmarkResults = struct();
benchmarkResults.metadata = struct( ...
    'benchmark_entry_point', 'examples/Performance/performance_comparison.m', ...
    'timing_note', ['total_wall_time is measured around the complete run_topopt_from_json call; ' ...
        'legacy_reconstructed_time is retained as average_iteration_time * iter_total.'], ...
    'na_representation', 'JSON null and CSV N/A mean not applicable or unavailable.', ...
    'iteration_fields', ['iter_total counts all optimization iterations; iter_stage1 and ' ...
        'iter_stage2 are Yuksel stages and sum to iter_total; shares are percentages; ' ...
        'outer and inner apply only to methods with a two-level loop (Olhoff: Fig. 1 outer ' ...
        'loop and subproblem (25) solves) and are null otherwise.'], ...
    'diagnostics_enabled', metaDiagnosticsEnabled, ...
    'yuksel_stage1_tolerance', metaYukselStage1Tol, ...
    'yuksel_stage1_iteration_cap', metaYukselStage1Cap);
benchmarkResults.metadata.field_definitions = struct( ...
    'initialization_time_s', 'Configuration parsing, dispatch, and solver setup before optimization.', ...
    'optimization_loop_time_s', 'Time measured inside optimization loops; Yuksel is Stage 1 plus Stage 2.', ...
    'postprocessing_time_s', 'Final modal analysis and other work after optimization loops.', ...
    'total_wall_time_s', 'Caller-side wall time around the complete run_topopt_from_json call.', ...
    'iter_total', 'All executed optimization iterations.', ...
    'iter_stage1', 'Yuksel compliance-stage iterations; N/A for single-stage methods.', ...
    'iter_stage2', 'Yuksel inertial-stage iterations; N/A for single-stage methods.', ...
    'outer', 'Outer-loop iterations for two-level methods; N/A otherwise.', ...
    'inner', 'Total subproblem solves across all outer iterations; N/A otherwise.', ...
    'inner_solver', 'Subproblem solver used for the inner loop, where applicable.', ...
    'final_max_density_change', 'Final maximum absolute design-density change.', ...
    'final_rms_density_change', 'Final RMS design-density change when available.', ...
    'final_relative_objective_change', ...
        'Final relative change in the convergence objective (frequency for Olhoff).', ...
    'final_grayness', 'Mean 4*x*(1-x) of the final physical density field.', ...
    'convergence_tolerance', 'Numerical convergence tolerance actually used; criteria are unchanged.');
% Record which solver actually produced each named column.  The Olhoff column
% is the Du-Olhoff 2007 clean-room reproduction, not analysis/OlhoffApproach;
% without this the JSON would not say so.
benchmarkResults.metadata.method_dispatch = struct();
for m = 1:nMethods
    benchmarkResults.metadata.method_dispatch.( ...
        matlab.lang.makeValidName(approaches{m})) = solverApproaches{m};
end
benchmarkResults.metadata.olhoff_column_note = ['The Olhoff column is produced by ' ...
    'Matlab/reproduction2007 (Du-Olhoff 2007 clean-room reproduction, Eq. 22 LP route). ' ...
    'While its LP subproblem solves successfully it is move-saturated and stops at its ' ...
    'outer-iteration budget, so iter_total is a fixed budget rather than a convergence ' ...
    'result.  A stop reason other than max_outer_iterations means the subproblem FAILED; ' ...
    'telemetry.stopping.status reports SOLVER_FAILURE in that case and the row is not a ' ...
    'timing result.'];
benchmarkResults.metadata.benchmark_profile_id = benchmarkProfileId;
benchmarkResults.metadata.benchmark_profile = benchmarkProfileMeta;
% The admission record for the Olhoff column: which meshes were proved to run
% the clean-room implementation through the benchmark path, and which were
% refused.  Refused meshes were not executed and carry no timing figures.
benchmarkResults.metadata.olhoff_equivalence_gate = olhoffGate;
benchmarkResults.metadata.olhoff_rows_admitted = olhoffAdmitted(:).';
benchmarkResults.metadata.olhoff_equivalence_note = ['An Olhoff timing/scaling row is ' ...
    'admitted only after verify_repro2007_benchmark_equivalence has proved, for that ' ...
    'mesh and this exact profile, source commit, frozen implementation and MATLAB ' ...
    'release, that the benchmark-dispatched path reproduces a direct call to the ' ...
    'clean-room reproduction bit for bit.  See OLHOFF_BENCHMARK_EQUIVALENCE_REPORT.md. ' ...
    'This gate applies to the LEGACY dispatched path only; the final campaign calls ' ...
    'the runner named in final_campaign_profile.json directly and is admitted by ' ...
    'final_campaign_preflight instead.'];
benchmarkResults.metadata.environment = struct( ...
    'matlab_version', version, 'computer', computer, ...
    'comp_threads_requested', threadsRequested, 'comp_threads_active', threadsActive, ...
    'n_samples', nSamples);
if isFinalCampaign
    benchmarkResults.metadata.method_dispatch = struct( ...
        'Olhoff', char(finalCampaignConfigs{1,1}.optimization.stabilization.runner), ...
        'Yuksel', 'tools/Matlab/run_topopt_from_json.m (Yuksel)', ...
        'OurApproach', 'tools/Matlab/run_topopt_from_json.m (ourApproach)');
    benchmarkResults.metadata.campaign = struct( ...
        'campaign_id', char(campaignManifest.campaign_id), ...
        'manifest', 'analysis/olhoff_stabilization_audit/final_campaign_profile.json', ...
        'profile_ids', struct( ...
            'Olhoff', finalCampaignProfileIds{1}, ...
            'Yuksel', finalCampaignProfileIds{2}, ...
            'Proposed', finalCampaignProfileIds{3}), ...
        'preflight', campaignGate, ...
        'warmup', warmupReport);
    benchmarkResults.metadata.olhoff_column_note = ['The Olhoff column is the ' ...
        'stabilized S1 profile from analysis/olhoff_stabilization_audit/: the ' ...
        'Du-Olhoff 2007 clean-room reproduction at r_min = 1.3 elements with a causal ' ...
        'move reduction 0.005 -> 0.0025, triggered by 100 consecutive native ' ...
        'evaluations with N == 2 and gap12 <= 1%. It ends at a FIXED 1600 outer ' ...
        'iterations. THE ENDPOINT IS NOT NATIVE CONVERGENCE and the trigger is not a ' ...
        'convergence test; no convergence-speed claim may be inferred from it. This is ' ...
        'NOT the legacy S0 / r_min = 2 benchmark profile.'];
    benchmarkResults.metadata.status_semantics = struct( ...
        'precedence', {cellstr(campaignManifest.status_precedence(:))}, ...
        'olhoff_success', 'VALID_STABILIZED_STATE_AT_FIXED_WORK', ...
        'native_success', 'NATIVE_CONVERGED', ...
        'censored', {{'SOLVER_FAILURE','CAP_HIT','UNRECOGNIZED_STOP','RUN_ERROR','NOT_RUN'}}, ...
        'note', ['Censored rows are reported in full in every table and artifact and ' ...
            'are excluded from every scaling fit. A cap hit is not convergence and a ' ...
            'solver failure is not a timing result.']);
    benchmarkResults.metadata.evaluator_separation = struct( ...
        'native', 'Solver-native omega under each method own material model.', ...
        'common_raw', 'study_evaluate_design E1/E2/E3 on the final raw density field.', ...
        'common_binary', ['study_evaluate_design E1/E2/E3 on the exact-count ' ...
            'volume-preserving binary projection.'], ...
        'file', 'common_evaluators.csv', ...
        'evaluator_source', 'analysis/three_method_parametric_study/study_evaluate_design.m', ...
        'evaluator_changed', false);
    benchmarkResults.metadata.timing_protocol = struct( ...
        'threads_per_run', threadsRequested, ...
        'samples_per_case', nSamples, ...
        'warm_up', 'one discarded solve per method at 40x5, outside the campaign mesh set', ...
        'olhoff_total_wall_time', ['solver-side wallclock inside olhoffOptStabilized; ' ...
            'excludes the evidence .mat save'], ...
        'olhoff_loop_time', 'sum of per-iteration eigensolve + gradient + subproblem time', ...
        'olhoff_init_post', 'not separable inside the frozen audit runner; reported null', ...
        'dispatched_total_wall_time', 'caller-side tic/toc around run_topopt_from_json', ...
        'comparability_warning', ['The stabilization audit measured its timings with ' ...
            'several single-thread MATLAB processes in parallel. Those numbers are ' ...
            'descriptive only and must NOT be compared directly with this serial ' ...
            'campaign.']);
end
% Per-case status and configuration, each row NAMING its method and mesh.
% A bare 2-D array would arrive in JSON as a flat list whose order a reader has
% to reconstruct, which is how a status gets attached to the wrong method.
caseStatus = struct('method', {}, 'mesh', {}, 'nelx', {}, 'nely', {}, ...
    'profile_id', {}, 'status', {}, 'status_note', {}, 'in_scaling_fit', {}, ...
    'total_wall_time_s', {}, 'iterations', {});
for r = 1:nRes
    for m = 1:nMethods
        if isFinalCampaign
            pid = finalCampaignProfileIds{m};
        else
            pid = benchmarkProfileId;
        end
        caseStatus(end+1) = struct( ...
            'method', displayNames{m}, ...
            'mesh', sprintf('%dx%d', resolutions(r,1), resolutions(r,2)), ...
            'nelx', resolutions(r,1), 'nely', resolutions(r,2), ...
            'profile_id', pid, ...
            'status', status_all{r,m}, 'status_note', statusNote_all{r,m}, ...
            'in_scaling_fit', fitEligible_all(r,m), ...
            'total_wall_time_s', tTotal_all(r,m), ...
            'iterations', nIter_all(r,m)); %#ok<SAGROW>
    end
end
% Assigned field by field: struct() with a struct-array value is ambiguous.
benchmarkResults.results = struct();
benchmarkResults.results.cases = caseStatus;
benchmarkResults.results.meshes = resolutions;
if isFinalCampaign
    perCase = struct('method', {}, 'mesh', {}, 'configuration', {});
    for r = 1:nRes
        for m = 1:nMethods
            perCase(end+1) = struct( ...
                'method', displayNames{m}, ...
                'mesh', sprintf('%dx%d', resolutions(r,1), resolutions(r,2)), ...
                'configuration', finalCampaignConfigs{r,m}); %#ok<SAGROW>
        end
    end
    benchmarkResults.configuration = struct();
    benchmarkResults.configuration.note = ...
        'One frozen configuration per (mesh, method); no shared task file.';
    benchmarkResults.configuration.meshes = resolutions;
    benchmarkResults.configuration.per_case = perCase;
else
    benchmarkResults.configuration = data;
end
benchmarkResults.runs = runRecords;
fid = fopen(jsonResultsPath, 'w');
assert(fid >= 0, 'performance_comparison:JsonOpenFailed', ...
    'Cannot open %s for writing.', jsonResultsPath);
fprintf(fid, '%s\n', jsonencode(benchmarkResults));
fclose(fid);
fprintf('Per-run benchmark results saved to: %s\n', jsonResultsPath);

% -------------------------------------------------------------------------
% Fit computational-complexity model T(N_e) = C * N_e^exp per method.
% N_e = nelx*nely is the number of finite elements in the mesh.
% -------------------------------------------------------------------------
Ne = resolutions(:,1) .* resolutions(:,2);
outDir = outputDir;

% ---- Table 2: free fit -- both C and exp estimated by least-squares
% linear regression on log(T) = log(C) + exp*log(N_e). ----
% tTotal_fit, not tTotal_all: censored rows are NaN here and drop out of the
% fit, while staying visible in every table above.
[complexity_C, complexity_exp, complexity_R2, complexity_n] = ...
    fit_complexity_model(Ne, tTotal_fit, 'free');

complexityCsvPath = fullfile(outDir, 'table1_complexity_fit.csv');
print_complexity_fit_table(methodLabels, displayNames, complexity_C, complexity_exp, ...
    complexity_R2, complexity_n, ...
    {'Table 2. Computational complexity fit  T(N_e) = C * N_e^exp', ...
     '(least-squares fit of log(T) vs log(N_e); N_e = nelx*nely)'}, ...
    complexityCsvPath);

% ---- Table 3: fixed-exponent fit -- exp is held fixed at an arbitrarily
% chosen value (default 1.5) and only C (the prefactor) is estimated by
% least squares. ----
fixedExp = 1.5;
[complexity_C_fixed, complexity_exp_fixed, complexity_R2_fixed, complexity_n_fixed] = ...
    fit_complexity_model(Ne, tTotal_fit, 'fixed', fixedExp);

complexityCsvPathFixed = fullfile(outDir, 'table1_complexity_fit_fixedexp.csv');
print_complexity_fit_table(methodLabels, displayNames, complexity_C_fixed, complexity_exp_fixed, ...
    complexity_R2_fixed, complexity_n_fixed, ...
    {sprintf('Table 3. Fixed-exponent complexity fit  T(N_e) = C * N_e^%.2f', fixedExp), ...
     '(exponent held fixed; only C estimated by linear-space least squares on T, i.e. minimizing', ...
     'absolute run-time error sum((T - C*N_e^exp)^2); R^2 is on T, not log(T))'}, ...
    complexityCsvPathFixed);

% -------------------------------------------------------------------------
% Plot measured run times (Table 1 points) together with the fitted
% power-law curves, on both log-log and linear axes -- once for the free
% fit (Table 2), once for the fixed-exponent fit (Table 3).
% -------------------------------------------------------------------------
plot_table1_complexity(Ne, methodLabels, tTotal_fit, complexity_C, complexity_exp, outDir);

plot_table1_complexity(Ne, methodLabels, tTotal_fit, complexity_C_fixed, complexity_exp_fixed, ...
    outDir, 'table1_complexity_fit_fixedexp', ...
    sprintf('Fixed-exponent fit (C estimated only):  T(N_e) = C \\cdot N_e^{%.2f}', fixedExp));

% -------------------------------------------------------------------------
% Campaign admission record and the closing manifest of what was written.
% -------------------------------------------------------------------------
if isFinalCampaign
    gatePath = fullfile(outputDir, 'campaign_gate.json');
    gateOut = campaignGate;
    gateOut.run_completed_at = datestr(now, 'yyyy-mm-ddTHH:MM:SS'); %#ok<TNOW1,DATST>
    gateOut.rows_in_scaling_fit = fitEligible_all;
    gateOut.rows_censored = sum(~fitEligible_all(:));
    fid = fopen(gatePath, 'w');
    if fid >= 0
        fprintf(fid, '%s\n', jsonencode(gateOut));
        fclose(fid);
    end

    fprintf('\n================================================================\n');
    fprintf(' FINAL CAMPAIGN COMPLETE -- artifacts written under\n   %s\n', outputDir);
    fprintf('   table1_performance.csv          per-case timing, status, censoring\n');
    fprintf('   common_evaluators.csv           native vs common raw vs common binary E1/E2/E3\n');
    fprintf('   benchmark_results.json          per-run records and frozen configurations\n');
    fprintf('   table1_paper_style.tex          paper-layout table\n');
    fprintf('   table1_complexity_fit*.csv/.png scaling fits, censored rows excluded\n');
    fprintf('   campaign_gate.json              preflight verdict and admission record\n');
    fprintf('   raw/olhoff/s1_<mesh>.mat        per-mesh Olhoff trajectory evidence\n');
    fprintf('   warmup/                         discarded warm-up solves\n');
    fprintf(' Nothing outside this directory was written.\n');
    fprintf('================================================================\n');
end

function record = make_run_record(methodName, methodLabel, nelx, nely, sample, ...
        x, omega, tIter, nIter, mem, nIterStage, telemetry, totalWallTime)
    if isfinite(nIterStage.stage1) && nIter > 0
        stage1Share = 100 * nIterStage.stage1 / nIter;
        stage2Share = 100 * nIterStage.stage2 / nIter;
    else
        stage1Share = NaN;
        stage2Share = NaN;
    end
    record = struct();
    record.method = methodName;
    record.method_label = methodLabel;
    record.mesh = struct('nelx', nelx, 'nely', nely, 'elements', nelx*nely);
    record.sample = sample;
    outerIters = NaN;
    innerIters = NaN;
    innerSolver = 'N/A';
    if isfield(telemetry, 'iterations')
        outerIters  = telemetry.iterations.outer;
        innerIters  = telemetry.iterations.inner;
        innerSolver = telemetry.iterations.inner_solver;
    end
    record.iterations = struct( ...
        'iter_total', nIter, ...
        'iter_stage1', nIterStage.stage1, ...
        'iter_stage2', nIterStage.stage2, ...
        'stage1_share_pct', stage1Share, ...
        'stage2_share_pct', stage2Share, ...
        'outer', outerIters, ...
        'inner', innerIters, ...
        'inner_solver', innerSolver);
    % Stage loop times are computed by the Yuksel solver and were previously
    % dropped here.  Without them the stage split is reported in iterations
    % only, and "how much of the run time is stage 1" cannot be answered from
    % the archive.  NaN for single-stage methods.
    stage1Time = NaN;
    stage2Time = NaN;
    if isfield(telemetry, 'yuksel')
        if isfield(telemetry.yuksel, 'stage1_loop_time')
            stage1Time = telemetry.yuksel.stage1_loop_time;
        end
        if isfield(telemetry.yuksel, 'stage2_loop_time')
            stage2Time = telemetry.yuksel.stage2_loop_time;
        end
    end
    eigTimeRec = NaN;
    if isfield(telemetry.timing, 'eigensolve_time')
        eigTimeRec = telemetry.timing.eigensolve_time;
    end
    record.timing = struct( ...
        'initialization_time_s', telemetry.timing.initialization_time, ...
        'optimization_loop_time_s', telemetry.timing.optimization_loop_time, ...
        'postprocessing_time_s', telemetry.timing.postprocessing_time, ...
        'total_wall_time_s', totalWallTime, ...
        'runner_reported_total_wall_time_s', telemetry.timing.total_wall_time, ...
        'legacy_reconstructed_time_s', tIter*nIter, ...
        'average_iteration_time_s', tIter, ...
        'eigensolve_time_s', eigTimeRec, ...
        'stage1_loop_time_s', stage1Time, ...
        'stage2_loop_time_s', stage2Time, ...
        ... % Identity checks, recorded rather than asserted: a broken identity
        ... % is a fact about the run that belongs in the archive.
        'stage_time_sum_residual_s', ...
            stage1Time + stage2Time - telemetry.timing.optimization_loop_time, ...
        'loop_within_total_s', totalWallTime - telemetry.timing.optimization_loop_time);
    record.stopping = telemetry.stopping;
    record.configuration = struct( ...
        'diagnostics_enabled', telemetry.diagnostics_enabled, ...
        'convergence_tolerance', telemetry.stopping.convergence_tolerance, ...
        'yuksel_stage1_max_iters', telemetry.yuksel.stage1_max_iters, ...
        'yuksel_stage1_tolerance', telemetry.yuksel.stage1_tolerance, ...
        'yuksel_stage2_tolerance', telemetry.yuksel.stage2_tolerance);
    record.results = struct( ...
        'objective_final', telemetry.objective_final, ...
        'objective_history_checksum', numeric_fingerprint(telemetry.objective_history), ...
        'final_frequencies_rad_s', omega(:)', ...
        'topology_checksum', numeric_fingerprint(x));
    record.max_ram_mb = mem;
end

function st = telemetryStatusOr(telemetry, fallback)
% Status as the solver reported it, or FALLBACK when it reported none.  The
% legacy modes predate the status vocabulary; they must not silently borrow it.
    st = fallback;
    if isstruct(telemetry) && isfield(telemetry, 'stopping') && ...
            isfield(telemetry.stopping, 'status') && ...
            ~isempty(telemetry.stopping.status) && ...
            ~strcmp(telemetry.stopping.status, 'N/A')
        st = char(string(telemetry.stopping.status));
    end
end

function name = display_method_name(approach)
    if strcmpi(approach, 'OurApproach')
        name = 'Proposed';
    else
        name = approach;
    end
end

function value = numeric_fingerprint(x)
    x = double(x(:));
    if isempty(x)
        value = 'N/A';
        return;
    end
    weights = (1:numel(x))';
    value = sprintf('n=%d;sum=%.17g;weighted=%.17g;l2=%.17g', ...
        numel(x), sum(x), sum(weights.*x), norm(x));
end

function value = metric_string(x)
    if isempty(x) || ~isfinite(x)
        value = 'N/A';
    else
        value = sprintf('%.5e', x);
    end
end

function value = csv_metric(x)
    if isempty(x) || ~isfinite(x)
        value = 'N/A';
    else
        value = sprintf('%.17g', x);
    end
end

function value = csv_text(s)
% Free text in a CSV cell: commas, quotes and newlines would otherwise shift
% every following column, which is exactly how a status note turns into a
% timing number in someone's spreadsheet.
    if isempty(s)
        value = '';
        return;
    end
    value = char(string(s));
    value = strrep(value, char(10), ' ');
    value = strrep(value, char(13), ' ');
    if any(value == ',') || any(value == '"')
        value = ['"' strrep(value, '"', '""') '"'];
    end
end
