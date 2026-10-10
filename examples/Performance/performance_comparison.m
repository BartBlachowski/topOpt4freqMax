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
%   counts and component times explain the architecture behind it. The table
%   reports Stage time = Time 1 + Time 2, Other, and Total wall time; Other
%   includes setup and final eigenanalysis. No solver timer is changed by
%   these derived columns. See
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

% ---- Guards --------------------------------------------------------------
% Running anything above 160x20 costs minutes to hours per row.  This must be
% true to launch the nine-resolution conference campaign selected above; the
% mesh list stays visible and editable either way.
cfg.confirmLongCampaign = true;

% Truncated outer budget for MECHANICS-ONLY smoke tests.  [] = the methods'
% own frozen budgets.  Any value here marks the whole run non-scientific.
cfg.maxOuterOverride = [];

% Yuksel per-stage SAFETY budget, applied to BOTH stages.  [] = the frozen
% profile value (1000).
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
% This is a SAFETY budget, not a stopping rule: a stage that reaches it is
% CAP_HIT and NOT converged, and confbench_run_case now detects that
% NUMERICALLY from the actual per-stage counts rather than from the overall
% textual stop reason.  RAISING a safety budget does not make a run
% non-scientific; LOWERING it below the frozen value is truncation, and is
% treated exactly like cfg.maxOuterOverride.
cfg.yukselMaxIters = 5000;

% ---- Stopping rules ------------------------------------------------------
% [] = the PRODUCTION setting: the frozen rule the Table 1 campaign ran with,
% whose value is given in the comment.  Setting a value changes WHERE a method
% stops, so the run is no longer the production regime; it stays a valid run
% and is recorded as such (run_class.production_stop_rules = false in
% benchmark_manifest.json, and in every Olhoff row's effective configuration
% hash).  As for cfg.yukselMaxIters, RAISING a safety budget keeps the run
% scientific and LOWERING it below the production value is truncation, treated
% like cfg.maxOuterOverride.  (2026-10-07, simply supported beam, 160x20 to
% 400x50: proposed.tol = 0.025 and olhoff.c = 0.2 stop where the design has
% stagnated; see examples/bimodality.)
%
% criterion selects WHAT each tolerance is compared with, on the design
% variable x (rho for Du-Olhoff); [] = the method's production criterion:
%   'max_change'          max|x - x_old|                  (Proposed, Yuksel production)
%   'relative_l2_change'  ||x - x_old||_2 / ||x_old||_2 < tol, x_old the design
%                         before the update.  It has NO production tolerance:
%                         the tolerances it uses must be set.
%   'l2_change'           Du-Olhoff only, its production rule (below)
%   'stagnation'          every method: stop once, over the last window+1
%                         analysed designs, the method's own objective has varied
%                         by < objectiveTol (relative to its latest value) AND the
%                         grayness 4*mean(x(1-x)) by < graynessTol (absolute).
%                         No tol/c is read.  Du-Olhoff's objective is omega_n.
cfg.stop = struct();

% ---- ACTIVE: one stopping rule for all three methods (2026-10-10) ----------
% Rule R1 of examples/Performance/stop_criterion_study/REPORT.tex: replayed on
% 23 recorded histories (160x20 ... 800x100) it fired on every one, 0.48-2.68x
% the stagnation iteration, losing at most 0.91 % of omega_1 against 300 (600 for
% Yuksel stage 2) further iterations.  The thresholds were calibrated on that
% benchmark and are to be kept frozen, not re-tuned per case.  A live run stops
% one iteration after the replayed k*: the window holds designs whose objective
% is known, which trails the update by one.
R1 = struct('window', 10, 'objectiveTol', 1e-3, 'graynessTol', 5e-3);

% Proposed
cfg.stop.proposed.criterion   = 'stagnation';   % production: 'max_change'
cfg.stop.proposed.tol         = [];             % not read by 'stagnation'
cfg.stop.proposed.window      = R1.window;
cfg.stop.proposed.objectiveTol = R1.objectiveTol;
cfg.stop.proposed.graynessTol = R1.graynessTol;
cfg.stop.proposed.maxIters    = [];   % production: 2000  (safety budget)

% Yuksel.  Stage 1 keeps the published rule max|x - x_old| < 0.01: its
% tolerance decides the design stage 2 starts from, so it is part of the method,
% and it is the stage-1 rule the replay was recorded with.  R1 ends stage 2.
cfg.stop.yuksel.criterion     = 'stagnation';   % production: 'max_change' (both stages)
cfg.stop.yuksel.stage1Criterion = 'max_change'; % stage 1 only
cfg.stop.yuksel.stage1Tol     = 0.01;           % production: 0.01 with max_change
cfg.stop.yuksel.stage2Tol     = [];             % not read by 'stagnation'
cfg.stop.yuksel.window        = R1.window;
cfg.stop.yuksel.objectiveTol  = R1.objectiveTol;
cfg.stop.yuksel.graynessTol   = R1.graynessTol;

% Du-Olhoff
cfg.stop.olhoff.criterion     = 'stagnation';   % production: 'l2_change'
cfg.stop.olhoff.c             = [];             % not read by 'stagnation'
cfg.stop.olhoff.tol           = [];             % not read by 'stagnation'
cfg.stop.olhoff.window        = R1.window;
cfg.stop.olhoff.objectiveTol  = R1.objectiveTol;
cfg.stop.olhoff.graynessTol   = R1.graynessTol;
cfg.stop.olhoff.maxOuter      = 1000;   % production: 400   (preset runtime default; safety budget)

% ---- PREVIOUS per-method settings (campaign_mac_convergence_relative_l2_change)
% Kept for reference; uncomment one block per method (and drop the matching
% ACTIVE block above) to return to it.
%
% % Proposed: stops when the criterion falls to tol (max_change: <= tol).
% cfg.stop.proposed.criterion = [];     % production: 'max_change'
% cfg.stop.proposed.tol      = 0.04;   % production: 0.01 with max_change  (profile proposed_practical_move02_tol001)
% cfg.stop.proposed.maxIters = [];   % production: 2000  (safety budget)
%
% % Yuksel: each stage stops when the criterion is below its tolerance, from its
% % second iteration on, and the run ends when stage 2 stops.  stage1Tol also
% % sets the design stage 2 starts from.  Per-stage budget: cfg.yukselMaxIters.
% cfg.stop.yuksel.criterion  = [];     % production: 'max_change' (both stages)
% cfg.stop.yuksel.stage1Tol  = 0.04;   % production: 0.01 with max_change  (profile yuksel_practical_move01_tol001)
% cfg.stop.yuksel.stage2Tol  = 0.04;   % production: 0.01 with max_change
%
% % Du-Olhoff.  criterion = 'l2_change' (production): stops when
% % ||drho||_2 < c*sqrt(NE/3200) (sec. 3.5 of the paper, which gives no value for
% % epsilon; c and the mesh scaling are this reconstruction's,
% % olh.config.epsilonForMesh); tol is ignored.  'max_change' (max|drho| <= tol)
% % or 'relative_l2_change' (||drho||_2/||rho||_2 < tol): no mesh scaling and no
% % guards; c is ignored.
% cfg.stop.olhoff.criterion  = 'relative_l2_change';   % production: 'l2_change'
% cfg.stop.olhoff.c          = 0.08;   % production: 0.05  (l2_change)
% cfg.stop.olhoff.tol        = 1e-3; %0.04;   % no production value (max_change / relative_l2_change)
% cfg.stop.olhoff.maxOuter   = 1000;   % production: 400   (preset runtime default; safety budget)

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

cfg.outputDir = '';                  % auto: examples/Performance/conference_benchmark/<runLabel>

% The label IS the output directory, so it is the only thing standing between a
% new campaign and the artifacts of an earlier one.  Every writer in this driver
% (confbench_export, confbench_complexity_plots, confbench_topology_images,
% confbench_complexity_diagnostics) is confined to cfg.outputDir, and the
% repoRoot/results figure writer in run_topopt_from_json stays off because
% confbench_method_config leaves postprocessing.save_frequency_iterations at its
% false default.  So a fresh label is SUFFICIENT to protect earlier evidence --
% and a reused one silently buries it, which the preflight can only warn about.
%
% 'campaign_9mesh_r2' is NOT reused here on purpose.  That directory already
% holds a completed nine-mesh campaign whose Olhoff column was produced at the
% SUPERSEDED preset duOlhoffFixedPenaltySensitivityFiltered (the M4 fixed-penalty
% formulation; see its benchmark_manifest.json).  This driver now runs
% duOlhoffPedersenAdaptiveBoxSensitivityFiltered, a DISTINCT formulation, so
% writing into that directory would leave one label covering two different
% material laws and controllers.  The comparable local reference is
% conference_benchmark/nine_mesh_pedersen_b21483b.
%
% 2026-09-28: recomputation on a second machine (Windows 11, MATLAB R2024a,
% hostname BIO-2_HELI) at a paper reviewer's request, to show the reported
% scaling is not an artifact of one host.  The label records the machine because
% the quantity being reported is wall-clock time, which is a property of the
% host as much as of the method.
% cfg.runLabel  = 'campaign_mac_convergence_relative_l2_change';   % previous stop rules
cfg.runLabel  = 'campaign_mac_convergence_stagnation_r1';

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
addpath(fullfile(scriptDir, 'benchmark_profile'));   % frozen Proposed/Yuksel profile, base config, common evaluator
% THE production Du-Olhoff implementation.  analysis/Olhoff (named
% analysis/OlhoffCurrent until the 2026-09-14 repository cleanup) is the ONLY
% Olhoff implementation this driver -- or any production script -- may execute.
% Its solver core lives under +impl/ and is reachable ONLY through
% olhoffcurrent_paths(), which proves that exactly one Olhoff implementation is
% visible before anything runs.  No historical Olhoff tree is added here, by
% design; all of them are archived under development/, which the gate forbids.
addpath(fullfile(repoRoot, 'analysis', 'Olhoff'));

% Not adding a superseded implementation is not enough: MATLAB paths are
% session state, and archived scripts call addpath(genpath(...)), which can
% leave historical Olhoff trees on the path for the rest of the session.
% olhoffOpt then resolves to whichever of the realizations came first -- a run
% that looks fine and is scientifically void.  This driver curates its own
% path, so it REMOVES what the session handed it rather than inheriting it.  The
% preflight below re-checks the result independently, so a scrub that missed
% something still fails closed; the scrub is recorded in the benchmark manifest.
pathScrub = olhoffcurrent_scrub_forbidden_paths(repoRoot);
if ~isempty(pathScrub)
    fprintf('Removed %d non-production Olhoff path entr%s inherited from this MATLAB session.\n', ...
        numel(pathScrub), pluralIes(numel(pathScrub)));
end

% Pinning threads is a property of the MEASUREMENT, not of the user's session,
% so the entry value is captured and restored again on the way out.
%
% This file is a SCRIPT, and an onCleanup object in a script's workspace is NOT
% destroyed when the script ends -- it survives in the base workspace until
% something clears it.  (Measured: after the script returns the pin is still in
% force; it lifts only on `clear`.)  So the guard below is a BACKSTOP for an
% uncaught error mid-campaign -- it fires at the `clear` on the next run -- and
% the deterministic restores are the two explicit calls to restoreThreads():
% one before the preflight bail-out, one at the end of the script.
entryThreads = maxNumCompThreads();
threadGuard = onCleanup(@() maxNumCompThreads(entryThreads)); %#ok<NASGU>
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

% The stopping rules.  The other two budgets follow the Yuksel rule above, read
% from the same frozen sources; a changed tolerance leaves the run scientific
% but not the production regime.
validateStop(cfg.stop);
budgetTruncated = yukselBudgetTruncated ...
    || belowBudget(cfg.stop.proposed.maxIters, confbench_frozen_budget('proposed')) ...
    || belowBudget(cfg.stop.olhoff.maxOuter, confbench_frozen_budget('olhoff'));
cfg.productionStopRules = isempty(cfg.stop.proposed.tol) && isempty(cfg.stop.yuksel.stage1Tol) ...
    && isempty(cfg.stop.yuksel.stage2Tol) && isempty(cfg.stop.olhoff.c) ...
    && isProductionCriterion(cfg.stop.proposed.criterion, 'max_change') ...
    && isProductionCriterion(cfg.stop.yuksel.criterion, 'max_change') ...
    && isProductionCriterion(fieldOr(cfg.stop.yuksel, 'stage1Criterion', []), 'max_change') ...
    && isProductionCriterion(cfg.stop.olhoff.criterion, 'l2_change');

cfg.scientificEvidence  = isempty(cfg.maxOuterOverride) && all(elementCounts >= 3200) ...
    && ~budgetTruncated;
cfg.performanceCampaign = cfg.scientificEvidence && isequal(cfg.resolutions, CAMPAIGN_MESHES);

if isempty(cfg.runLabel)
    if ~isempty(cfg.maxOuterOverride) || budgetTruncated
        cfg.runLabel = 'smoke';
    elseif cfg.performanceCampaign
        cfg.runLabel = 'campaign_9mesh';
    elseif size(cfg.resolutions,1) == 1
        cfg.runLabel = sprintf('preflight_%dx%d', cfg.resolutions(1,1), cfg.resolutions(1,2));
    else
        cfg.runLabel = sprintf('partial_%dmesh', size(cfg.resolutions,1));
    end
    if ~cfg.productionStopRules
        cfg.runLabel = [cfg.runLabel '_stoprules'];
    end
end
if isempty(cfg.outputDir)
    cfg.outputDir = fullfile(scriptDir, 'conference_benchmark', cfg.runLabel);
end

% EVERY campaign gets a directory of its own.  A rerun under the same label
% otherwise lands on top of the previous attempt, and the preflight can only
% WARN about that -- which is no protection for an 8-hour campaign launched over
% ssh and restarted after a dropped connection, precisely when a restart is most
% likely and the earlier artifacts are most valuable.  So exclusivity is
% enforced here instead of trusted: a non-empty target is not written into, the
% next free _2, _3, ... is taken, and cfg.runLabel follows the directory so that
% the manifest, the run label in runOpts and the artifacts all name the same
% place.  Candidates are derived from the target's own parent, so this works
% whether cfg.outputDir was auto-derived above or set by hand.
if ~isEmptyDir(cfg.outputDir)
    [parentDir, baseName] = fileparts(cfg.outputDir);
    n = 1;
    while true
        n = n + 1;
        candidate = fullfile(parentDir, sprintf('%s_%d', baseName, n));
        if isEmptyDir(candidate); break; end
    end
    fprintf(['Output directory %s already holds artifacts; this run writes to ' ...
        '%s instead, so the earlier run is preserved.\n'], cfg.outputDir, candidate);
    cfg.outputDir = candidate;
    cfg.runLabel  = sprintf('%s_%d', baseName, n);
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
fprintf('  stopping rules       : %s\n', stopSummary(cfg.stop));
fprintf('  run label            : %s\n', cfg.runLabel);
fprintf('  output directory     : %s\n', cfg.outputDir);
fprintf('  DERIVED scientific_evidence  : %d\n', cfg.scientificEvidence);
fprintf('  DERIVED performance_campaign : %d\n', cfg.performanceCampaign);
fprintf('  DERIVED production_stop_rules: %d\n', cfg.productionStopRules);
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
        stopM = cfg.stop.(methodKeys{m});
        if strcmp(methodKeys{m}, 'yuksel')
            % so the configuration printed and recorded carries the budget the
            % run uses (confbench_run_case applies the same value)
            stopM.maxIters = cfg.yukselMaxIters;
        end
        [mc, pid] = confbench_method_config(methodKeys{m}, ...
            cfg.resolutions(r,1), cfg.resolutions(r,2), cfg.outputDir, stopM);
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
if ~cfg.productionStopRules
    fprintf('  ...with NON-PRODUCTION stopping rules: %s\n', stopSummary(cfg.stop));
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
    maxNumCompThreads(entryThreads);   % nothing was solved; hand the session back
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
    wSchemaKeys = {}; wSchemas = {};
    for m = 1:nMet
        try
            wc = confbench_method_config(methodKeys{m}, wNelx, wNely, ...
                fullfile(cfg.outputDir, 'warmup'));
            wr = confbench_run_case(methodKeys{m}, wc, struct( ...
                'max_outer_override', wOuter, 'warmup', true, 'label', 'warmup'));
            % The warm-up is also the cheapest place to prove that every method
            % returns the SAME top-level record field set.  The main loop appends
            % with records(end+1) = rec, which requires that; discovering a
            % mismatch there costs the meshes already solved, discovering it here
            % costs seconds.
            wSchemaKeys{end+1} = methodKeys{m};          %#ok<SAGROW>
            wSchemas{end+1} = fieldnames(orderfields(wr)); %#ok<SAGROW>
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
    for m = 2:numel(wSchemas)
        if isequal(wSchemas{m}, wSchemas{1}); continue; end
        onlyHere  = setdiff(wSchemas{m}, wSchemas{1});
        onlyFirst = setdiff(wSchemas{1}, wSchemas{m});
        maxNumCompThreads(entryThreads);   % no measured row was produced
        error('performance_comparison:RecordSchemaMismatch', ...
            ['Record field sets differ between methods, so the campaign cannot ' ...
             'build one struct array.\n  %s has, and %s lacks: %s\n  %s has, and ' ...
             '%s lacks: %s\nDeclare the missing field(s) in the shared block at ' ...
             'the top of confbench_run_case so every method carries them.'], ...
            wSchemaKeys{m}, wSchemaKeys{1}, strjoin(reshape(onlyHere,1,[]), ', '), ...
            wSchemaKeys{1}, wSchemaKeys{m}, strjoin(reshape(onlyFirst,1,[]), ', '));
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
    if isfield(scaling, 'per_outer') && ~isempty(scaling.per_outer.methods)
        fprintf('  per outer iteration, %s:\n', scaling.per_outer.model);
        for i = 1:numel(scaling.per_outer.methods)
            s = scaling.per_outer.methods(i);
            fprintf('    %-30s %-44s C = %.6e   p = %.4f   R^2 = %.4f   (%d points)\n', ...
                s.method, s.quantity, s.C, s.p, s.R2, s.n);
        end
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

% Whether a budget was ACTUALLY reached, recorded next to the budget itself so
% a reader of the manifest never has to reconstruct censoring from the rows.
% confbench_classify decides CAP_HIT numerically from the per-stage counts, so
% this summary is exactly what the scaling fit excluded.
capRows = arrayfun(@(r) strcmp(r.status, 'CAP_HIT'), records);
manifest.cap_summary = struct( ...
    'any_cap_hit', any(capRows), ...
    'n_cap_hit', sum(capRows), ...
    'n_records', numel(records), ...
    'cap_hit_rows', {arrayfun(@(r) sprintf('%s %dx%d', r.method, r.mesh(1), r.mesh(2)), ...
        records(capRows), 'UniformOutput', false)}, ...
    'meaning', ['CAP_HIT means a method reached a safety budget instead of its ' ...
        'own stopping rule. Such rows are reported but are excluded from every ' ...
        'scaling fit and must not be described as converged.']);
% Assigned field by field: struct('removed_entries', {c}) collapses to a 0x0
% struct array when c is an empty cell, which is exactly the common case here.
manifest.path_scrub = struct();
manifest.path_scrub.removed_entries = pathScrub;
manifest.path_scrub.rationale = ['Non-production Olhoff implementations ' ...
    'inherited from this MATLAB session were removed from the path before ' ...
    'preflight, so dispatch is a property of this driver rather than of ' ...
    'whatever ran earlier in the session. The sole production implementation ' ...
    'is analysis/Olhoff.'];

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

% Every measurement is complete; give the session its thread setting back.
maxNumCompThreads(entryThreads);
fprintf('\nmaxNumCompThreads restored to %d.\n', maxNumCompThreads());

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

function validateStop(stop)
% Empty = production; anything else must be a usable criterion, tolerance or
% budget.  A criterion without a production tolerance needs its tolerances.
if isfield(stop.olhoff, 'useC')
    error('performance_comparison:UseCRetired', ['cfg.stop.olhoff.useC was replaced by ' ...
        'cfg.stop.olhoff.criterion (''l2_change'' = useC true, ''max_change'' = useC false).']);
end
crits = {'proposed', {'max_change','relative_l2_change','stagnation'}; ...
         'yuksel',   {'max_change','relative_l2_change','stagnation'}; ...
         'olhoff',   {'l2_change','max_change','relative_l2_change','stagnation'}};
for k = 1:size(crits, 1)
    c = stop.(crits{k,1}).criterion;
    if ~isempty(c) && ~any(strcmp(char(c), crits{k,2}))
        error('performance_comparison:UnknownCriterion', ...
            'cfg.stop.%s.criterion must be one of %s (got ''%s'').', ...
            crits{k,1}, strjoin(crits{k,2}, ', '), char(c));
    end
end
s1 = fieldOr(stop.yuksel, 'stage1Criterion', []);
if ~isempty(s1) && ~any(strcmp(char(s1), crits{2,2}))
    error('performance_comparison:UnknownCriterion', ...
        'cfg.stop.yuksel.stage1Criterion must be one of %s (got ''%s'').', ...
        strjoin(crits{2,2}, ', '), char(s1));
end
if any(strcmp(char(stop.olhoff.criterion), {'max_change','relative_l2_change'})) && isempty(stop.olhoff.tol)
    error('performance_comparison:OlhoffTolRequired', ...
        'cfg.stop.olhoff.criterion = ''%s'' requires cfg.stop.olhoff.tol (it has no production value).', ...
        char(stop.olhoff.criterion));
end
if strcmp(char(stop.proposed.criterion), 'relative_l2_change') && isempty(stop.proposed.tol)
    error('performance_comparison:RelativeTolRequired', ...
        'cfg.stop.proposed.criterion = ''relative_l2_change'' requires cfg.stop.proposed.tol.');
end
s2 = char(stop.yuksel.criterion);
if isempty(s1); s1 = s2; end
if (strcmp(char(s1), 'relative_l2_change') && isempty(stop.yuksel.stage1Tol)) ...
        || (strcmp(s2, 'relative_l2_change') && isempty(stop.yuksel.stage2Tol))
    error('performance_comparison:RelativeTolRequired', ['cfg.stop.yuksel: a ' ...
        '''relative_l2_change'' stage requires its stage1Tol / stage2Tol.']);
end
meths = {'proposed', 'yuksel', 'olhoff'};
stagFields = {'window', 'objectiveTol', 'graynessTol'};
for i = 1:numel(meths)
    for j = 1:numel(stagFields)
        v = fieldOr(stop.(meths{i}), stagFields{j}, []);
        if isempty(v); continue; end
        attrs = {'scalar','positive','finite'};
        if strcmp(stagFields{j}, 'window'); attrs = [attrs, {'integer'}]; end %#ok<AGROW>
        validateattributes(v, {'numeric'}, attrs, mfilename, ...
            sprintf('cfg.stop.%s.%s', meths{i}, stagFields{j}));
    end
end
tols = {stop.proposed.tol, stop.yuksel.stage1Tol, stop.yuksel.stage2Tol, stop.olhoff.c, stop.olhoff.tol};
names = {'proposed.tol', 'yuksel.stage1Tol', 'yuksel.stage2Tol', 'olhoff.c', 'olhoff.tol'};
for k = 1:numel(tols)
    if ~isempty(tols{k})
        validateattributes(tols{k}, {'numeric'}, {'scalar','positive','finite'}, ...
            mfilename, ['cfg.stop.' names{k}]);
    end
end
budgets = {stop.proposed.maxIters, stop.olhoff.maxOuter};
names = {'proposed.maxIters', 'olhoff.maxOuter'};
for k = 1:numel(budgets)
    if ~isempty(budgets{k})
        validateattributes(budgets{k}, {'numeric'}, {'scalar','integer','positive','finite'}, ...
            mfilename, ['cfg.stop.' names{k}]);
    end
end
end

function tf = isProductionCriterion(c, production)
tf = isempty(c) || strcmp(char(c), production);
end

function tf = belowBudget(used, frozen)
tf = ~isempty(used) && used < frozen;
end

function s = stopSummary(stop)
% "production", or every setting that differs from it.  A production
% criterion is not reported, nor is the Olhoff field its criterion leaves
% unused (c outside l2_change, tol inside it).
parts = {};
production = struct('proposed', 'max_change', 'yuksel', 'max_change', 'olhoff', 'l2_change');
meths = fieldnames(stop);
for i = 1:numel(meths)
    f = fieldnames(stop.(meths{i}));
    for j = 1:numel(f)
        v = stop.(meths{i}).(f{j});
        if strcmp(f{j}, 'criterion')
            if ~isProductionCriterion(v, production.(meths{i}))
                parts{end+1} = sprintf('%s.criterion = %s', meths{i}, char(v)); %#ok<AGROW>
            end
            continue
        end
        if strcmp(f{j}, 'stage1Criterion')
            if ~isempty(v) && ~strcmp(char(v), char(stop.(meths{i}).criterion))
                parts{end+1} = sprintf('%s.stage1Criterion = %s', meths{i}, char(v)); %#ok<AGROW>
            end
            continue
        end
        if strcmp(meths{i}, 'olhoff')
            olhL2 = isProductionCriterion(stop.olhoff.criterion, 'l2_change');
            if (olhL2 && strcmp(f{j}, 'tol')) || (~olhL2 && strcmp(f{j}, 'c'))
                continue
            end
        end
        if ~isempty(v)
            parts{end+1} = sprintf('%s.%s = %g', meths{i}, f{j}, v); %#ok<AGROW>
        end
    end
end
if isempty(parts)
    s = 'production';
else
    s = ['CHANGED: ' strjoin(parts, ', ')];
end
end

function tf = isEmptyDir(d)
% True when d does not exist, or exists and contains nothing.  A directory that
% exists but is empty is safe to write into: mkdir leaves one behind when a run
% dies before its first artifact, and refusing that would push every restart to
% a new suffix for no reason.
if exist(d, 'dir') ~= 7
    tf = true;
    return;
end
entries = dir(d);
tf = isempty(setdiff({entries.name}, {'.', '..'}));
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
    'stage1_eigenanalysis_and_preparation_s',   'T_prep+eig'; ...   % timing schema 1 (recorded campaigns)
    'stage1_reference_eigenanalysis_s',         'T_eig'; ...        % timing schema 2
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
            k = mc.canonical;
            gp = @(q) dotGet(k, q);   % plain field access: the olh package is not on the path here
            pr = olhoffcurrent_preset(mc.olhoff_preset);
            fprintf('    preset: %s (%s; upstream %s @ %s)\n', pr.name, pr.role, ...
                pr.upstreamPreset, pr.upstreamCommit(1:7));
            fprintf('    stiffness: %s\n', pr.formulation.stiffness);
            fprintf('    mass:      %s\n', pr.formulation.mass);
            fprintf('    resolved:  material.stiffness.model=%s linearBelow=%g, material.mass.model=%s\n', ...
                gp('material.stiffness.model'), gp('material.stiffness.linearBelow'), gp('material.mass.model'));
            fprintf(['    nested MMA (innerSolver=%s innerVar=%s variant=%s offDiag=%d ' ...
                'tolInner=%g maxInner=%d)\n'], mc.innerSolver, mc.innerVar, ...
                mc.mmaVariant, mc.offDiag, mc.tolInner, mc.maxInner);
            fprintf('    multiplicity: multRule=%s subN=%d (no threshold classifier)\n', ...
                mc.multRule, mc.subN);
            fprintf('    filter: physical R=%g -> rminEl=%g derived, mode=%s\n', ...
                mc.rminPhys, mc.rminPhys/(mc.b/mc.nely), mc.filterMode);
            switch gp('move.policy')
                case 'adaptive'
                    fprintf(['    move: ADAPTIVE per-element box, initial %g, floor %g, ' ...
                        'x%g monotone / x%g reversal; stage exhaustion %s\n'], mc.move, mc.moveMin, ...
                        mc.sAGrow, mc.sAShrink, onOff(strcmp(gp('move.continuation.signal'),'stageExhaustion')));
                case 'ladder'
                    fprintf('    move: global ladder %s, signal %s, window %d, tol %g\n', ...
                        mat2str(mc.s2Levels), gp('move.continuation.signal'), mc.s2Window, mc.s2Tol);
                otherwise
                    fprintf('    move: policy %s, initial %g\n', gp('move.policy'), mc.move);
            end
            if strcmp(gp('stop.rule'), 'stagnation')
                fprintf(['    outer stop: rule=stagnation over %d+1 designs, range(omega)/omega ' ...
                    '< %g and range(Mnd) < %g, cap=%d\n'], gp('stop.stagnation.window'), ...
                    gp('stop.stagnation.objectiveTolerance'), ...
                    gp('stop.stagnation.graynessTolerance'), mc.maxOuter);
            else
                if strcmp(mc.outerNorm, 'l2')
                    rmsNote = sprintf('  (per-element RMS %.6e)', mc.tolOuter/sqrt(mc.nelx*mc.nely));
                else
                    rmsNote = '';   % max and relativeL2 tolerances have no RMS reading
                end
                fprintf('    outer stop: rule=%s, %s norm < %.6g%s, guard=%s, cap=%d\n', ...
                    gp('stop.rule'), mc.outerNorm, mc.tolOuter, rmsNote, mc.outerGuard, mc.maxOuter);
            end
            fprintf('    threads=%d, diagnostics recorder=%d\n', mc.threads, mc.diag);
        otherwise
            o = mc.optimization;
            crit = 'max_change';
            if isfield(o, 'stop_criterion'); crit = o.stop_criterion; end
            fprintf('    optimizer=%s move=%g rmin=%g el stop=%s tol=%g maxIters=%d volfrac=%g p=%g\n', ...
                o.optimizer, o.move_limit, o.filter.radius, crit, o.convergence_tol, ...
                o.max_iters, o.volume_fraction, o.penalization);
            if isfield(o, 'yuksel')
                crit1 = crit;
                if isfield(o.yuksel, 'stage1_stop_criterion'); crit1 = o.yuksel.stage1_stop_criterion; end
                fprintf('    stage1: stop=%s tol=%g maxIters=%d | stage2: stop=%s tol=%g\n', ...
                    crit1, o.yuksel.stage1_tol, o.yuksel.stage1_max_iters, crit, o.yuksel.stage2_tol);
            end
            if isfield(o, 'stagnation')
                fprintf('    stagnation: %s (unlisted = solver default 10 / 1e-3 / 5e-3)\n', ...
                    jsonencode(o.stagnation));
            end
            if isfield(o, 'semi_harmonic_baseline')
                fprintf('    semi-harmonic baseline=%s, load sensitivity=%d\n', ...
                    o.semi_harmonic_baseline, o.semi_harmonic_load_sensitivity);
            end
    end
end
end

function v = dotGet(S, dotted)
parts = strsplit(dotted, '.');
v = getfield(S, parts{:}); %#ok<GFLD>
end

function s = onOff(tf)
if tf; s = 'ON'; else; s = 'off'; end
end

function writeJsonFile(path, s)
fid = fopen(path, 'w');
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(s, 'PrettyPrint', true));
end
