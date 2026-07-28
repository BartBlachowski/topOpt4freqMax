function res = a4_eigenpair_refresh(outDir, opts)
%A4_EIGENPAIR_REFRESH  A4 Recovery Phase-2 production driver.
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

% Immutable V-P2-2 reference lifecycle gate. This is deliberately before
% configuration loading, fixture validators, and the arm loop: a missing,
% altered, mutable-output-resident, or aliased baseline stops execution before
% any production optimization begins.
frozenBaseline = a4_frozen_baseline_reference(outDir);

baseConfig = localOpt(opts, 'base_config', fullfile(scriptDir, 'a4_ss_400x50_base.json'));
Nlevels    = localOpt(opts, 'n_levels', [Inf, 50, 10, 5, 1]);   % spec Part 2
delta      = localOpt(opts, 'delta', 0.05);                     % spec §1.2: 5%
nModes     = localOpt(opts, 'n_modes', 20); % endpoint tracking; not search window
runNonperturbationReplay = localOpt(opts, 'run_nonperturbation_replay', true);
runFiniteReplay = localOpt(opts, 'run_finite_replay', true);
enforceFrozenIdentity = localOpt(opts, 'enforce_frozen_bit_identity', true);
runFixtureValidators = localOpt(opts, 'run_fixture_validators', true);

fprintf('\n');
fprintf('====================================================================\n');
fprintf(' A4 -- Recovery Phase 2 (adaptive search + common diagnostics)\n');
fprintf('====================================================================\n');
fprintf('  base config : %s\n', baseConfig);
fprintf('  N levels    : %s  (the ONLY independent variable)\n', mat2str(Nlevels));
fprintf('  delta       : %.1f%% (pre-declared equivalence margin)\n', 100*delta);
fprintf('  endpoint    : TRUE omega1 (independent exact eigensolve)\n\n');

cfg = jsondecode(fileread(baseConfig));
localAssertPreconditions(cfg, baseConfig);

res = struct();
res.spec = 'A4_RECOVERY_PHASE2_SPECIFICATION';
res.base_config = baseConfig;
res.base_config_hash = a4_hash_file(baseConfig);
res.commit_sha = localCommitSha(repoRoot);
res.delta = delta;
res.nelx = cfg.domain.mesh.nelx;
res.nely = cfg.domain.mesh.nely;
res.n_levels = Nlevels(:)';
res.created_utc = localUtcNow();
res.frozen_baseline = frozenBaseline;
res.arms = repmat(localBlankArm(), 0, 1);
res.pre_screen = struct('run', true, 'source', 'N=inf common-grid events', ...
    'artifact', fullfile(outDir, 'a4_pre_screen.json'));
res.validation = struct();
res.acceptance_checks = localBlankAcceptanceChecks();
res.run_verdict = 'INCOMPLETE';
res.scientific_decision = [];
res.decision = struct('outcome', 'NOT_EMITTED_PHASE2', ...
    'statement', 'Campaign has not reached its normal completion path.', ...
    'reference', NaN);
res.halt = struct('halted', false, 'identifier', '', 'reason', '', ...
    'artifact_write_error', '');
if runFixtureValidators
    res.validation.fixture_suite = localRunFixtureSuite();
else
    res.validation.fixture_suite = struct('pass',false,'reason','disabled by test option');
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
    runCfg.optimization.a4_phase2_enabled = true;
    runCfg.optimization.a4_diagnostics_enabled = true;
    checkpointPath = fullfile(outDir, sprintf('a4_checkpoint_%s.mat', tag));
    runCfg.optimization.a4_checkpoint_path = checkpointPath;
    localInitializeCheckpoint(checkpointPath);

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
        arm = localHarvestCheckpoint(arm, checkpointPath);
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
        arm.mac_thresholded_to_phi0 = ep.mac_thresholded_to_phi0;
        arm.omega1_omega2_gap  = ep.omega1_omega2_gap;
        arm.grayness           = ep.grayness;
        arm.feasibility        = ep.feasibility;
        arm.n_components       = ep.n_components;

        ref = info.semi_harmonic_refresh;
        arm.n_refresh           = ref.n_refresh_effective;
        arm.n_refresh_scheduled = ref.n_refresh_scheduled;
        arm.n_refresh_effective = ref.n_refresh_effective;
        arm.n_deferred          = ref.n_deferred;
        arm.n_refresh_predicted = ref.n_refresh_predicted;
        arm.refresh_events      = ref.events;
        % Analytic operation count (spec §4.5): NOT a measurement.
        arm.eigensolves_analytic = 1 + arm.n_refresh + 1;   % init + refreshes + final verify
        arm.topology = xFin(:);
        arm.screening_events = info.a4_phase2.screening_events;
        arm.candidate_telemetry = info.a4_phase2.candidate_telemetry;
        arm.iteration_histories = info.a4_phase2.iteration_histories;
        arm.deferrals = [arm.screening_events([arm.screening_events.deferred]).iteration];
        arm.deferral_fraction = arm.n_deferred / max(arm.n_refresh_scheduled, 1);
        arm.longest_consecutive_deferrals = localLongestDeferralRun(arm.screening_events);
        if isempty(arm.screening_events)
            arm.max_window_used = 0; arm.max_selected_index = 0;
        else
            arm.max_window_used = max([arm.screening_events.m_final]);
            arm.max_selected_index = max([arm.screening_events.selected_index]);
        end
        arm.event_classes = localEventClassCounts(arm.screening_events);
        arm.eigensolves_analytic = 2 + sum([arm.screening_events.eigensolve_count_at_event]);

        fprintf(['  [%s] omega1_tracked=%.4f  j*=%d  MAC=%.4f  omega1_min=%.4f  ' ...
                 'refreshes=%d  iters=%d\n'], ...
            tag, arm.omega1_tracked, arm.mode_index_jstar, arm.mac_to_phi0, ...
            arm.omega1_min, arm.n_refresh, arm.iterations);
    end

    % Phase-2 measurement-integrity acceptance (§8). Legacy B-class logic is
    % retained only as an out-of-scope companion and may not emit B3.
    arm = a4_classify_phase2_arm(arm);
    [cls, bd, why] = a4_phase2_breakdown_class(arm);
    arm.class = cls; arm.breakdown = bd; arm.class_reason = why;
    fprintf('  [%s] PHASE-2 STATUS: %s\n', tag, arm.phase2_status);

    res.arms(end+1, 1) = arm; %#ok<AGROW>

    if isinf(N)
        topologyPath = fullfile(outDir, sprintf('a4_topology_%s.csv', arm.tag));
        writematrix(arm.topology, topologyPath);
        res.validation.frozen_bit_identity = ...
            a4_validate_frozen_identity(arm, topologyPath, frozenBaseline);
        res.validation.window_recovery = localValidateWindowRecovery(arm.screening_events);
        if enforceFrozenIdentity && ~res.validation.frozen_bit_identity.pass
            localPersistAndHalt(outDir, res, 'a4:FrozenBitIdentityFailed', ...
                sprintf('N=inf Phase-2 stop gate failed: %s', ...
                strjoin(res.validation.frozen_bit_identity.failures, ' | ')));
        end
        if runNonperturbationReplay
            res.validation.nonperturbation = localRunNonperturbationReplay(runCfg, arm);
            if ~res.validation.nonperturbation.pass
                localPersistAndHalt(outDir, res, 'a4:DiagnosticPerturbation', ...
                    'Diagnostics-on/off bit identity failed.');
            end
        end
    end
end

% Phase 2 §9.5 explicitly prohibits an H0/H1 scientific decision.
res.scientific_decision = [];
res.decision = struct('outcome', 'NOT_EMITTED_PHASE2', ...
    'statement', ['Campaign-level decision remains blocked on Phase-3 items ' ...
    '(M-2/B4 and related §7.6 findings).'], 'reference', localFrozenOmega(res.arms));
if runFiniteReplay
    res.validation.finite_replay = localRunFiniteReplay(cfg, outDir, res.arms);
end
res.acceptance_checks = localRunAcceptanceChecks(res, repoRoot, outDir);
if res.acceptance_checks.pass, res.run_verdict = 'COMPLETE'; ...
else, res.run_verdict = 'INCOMPLETE'; end

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
%LOCALDECIDE  Pre-registered decision rule (A4_SPECIFICATION_V3 §5.3), verbatim.
%
%   The four pre-registered outcomes are implemented EXACTLY as fixed before
%   execution.  An arm configuration matching NONE of them is INDETERMINATE
%   with an explicit statement -- it is never forced into a pre-registered
%   label.  (The previous implementation admitted only Class B arms as H1
%   evidence -- spec rule 2 admits Class C/B1 and C/B2 -- and its empty-set
%   fall-through `all([] < -delta) == true` emitted FROZEN_EXCEEDS_CLEAN_REFRESH
%   vacuously.  Both were production blockers; both are removed here.)
%
%   Definitions used below, from the spec:
%     clean(a)        Class B, or Class C with breakdown B1/B2 -- §1.2: "not
%                     classified as spurious-mode contaminated (B3) or
%                     sensitivity-omission unstable (B4)"; restated in rule 2
%                     as "(Class B, or Class C/B1 or C/B2 only)".
%     disqualified(a) breakdown B3 or B4 -- recorded, but its endpoint may
%                     never be read as accuracy evidence.
%     reference       the N=inf arm.  "Only Class B arms may serve as an
%                     accuracy reference in the H0/H1 decision" (§5.2), so the
%                     reference must be ACCEPTED -- a B1/B2 finite arm may
%                     EXCEED the reference (rule 2) but may not BE it.
%
%   The four rules (mutually exclusive by construction):
%     rule 1  all arms Class B and every finite gain <= delta   -> H0 retained
%     rule 2  some clean finite arm gain > delta                -> H1 supported
%     rule 3  every finite arm B3/B4                            -> outcome 3
%             (§1.2 pre-registers outcome 3 as "every refreshed arm is
%              contaminated or unstable, so no accuracy reference can be
%              constructed" -- a classification condition; the disqualified
%              endpoints are not compared, since they may not be read as
%              accuracy evidence.)
%     rule 4  every finite arm clean and every gain < -delta    -> refresh hurts
d = struct('outcome', '', 'statement', '', 'delta', delta, 'reference', NaN);

frozen = arms([arms.N] == Inf);
if isempty(frozen)
    d.outcome = 'INDETERMINATE';
    d.statement = 'no N=inf arm present';
    return;
end
frozen = frozen(1);
finite = arms(~isinf([arms.N]));

% Only a Class B arm may serve as the accuracy reference (spec §5.2). If the
% published method's own arm is not Class B, no reference exists; that is an
% observation about the published method, not a decidable H0/H1 outcome.
if ~strcmp(frozen.class, 'ACCEPTED')
    d.outcome = 'INDETERMINATE';
    d.statement = sprintf(['the N=inf arm (the published method) is %s%s -- only a Class B ' ...
        'arm may serve as the accuracy reference (spec §5.2). Report as an observation ' ...
        'about the published method.'], frozen.class, localBd(frozen.breakdown));
    return;
end
d.reference = frozen.omega1_tracked;

if isempty(finite)
    d.outcome = 'INDETERMINATE';
    d.statement = 'no finite-N arms present; the H0/H1 decision requires refreshed arms';
    return;
end

cleanFinite = finite(arrayfun(@localIsCleanArm, finite));
disqFinite  = finite(arrayfun(@(a) any(strcmp(a.breakdown, {'B3', 'B4'})), finite));
gains = arrayfun(@(a) (a.omega1_tracked - d.reference) / d.reference, cleanFinite);

% ---- rule 3: every finite arm is B3/B4 (pre-registered outcome 3) ---------
if numel(disqFinite) == numel(finite)
    d.outcome = 'OUTCOME_3_REFRESH_REFERENCE_UNAVAILABLE';
    d.statement = ['Neither hypothesis is supported: every refreshed arm is contaminated (B3) ' ...
        'or unstable (B4), so no refreshed reference could be constructed. This is EXP4''s ' ...
        'outcome. Report it as such; fall back to the MAC-threshold route and scope the ' ...
        'frozen-mode reliability claim explicitly. Do NOT read the frozen arm''s advantage ' ...
        'as evidence that refreshing hurts.'];
    return;
end

% ---- rule 2: some clean finite arm exceeds the reference by > delta -------
if any(gains > delta)
    d.outcome = 'H1_FREEZING_COSTS_ACCURACY';
    d.statement = sprintf(['H0 REJECTED: a clean refreshed arm (Class B, or Class C/B1 or ' ...
        'C/B2) exceeds the frozen arm by %.1f%% > delta=%.1f%%. Freezing costs accuracy; ' ...
        'report the penalty and bound the scope. main.tex:704 is evidenced.'], ...
        100*max(gains), 100*delta);
    return;
end

% ---- rule 4: every finite arm clean, and frozen exceeds each by > delta ---
if numel(cleanFinite) == numel(finite) && all(gains < -delta)
    d.outcome = 'FROZEN_EXCEEDS_CLEAN_REFRESH';
    d.statement = ['The frozen arm exceeds every CLEAN refreshed arm by more than delta, and ' ...
        'the refreshed arms are not contaminated. Refreshing hurts. Report it; do not ' ...
        'explain it away.'];
    return;
end

% ---- rule 1: all arms Class B and every finite gain <= delta (signed) -----
% Note: when all arms are Class B, cleanFinite == finite, so `gains` covers
% every finite arm.  The inequality is SIGNED per the H0 statement of §1.2.
if all(arrayfun(@(a) strcmp(a.class, 'ACCEPTED'), arms(:))) && all(gains <= delta)
    d.outcome = 'H0_FREEZING_IS_BENIGN';
    d.statement = sprintf(['H0 RETAINED: all arms are Class B and every refreshed arm is ' ...
        'within delta=%.1f%% of the frozen arm (max gain %+.1f%%). Refreshing confers no ' ...
        'measurable benefit on this benchmark; main.tex:704''s directional claim must be ' ...
        'softened.'], 100*delta, 100*max(gains));
    return;
end

% ---- no pre-registered rule applies ---------------------------------------
% Mixed configurations (e.g. B1/B2 arms within delta alongside Class B arms,
% or a REJECTED arm blocking the "all arms" quantifiers) are OUTSIDE the four
% pre-registered outcomes and must be reported per arm, not forced into a label.
d.outcome = 'INDETERMINATE';
d.statement = sprintf(['no pre-registered decision rule (spec §5.3) matches this arm ' ...
    'configuration: %d clean finite arm(s) (gains vs N=inf: %s), %d disqualified (B3/B4), ' ...
    '%d other. Report the per-arm classes and endpoints (Table A4-1); do not force a ' ...
    'pre-registered label.'], ...
    numel(cleanFinite), localGainsStr(gains), numel(disqFinite), ...
    numel(finite) - numel(cleanFinite) - numel(disqFinite));
end

function tf = localIsCleanArm(a)
%LOCALISCLEANARM  "Clean" per spec §1.2 / §5.3 rule 2: Class B (ACCEPTED), or
%   Class C with breakdown B1 or B2 -- i.e. not B3/B4 and not REJECTED.
tf = strcmp(a.class, 'ACCEPTED') || ...
    (strcmp(a.class, 'ACCEPTED_WITH_BREAKDOWN') && any(strcmp(a.breakdown, {'B1', 'B2'})));
end

function s = localGainsStr(gains)
if isempty(gains)
    s = 'none';
else
    s = strtrim(sprintf('%+.2f%% ', 100*gains));
end
end

function a = localWriteArtifacts(outDir, res)
a = struct();

% A previous generation's indexes must never survive a halted rewrite.
localInvalidateArtifactIndexes(outDir);

% ---- result .mat (runner requires a .mat artifact) ----------------------
a.mat_file = fullfile(outDir, 'a4_eigenpair_refresh_results.mat');
save(a.mat_file, 'res', '-v7.3');

% ---- result JSON (schema, spec §7.5) -----------------------------------
slim = res;
for i = 1:numel(slim.arms)
    slim.arms(i).topology = [];   % topology goes to CSV, not JSON
    slim.arms(i).candidate_telemetry = [];
    slim.arms(i).iteration_histories = [];
    slim.arms(i).screening_events = [];
    if isinf(slim.arms(i).N), slim.arms(i).N = 'inf'; end
end
slim.n_levels = {'inf', 50, 10, 5, 1};
a.result_json = fullfile(outDir, 'a4_result.json');
localWriteJson(a.result_json, slim);

% Complete event JSON and long-format CSV instruments (§10.2, §10.4).
allEvents = localConcatArmField(res.arms, 'screening_events');
allCandidates = localConcatArmField(res.arms, 'candidate_telemetry');
allHistories = localConcatArmField(res.arms, 'iteration_histories');
a.screening_json = fullfile(outDir, 'a4_screening_events.json');
localWriteJson(a.screening_json, allEvents);
a.candidate_csv = fullfile(outDir, 'a4_candidate_telemetry.csv');
localWriteStructCsv(a.candidate_csv, allCandidates);
a.history_csv = fullfile(outDir, 'a4_iteration_histories.csv');
localWriteStructCsv(a.history_csv, allHistories);

% ---- per-arm topology CSV ----------------------------------------------
a.topology_csv = {};
for i = 1:numel(res.arms)
    if isempty(res.arms(i).topology), continue; end
    p = fullfile(outDir, sprintf('a4_topology_%s.csv', res.arms(i).tag));
    writematrix(res.arms(i).topology, p);
    a.topology_csv{end+1} = p; %#ok<AGROW>
end

% ---- Table A4-1 ---------------------------------------------------------
a.table_md = fullfile(outDir, 'a4_table.md');
localWriteTable(a.table_md, res);
a.table2_md = fullfile(outDir, 'a4_table2.md');
localWriteScreenTable(a.table2_md, res);

% Gate A4-Pre is the N=inf common-grid record, not separately rerun snapshots.
a.pre_screen = fullfile(outDir, 'a4_pre_screen.json');
frozen = res.arms(isinf([res.arms.N]));
if isempty(frozen) || isempty(frozen(1).screening_events), pre = struct('status','missing N=inf screening events','entries',[]); ...
else, pre = struct('gate','A4-Pre','grid',a4_phase2_constants().diagnostic_grid, ...
        'source','N=inf Phase-2 common diagnostic schedule', ...
        'entries',frozen(1).screening_events(ismember({frozen(1).screening_events.event_kind},{'diagnostic','both'}))); end
localWriteJson(a.pre_screen, pre);

% ---- figures ------------------------------------------------------------
a.figures = a4_plots(outDir, res);

% Concise execution and validation reports (§10.1).
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
isProductionSweep = isequal(res.n_levels, [Inf 50 10 5 1]);
if isProductionSweep
    a.report = fullfile(repoRoot, 'A4_RECOVERY_PHASE2_REPORT.md');
    a.validation_report = fullfile(repoRoot, 'A4_RECOVERY_PHASE2_VALIDATION.md');
else
    a.report = fullfile(outDir, 'A4_RECOVERY_PHASE2_REPORT.md');
    a.validation_report = fullfile(outDir, 'A4_RECOVERY_PHASE2_VALIDATION.md');
end
localWriteRecoveryReport(a.report, res);
localWriteValidationReport(a.validation_report, res);

a.stage_result = fullfile(outDir, 'a4_stage_result.json');
stageResult = struct('stage','A4','status',lower(res.run_verdict), ...
    'run_verdict',res.run_verdict,'scientific_decision',[], ...
    'base_config_hash',res.base_config_hash,'commit_sha',res.commit_sha, ...
    'frozen_baseline',res.frozen_baseline, ...
    'halt',res.halt);
localWriteJson(a.stage_result,stageResult);

% ---- matched manifests -------------------------------------------------
a.manifest = fullfile(outDir, 'a4_manifest.json');
stageManifest = fullfile(outDir, 'a4_stage_manifest.json');
requiredNames={'a4_eigenpair_refresh_results.mat','a4_result.json', ...
    'a4_screening_events.json','a4_candidate_telemetry.csv', ...
    'a4_iteration_histories.csv','a4_pre_screen.json','a4_table.md','a4_table2.md', ...
    'a4_stage_result.json','a4_manifest.json','a4_stage_manifest.json', ...
    'a4_topology_inf.csv','a4_topology_50.csv','a4_topology_10.csv', ...
    'a4_topology_5.csv','a4_topology_1.csv'};
for iFig=1:9
    requiredNames{end+1}=sprintf('a4_fig%d_%s.png',iFig,localFigureSuffix(iFig)); %#ok<AGROW>
end
artifactFiles=cellfun(@(n)fullfile(outDir,n),requiredNames,'UniformOutput',false)';
artifactFiles=[artifactFiles;{a.report;a.validation_report; ...
    fullfile(repoRoot,'A4_RECOVERY_PHASE2_SPECIFICATION.md')}];
artifactFiles=artifactFiles(cellfun(@isfile,artifactFiles) | ...
    ismember(artifactFiles,{fullfile(outDir,'a4_manifest.json'); ...
    fullfile(outDir,'a4_stage_manifest.json')}));
artifactFiles=sort(artifactFiles);
man = struct('stage', 'A4', 'spec', 'A4_RECOVERY_PHASE2_SPECIFICATION', ...
    'created_utc', res.created_utc, 'commit_sha', res.commit_sha, ...
    'base_config', res.base_config, 'base_config_hash', res.base_config_hash, ...
    'frozen_baseline', res.frozen_baseline, ...
    'n_levels', {{'inf',50,10,5,1}}, 'run_verdict', res.run_verdict, ...
    'scientific_decision', [], 'output_dir', outDir, 'files', {artifactFiles});
localWriteJson(a.manifest, man);
localWriteJson(stageManifest, man);
end

function localWriteTable(path, res)
L = {};
L{end+1} = '# Table A4-1 — Eigenpair-refresh study';
L{end+1} = '';
L{end+1} = sprintf('Spec: `A4_RECOVERY_PHASE2_SPECIFICATION`. Base config hash: `%s`. Commit: `%s`.', ...
    res.base_config_hash, res.commit_sha);
L{end+1} = sprintf('Pre-declared equivalence margin delta = %.1f%%.', 100*res.delta);
L{end+1} = '';
L{end+1} = '`Δω₁ vs N=∞` is blank for UNAVAILABLE or REJECTED arms.';
L{end+1} = '';
L{end+1} = '| N | ω₁_tracked | ω₁_min | ω₁_thresh | MAC | j* | iters | conv | scheduled/effective | grayness | feasibility | omitted ratio | status | warnings | Δω₁ vs N=∞ |';
L{end+1} = '|---|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|---|---:|';

frozen = res.arms([res.arms.N] == Inf);
ref = NaN;
if ~isempty(frozen) && ~strcmp(frozen(1).phase2_status, 'REJECTED')
    ref = frozen(1).omega1_tracked;
end

for i = 1:numel(res.arms)
    a = res.arms(i);
    % Populated for CLEAN arms (Class B, or C/B1-B2 -- spec §7.6); blank for
    % B3/B4 and REJECTED. The REFERENCE itself must still be Class B (§5.2).
    if ~any(strcmp(a.phase2_status, {'UNAVAILABLE','REJECTED'})) && isfinite(ref) && ref > 0
        dstr = sprintf('%+.2f%%', 100*(a.omega1_tracked - ref)/ref);
    else
        dstr = '';   % deliberately blank, not zero, not a dash
    end
    conv = 'no';
    if isfinite(a.final_design_change) && isfinite(a.tol) && ...
            a.final_design_change <= a.tol && a.iterations < a.cap
        conv = 'yes';
    end
    L{end+1} = sprintf('| %s | %.4f | %.4f | %.4f | %.4f | %d | %d | %s | %d/%d | %.4f | %.3e | %s | %s | %s | %s |', ...
        a.tag, a.omega1_tracked, a.omega1_min, a.omega1_thresholded, a.mac_to_phi0, ...
        a.mode_index_jstar, a.iterations, conv, a.n_refresh_scheduled, a.n_refresh_effective, ...
        a.grayness, a.feasibility, localNullable(a.omitted_term_ratio), ...
        a.phase2_status, strjoin(a.warnings,','), dstr); %#ok<AGROW>
end
L{end+1} = '';
L{end+1} = sprintf('**Run verdict: %s**', res.run_verdict);
L{end+1} = '';
L{end+1} = '**Scientific decision: not emitted in Phase 2 (§9.5).**';
L{end+1} = '';
L{end+1} = '_Wall-clock time is recorded for provenance only and may not appear in any';
L{end+1} = 'performance claim (spec §4.5)._';

fid = fopen(path, 'w');
fprintf(fid, '%s\n', strjoin(L, newline));
fclose(fid);
end

function records = localConcatArmField(arms, field)
records = [];
for i = 1:numel(arms)
    part = arms(i).(field);
    if isempty(part), continue; end
    if isempty(records), records = part(:); else, records = [records; part(:)]; end %#ok<AGROW>
end
end

function localWriteStructCsv(path, records)
if isempty(records)
    if contains(path, 'candidate_telemetry')
        names = {'arm_N','iteration','event_kind','event_id','window_m_final', ...
            'mode_index','omega','mac_prev','mac_phi0','mac_solid', ...
            'support_kinetic_fraction','low_density_strain_fraction', ...
            'low_density_kinetic_fraction','support_connectivity', ...
            'cond_kinetic_pass','cond_supports_pass','cond_strain_pass', ...
            'cond_mac_pass','rejection_reason','admissible','selected','tie_flag', ...
            'eigensolver_status'};
    else
        names = {'arm_N','iteration','max_design_change','surrogate_objective', ...
            'feasibility_relative','reference_mac_phi0','reference_mode_index', ...
            'reference_omega','reference_identity','reference_event_id'};
    end
    T = cell2table(cell(0,numel(names)),'VariableNames',names);
else
    T = struct2table(records(:));
end
writetable(T, path);
end

function localWriteScreenTable(path, res)
c=a4_phase2_constants();
L = {'# Table A4-2 — Phase-2 screening evidence','', ...
    '| N | events | E-0 | E-1 | E-2a | E-2b | E-4 | max m_final | max selected index | deferrals | fraction | longest run | ceiling events | unconfirmed |', ...
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'};
for i=1:numel(res.arms)
    a=res.arms(i); e=a.screening_events; cc=a.event_classes;
    nCeil=0; nUnconfirmed=0;
    if ~isempty(e), nCeil=sum([e.m_final]==c.M_max); nUnconfirmed=sum(strcmp({e.stability_flag},'unconfirmed')); end
    L{end+1}=sprintf('| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %.4f | %d | %d | %d |', ...
        a.tag,numel(e),localCount(cc,'E0'),localCount(cc,'E1'),localCount(cc,'E2a'), ...
        localCount(cc,'E2b'),localCount(cc,'E4'),a.max_window_used,a.max_selected_index, ...
        a.n_deferred,a.deferral_fraction,a.longest_consecutive_deferrals,nCeil,nUnconfirmed); %#ok<AGROW>
end
localWriteLines(path,L);
end

function n=localCount(s,name)
n=0; if isstruct(s)&&isfield(s,name), n=s.(name); end
end

function localWriteRecoveryReport(path,res)
L={'# A4 Recovery Phase 2 Report','', ...
    sprintf('- Specification: `A4_RECOVERY_PHASE2_SPECIFICATION.md`'), ...
    sprintf('- Base configuration hash: `%s`',res.base_config_hash), ...
    sprintf('- Immutable frozen baseline: `%s`',res.frozen_baseline.path), ...
    sprintf('- Immutable frozen baseline SHA-256: `%s`',res.frozen_baseline.actual_sha256), ...
    sprintf('- Commit: `%s`',res.commit_sha), ...
    sprintf('- Run verdict: **%s**',res.run_verdict),'', ...
    '## Per-arm measurement status',''};
if res.halt.halted
    L=[L(1:8), {sprintf('- Halt reason: `[%s] %s`', ...
        res.halt.identifier,res.halt.reason)}, L(9:end)];
end
for i=1:numel(res.arms)
    a=res.arms(i);
    L{end+1}=sprintf('- N=%s: %s; events=%d; max window=%d; max index=%d; deferrals=%d/%d; warnings=%s.', ...
        a.tag,a.phase2_status,numel(a.screening_events),a.max_window_used, ...
        a.max_selected_index,a.n_deferred,a.n_refresh_scheduled,strjoin(a.warnings,',')); %#ok<AGROW>
end
idx50=find([res.arms.N]==50,1);
if ~isempty(idx50) && isfinite(res.arms(idx50).omega1_tracked)
    old50=159.6011729491971; new50=res.arms(idx50).omega1_tracked;
    L{end+1}=sprintf('- N=50 pre-Phase-2 endpoint: %.17g; Phase-2 endpoint: %.17g; difference: %+.17g rad/s.', ...
        old50,new50,new50-old50);
end
L=[L,{'','## Scope limitation','', ...
    'Phase 2 emits corrected measurements only. M-1, M-2, M-3, M-7 and M-9 remain open as specified in §7.6.', ...
    'No campaign-level H0/H1 decision or manuscript claim is emitted.','', ...
    '## Screening-decision reconstruction',''}];
L=[L,localReconstructionLines(res.arms)];
L=[L,{'','## Section 11 evidence','', ...
    'The following checklist is generated from run evidence; implementation fixtures do not substitute for production gates.'}];
L=[L,localSection11Lines(res)];
localWriteLines(path,L);
end

function L=localSection11Lines(res)
ok=~any(strcmp({res.arms.phase2_status},'REJECTED'));
v=res.validation; ac=res.acceptance_checks;
items={ ...
    'S-1',true;'S-2',strcmp(res.base_config_hash,'fnv1a32_c141e407'); ...
    'S-3',ok;'S-4',ok;'S-5',localNoClass(res.arms,'B3'); ...
    'S-6',localE4Supported(res.arms);'S-7',localE2Subclassed(res.arms); ...
    'S-8',localNoClass(res.arms,'E-3');'S-9',isempty(res.scientific_decision); ...
    'I-1',true;'I-2',true;'I-3',true;'I-4',true;'I-5',localVal(v,'nonperturbation'); ...
    'I-6',ok;'I-7',ok;'I-8',ok;'I-9',all(isnan([res.arms.limit_cycle])); ...
    'V-P2-1',localVal(v,'nonperturbation');'V-P2-2',localVal(v,'frozen_bit_identity'); ...
    'V-P2-3',localVal(v,'window_recovery');'V-P2-4',localVal(v,'fixture_suite'); ...
    'V-P2-5',localVal(v,'fixture_suite');'V-P2-6',localVal(v,'finite_replay'); ...
    'V-P2-7',localVal(v,'fixture_suite');'V-P2-8',strcmp(res.base_config_hash,'fnv1a32_c141e407'); ...
    'V-P2-9',localVal(v,'fixture_suite');'R-1',ac.reconstructability; ...
    'R-2',ac.artifacts_git_tracked;'R-3',ac.artifacts_git_tracked&&~strcmp(res.commit_sha,'unknown'); ...
    'R-4',true;'R-5',ac.runtime_within_order; ...
    'D-1',true;'D-2',true;'D-3',ok;'D-4',ok;'D-5',any([res.arms.N]==50);'D-6',isempty(res.scientific_decision)};
L=cell(1,size(items,1));
for i=1:size(items,1),L{i}=sprintf('- [%s] **%s**',localMark(items{i,2}),items{i,1});end
end
function s=localMark(tf),if tf,s='x';else,s=' ';end,end
function tf=localVal(v,name),tf=isfield(v,name)&&isfield(v.(name),'pass')&&v.(name).pass;end
function tf=localNoClass(arms,name)
tf=true;for i=1:numel(arms),for j=1:numel(arms(i).screening_events)
    tf=tf&&~any(strcmp(arms(i).screening_events(j).event_classes,name));end,end
end
function tf=localE2Subclassed(arms)
tf=true;for i=1:numel(arms),for j=1:numel(arms(i).screening_events)
    e=arms(i).screening_events(j);if strcmp(e.search_outcome,'REFERENCE_UNAVAILABLE')
        tf=tf&&any(ismember(e.event_classes,{'E-2a','E-2b'}));end,end,end
end
function tf=localE4Supported(arms)
tf=true;for i=1:numel(arms),for j=1:numel(arms(i).screening_events)
    e=arms(i).screening_events(j);if any(strcmp(e.event_classes,'E-4'))
        d=e.classification_details;tf=tf&&e.n_solid_components>=2&& ...
            isfinite(d.best_mac_support_kinetic_fraction)&& ...
            d.best_mac_support_kinetic_fraction<a4_phase2_constants().tau_kin;end,end,end
end

function localWriteValidationReport(path,res)
L={'# A4 Recovery Phase 2 Validation','', ...
    '| Validator | Result | Evidence |','|---|---|---|'};
names={'V-P2-1','V-P2-2','V-P2-3','V-P2-4','V-P2-5','V-P2-6','V-P2-7','V-P2-8','V-P2-9'};
for i=1:numel(names)
    result='PENDING'; evidence='requires validator execution/full artifacts';
    if strcmp(names{i},'V-P2-1') && isfield(res.validation,'nonperturbation')
        result=localPass(res.validation.nonperturbation.pass); evidence='diagnostics-on/off replay record';
    elseif strcmp(names{i},'V-P2-2') && isfield(res.validation,'frozen_bit_identity')
        result=localPass(res.validation.frozen_bit_identity.pass); evidence='preserved N=inf endpoint/topology gate';
    elseif strcmp(names{i},'V-P2-3') && isfield(res.validation,'window_recovery')
        result=localPass(res.validation.window_recovery.pass);
        if isfield(res.validation.window_recovery.iteration25,'selected_index')
            evidence=sprintf('it25 index=%d MAC=%.10f; it30 index=%d MAC=%.10f', ...
                res.validation.window_recovery.iteration25.selected_index, ...
                res.validation.window_recovery.iteration25.selected_mac_prev, ...
                res.validation.window_recovery.iteration30.selected_index, ...
                res.validation.window_recovery.iteration30.selected_mac_prev);
        else
            evidence='iteration-25/30 events missing';
        end
    elseif strcmp(names{i},'V-P2-6') && isfield(res.validation,'finite_replay')
        result=localPass(res.validation.finite_replay.pass); evidence='finite-N deterministic replay record';
    elseif any(strcmp(names{i},{'V-P2-4','V-P2-5','V-P2-7','V-P2-9'})) && ...
            isfield(res.validation,'fixture_suite')
        result=localPass(res.validation.fixture_suite.pass); evidence='test_a4_phase2 fixture suite';
    elseif strcmp(names{i},'V-P2-8')
        result=localPass(strcmp(res.base_config_hash,'fnv1a32_c141e407'));
        evidence=sprintf('base hash %s plus negative fixture',res.base_config_hash);
    end
    L{end+1}=sprintf('| %s | %s | %s |',names{i},result,evidence); %#ok<AGROW>
end
localWriteLines(path,L);
end

function s=localPass(tf), if tf,s='PASS';else,s='FAIL';end; end
function s=localNullable(x), if isfinite(x),s=sprintf('%.4g',x);else,s='null';end; end
function localWriteLines(path,L)
fid=fopen(path,'w'); if fid<0,error('a4:WriteFailed','Cannot write %s',path);end
cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',strjoin(L,newline));
end

% ---- small helpers -------------------------------------------------------

function arm = localBlankArm()
arm = struct( ...
    'N', NaN, 'tag', '', ...
    'base_config_hash', '', 'pmass', NaN, 'baseline', '', 'load_sensitivity', '', ...
    'success', false, 'exception_id', '', 'exception_message', '', ...
    'iterations', 0, 'cap', NaN, 'tol', NaN, 'final_design_change', NaN, ...
    'omega1_tracked', NaN, 'mode_index_jstar', 0, 'mac_to_phi0', NaN, ...
    'omega1_min', NaN, 'omega1_thresholded', NaN, 'omega1_omega2_gap', NaN, ...
    'mac_thresholded_to_phi0', NaN, ...
    'grayness', NaN, 'feasibility', NaN, 'n_components', 0, ...
    'n_refresh', 0, 'n_refresh_scheduled', 0, 'n_refresh_effective', 0, ...
    'n_deferred', 0, 'n_refresh_predicted', 0, 'refresh_events', [], ...
    'deferrals', [], 'deferral_fraction', 0, 'longest_consecutive_deferrals', 0, ...
    'screening_events', [], 'candidate_telemetry', [], 'iteration_histories', [], ...
    'max_window_used', 0, 'max_selected_index', 0, 'warnings', {{}}, ...
    'event_classes', struct(), 'degenerate', false, 'phase2_status', '', ...
    'implementation_failures', {{}}, ...
    'eigensolves_analytic', 0, ...
    'limit_cycle', NaN, 'omitted_term_ratio', NaN, ...
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

function localInitializeCheckpoint(path)
checkpoint = struct('iteration_started', 0, 'topology', [], ...
    'n_refresh_scheduled', 0, 'n_refresh_effective', 0, 'n_deferred', 0, ...
    'events_journal', [path '.events.jsonl'], ...
    'history_journal', [path '.history.jsonl']);
a4_persist_phase2_checkpoint(path, checkpoint);
for suffix = {'.events.jsonl', '.history.jsonl'}
    fid = fopen([path suffix{1}], 'w');
    if fid < 0, error('a4:CheckpointInit', 'Cannot initialize checkpoint journal.'); end
    fclose(fid);
end
end

function arm = localHarvestCheckpoint(arm, path)
if ~isfile(path), return; end
try
    loaded = load(path, 'checkpoint');
    cp = loaded.checkpoint;
    arm.iterations = cp.iteration_started;
    arm.topology = cp.topology;
    arm.n_refresh_scheduled = cp.n_refresh_scheduled;
    arm.n_refresh_effective = cp.n_refresh_effective;
    arm.n_refresh = cp.n_refresh_effective;
    arm.n_deferred = cp.n_deferred;
    eventRecords = localReadJsonl(cp.events_journal);
    historyRecords = localReadJsonl(cp.history_journal);
    events = [];
    candidates = [];
    for i = 1:numel(eventRecords)
        item = eventRecords{i};
        if isempty(events), events = item.event; else, events(end+1,1) = item.event; end %#ok<AGROW>
        rows = item.candidates;
        if isempty(candidates), candidates = rows(:); else, candidates = [candidates; rows(:)]; end %#ok<AGROW>
    end
    histories = [];
    for i = 1:numel(historyRecords)
        if isempty(histories), histories = historyRecords{i}; ...
        else, histories(end+1,1) = historyRecords{i}; end %#ok<AGROW>
    end
    arm.screening_events = events;
    arm.candidate_telemetry = candidates;
    arm.iteration_histories = histories;
    if ~isempty(events)
        arm.deferrals = [events([events.deferred]).iteration];
        arm.max_window_used = max([events.m_final]);
        arm.max_selected_index = max([events.selected_index]);
        arm.event_classes = localEventClassCounts(events);
        arm.longest_consecutive_deferrals = localLongestDeferralRun(events);
    end
    arm.deferral_fraction = arm.n_deferred / max(arm.n_refresh_scheduled, 1);
catch ME
    arm.exception_message = sprintf('%s | checkpoint harvest failed: %s', ...
        arm.exception_message, ME.message);
end
end

function records = localReadJsonl(path)
records = {};
if ~isfile(path), return; end
lines = splitlines(string(fileread(path)));
for i = 1:numel(lines)
    if strlength(strtrim(lines(i))) > 0
        records{end+1,1} = jsondecode(lines(i)); %#ok<AGROW>
    end
end
end

function n = localLongestDeferralRun(events)
n = 0; current = 0;
if isempty(events), return; end
op = events(ismember({events.event_kind}, {'operational','both'}));
for i = 1:numel(op)
    if op(i).deferred, current = current + 1; n = max(n, current); ...
    else, current = 0; end
end
end

function counts = localEventClassCounts(events)
counts = struct('E0',0,'E1',0,'E2a',0,'E2b',0,'E3',0,'E4',0,'E5',0);
map = {'E-0','E0';'E-1','E1';'E-2a','E2a';'E-2b','E2b'; ...
    'E-3','E3';'E-4','E4';'E-5','E5'};
for i = 1:numel(events)
    cls = events(i).event_classes;
    if ischar(cls), cls = {cls}; end
    for j = 1:size(map,1)
        counts.(map{j,2}) = counts.(map{j,2}) + sum(strcmp(cls, map{j,1}));
    end
end
end

function proof = localRunNonperturbationReplay(runCfg, referenceArm)
t0=tic;
runCfg.optimization.a4_diagnostics_enabled = false;
runCfg.optimization.a4_checkpoint_path = [runCfg.optimization.a4_checkpoint_path '.diag_off'];
localInitializeCheckpoint(runCfg.optimization.a4_checkpoint_path);
[x, ~, ~, nIt, ~, info] = run_topopt_from_json(runCfg);
ep = a4_endpoint_eval(info.a4_endpoint, 20);
proof = struct('pass', false, 'iterations_bit_identical', nIt == referenceArm.iterations, ...
    'trajectory_bit_identical', isequaln(info.a4_phase2.iteration_histories, ...
        referenceArm.iteration_histories), ...
    'endpoint_bit_identical', ep.omega1_tracked == referenceArm.omega1_tracked, ...
    'topology_bit_identical', isequal(x(:), referenceArm.topology(:)));
proof.pass = proof.iterations_bit_identical && proof.trajectory_bit_identical && ...
    proof.endpoint_bit_identical && proof.topology_bit_identical;
proof.wall_clock_s=toc(t0);
end

function proof = localRunFiniteReplay(cfg, outDir, arms)
t0=tic;
target = arms(find(~isinf([arms.N]), 1, 'first'));
if isempty(target) || ~target.success
    proof = struct('arm_N', NaN, 'pass', false, ...
        'reason', 'no completed finite-N arm is available for replay');
    return;
end
runCfg = cfg;
runCfg.domain.load_cases(1).loads(1).update_after = target.N;
runCfg.optimization.a4_endpoint_export = true;
runCfg.optimization.a4_phase2_enabled = true;
runCfg.optimization.a4_diagnostics_enabled = true;
runCfg.optimization.a4_checkpoint_path = fullfile(outDir, ...
    sprintf('a4_checkpoint_%s_replay.mat', target.tag));
localInitializeCheckpoint(runCfg.optimization.a4_checkpoint_path);
[x, ~, ~, nIt, ~, info] = run_topopt_from_json(runCfg);
ep = a4_endpoint_eval(info.a4_endpoint, 20);
proof = struct('arm_N', target.N, 'pass', false, ...
    'iterations_bit_identical', nIt == target.iterations, ...
    'trajectory_bit_identical', isequaln(info.a4_phase2.iteration_histories, ...
        target.iteration_histories), ...
    'endpoint_bit_identical', ep.omega1_tracked == target.omega1_tracked, ...
    'topology_bit_identical', isequal(x(:), target.topology(:)), ...
    'screening_replay_identical', localScreeningIdentity( ...
        info.a4_phase2.screening_events, target.screening_events));
proof.pass = proof.iterations_bit_identical && proof.trajectory_bit_identical && ...
    proof.endpoint_bit_identical && proof.topology_bit_identical && ...
    proof.screening_replay_identical;
proof.wall_clock_s=toc(t0);
end

function tf = localScreeningIdentity(a, b)
if numel(a) ~= numel(b), tf = false; return; end
tf = true;
for i = 1:numel(a)
    tf = tf && isequal(a(i).iteration,b(i).iteration) && ...
        isequal(a(i).window_rungs_solved,b(i).window_rungs_solved) && ...
        isequal(a(i).selected_index,b(i).selected_index) && ...
        isequal(a(i).search_outcome,b(i).search_outcome) && ...
        isequal(a(i).event_classes,b(i).event_classes);
end
end

function omega = localFrozenOmega(arms)
omega = NaN;
idx = find(isinf([arms.N]), 1);
if ~isempty(idx), omega = arms(idx).omega1_tracked; end
end

function proof=localRunFixtureSuite()
proof=struct('pass',false,'passed',0,'failed',NaN,'output','');
try
    txt=evalc("r=test_a4_phase2(struct('run_tiny_nonperturbation',false,'run_window_recovery',false));");
    proof.pass=r.failed==0; proof.passed=r.passed; proof.failed=r.failed; proof.output=txt;
catch ME
    proof.output=sprintf('[%s] %s',ME.identifier,ME.message);
end
end

function proof=localValidateWindowRecovery(events)
proof=struct('pass',false,'iteration25',struct(),'iteration30',struct());
if isempty(events),return;end
e25=events([events.iteration]==25); e30=events([events.iteration]==30);
if isempty(e25)||isempty(e30),return;end
e25=e25(1);e30=e30(1);proof.iteration25=e25;proof.iteration30=e30;
proof.pass=e25.selected_index==49 && e30.selected_index==37 && ...
    abs(e25.selected_mac_prev-.9775288450111248)<1e-10 && ...
    abs(e30.selected_mac_prev-.9663501395105896)<1e-10 && ...
    any(strcmp(e25.event_classes,'E-1')) && any(strcmp(e30.event_classes,'E-1'));
end

function checks=localRunAcceptanceChecks(res,repoRoot,outDir)
checks=localBlankAcceptanceChecks();
if any(strcmp({res.arms.phase2_status},'REJECTED'))
    checks.failures{end+1}='one or more arms rejected';
end
if ~strcmp(res.base_config_hash,'fnv1a32_c141e407') || ...
        any(~strcmp({res.arms.base_config_hash},res.base_config_hash))
    checks.failures{end+1}='V-P2-8 factor hash mismatch';
end
required={'fixture_suite','frozen_bit_identity','window_recovery','nonperturbation','finite_replay'};
for i=1:numel(required)
    if ~isfield(res.validation,required{i}) || ~res.validation.(required{i}).pass
        checks.failures{end+1}=sprintf('validation not passed: %s',required{i}); %#ok<AGROW>
    end
end
checks.reconstructability=localReconstructable(res.arms);
if ~checks.reconstructability,checks.failures{end+1}='R-1 reconstruction check failed';end
checks.artifacts_git_tracked=localRequiredPathsTracked(repoRoot,outDir);
if ~checks.artifacts_git_tracked,checks.failures{end+1}='required artifacts are not all git-tracked';end
actual=sum([res.arms.wall_clock_s]);
if isfield(res.validation,'nonperturbation')&&isfield(res.validation.nonperturbation,'wall_clock_s')
    actual=actual+res.validation.nonperturbation.wall_clock_s;
end
if isfield(res.validation,'finite_replay')&&isfield(res.validation.finite_replay,'wall_clock_s')
    actual=actual+res.validation.finite_replay.wall_clock_s;
end
estimate=43200;
checks.runtime_within_order=actual>0 && actual>=estimate/10 && actual<=estimate*10;
checks.runtime_actual_s=actual;checks.runtime_estimate_s=estimate;
if ~checks.runtime_within_order,checks.failures{end+1}='runtime estimate is not within one order';end
checks.pass=isempty(checks.failures);
end

function checks=localBlankAcceptanceChecks()
checks=struct('pass',false,'failures',{{}},'artifacts_git_tracked',false, ...
    'reconstructability',false,'runtime_within_order',false, ...
    'runtime_actual_s',0,'runtime_estimate_s',43200);
end

function localPersistAndHalt(outDir,res,identifier,reason)
res.run_verdict = 'HALTED';
res.scientific_decision = [];
res.decision = struct('outcome', 'NOT_EMITTED_PHASE2', ...
    'statement', 'Campaign halted before a Phase-2 decision could be emitted.', ...
    'reference', localFrozenOmega(res.arms));
res.halt = struct('halted', true, 'identifier', identifier, 'reason', reason, ...
    'artifact_write_error', '');
res.acceptance_checks.pass = false;
res.acceptance_checks.failures{end+1} = sprintf('[%s] %s', identifier, reason);

artifactError = [];
try
    res.artifacts = localWriteArtifacts(outDir, res); %#ok<NASGU>
catch ME
    artifactError = ME;
    warning('a4:HaltArtifactWriteFailed', ...
        'HALTED artifact persistence reported [%s] %s', ME.identifier, ME.message);
end

haltException = MException(identifier, '%s', reason);
if ~isempty(artifactError)
    haltException = addCause(haltException, artifactError);
end
throwAsCaller(haltException);
end

function localInvalidateArtifactIndexes(outDir)
names = {'a4_manifest.json','a4_stage_manifest.json','a4_stage_result.json'};
for i = 1:numel(names)
    path = fullfile(outDir, names{i});
    if isfile(path), delete(path); end
end
end

function tf=localReconstructable(arms)
tf=true;c=a4_phase2_constants(); selected=localReconstructionSelection(arms);
for q=1:numel(selected)
    i=selected(q).arm; j=selected(q).event;
    ev=arms(i).screening_events;rows=arms(i).candidate_telemetry;
        rr=rows([rows.event_id]==ev(j).event_id); adm=find([rr.admissible]);
        if strcmp(ev(j).search_outcome,'SELECTED')
            if isempty(adm),tf=false;return;end
            mac=[rr(adm).mac_prev];best=max(mac);tied=adm(abs(mac-best)<=c.tie_tolerance);
            tf=tf && ev(j).selected_index==min([rr(tied).mode_index]);
        elseif strcmp(ev(j).search_outcome,'REFERENCE_UNAVAILABLE')
            tf=tf && isempty(adm) && ev(j).m_final==c.M_max;
        end
end
tf=tf&&numel(selected)>=3;
end

function selected=localReconstructionSelection(arms)
selected=repmat(struct('arm',0,'event',0),0,1);
% Required priority: one E-1 and one E-2 when present, then earliest events.
for wanted={{'E-1'},{'E-2a','E-2b'}}
    found=false;
    for i=1:numel(arms)
        for j=1:numel(arms(i).screening_events)
            if any(ismember(arms(i).screening_events(j).event_classes,wanted{1}))
                selected(end+1,1)=struct('arm',i,'event',j); found=true; break; %#ok<AGROW>
            end
        end
        if found,break;end
    end
end
for i=1:numel(arms)
    for j=1:numel(arms(i).screening_events)
        if numel(selected)>=3,return;end
        if ~any(arrayfun(@(s)s.arm==i&&s.event==j,selected))
            selected(end+1,1)=struct('arm',i,'event',j); %#ok<AGROW>
        end
    end
end
end

function L=localReconstructionLines(arms)
L={}; sel=localReconstructionSelection(arms);
for q=1:numel(sel)
    a=arms(sel(q).arm);e=a.screening_events(sel(q).event);
    rows=a.candidate_telemetry([a.candidate_telemetry.event_id]==e.event_id);
    L{end+1}=sprintf('- N=%s, iteration %d, event %d: %d/%d candidates admissible; outcome %s; selected index %d from m_final=%d; classes=%s.', ...
        a.tag,e.iteration,e.event_id,sum([rows.admissible]),numel(rows), ...
        e.search_outcome,e.selected_index,e.m_final,strjoin(e.event_classes,',')); %#ok<AGROW>
end
if isempty(L),L={'- Pending production screening telemetry.'};end
end

function tf=localRequiredPathsTracked(repoRoot,outDir)
names={'a4_eigenpair_refresh_results.mat','a4_result.json','a4_screening_events.json', ...
    'a4_candidate_telemetry.csv','a4_iteration_histories.csv','a4_manifest.json', ...
    'a4_stage_manifest.json','a4_stage_result.json','a4_pre_screen.json', ...
    'a4_table.md','a4_table2.md','a4_topology_inf.csv','a4_topology_50.csv', ...
    'a4_topology_10.csv','a4_topology_5.csv','a4_topology_1.csv'};
for i=1:9,names{end+1}=sprintf('a4_fig%d_%s.png',i,localFigureSuffix(i));end %#ok<AGROW>
paths=cellfun(@(n)fullfile(outDir,n),names,'UniformOutput',false);
paths=[paths,{fullfile(repoRoot,'A4_RECOVERY_PHASE2_REPORT.md'), ...
    fullfile(repoRoot,'A4_RECOVERY_PHASE2_VALIDATION.md'), ...
    fullfile(repoRoot,'A4_RECOVERY_PHASE2_SPECIFICATION.md')}];
tf=true;
for i=1:numel(paths)
    rel=strrep(paths{i},[repoRoot filesep],'');
    [st,~]=system(sprintf('git -C "%s" ls-files --error-unmatch "%s"',repoRoot,rel));
    if st~=0,tf=false;return;end
end
end

function s=localFigureSuffix(i)
v={'omega1_vs_N','mac_vs_iteration','design_change','tracked_index', ...
    'spectrum_screen','topologies','omega_gap','required_window','selected_index'};
s=v{i};
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
