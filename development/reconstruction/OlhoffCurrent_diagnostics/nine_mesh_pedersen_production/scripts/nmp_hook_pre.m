function tf = nmp_hook_pre(nelx, nely, cfg, out)
%NMP_HOOK_PRE  PRECHECK, evaluated as the condition of a breakpoint on
%   olhoffcurrent_run.m line 124 ("tCall = tic;"): after the configuration the
%   solver will receive has been resolved and hashed, BEFORE the solver timer
%   starts.  Always returns false, so execution never stops.  It reads its
%   arguments and writes files; it modifies nothing the solver sees.
%
%   Campaign mode: a failed precheck writes PRECHECK_FAILED.txt and terminates
%   MATLAB with exit code 3, so the mesh is not solved and nothing after it runs.
%   Smoke mode: the result is recorded and execution continues.
%
%   A breakpoint condition that throws halts a -batch MATLAB process in the
%   debugger, so every statement here is inside try/catch.
tf = false;
t0 = tic;
mode = 'unknown';
dirp = '';
try
    ctx = nmp_ctx();
    mode = ctx.mode;
    tag = sprintf('%dx%d', nelx, nely);
    isW = logical(out.is_warmup);
    if isW; runName = ['warmup_' tag]; else; runName = tag; end
    dirp = fullfile(ctx.run_root_abs, runName);
    existed = exist(dirp, 'dir') == 7;
    if ~existed; mkdir(dirp); end

    P = struct();
    P.schema = 'nmp_precheck/1';
    P.mode = mode;
    P.mesh = tag;
    P.is_warmup = isW;
    P.when = nmp_now();
    P.hook = 'breakpoint condition at analysis/OlhoffCurrent/olhoffcurrent_run.m:124 (tCall = tic;), before the solver timer';
    P.run_dir_existed_before = existed;
    P.identity = nmp_identity_snapshot(ctx);

    h = olhoffcurrent_config_hash(cfg);
    key = matlab.lang.makeValidName(tag);
    fz = '';
    if isfield(ctx.nine_config_hashes, key); fz = ctx.nine_config_hashes.(key); end
    P.config_hash_of_cfg_about_to_be_solved = h;
    P.config_hash_recorded_by_olhoffcurrent_run = out.effective_config_hash;
    P.frozen_config_hash = fz;
    P.preset = out.preset;
    P.production_preset = out.production_preset;
    P.olhoffcurrent_preset_stamp = cfg.provenance.olhoffCurrentPreset;

    g = @(p) olh.config.getPath(cfg, p);
    F = struct( ...
        'stiffness_model', g('material.stiffness.model'), 'linearBelow', g('material.stiffness.linearBelow'), ...
        'p', g('material.stiffness.p'), 'p_continuation', g('material.stiffness.continuation.enabled'), ...
        'mass_model', g('material.mass.model'), 'mass_q', g('material.mass.q'), ...
        'mass_continuation', g('material.mass.continuation.enabled'), ...
        'filter_type', g('filter.type'), 'radiusPhysical', g('filter.radiusPhysical'), 'domain_b', g('domain.b'), ...
        'radiusElements_derived', g('filter.radiusPhysical')/(g('domain.b')/nely), ...
        'projection', g('projection.enabled'), 'move_policy', g('move.policy'), ...
        'continuation_signal', g('move.continuation.signal'), 'stop_rule', g('stop.rule'), ...
        'stop_tolerance', g('stop.tolerance'), 'inner_type', g('optimizer.inner.type'), ...
        'maxOuter', g('runtime.maxOuter'), 'singleThread', g('runtime.singleThread'));
    P.formulation = F;
    formulationOk = strcmp(F.stiffness_model, 'pedersen') && F.linearBelow == 0.1 && F.p == 3 && ...
        ~F.p_continuation && strcmp(F.mass_model, 'eq2') && F.mass_q == 1 && ~F.mass_continuation && ...
        strcmp(F.filter_type, 'sensitivity') && F.radiusPhysical == 0.06 && F.domain_b == 1 && ...
        ~F.projection && strcmp(F.move_policy, 'adaptive') && ...
        ~strcmp(F.continuation_signal, 'stageExhaustion') && strcmp(F.stop_rule, 'designChange') && ...
        strcmp(F.inner_type, 'mma') && F.singleThread;

    outRoot = ctx.output_root_abs;
    P.runner_output_fresh = struct( ...
        'manifest_absent', exist(fullfile(outRoot, 'benchmark_manifest.json'), 'file') ~= 2, ...
        'records_absent', exist(fullfile(outRoot, 'benchmark_records.mat'), 'file') ~= 2, ...
        'topology_absent', exist(fullfile(outRoot, 'topologies', ['topology_olhoff_' tag '.png']), 'file') ~= 2);
    P.maxNumCompThreads = maxNumCompThreads();
    P.which_olhoffSolve = which('olhoffSolve');
    P.which_mmasub = which('mmasub');
    P.rng_state_note = 'recorded for completeness; +impl uses no random numbers (eigs start vector is fixed)';
    r = rng(); P.rng = struct('Type', r.Type, 'Seed', r.Seed);
    implRoot = fullfile(ctx.repo_abs, 'analysis', 'OlhoffCurrent', '+impl');

    P.checks = struct( ...
        'identity', P.identity.pass, ...
        'config_hash_equals_frozen', strcmp(h, fz) && strcmp(h, out.effective_config_hash), ...
        'preset', strcmp(out.preset, ctx.preset) && strcmp(out.production_preset, ctx.preset) && ...
            strcmp(cfg.provenance.olhoffCurrentPreset, ctx.preset), ...
        'formulation', formulationOk, ...
        'run_dir_fresh', ~existed, ...
        'runner_output_fresh', all(struct2array(P.runner_output_fresh)), ...
        'single_thread', P.maxNumCompThreads == 1, ...
        'dispatch_inside_impl', strncmp(P.which_olhoffSolve, implRoot, numel(implRoot)) && ...
            strncmp(P.which_mmasub, implRoot, numel(implRoot)));
    if isW
        % The warm-up mesh (48x6, 5 outer) is the runner's own discarded solve;
        % it is outside the frozen list, so only identity and dispatch apply.
        P.checks.config_hash_equals_frozen = true;
        P.checks.runner_output_fresh = true;
        P.warmup_note = 'runner warm-up (discarded by performance_comparison.m); frozen-hash check not applicable';
    end
    P.pass = all(struct2array(P.checks));
    P.hook_seconds = toc(t0);
    nmp_util('json', fullfile(dirp, 'PRECHECK.json'), P);
    nmp_util('event', ctx, sprintf('PRECHECK %-14s pass=%d (%.3f s, outside solver timer)', runName, P.pass, P.hook_seconds));

    if ~P.pass && ~strcmp(mode, 'smoke')
        failed = fieldnames(P.checks);
        failed = failed(~struct2array(P.checks));
        nmp_util('text', fullfile(dirp, 'PRECHECK_FAILED.txt'), ...
            sprintf('PRECHECK FAILED for %s at %s: %s. MATLAB terminated with exit code 3 before the solve.', ...
            runName, nmp_now(), strjoin(failed, ', ')));
        nmp_util('event', ctx, sprintf('EXIT 3 (precheck failed: %s)', strjoin(failed, ', ')));
        exit(3);
    end
catch ME
    try
        if isempty(dirp); dirp = tempdir; end
        nmp_util('text', fullfile(dirp, 'PRECHECK_ERROR.txt'), getReport(ME, 'extended', 'hyperlinks', 'off'));
    catch
    end
    if ~strcmp(mode, 'smoke')
        exit(3);
    end
end
end
