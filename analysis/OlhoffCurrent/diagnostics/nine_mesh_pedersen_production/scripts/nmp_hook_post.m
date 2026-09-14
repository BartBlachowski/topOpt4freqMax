function tf = nmp_hook_post(nelx, nely, out)
%NMP_HOOK_POST  POSTCHECK, evaluated as the condition of a breakpoint on
%   olhoffcurrent_run.m line 249 ("if out.is_warmup"), reached on every path
%   after the result struct (status, stopping, accounting, provenance) is built.
%
%   Saves the complete olhoffcurrent_run result, re-checks repository identity
%   and the configuration hash, and cross-checks the history tap against the
%   runner's own accounting.  Campaign mode: an identity change or an
%   unverifiable identity writes CAMPAIGN_INTEGRITY_FAILURE.txt and terminates
%   MATLAB with exit code 4 so no later mesh runs.  Always returns false.
tf = false;
mode = 'unknown';
dirp = '';
try
    ctx = nmp_ctx();
    mode = ctx.mode;
    tag = sprintf('%dx%d', nelx, nely);
    isW = logical(out.is_warmup);
    if isW; runName = ['warmup_' tag]; else; runName = tag; end
    dirp = fullfile(ctx.run_root_abs, runName);
    if exist(dirp, 'dir') ~= 7; mkdir(dirp); end
    t0 = tic;

    outFile = fullfile(dirp, 'RUN_OUT.mat');
    save(outFile, 'out', '-v7.3');

    Q = struct();
    Q.schema = 'nmp_postcheck/1';
    Q.mode = mode;
    Q.mesh = tag;
    Q.is_warmup = isW;
    Q.when = nmp_now();
    Q.hook = 'breakpoint condition at analysis/OlhoffCurrent/olhoffcurrent_run.m:249 (if out.is_warmup), after the result is assembled';
    Q.identity = nmp_identity_snapshot(ctx);
    Q.status = out.status;
    Q.status_note = out.status_note;
    Q.ok = out.ok;
    Q.preset = out.preset;
    Q.production_preset = out.production_preset;

    key = matlab.lang.makeValidName(tag);
    fz = '';
    if isfield(ctx.nine_config_hashes, key); fz = ctx.nine_config_hashes.(key); end
    Q.frozen_config_hash = fz;
    Q.effective_config_hash = '';
    Q.config_hash_rehashed_from_out_configuration = '';
    if isfield(out, 'effective_config_hash'); Q.effective_config_hash = out.effective_config_hash; end
    if isfield(out, 'configuration'); Q.config_hash_rehashed_from_out_configuration = olhoffcurrent_config_hash(out.configuration); end
    Q.config_hash_rehashed_from_solver_effective_cfg = '';
    if isfield(out, 'effective_cfg'); Q.config_hash_rehashed_from_solver_effective_cfg = olhoffcurrent_config_hash(out.effective_cfg); end

    Q.x_sha256 = '';
    if ~isempty(out.x); Q.x_sha256 = nmp_util('sha256double', out.x); end
    tapJson = fullfile(dirp, 'TAP.json');
    Q.tap_present = exist(tapJson, 'file') == 2;
    tapOk = false;
    if Q.tap_present && isfield(out, 'accounting')
        T = jsondecode(fileread(tapJson));
        a = out.accounting;
        nw = min(3, numel(out.omega));
        Q.tap_crosscheck = struct( ...
            'rho_sha256_equals_out_x', strcmp(T.rho_sha256, Q.x_sha256), ...
            'outer_equals_accounting', T.hist_numel_N == a.outer_iterations, ...
            'inner_equals_accounting', T.hist_sum_nInner == a.inner_iterations_total, ...
            'sum_tOuter_equals_accounting', strcmp(T.hex.sum_tOuter, num2hex(a.outer_time_total_s)), ...
            'sum_tInner_equals_accounting', strcmp(T.hex.sum_tInner, num2hex(a.inner_time_total_s)), ...
            'sum_tEig_equals_accounting', strcmp(T.hex.sum_tEig, num2hex(a.eigen_time_s)), ...
            'callWall_equals_accounting', strcmp(T.hex.callWall, num2hex(a.total_wall_time_s)), ...
            'omega123_equals_out', strcmp(T.omega123_sha256, nmp_util('sha256double', double(out.omega(1:nw)))));
        tapOk = all(struct2array(Q.tap_crosscheck));
    end
    Q.file = 'RUN_OUT.mat';
    Q.file_sha256 = olhoffcurrent_sha256_file(outFile);

    hashOk = strcmp(Q.effective_config_hash, fz) && strcmp(Q.config_hash_rehashed_from_out_configuration, fz);
    Q.checks = struct( ...
        'identity', Q.identity.pass, ...
        'config_hash_unchanged', hashOk, ...
        'preset_unchanged', strcmp(out.preset, ctx.preset) && strcmp(out.production_preset, ctx.preset), ...
        'tap_consistent_with_runner_accounting', tapOk);
    if isW
        Q.checks.config_hash_unchanged = true;
        Q.warmup_note = 'runner warm-up (discarded); frozen-hash check not applicable';
    end
    Q.pass = all(struct2array(Q.checks));
    Q.hook_seconds = toc(t0);
    nmp_util('json', fullfile(dirp, 'POSTCHECK.json'), Q);
    nmp_util('event', ctx, sprintf('POSTCHECK %-13s pass=%d status=%s (%.3f s)', runName, Q.pass, out.status, Q.hook_seconds));

    integrityBroken = ~(Q.checks.identity && Q.checks.config_hash_unchanged && Q.checks.preset_unchanged);
    if integrityBroken && ~strcmp(mode, 'smoke')
        nmp_util('text', fullfile(dirp, 'CAMPAIGN_INTEGRITY_FAILURE.txt'), ...
            sprintf('Repository scientific identity changed or could not be confirmed after %s at %s. MATLAB terminated with exit code 4.', ...
            runName, nmp_now()));
        nmp_util('event', ctx, 'EXIT 4 (postcheck identity failure)');
        exit(4);
    end
catch ME
    try
        if isempty(dirp); dirp = tempdir; end
        nmp_util('text', fullfile(dirp, 'POSTCHECK_ERROR.txt'), getReport(ME, 'extended', 'hyperlinks', 'off'));
    catch
    end
    if ~strcmp(mode, 'smoke')
        exit(4);
    end
end
end
