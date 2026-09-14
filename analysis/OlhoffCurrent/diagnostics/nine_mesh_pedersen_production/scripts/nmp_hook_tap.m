function tf = nmp_hook_tap(nelx, nely, res, callWall, out)
%NMP_HOOK_TAP  HISTORY TAP, evaluated as the condition of a breakpoint on
%   olhoffcurrent_run.m line 128 ("out.x = double(res.rho(:));"): immediately
%   after "callWall = toc(tCall);", i.e. after the solver timer has stopped.
%
%   olhoffcurrent_run keeps only aggregates of res.hist; this saves the solver's
%   own result struct (every field except the FE model mdl, whose sizes are
%   recorded instead) so the per-iteration eigenvalue, adaptive-box, inner and
%   timing histories of THE production solve are retained.  Always returns
%   false; never terminates MATLAB (a lost tap is recorded, not fatal).
tf = false;
dirp = '';
try
    ctx = nmp_ctx();
    tag = sprintf('%dx%d', nelx, nely);
    isW = logical(out.is_warmup);
    if isW; runName = ['warmup_' tag]; else; runName = tag; end
    dirp = fullfile(ctx.run_root_abs, runName);
    if exist(dirp, 'dir') ~= 7; mkdir(dirp); end
    t0 = tic;

    mdlInfo = struct('nele', NaN, 'nnode', NaN, 'ndof', NaN, 'nfree', NaN);
    try
        mdlInfo = struct('nele', res.mdl.nele, 'nnode', res.mdl.nnode, 'ndof', res.mdl.ndof, ...
            'nfree', numel(res.mdl.free));
    catch
    end
    resTap = res;
    if isfield(resTap, 'mdl'); resTap = rmfield(resTap, 'mdl'); end
    tapFile = fullfile(dirp, 'SOLVER_RESULT.mat');
    save(tapFile, 'resTap', 'callWall', 'mdlInfo', '-v7.3');

    h = res.hist;
    T = struct();
    T.schema = 'nmp_tap/1';
    T.mode = ctx.mode;
    T.mesh = tag;
    T.is_warmup = isW;
    T.when = nmp_now();
    T.hook = 'breakpoint condition at analysis/OlhoffCurrent/olhoffcurrent_run.m:128, after callWall = toc(tCall)';
    T.callWall_s = callWall;
    T.res_wallclock_s = res.wallclock;
    T.res_status = res.status;
    T.res_nOuter = res.nOuter;
    T.omega = res.omega(:).';
    w3 = double(res.omega(:)); w3 = w3(1:min(3, numel(w3)));
    T.omega123_sha256 = nmp_util('sha256double', w3);
    T.rho_sha256 = nmp_util('sha256double', res.rho);
    T.hist_numel_N = numel(h.N);
    T.hist_sum_nInner = sum(h.nInner);
    T.hist_sum_tOuter = sum(double(h.tOuter(:)));
    T.hist_sum_tInner = sum(double(h.tInner(:)));
    T.hist_sum_tEig = sum(double(h.tEig(:)));
    % bit-exact carriers for the postcheck cross-check (JSON decimal text may round)
    T.hex = struct('callWall', num2hex(callWall), 'sum_tOuter', num2hex(T.hist_sum_tOuter), ...
        'sum_tInner', num2hex(T.hist_sum_tInner), 'sum_tEig', num2hex(T.hist_sum_tEig));
    T.hist_fields = fieldnames(h);
    T.mdl = mdlInfo;
    T.file = 'SOLVER_RESULT.mat';
    T.file_sha256 = olhoffcurrent_sha256_file(tapFile);
    T.hook_seconds = toc(t0);
    nmp_util('json', fullfile(dirp, 'TAP.json'), T);
    nmp_util('event', ctx, sprintf('TAP      %-14s outer=%d status=%s omega1=%.10g (%.3f s, outside solver timer)', ...
        runName, res.nOuter, res.status, res.omega(1), T.hook_seconds));
catch ME
    try
        if isempty(dirp); dirp = tempdir; end
        nmp_util('text', fullfile(dirp, 'TAP_ERROR.txt'), getReport(ME, 'extended', 'hyperlinks', 'off'));
    catch
    end
end
end
