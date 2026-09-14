function A = nmp_arm_hooks()
%NMP_ARM_HOOKS  Arm the three observation breakpoints in olhoffcurrent_run.m.
%   Called by nmp_launch.sh in the SAME MATLAB process, immediately before
%   run('examples/Performance/performance_comparison.m').  Refuses (errors, so
%   the runner is never started) unless:
%     - the lock named by NMP_LOCK hashes to NMP_LOCK_SHA256;
%     - repository identity equals the lock (nmp_identity_snapshot);
%     - olhoffcurrent_run.m is the committed file and lines 124/128/249 carry
%       the exact statements the hooks are written against;
%     - the hook functions resolve to this directory;
%     - the runner output root does not exist yet (fresh output);
%     - all three breakpoints are confirmed by dbstatus with their conditions.
%   No file is modified; breakpoints are MATLAB session state only.
here = fileparts(mfilename('fullpath'));
ctx = nmp_ctx();
repo = ctx.repo_abs;
runFile = fullfile(repo, ctx.olhoffcurrent_run.path);

A = struct();
A.schema = 'nmp_hooks_armed/1';
A.mode = ctx.mode;
A.when = nmp_now();
A.matlab = version();
A.computer = computer();
A.pid = feature('getpid');
A.identity = nmp_identity_snapshot(ctx);
assert(A.identity.pass, 'nmp_arm_hooks:Identity', 'identity differs from the lock: %s', ...
    jsonencode(A.identity.checks));

L = readlines(runFile);
want = {124, 'tCall = tic;'; 128, 'out.x = double(res.rho(:));'; 249, 'if out.is_warmup'};
for k = 1:size(want, 1)
    got = strtrim(char(L(want{k, 1})));
    assert(strcmp(got, want{k, 2}), 'nmp_arm_hooks:LineDrift', ...
        'olhoffcurrent_run.m:%d is "%s", expected "%s"', want{k, 1}, got, want{k, 2});
end
A.hook_lines = cell2struct(want(:, 2), {'line124', 'line128', 'line249'}, 1);

assert(strcmp(which('olhoffcurrent_run'), runFile), 'nmp_arm_hooks:Dispatch', ...
    'olhoffcurrent_run resolves to %s', which('olhoffcurrent_run'));
for f = {'nmp_hook_pre', 'nmp_hook_tap', 'nmp_hook_post', 'nmp_ctx', 'nmp_identity_snapshot', 'nmp_util', 'nmp_now'}
    w = which(f{1});
    assert(strcmp(fileparts(w), here), 'nmp_arm_hooks:HookPath', '%s resolves to %s', f{1}, w);
end
assert(exist(ctx.output_root_abs, 'file') == 0, 'nmp_arm_hooks:OutputExists', ...
    'output root already exists: %s', ctx.output_root_abs);

dbclear all
cond = {'124', 'nmp_hook_pre(nelx,nely,cfg,out)'; ...
        '128', 'nmp_hook_tap(nelx,nely,res,callWall,out)'; ...
        '249', 'nmp_hook_post(nelx,nely,out)'};
for k = 1:size(cond, 1)
    dbstop('in', runFile, 'at', cond{k, 1}, 'if', cond{k, 2});
end
s = dbstatus(runFile);
assert(numel(s) == 1 && isequal(sort(s.line(:)'), [124 128 249]), 'nmp_arm_hooks:NotArmed', ...
    'dbstatus does not show exactly lines 124, 128, 249');
for k = 1:size(cond, 1)
    i = find(s.line == str2double(cond{k, 1}), 1);
    assert(strcmp(s.expression{i}, cond{k, 2}), 'nmp_arm_hooks:Condition', ...
        'line %s condition is "%s"', cond{k, 1}, s.expression{i});
end
A.breakpoints = struct('file', s.file, 'lines', s.line, 'expressions', {s.expression});
A.output_root_abs = ctx.output_root_abs;
A.run_root_abs = ctx.run_root_abs;
nmp_util('json', fullfile(ctx.logs_dir_abs, [ctx.mode '_HOOKS_ARMED.json']), A);
fprintf('[nmp] %s hooks armed at %s (pid %d); running performance_comparison.m\n', ctx.mode, A.when, A.pid);
end
