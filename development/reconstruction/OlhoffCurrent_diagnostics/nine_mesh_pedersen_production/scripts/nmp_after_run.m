function E = nmp_after_run()
%NMP_AFTER_RUN  Reached only if performance_comparison.m returned normally.
%   Records that the breakpoints were still armed at the end and a final
%   identity snapshot, then clears the breakpoints.
ctx = nmp_ctx();
runFile = fullfile(ctx.repo_abs, ctx.olhoffcurrent_run.path);
E = struct();
E.schema = 'nmp_after_run/1';
E.mode = ctx.mode;
E.when = nmp_now();
s = dbstatus(runFile);
E.breakpoints_still_armed = numel(s) == 1 && isequal(sort(s.line(:)'), [124 128 249]);
E.identity = nmp_identity_snapshot(ctx);
dbclear all
nmp_util('json', fullfile(ctx.logs_dir_abs, [ctx.mode '_AFTER_RUN.json']), E);
fprintf('[nmp] %s runner returned at %s; breakpoints still armed = %d; identity pass = %d\n', ...
    ctx.mode, E.when, E.breakpoints_still_armed, E.identity.pass);
end
