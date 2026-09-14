function sd_m1_run()
%SD_M1_RUN  The ONE preregistered source-side 480x60 trajectory (Level M1).
%
%   Source snapshot code, committed preset duOlhoffAdaptiveMove with the
%   committed sweep override move.initial = 0.10 (the A2/C3 configuration),
%   native diagnostics recorder on.  Fail-closed preflight first.  The solver is
%   called unmodified; nothing here touches its arithmetic.
P = sd_use_source();
if ~isfolder(P.m1dir), mkdir(P.m1dir); end
[cM, ~, ~, rep] = sd_m1_config();
metaAllowed = {'provenance.preset','provenance.overrides','provenance.resolvedAt'}; % Amendment 1
bad = rep(~[rep.allowed] & ~ismember({rep.path}, metaAllowed));
pre = struct('when', datestr(now, 'yyyy-mm-ddTHH:MM:SS'), 'diff', rep, ...
    'blocking', {{bad.path}}, 'olhoffSolve', which('olhoffSolve'), ...
    'innerLoop', which('innerLoop'), 'limit', which('olh.move.limit'), ...
    'matlab', version, 'host', getenv('HOSTNAME'));
fid = fopen(fullfile(P.m1dir, 'M1_preflight.json'), 'w');
fprintf(fid, '%s\n', jsonencode(pre, 'PrettyPrint', true)); fclose(fid);
if ~isempty(bad)
    error('sd:m1:preflight', 'M1 preflight FAILED on: %s', strjoin({bad.path}, ', '));
end
fprintf('M1 preflight PASS (%d leaves differ, all allowed)\n', numel(rep));

t0 = tic;
res = olhoffSolve(cM);
wall = toc(t0);
cfg = cM;
save(fullfile(P.m1dir, 'M1_480x60_res.mat'), 'res', 'cfg', 'wall', '-v7.3');
fprintf('M1 DONE status=%s nOuter=%d wall=%.1f s omega1=%.12g\n', res.status, res.nOuter, wall, res.omega(1));
for i = 1:numel(res.log), fprintf('LOG: %s\n', res.log{i}); end
end
