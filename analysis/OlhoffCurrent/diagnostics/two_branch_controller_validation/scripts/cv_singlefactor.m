function G = cv_singlefactor()
%CV_SINGLEFACTOR  Phase 6.  Prove candidate and production differ ONLY in the
%   declared controller intervention and its necessary telemetry.
%
%   Compares every field of the schema, at every scientific mesh, between the
%   candidate configuration and the PRODUCTION ENTRY POINT olhoffcurrent_config
%   at the same cap and recorder setting.  Anything differing outside the
%   declared list is CONTROLLER_SINGLE_FACTOR_FAIL.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
guard = olhoffcurrent_paths(); %#ok<NASGU>

ALLOWED = {'move.continuation.signal', 'stop.rule', 'runtime.name'};
meshes  = [160 20; 320 40; 400 50];
S = olh.config.schema();

G = struct('allowed', {ALLOWED}, 'meshes', meshes, 'pass', true, 'rows', struct([]));
fprintf('\n%s\nCV_SINGLEFACTOR  (Phase 6)\n%s\n', repmat('=',1,72), repmat('=',1,72));

for r = 1:size(meshes,1)
    nelx = meshes(r,1); nely = meshes(r,2);
    cP = olhoffcurrent_config(nelx, nely, 'MaxOuter', 1600, 'Diagnostics', true);
    cC = cv_config('C', nelx, nely);

    diffs = {};
    for k = 1:size(S,1)
        p = S{k,1};
        a = olh.config.getPath(cP, p);
        b = olh.config.getPath(cC, p);
        if ~isequaln(a, b), diffs{end+1} = p; end %#ok<AGROW>
    end
    unexpected = setdiff(diffs, ALLOWED);
    missing    = setdiff({'move.continuation.signal','stop.rule'}, diffs);

    ok = isempty(unexpected) && isempty(missing);
    G.pass = G.pass && ok;
    G.rows(r).mesh        = [nelx nely];
    G.rows(r).diffs       = diffs;
    G.rows(r).unexpected  = unexpected;
    G.rows(r).missing     = missing;
    G.rows(r).ok          = ok;
    G.rows(r).cfgHashProd = olhoffcurrent_config_hash(cP);
    G.rows(r).cfgHashCand = olhoffcurrent_config_hash(cC);

    % the locked scientific identity, spelled out rather than implied
    lock = {'material.stiffness.p','material.stiffness.continuation.enabled', ...
            'material.mass.model','material.mass.q','filter.type','filter.applyTo', ...
            'filter.radiusPhysical','projection.enabled','multiplicity.method', ...
            'multiplicity.subspaceSize','multiplicity.diagonalOffsets', ...
            'multiplicity.offDiagonal','optimizer.inner.type','optimizer.inner.variant', ...
            'optimizer.inner.variable','optimizer.inner.tolerance', ...
            'optimizer.inner.maxIterations','optimizer.inner.minIterations', ...
            'eigen.solver','eigen.targetMode','eigen.maxCluster','design.initial', ...
            'design.minimum','design.volumeFraction','domain.a','domain.b', ...
            'domain.mesh.nelx','domain.mesh.nely','move.policy','move.initial', ...
            'move.levels','move.minimum','stop.norm','stop.tolerance', ...
            'stop.toleranceRule','stop.field','stop.guards.settledMove', ...
            'stop.guards.ladderExhausted','stop.guards.maxDesignChange', ...
            'runtime.maxOuter','runtime.singleThread','runtime.diagnostics'};
    lockOk = true; lockBad = {};
    for k = 1:numel(lock)
        if ~isequaln(olh.config.getPath(cP,lock{k}), olh.config.getPath(cC,lock{k}))
            lockOk = false; lockBad{end+1} = lock{k}; %#ok<AGROW>
        end
    end
    G.rows(r).lockOk = lockOk; G.rows(r).lockBad = lockBad;
    G.pass = G.pass && lockOk;

    fprintf('  %3dx%-3d  diffs={%s}  unexpected={%s}  lock=%s\n', nelx, nely, ...
        strjoin(diffs,', '), strjoin(unexpected,', '), local_tf(lockOk));
    fprintf('           prod cfgHash %s\n           cand cfgHash %s\n', ...
        G.rows(r).cfgHashProd, G.rows(r).cfgHashCand);
end

G.verdict = 'CONTROLLER_SINGLE_FACTOR_PASS';
if ~G.pass, G.verdict = 'CONTROLLER_SINGLE_FACTOR_FAIL'; end
fprintf('%s\n  %s\n', repmat('-',1,72), G.verdict);

outF = fullfile(study,'evidence','single_factor.json');
fid = fopen(outF,'w'); fwrite(fid, jsonencode(G,'PrettyPrint',true)); fclose(fid);
fprintf('  wrote %s\n', outF);
end

function s = local_tf(t), if t, s='OK'; else, s='VIOLATED'; end, end
