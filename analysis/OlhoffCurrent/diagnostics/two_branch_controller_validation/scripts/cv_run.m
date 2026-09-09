function out = cv_run(nelx, nely)
%CV_RUN  ONE authorized candidate scientific run.
%
%   Runs the frozen two-branch stage-exhaustion controller at one mesh, writes
%   the raw trajectory to the durable evidence root and the per-iteration
%   telemetry to runs/, and returns the scalar record.
%
%   Only 160x20, 320x40 and 400x50 are authorized (PREREGISTRATION.md sec. 13).
%   Any other mesh is refused here rather than trusted to discipline.

AUTHORIZED = [160 20; 320 40; 400 50];
if ~any(all(AUTHORIZED == [nelx nely], 2))
    error('cv_run:UnauthorizedMesh', ...
        ['%dx%d is not an authorized candidate mesh. Exactly three runs are ' ...
         'authorized: 160x20, 320x40, 400x50.'], nelx, nely);
end

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','dynamical_regime','scripts'));

guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);

[cfg, meta] = cv_config('C', nelx, nely);
NE   = nelx*nely;
rho0 = olh.config.getPath(cfg,'design.initial');
tag  = sprintf('C%dx%d', nelx, nely);

fprintf('\n%s\n[cv_run] CANDIDATE %dx%d  NE=%d  cap=%d\n%s\n', ...
    repmat('=',1,72), nelx, nely, NE, meta.maxOuter, repmat('=',1,72));
fprintf('  signal=%s  stop.rule=%s  levels=%s  tol=%.6g\n', ...
    cfg.move.continuation.signal, cfg.stop.rule, mat2str(cfg.move.levels), cfg.stop.tolerance);
cfgHash = olhoffcurrent_config_hash(cfg);
fprintf('  cfgHash=%s\n', cfgHash);

% ---- the scope lock, asserted in code before the solve ------------------
g = @(p) olh.config.getPath(cfg,p);
assert(g('material.stiffness.p') == 3);
assert(~g('material.stiffness.continuation.enabled'));
assert(strcmp(g('material.mass.model'),'eq4b'));
assert(g('material.mass.q') == 1);
assert(strcmp(g('filter.type'),'sensitivity') && strcmp(g('filter.applyTo'),'all'));
assert(g('filter.radiusPhysical') == 0.06);
assert(~g('projection.enabled'));
assert(strcmp(g('multiplicity.method'),'subspace') && g('multiplicity.subspaceSize') == 2);
assert(g('multiplicity.diagonalOffsets') && g('multiplicity.offDiagonal'));
assert(strcmp(g('optimizer.inner.variant'),'published'));
assert(isequal(g('move.levels'), [0.04 0.02 0.01 0.005]));
assert(g('stop.tolerance') == 0.05*sqrt(NE/3200));

t0 = tic;
res = olhoffSolve(cfg);
wall = toc(t0);

n = numel(res.hist.N);
fprintf('[cv_run] status=%s nOuter=%d wall=%.1fs inner=%d\n', ...
    res.status, n, wall, sum(res.hist.nInner));

% ---- rebuild the raw trajectory, and prove the rebuild is exact ---------
assert(isfield(res,'diag') && ~isempty(res.diag.drho), 'cv_run:NoDiag', ...
    'runtime.diagnostics must be on: the raw trajectory is rebuilt from res.diag.drho');
rhomin = g('design.minimum');
RHO  = zeros(NE, n);  DRHO = zeros(NE, n);
r = rho0*ones(NE,1);
for k = 1:n
    d = res.diag.drho{k};
    DRHO(:,k) = d;
    r = min(1, max(rhomin, r + d));
    RHO(:,k) = r;
end
assert(isequal(RHO(:,end), res.rho), 'cv_run:RebuildMismatch', ...
    'the rebuilt trajectory does not end at res.rho');
clampErr = max(max(abs(diff([rho0*ones(NE,1) RHO],1,2) - DRHO)));
fprintf('[cv_run] trajectory rebuilt exactly; clamp displacement max = %.3e\n', clampErr);

% ---- telemetry ----------------------------------------------------------
per = cv_telemetry(res, RHO, NE, rho0);

% ---- durable raw evidence ----------------------------------------------
evDir = fullfile(root,'evidence','two_branch_controller_validation');
if ~isfolder(evDir), mkdir(evDir); end
trajFile = fullfile(evDir, sprintf('%s_trajectory.mat', tag));
hist = res.hist; exh = res.exhaustion; log = res.log; %#ok<NASGU>
move = res.hist.move(:); %#ok<NASGU>
meta.matlab = version; meta.cfgHash = cfgHash; meta.wall_s = wall;
meta.implTree = olhoffcurrent_source_manifest('Verify',false).treeHash;
save(trajFile, 'RHO','DRHO','move','hist','cfg','meta','exh','log','-v7.3');
d = dir(trajFile);
fprintf('[cv_run] wrote %s (%.1f MB)\n', trajFile, d.bytes/1e6);

% ---- per-iteration CSV (tracked metadata) ------------------------------
csvFile = fullfile(study,'runs',sprintf('%s_iterations.csv', tag));
cv_export(per, csvFile);
fprintf('[cv_run] wrote %s\n', csvFile);

% ---- scalar record ------------------------------------------------------
out = struct();
out.tag = tag; out.mesh = [nelx nely]; out.NE = NE;
out.status = res.status; out.nOuter = n; out.wall_s = wall;
out.innerTotal = sum(res.hist.nInner);
out.innerMax   = max(res.hist.nInner);
out.innerNonConv = sum(~res.hist.innerConv);
out.cfgHash = cfgHash; out.cap = meta.maxOuter;
out.tol = cfg.stop.tolerance;
out.omega = res.omega(1:min(5,numel(res.omega)));
out.omega1 = res.omega(1); out.omega2 = res.omega(2);
out.gap12 = (res.omega(2)-res.omega(1))/res.omega(1);
out.volume_final = mean(res.rho);
out.Mnd_final = per.Mnd(end); out.gray_final = per.gray(end); out.mid_final = per.mid(end);
out.move_final = per.move(end); out.stage_final = per.stage(end);
out.rho_sha256 = local_vecHash(res.rho);
out.trajectory = strrep(trajFile, [repo filesep], '');
out.trajectoryBytes = d.bytes;
out.csv = strrep(csvFile, [repo filesep], '');
out.exhaustion = exh;
out.descents = exh.descents;      % [iterApplied stageFrom declIter declBegin]
out.eventBranch = exh.eventBranch;
out.terminalDeclared = exh.terminalDeclared;
out.terminalDeclIter = exh.terminalDeclIter;
out.terminalDeclBegin = exh.terminalDeclBegin;
out.terminalBranch = exh.terminalBranch;
out.betaStallFirst = local_first(per.betaStallFires);
out.prodStopFirst  = local_first(per.prodStopAdmit);
out.prodShadowDescents = find(diff([1; per.prodStageShadow]) > 0).';
out.log = res.log;
out.matlab = version; out.implTree = meta.implTree;
out.clampDisplacementMax = clampErr;

recFile = fullfile(study,'runs',sprintf('%s_record.json', tag));
fid = fopen(recFile,'w'); fwrite(fid, jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[cv_run] wrote %s\n', recFile);

fprintf('[cv_run] descents (iter stageFrom declIter declBegin):\n'); disp(exh.descents);
fprintf('[cv_run] branches: %s\n', strjoin(exh.eventBranch, ', '));
fprintf('[cv_run] final: move=%g stage=%d omega1=%.9f Mnd=%.6f vol=%.9f\n', ...
    out.move_final, out.stage_final, out.omega1, out.Mnd_final, out.volume_final);
end

function k = local_first(v), k = find(v,1); if isempty(k), k = NaN; end, end

function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
