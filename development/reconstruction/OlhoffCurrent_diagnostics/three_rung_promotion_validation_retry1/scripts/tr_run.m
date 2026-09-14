function out = tr_run()
%TR_RUN  THE ONE authorized scientific run of this retry.
%
%   320x40, three-rung ladder [0.04 0.02 0.01], frozen two-branch stage
%   exhaustion E = A OR B.  Exactly one mesh is authorized and the function
%   refuses to be told otherwise -- the mesh is not a parameter.
%
%   Structure, telemetry and evidence handling are REUSED from
%   two_branch_controller_validation (cv_telemetry, cv_export), called rather
%   than re-typed, so the candidate's CSV is column-for-column the oracle's.

NELX = 320; NELY = 40;                     % not a parameter, by design

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
addpath(fullfile(root,'diagnostics','dynamical_regime','scripts'));

guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);

[cfg, meta] = tr_config(NELX, NELY);
NE   = NELX*NELY;
rho0 = olh.config.getPath(cfg,'design.initial');
tag  = sprintf('C%dx%d_three_rung', NELX, NELY);

fprintf('\n%s\n[tr_run] THREE-RUNG CANDIDATE %dx%d  NE=%d  cap=%d\n%s\n', ...
    repmat('=',1,72), NELX, NELY, NE, meta.maxOuter, repmat('=',1,72));
fprintf('  signal=%s  stop.rule=%s  levels=%s  tol=%.6g\n', ...
    cfg.move.continuation.signal, cfg.stop.rule, mat2str(cfg.move.levels), cfg.stop.tolerance);
cfgHash = olhoffcurrent_config_hash(cfg);
fprintf('  cfgHash=%s\n', cfgHash);

% ---- the scope lock, asserted in code before the solve ------------------
% Identical to cv_run's lock except for the one field under test.
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
assert(strcmp(g('move.continuation.signal'),'stageExhaustion'));
assert(strcmp(g('stop.rule'),'stageExhaustion'));
assert(isequal(g('move.levels'), [0.04 0.02 0.01]), 'tr_run:LadderNotThreeRung', ...
    'the candidate must carry the three-rung ladder and nothing else');
assert(g('stop.tolerance') == 0.05*sqrt(NE/3200));
assert(g('runtime.maxOuter') == 1600);
assert(g('runtime.singleThread'));

t0 = tic;
res = olhoffSolve(cfg);
wall = toc(t0);

n = numel(res.hist.N);
fprintf('[tr_run] status=%s nOuter=%d wall=%.1fs inner=%d\n', ...
    res.status, n, wall, sum(res.hist.nInner));

% ---- rebuild the raw trajectory, and prove the rebuild is exact ---------
assert(isfield(res,'diag') && ~isempty(res.diag.drho), 'tr_run:NoDiag', ...
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
assert(isequal(RHO(:,end), res.rho), 'tr_run:RebuildMismatch', ...
    'the rebuilt trajectory does not end at res.rho');
clampErr = max(max(abs(diff([rho0*ones(NE,1) RHO],1,2) - DRHO)));
fprintf('[tr_run] trajectory rebuilt exactly; clamp displacement max = %.3e\n', clampErr);

% ---- telemetry, by the oracle study's own definitions -------------------
per = cv_telemetry(res, RHO, NE, rho0);

% ---- durable raw evidence ----------------------------------------------
evDir = fullfile(root,'evidence','three_rung_promotion_validation_retry1');
if ~isfolder(evDir), mkdir(evDir); end
trajFile = fullfile(evDir, sprintf('%s_trajectory.mat', tag));
hist = res.hist; exh = res.exhaustion; log = res.log; %#ok<NASGU>
move = res.hist.move(:); %#ok<NASGU>
meta.matlab = version; meta.cfgHash = cfgHash; meta.wall_s = wall;
meta.implTree = olhoffcurrent_source_manifest('Verify',false).treeHash;
save(trajFile, 'RHO','DRHO','move','hist','cfg','meta','exh','log','-v7.3');
d = dir(trajFile);
fprintf('[tr_run] wrote %s (%.1f MB)\n', trajFile, d.bytes/1e6);

% ---- per-iteration CSV --------------------------------------------------
csvFile = fullfile(study,'runs',sprintf('%s_iterations.csv', tag));
cv_export(per, csvFile);
fprintf('[tr_run] wrote %s\n', csvFile);

% ---- scalar record ------------------------------------------------------
out = struct();
out.tag = tag; out.mesh = [NELX NELY]; out.NE = NE;
out.status = res.status; out.nOuter = n; out.wall_s = wall;
out.innerTotal = sum(res.hist.nInner);
out.innerMax   = max(res.hist.nInner);
out.innerNonConv = sum(~res.hist.innerConv);
out.cfgHash = cfgHash; out.cap = meta.maxOuter;
out.tol = cfg.stop.tolerance;
out.levels = cfg.move.levels;
out.omega = res.omega(1:min(5,numel(res.omega)));
out.omega1 = res.omega(1); out.omega2 = res.omega(2);
out.gap12 = (res.omega(2)-res.omega(1))/res.omega(1);
out.volume_final = mean(res.rho);
out.Mnd_final = per.Mnd(end); out.gray_final = per.gray(end); out.mid_final = per.mid(end);
out.move_final = per.move(end); out.stage_final = per.stage(end);
out.rho_sha256 = local_vecHash(res.rho);
out.rho_prefix352_sha256 = '';
out.omega_prefix352_sha256 = '';
if n >= 352
    out.rho_prefix352_sha256   = local_vecHash(RHO(:,1:352));
    out.omega_prefix352_sha256 = local_vecHash(res.hist.omega(1:2,1:352));
end
out.trajectory = strrep(trajFile, [repo filesep], '');
out.trajectoryBytes = d.bytes;
out.trajectorySha256 = olhoffcurrent_sha256_file(trajFile);
out.csv = strrep(csvFile, [repo filesep], '');
out.csvSha256 = olhoffcurrent_sha256_file(csvFile);
out.exhaustion = exh;
out.descents = exh.descents;      % [iterApplied stageFrom declIter declBegin]
out.eventBranch = exh.eventBranch;
out.stageStarts = exh.stageStarts;
out.implTree = meta.implTree;
out.matlab = version;
out.log = res.log;

recFile = fullfile(study,'runs',sprintf('%s_record.json', tag));
fid = fopen(recFile,'w'); c = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));
fprintf('[tr_run] wrote %s\n', recFile);

fprintf('\n  status      : %s @ %d\n', out.status, out.nOuter);
fprintf('  stage_final : %d   move_final : %.4g\n', out.stage_final, out.move_final);
fprintf('  innerTotal  : %d  (max %d, nonconv %d)\n', out.innerTotal, out.innerMax, out.innerNonConv);
fprintf('  rho_sha256  : %s\n', out.rho_sha256);
fprintf('  descents    :\n'); disp(out.descents);
fprintf('  branches    : %s\n', strjoin(out.eventBranch,' '));
end

function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
