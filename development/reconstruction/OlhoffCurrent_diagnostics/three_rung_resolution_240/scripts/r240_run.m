function out = r240_run()
%R240_RUN  THE ONE authorized scientific run: C240x30, frozen four-rung controller.
%
%   This mirrors two_branch_controller_validation/scripts/cv_run.m line for line.
%   cv_run.m itself refuses any mesh outside its three authorized ones -- a guard
%   from that study's preregistration and a hashed artifact of a sealed study --
%   so it is NOT edited.  Instead cv_config, cv_telemetry and cv_export are CALLED
%   unchanged, so the controller and the telemetry are identical by reference.
%
%   Exactly one mesh is authorized here: 240x30.  Anything else is refused.

NELX = 240; NELY = 30;

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
addpath(fullfile(root,'diagnostics','dynamical_regime','scripts'));

guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);

% ---- refuse to run more than the one authorized experiment -------------
tag = sprintf('C%dx%d', NELX, NELY);
trajFile = fullfile(root,'evidence','three_rung_resolution_240', ...
                    sprintf('%s_trajectory.mat', tag));
if isfile(trajFile)
    error('r240_run:AlreadyRun', ...
        ['%s already exists.  Exactly ONE scientific run is authorized; ' ...
         'refusing to overwrite it.'], trajFile);
end

[cfg, meta] = cv_config('C', NELX, NELY);     % the SAME function as the prior arms
NE   = NELX*NELY;
rho0 = olh.config.getPath(cfg,'design.initial');

fprintf('\n%s\n[r240_run] C240x30  NE=%d  cap=%d\n%s\n', ...
    repmat('=',1,72), NE, meta.maxOuter, repmat('=',1,72));
fprintf('  signal=%s  stop.rule=%s  levels=%s  tol=%.6g\n', ...
    cfg.move.continuation.signal, cfg.stop.rule, mat2str(cfg.move.levels), cfg.stop.tolerance);
cfgHash = olhoffcurrent_config_hash(cfg);
fprintf('  cfgHash=%s\n', cfgHash);
assert(strcmp(cfgHash,'33833323efa08facaa5849c24fe32d6c9c47b5924f88c00d34fb65f7140d54d6'), ...
    'r240_run:ConfigHash', 'resolved config differs from the preregistered one');

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
assert(strcmp(g('move.policy'),'ladder'));
assert(strcmp(g('move.continuation.signal'),'stageExhaustion'));
assert(strcmp(g('stop.rule'),'stageExhaustion'));
assert(g('stop.tolerance') == 0.05*sqrt(NE/3200));
assert(g('runtime.maxOuter') == 1600);
assert(maxNumCompThreads() == 1);

t0 = tic;
res = olhoffSolve(cfg);
wall = toc(t0);

n = numel(res.hist.N);
fprintf('[r240_run] status=%s nOuter=%d wall=%.1fs inner=%d\n', ...
    res.status, n, wall, sum(res.hist.nInner));

% ---- rebuild the raw trajectory, and prove the rebuild is exact ---------
assert(isfield(res,'diag') && ~isempty(res.diag.drho), 'r240_run:NoDiag', ...
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
assert(isequal(RHO(:,end), res.rho), 'r240_run:RebuildMismatch', ...
    'the rebuilt trajectory does not end at res.rho');
clampErr = max(max(abs(diff([rho0*ones(NE,1) RHO],1,2) - DRHO)));
fprintf('[r240_run] trajectory rebuilt exactly; clamp displacement max = %.3e\n', clampErr);

% ---- telemetry, by the SAME cv_telemetry ------------------------------
per = cv_telemetry(res, RHO, NE, rho0);

% ---- durable raw evidence ----------------------------------------------
evDir = fileparts(trajFile);
if ~isfolder(evDir), mkdir(evDir); end
hist = res.hist; exh = res.exhaustion; log = res.log; %#ok<NASGU>
move = res.hist.move(:); %#ok<NASGU>
meta.matlab = version; meta.cfgHash = cfgHash; meta.wall_s = wall;
meta.implTree = olhoffcurrent_source_manifest('Verify',false).treeHash;
meta.preregSha256 = 'f86d022e5259beb0936072d204e54e3761f14b2fd903c1794a6d4ccb2c5652cc';
save(trajFile, 'RHO','DRHO','move','hist','cfg','meta','exh','log','-v7.3');
d = dir(trajFile);
fprintf('[r240_run] wrote %s (%.1f MB)\n', trajFile, d.bytes/1e6);

% ---- per-iteration CSV, by the SAME cv_export -------------------------
csvFile = fullfile(study,'runs',sprintf('%s_iterations.csv', tag));
cv_export(per, csvFile);
fprintf('[r240_run] wrote %s\n', csvFile);

% ---- scalar record ------------------------------------------------------
out = struct();
out.tag = tag; out.mesh = [NELX NELY]; out.NE = NE;
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
out.descents = exh.descents;
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
out.preregSha256 = meta.preregSha256;

recFile = fullfile(study,'runs',sprintf('%s_record.json', tag));
fid = fopen(recFile,'w'); fwrite(fid, jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[r240_run] wrote %s\n', recFile);

fprintf('[r240_run] stageStarts: %s\n', mat2str(exh.stageStarts));
fprintf('[r240_run] descents (iter stageFrom declIter declBegin):\n'); disp(exh.descents);
fprintf('[r240_run] branches: %s\n', strjoin(exh.eventBranch, ', '));
fprintf('[r240_run] final: status=%s move=%g stage=%d omega1=%.9f Mnd=%.6f vol=%.9f\n', ...
    res.status, out.move_final, out.stage_final, out.omega1, out.Mnd_final, out.volume_final);
end

function k = local_first(v), k = find(v,1); if isempty(k), k = NaN; end, end

function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
