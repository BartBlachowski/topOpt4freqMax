function out = cp_run(nelx, nely)
%CP_RUN  ONE scientific canary.  Fail-closed: the preflight runs FIRST.
%
%   out = CP_RUN(480, 60)    canary 1
%   out = CP_RUN(800, 100)   canary 2 -- authorized ONLY if canary 1 passed
%                                        its preregistered gate
%
%   Exactly two meshes are accepted and the function refuses to be told
%   otherwise: the mesh is not a free parameter, and no third canary, no
%   extension of a finished run and no restart from an intermediate design is
%   reachable through this entry point.
%
%   ORDER OF OPERATIONS, and there is no other:
%
%       cp_preflight (THROWS on any mismatch)   <- no compute has happened yet
%           -> host-load probe                  <- Part G
%           -> olhoffSolve                      <- THE one solve
%           -> trajectory rebuild + exactness proof
%           -> telemetry (cv_telemetry, by reference)
%           -> durable evidence, then CSV, then scalar record
%
%   If the preflight throws, this function has executed ZERO optimization.
%   That is the whole point: the mismatch is refused BEFORE compute, not
%   reported after it.
%
%   Structure, telemetry and evidence handling are REUSED from
%   two_branch_controller_validation (cv_telemetry, cv_export) and from
%   three_rung_promotion_validation_retry1 (tr_config), called rather than
%   re-typed, so this canary's CSV is column-for-column the validated C320's.
%
%   See also CP_PREFLIGHT, CP_CONFIG, CP_FIXEDWORK.

assert(nargin == 2, 'cp_run:Args', 'cp_run(nelx, nely)');
ok = (nelx == 480 && nely == 60) || (nelx == 800 && nely == 100);
assert(ok, 'cp_run:MeshNotAuthorized', ...
    ['this study authorizes exactly two canaries, 480x60 and 800x100; ' ...
     'got %dx%d.  Running a third mesh here would make the campaign ' ...
     'decision unpreregistered.'], nelx, nely);

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
addpath(fullfile(root,'diagnostics','three_rung_promotion_validation_retry1','scripts'));
addpath(fullfile(root,'diagnostics','dynamical_regime','scripts'));

% =====================================================================
% GATE.  Throws before any optimization if the deployment is not proved.
% =====================================================================
pf = cp_preflight(nelx, nely, 'Throw', true);
cfg = pf.cfg; meta = pf.meta;

guard = olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);

NE   = nelx*nely;
rho0 = olh.config.getPath(cfg,'design.initial');
tag  = sprintf('C%dx%d_three_rung', nelx, nely);

fprintf('\n%s\n[cp_run] THREE-RUNG CANARY %dx%d  NE=%d  cap=%d\n%s\n', ...
    repmat('=',1,72), nelx, nely, NE, olh.config.getPath(cfg,'runtime.maxOuter'), ...
    repmat('=',1,72));
fprintf('  signal=%s  stop.rule=%s  levels=%s  eps=%.6g\n', ...
    cfg.move.continuation.signal, cfg.stop.rule, ...
    mat2str(cfg.move.levels), cfg.stop.tolerance);
fprintf('  cfgHash=%s  (frozen %s)\n', pf.record.cfgHash, pf.record.cfgHash_frozen);

% ---- Part G: host-load probe, immediately before the solve --------------
pre = cp_hostprobe(sprintf('%s_pre', tag), study);

% =====================================================================
% THE ONE SOLVE.  Its outcome is FINAL for this task.
% =====================================================================
t0 = tic;
res = olhoffSolve(cfg);
wall = toc(t0);

post = cp_hostprobe(sprintf('%s_post', tag), study);

n = numel(res.hist.N);
fprintf('[cp_run] status=%s nOuter=%d wall=%.1fs inner=%d\n', ...
    res.status, n, wall, sum(res.hist.nInner));

% ---- rebuild the raw trajectory, and prove the rebuild is exact ---------
assert(isfield(res,'diag') && ~isempty(res.diag.drho), 'cp_run:NoDiag', ...
    'runtime.diagnostics must be on: the raw trajectory is rebuilt from res.diag.drho');
rhomin = olh.config.getPath(cfg,'design.minimum');
RHO  = zeros(NE, n);  DRHO = zeros(NE, n);
r = rho0*ones(NE,1);
for k = 1:n
    d = res.diag.drho{k};
    DRHO(:,k) = d;
    r = min(1, max(rhomin, r + d));
    RHO(:,k) = r;
end
assert(isequal(RHO(:,end), res.rho), 'cp_run:RebuildMismatch', ...
    'the rebuilt trajectory does not end at res.rho');
clampErr = max(max(abs(diff([rho0*ones(NE,1) RHO],1,2) - DRHO)));
fprintf('[cp_run] trajectory rebuilt exactly; clamp displacement max = %.3e\n', clampErr);

% ---- telemetry, by the validated study's own definitions ----------------
per = cv_telemetry(res, RHO, NE, rho0);

% ---- SAVED STATE for the Part F fixed-work benchmark --------------------
% The terminal design and the configuration, and nothing else: the benchmark
% re-evaluates kernels AT this state and never updates it.
stateFile = fullfile(root,'evidence','three_rung_canary_preflight', ...
                     sprintf('%s_state.mat', tag));
if ~isfolder(fileparts(stateFile)), mkdir(fileparts(stateFile)); end
state = struct('rho', res.rho, 'cfg', cfg, 'mesh', [nelx nely], 'NE', NE, ...
               'nOuter', n, 'tag', tag); %#ok<NASGU>
save(stateFile, 'state', '-v7.3');

% ---- durable raw evidence ----------------------------------------------
evDir = fullfile(root,'evidence','three_rung_canary_preflight');
trajFile = fullfile(evDir, sprintf('%s_trajectory.mat', tag));
hist = res.hist; exh = res.exhaustion; log = res.log; %#ok<NASGU>
move = res.hist.move(:); %#ok<NASGU>
meta.matlab = version; meta.cfgHash = pf.record.cfgHash; meta.wall_s = wall;
meta.implTree = pf.record.implTree;
meta.preflight = rmfield(pf, {'cfg','meta'});
meta.hostPre = pre; meta.hostPost = post;
save(trajFile, 'RHO','DRHO','move','hist','cfg','meta','exh','log','-v7.3');
d = dir(trajFile);
fprintf('[cp_run] wrote %s (%.1f MB)\n', trajFile, d.bytes/1e6);

% ---- per-iteration CSV --------------------------------------------------
csvFile = fullfile(study,'runs',sprintf('%s_iterations.csv', tag));
if ~isfolder(fileparts(csvFile)), mkdir(fileparts(csvFile)); end
cv_export(per, csvFile);
fprintf('[cp_run] wrote %s\n', csvFile);

% ---- scalar record ------------------------------------------------------
out = struct();
out.tag = tag; out.mesh = [nelx nely]; out.NE = NE;
out.freeDOF = res.mdl.ndof - numel(res.mdl.fixed);
out.status = res.status; out.nOuter = n; out.wall_s = wall;
out.innerTotal = sum(res.hist.nInner);
out.innerMax   = max(res.hist.nInner);
out.innerNonConv = sum(~res.hist.innerConv);
out.cfgHash = pf.record.cfgHash;
out.implTree = pf.record.implTree;
out.cap = olh.config.getPath(cfg,'runtime.maxOuter');
out.tol = cfg.stop.tolerance;
out.levels = cfg.move.levels;
out.omega = res.omega(1:min(5,numel(res.omega)));
out.omega1 = res.omega(1); out.omega2 = res.omega(2);
out.omega3 = res.omega(min(3,numel(res.omega)));
out.gap12 = (res.omega(2)-res.omega(1))/res.omega(1);
out.gap23 = (res.omega(3)-res.omega(2))/res.omega(2);
out.volume_final = mean(res.rho);
out.Mnd_final = per.Mnd(end); out.gray_final = per.gray(end); out.mid_final = per.mid(end);
out.move_final = per.move(end); out.stage_final = per.stage(end);
out.multN_final = res.hist.N(end);
out.multJ_count = sum(res.hist.multJ);
out.multJ_first = find(res.hist.multJ, 1);
out.terminal_l2 = res.hist.dxNorm2(end);
out.terminal_max = res.hist.dxOuter(end);

% ---- controller events, read off the solver's own exhaustion record -----
out.stageStarts = res.exhaustion.stageStarts;
out.descents    = res.exhaustion.descents;
out.terminalDeclared  = res.exhaustion.terminalDeclared;
out.terminalDeclIter  = res.exhaustion.terminalDeclIter;
out.terminalDeclBegin = res.exhaustion.terminalDeclBegin;
out.terminalBranch    = res.exhaustion.terminalBranch;
out.eventBranch       = res.exhaustion.eventBranch;
out.signalDrivesMove  = res.exhaustion.signalDrivesMove;
out.ruleAdmitsStop    = res.exhaustion.ruleAdmitsStop;

% ---- timing decomposition, Part E --------------------------------------
out.t = struct('total', wall, ...
    'eig',   sum(res.hist.tEig),   'grad',  sum(res.hist.tGrad), ...
    'inner', sum(res.hist.tInner), 'outer', sum(res.hist.tOuter));
out.t.other = out.t.outer - out.t.eig - out.t.grad - out.t.inner;
out.t.per_outer       = out.t.outer/n;
out.t.eig_per_outer   = out.t.eig/n;
out.t.grad_per_outer  = out.t.grad/n;
out.t.inner_per_outer = out.t.inner/n;
out.t.inner_per_mma   = out.t.inner/out.innerTotal;
out.t.mean_inner_per_outer = out.innerTotal/n;

out.rho_sha256 = local_vecHash(res.rho);
out.preflight = rmfield(pf, {'cfg','meta'});
out.hostPre = pre; out.hostPost = post;
out.files = struct('trajectory', trajFile, 'state', stateFile, 'csv', csvFile);

recFile = fullfile(study,'runs',sprintf('%s_record.json', tag));
fid = fopen(recFile,'w'); fprintf(fid,'%s', jsonencode(out,'PrettyPrint',true)); fclose(fid);
fprintf('[cp_run] wrote %s\n', recFile);
end

% =========================================================================
function h = local_vecHash(v)
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
