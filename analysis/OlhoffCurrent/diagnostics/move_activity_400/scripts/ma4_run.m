function out = ma4_run(arm, nelx, nely)
%MA4_RUN  Execute one arm of the 400x50 measurement with DURABLE raw retention.
%
%   The solver is NOT modified and no move controller is added: this study
%   measures the existing production mechanism, it does not test a new one.
%
%   What distinguishes this from the three studies whose trajectories were lost:
%   the element-level history is written to the declared durable evidence root
%   and registered in EVIDENCE.json with a SHA-256, so olhoffcurrent_evidence_gate
%   will FAIL the study if it ever goes missing again.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = olhoffcurrent_root();
repo  = fileparts(fileparts(root));

guard = olhoffcurrent_paths(); %#ok<NASGU>
dsp   = olhoffcurrent_assert_dispatch(); %#ok<NASGU>
man   = olhoffcurrent_source_manifest('Verify', true);
cur   = olhoffcurrent_currentness('Verbose', false);
assert(man.ok, 'ma4_run:Integrity', 'OlhoffCurrent source integrity FAILED');
assert(strcmp(cur.state,'CURRENT'), 'ma4_run:NotCurrent', 'currentness = %s', cur.state);
maxNumCompThreads(1);

[cfg, meta] = ma4_config(arm, nelx, nely);

% ---- the ONLY permitted difference from production ----------------------
prod = olhoffcurrent_config(nelx, nely, 'Diagnostics', true);
S = olh.config.schema();
allowed = {'runtime.name','runtime.maxOuter','move.policy','move.initial'};
if strcmp(meta.arm,'P'); allowed = {'runtime.name'}; end
bad = {};
for k = 1:size(S,1)
    p = S{k,1};
    if any(strcmp(p, allowed)); continue; end
    if ~isequaln(olh.config.getPath(cfg,p), olh.config.getPath(prod,p))
        bad{end+1} = p; %#ok<AGROW>
    end
end
assert(isempty(bad), 'ma4_run:UnexpectedDrift', ...
    'arm %s differs from production outside its declared overrides: %s', ...
    meta.arm, strjoin(bad,', '));

% Scope lock: the fields this study must not touch, asserted not assumed.
lock = {'material.stiffness.p',3; 'material.mass.model','eq4b'; 'material.mass.q',1; ...
        'filter.type','sensitivity'; 'filter.radiusPhysical',0.06; ...
        'projection.enabled',false; 'optimizer.inner.variant','published'; ...
        'multiplicity.method','subspace'; 'multiplicity.subspaceSize',2; ...
        'multiplicity.offDiagonal',true; 'move.levels',[0.04 0.02 0.01 0.005]};
for k = 1:size(lock,1)
    v = olh.config.getPath(cfg, lock{k,1});
    assert(isequaln(v, lock{k,2}), 'ma4_run:ScopeLock', ...
        'scope-locked field %s = %s', lock{k,1}, mat2str(v));
end

fprintf('[ma4_run] %s  %dx%d  cap=%d  move.policy=%s\n', meta.label, nelx, nely, ...
        meta.maxOuter, olh.config.getPath(cfg,'move.policy'));
tS = tic; res = olhoffSolve(cfg); wall = toc(tS);

% ---- exact per-iteration density reconstruction -------------------------
NE     = nelx*nely;
rhoMin = olh.config.getPath(cfg,'design.minimum');
rho0   = olh.config.getPath(cfg,'design.initial');
h  = res.hist; nO = numel(h.N);
assert(isfield(res,'diag') && numel(res.diag.drho)==nO, 'ma4_run:NoDiag','recorder truncated');

RHO  = zeros(NE, nO);
DRHO = zeros(NE, nO);
rho  = rho0*ones(NE,1);
vErr = 0; clampErr = 0;
for k = 1:nO
    d   = res.diag.drho{k};
    new = min(1, max(rhoMin, rho + d));
    % Is the update's own clamp inert, so that the recorded increment IS the
    % applied increment?  Prior studies asserted this in prose; measure it.
    clampErr    = max(clampErr, max(abs((new - rho) - d)));
    rho         = new;
    RHO(:,k)    = rho;
    DRHO(:,k)   = d;
    vErr        = max(vErr, abs(mean(rho) - h.vol(k)));
end
assert(vErr < 1e-12, 'ma4_run:ReconstructionFailed', 'volume mismatch %.3e', vErr);
assert(isequaln(RHO(:,end), double(res.rho(:))), 'ma4_run:DesignNotPhysical', ...
    'reconstructed final design is not the physical density');

% ---- durable raw evidence ----------------------------------------------
evRel = fullfile('analysis','OlhoffCurrent','evidence','move_activity_400');
evAbs = fullfile(repo, evRel);
if exist(evAbs,'dir')~=7; mkdir(evAbs); end
trajName = sprintf('%s400_%dx%d_trajectory.mat', meta.arm, nelx, nely);
trajPath = fullfile(evAbs, trajName);
move = h.move(:); hist = h; %#ok<NASGU>
save(trajPath, 'RHO', 'DRHO', 'move', 'hist', 'cfg', 'meta', '-v7.3');
dd = dir(trajPath);
fprintf('[ma4_run] trajectory -> %s  (%.1f MB)\n', trajPath, dd(1).bytes/1e6);

out = struct('arm', meta.arm, 'label', meta.label, 'mesh', [nelx nely], 'NE', NE, ...
    'nOuter', nO, 'innerTotal', sum(h.nInner), 'wall_s', wall, 'status', res.status, ...
    'omega', double(res.omega(:)), 'volReconErrMax', vErr, 'clampInertMaxErr', clampErr, ...
    'trajectoryPath', fullfile(evRel, trajName), 'trajectoryBytes', dd(1).bytes, ...
    'cfgHash', olhoffcurrent_config_hash(cfg), 'sourceTree', man.treeHash, ...
    'matlab', version, 'meta', meta, 'log', {res.log}, 'maxOuter', meta.maxOuter);

% per-iteration telemetry + activity distribution
P = ma4_telemetry(h, RHO, DRHO, NE);
out.per = P;
ma4_export(P, fullfile(study,'runs', sprintf('%s400_%dx%d_iterations.csv', meta.arm, nelx, nely)));

fprintf(['[ma4_run] %s  nOuter=%d  status=%s  omega1=%.10g  M_nd=%.4f%%  ' ...
         'clampInert=%.3e  wall=%.0fs\n'], meta.arm, nO, res.status, out.omega(1), ...
        P.Mnd(end), clampErr, wall);
end
