function out = fm_run(nelx, nely, outDir)
%FM_RUN  Execute RUN C with full raw retention and the native-stop replay.
%
%   Nothing under +impl/ is modified and NO solver copy exists: the run calls
%   olhoffSolve through the canonical configuration route.  Per-iteration
%   telemetry is produced by dr_telemetry from the preceding dynamical_regime
%   study, called BY REFERENCE so the dynamical definitions are provably the
%   frozen ones rather than a re-typed copy.

guard = olhoffcurrent_paths(); %#ok<NASGU>
man = olhoffcurrent_source_manifest('Verify', true);
cur = olhoffcurrent_currentness('Verbose', false);
assert(man.ok, 'fm_run:Integrity', 'OlhoffCurrent source integrity FAILED');
assert(strcmp(cur.state,'CURRENT'), 'fm_run:NotCurrent', 'currentness = %s', cur.state);
maxNumCompThreads(1);

[cfg, meta] = fm_config(nelx, nely);

% ---- single-factor gate: only the DECLARED overrides may differ ----------
prod = olhoffcurrent_config(nelx, nely, 'MaxOuter', meta.maxOuter, 'Diagnostics', true);
declared = {'runtime.name'};
for i = 1:2:numel(meta.overrides), declared{end+1} = meta.overrides{i}; end %#ok<AGROW>
S = olh.config.schema(); bad = {};
for k = 1:size(S,1)
    p = S{k,1};
    if any(strcmp(p, declared)); continue; end
    if ~isequaln(olh.config.getPath(cfg,p), olh.config.getPath(prod,p))
        bad{end+1} = p; %#ok<AGROW>
    end
end
assert(isempty(bad), 'fm_run:UnexpectedDrift', ...
    'RUN C differs from production outside its declared overrides: %s', strjoin(bad,', '));

% ---- preregistered scope lock, asserted not assumed ---------------------
lock = {'material.stiffness.p',3; 'material.mass.model','eq4b'; 'material.mass.q',1; ...
        'filter.type','sensitivity'; 'filter.radiusPhysical',0.06; ...
        'projection.enabled',false; 'optimizer.inner.variant','published'; ...
        'multiplicity.method','subspace'; 'multiplicity.subspaceSize',2; ...
        'multiplicity.offDiagonal',true; 'move.levels',[0.04 0.02 0.01 0.005]};
for k = 1:size(lock,1)
    v = olh.config.getPath(cfg, lock{k,1});
    assert(isequaln(v, lock{k,2}), 'fm_run:ScopeLock', ...
        'scope-locked field %s = %s', lock{k,1}, mat2str(v));
end

prodTol  = olh.config.getPath(prod,'stop.tolerance');
prodNorm = olh.config.getPath(prod,'stop.norm');
fprintf('[fm_run] %s  cap=%d  movePolicy=%s  stopTol=%g  (production tol would be %g, norm %s)\n', ...
    meta.label, meta.maxOuter, olh.config.getPath(cfg,'move.policy'), ...
    olh.config.getPath(cfg,'stop.tolerance'), prodTol, prodNorm);

tS = tic;  res = olhoffSolve(cfg);  wall = toc(tS);

% ---- exact per-iteration density reconstruction --------------------------
NE = nelx*nely;
rhoMin = olh.config.getPath(cfg,'design.minimum');
rho0   = olh.config.getPath(cfg,'design.initial');
h = res.hist;  nO = numel(h.N);
assert(isfield(res,'diag') && numel(res.diag.drho)==nO, 'fm_run:NoDiag','recorder truncated');

rho = rho0*ones(NE,1);  RHO = zeros(NE,nO);  vErr = 0;
for k = 1:nO
    rho = min(1, max(rhoMin, rho + res.diag.drho{k}));
    RHO(:,k) = rho;  vErr = max(vErr, abs(mean(rho)-h.vol(k)));
end
assert(vErr < 1e-12, 'fm_run:ReconstructionFailed','vol mismatch %.3e', vErr);
assert(isequaln(RHO(:,end), double(res.rho(:))), 'fm_run:DesignNotPhysical', ...
    'reconstructed design is not the physical density');

% ---- telemetry: the FROZEN definitions, called by reference -------------
P = dr_telemetry(h, RHO, NE, rho0);

% ---- inherited native-stop predicate, replayed offline ------------------
% Validated before freezing against fixedmove_320x40 (216) and runA_400x50 (139).
P.nativeStopTol  = prodTol;
P.nativeStop     = false(nO,1);
for k = 2:nO
    P.nativeStop(k) = (P.l2(k) < prodTol) && (P.move(k) == P.move(k-1));
end
kNative = find(P.nativeStop,1);
if isempty(kNative), kNative = NaN; end

out = struct('run',meta.run,'label',meta.label,'mesh',[nelx nely],'NE',NE, ...
    'nOuter',nO,'innerTotal',sum(h.nInner),'wall_s',wall,'status',res.status, ...
    'omega',double(res.omega(:)),'rhoFinal',RHO(:,end), ...
    'Mnd_final',P.Mnd(end),'gray_final',P.gray(end),'mid_final',P.mid(end), ...
    'volume_final',P.volume(end),'per',P,'log',{res.log}, ...
    'cfgHash',olhoffcurrent_config_hash(cfg),'meta',meta, ...
    'sourceTree',man.treeHash,'volReconErrMax',vErr, ...
    'prodTol',prodTol,'prodNorm',prodNorm,'nativeStopIter',kNative);

if nargin>=3 && ~isempty(outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    f = fullfile(outDir, sprintf('runC_%dx%d.mat', nelx, nely));
    save(f,'out','cfg','RHO','-v7.3');
    fprintf('[fm_run] saved %s\n', f);
end
fprintf('[fm_run] C nOuter=%d status=%s omega1=%.12g M_nd=%.4f nativeStopWouldBe=%s wall=%.0fs\n', ...
    nO, res.status, out.omega(1), out.Mnd_final, mat2str(kNative), wall);
end
