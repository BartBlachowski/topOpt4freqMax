function out = dr_run(run, nelx, nely, outDir)
%DR_RUN  Execute one preregistered scientific run with full raw retention.
%
%   Nothing under +impl/ is modified and NO solver copy exists: both runs call
%   olhoffSolve through the canonical configuration route.  Per-iteration
%   density is reconstructed exactly as olhoffSolve forms it and validated
%   against hist.vol; a reconstruction failure aborts.

guard = olhoffcurrent_paths(); %#ok<NASGU>
man = olhoffcurrent_source_manifest('Verify', true);
cur = olhoffcurrent_currentness('Verbose', false);
assert(man.ok, 'dr_run:Integrity', 'OlhoffCurrent source integrity FAILED');
assert(strcmp(cur.state,'CURRENT'), 'dr_run:NotCurrent', 'currentness = %s', cur.state);
maxNumCompThreads(1);

[cfg, meta] = dr_config(run, nelx, nely);

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
assert(isempty(bad), 'dr_run:UnexpectedDrift', ...
    'run %s differs from production outside its declared overrides: %s', ...
    meta.run, strjoin(bad,', '));

% ---- preregistered scope lock, asserted not assumed ---------------------
lock = {'material.stiffness.p',3; 'material.mass.model','eq4b'; 'material.mass.q',1; ...
        'filter.type','sensitivity'; 'filter.radiusPhysical',0.06; ...
        'projection.enabled',false; 'optimizer.inner.variant','published'; ...
        'multiplicity.method','subspace'; 'multiplicity.subspaceSize',2; ...
        'multiplicity.offDiagonal',true; 'move.levels',[0.04 0.02 0.01 0.005]};
for k = 1:size(lock,1)
    v = olh.config.getPath(cfg, lock{k,1});
    assert(isequaln(v, lock{k,2}), 'dr_run:ScopeLock', ...
        'scope-locked field %s = %s', lock{k,1}, mat2str(v));
end

fprintf('[dr_run] %s  %dx%d  cap=%d  movePolicy=%s  stopTol=%g\n', meta.label, ...
    nelx, nely, meta.maxOuter, olh.config.getPath(cfg,'move.policy'), ...
    olh.config.getPath(cfg,'stop.tolerance'));

tS = tic;  res = olhoffSolve(cfg);  wall = toc(tS);

% ---- exact per-iteration density reconstruction --------------------------
NE = nelx*nely;
rhoMin = olh.config.getPath(cfg,'design.minimum');
rho0   = olh.config.getPath(cfg,'design.initial');
h = res.hist;  nO = numel(h.N);
assert(isfield(res,'diag') && numel(res.diag.drho)==nO, 'dr_run:NoDiag','recorder truncated');

rho = rho0*ones(NE,1);  RHO = zeros(NE,nO);  vErr = 0;
for k = 1:nO
    rho = min(1, max(rhoMin, rho + res.diag.drho{k}));
    RHO(:,k) = rho;  vErr = max(vErr, abs(mean(rho)-h.vol(k)));
end
assert(vErr < 1e-12, 'dr_run:ReconstructionFailed','vol mismatch %.3e', vErr);
assert(isequaln(RHO(:,end), double(res.rho(:))), 'dr_run:DesignNotPhysical', ...
    'reconstructed design is not the physical density');

P = dr_telemetry(h, RHO, NE, rho0);

out = struct('run',meta.run,'label',meta.label,'mesh',[nelx nely],'NE',NE, ...
    'nOuter',nO,'innerTotal',sum(h.nInner),'wall_s',wall,'status',res.status, ...
    'omega',double(res.omega(:)),'rhoFinal',RHO(:,end), ...
    'Mnd_final',P.Mnd(end),'gray_final',P.gray(end),'mid_final',P.mid(end), ...
    'volume_final',P.volume(end),'per',P,'log',{res.log}, ...
    'cfgHash',olhoffcurrent_config_hash(cfg),'meta',meta, ...
    'sourceTree',man.treeHash,'volReconErrMax',vErr, ...
    'prodTol',olh.config.getPath(prod,'stop.tolerance'), ...
    'prodNorm',olh.config.getPath(prod,'stop.norm'));

if nargin>=4 && ~isempty(outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    f = fullfile(outDir, sprintf('run%s_%dx%d.mat', meta.run, nelx, nely));
    save(f,'out','cfg','RHO','-v7.3');
    fprintf('[dr_run] saved %s\n', f);
end
fprintf('[dr_run] %s nOuter=%d status=%s omega1=%.12g M_nd=%.4f wall=%.0fs\n', ...
    meta.run, nO, res.status, out.omega(1), out.Mnd_final, wall);
end
