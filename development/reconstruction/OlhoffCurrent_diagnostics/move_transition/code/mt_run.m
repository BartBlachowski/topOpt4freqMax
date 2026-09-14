function out = mt_run(arm, nelx, nely, outDir)
%MT_RUN  Execute one arm of the move-transition study with full telemetry.
%
%   No file under +impl/ is modified.  The solver used is mt_olhoffSolveT, a
%   copy of +impl/architecture/olhoffSolve.m differing in exactly two lines (the
%   function signature and the one call to the move controller); the copy is
%   validated by the baseline regression gate, not merely asserted.
%
%   Per-iteration density is reconstructed exactly as olhoffSolve forms it and
%   validated against hist.vol to 1e-12, and the reconstructed final design is
%   asserted identical to res.rho.

guard = olhoffcurrent_paths(); %#ok<NASGU>
man = olhoffcurrent_source_manifest('Verify', true);
cur = olhoffcurrent_currentness('Verbose', false);
assert(man.ok, 'mt_run:Integrity', 'OlhoffCurrent source integrity FAILED');
assert(strcmp(cur.state,'CURRENT'), 'mt_run:NotCurrent', 'currentness = %s', cur.state);
maxNumCompThreads(1);

[cfg, tcfg, meta] = mt_config(arm, nelx, nely);

% ---- the ONLY permitted configuration differences from production -------
prod = olhoffcurrent_config(nelx, nely, 'MaxOuter', 600, 'Diagnostics', true);
S = olh.config.schema();
allowed = {'runtime.name','stop.tolerance','stop.toleranceRule'};
bad = {};
for k = 1:size(S,1)
    p = S{k,1};
    if any(strcmp(p, allowed)); continue; end
    if ~isequaln(olh.config.getPath(cfg,p), olh.config.getPath(prod,p))
        bad{end+1} = p; %#ok<AGROW>
    end
end
assert(isempty(bad), 'mt_run:UnexpectedDrift', ...
    'arm differs from production outside the stopping policy: %s', strjoin(bad,', '));

% Explicitly reassert the scope lock on the fields this study must not touch.
lock = {'material.stiffness.p',3; 'material.mass.model','eq4b'; 'material.mass.q',1; ...
        'filter.type','sensitivity'; 'filter.radiusPhysical',0.06; ...
        'projection.enabled',false; 'move.policy','ladder'; ...
        'optimizer.inner.variant','published'; 'multiplicity.method','subspace'; ...
        'multiplicity.subspaceSize',2; 'multiplicity.offDiagonal',true; 'move.levels',[0.04 0.02 0.01 0.005]};
for k = 1:size(lock,1)
    v = olh.config.getPath(cfg, lock{k,1});
    assert(isequaln(v, lock{k,2}), 'mt_run:ScopeLock', ...
        'scope-locked field %s = %s', lock{k,1}, mat2str(v));
end

fprintf('[mt_run] ARM %s %dx%d  metric=%s  cap=%d\n', ...
        meta.arm, nelx, nely, tcfg.metric, meta.maxOuter);
tS = tic;  res = mt_olhoffSolveT(cfg, tcfg);  wall = toc(tS);

% ---- exact per-iteration density reconstruction -------------------------
NE = nelx*nely;
rhoMin = olh.config.getPath(cfg,'design.minimum');
rho0   = olh.config.getPath(cfg,'design.initial');
h = res.hist;  nO = numel(h.N);
assert(isfield(res,'diag') && numel(res.diag.drho)==nO, 'mt_run:NoDiag','recorder truncated');

rho = rho0*ones(NE,1);  RHO = zeros(NE,nO);  vErr = 0;
for k = 1:nO
    rho = min(1, max(rhoMin, rho + res.diag.drho{k}));
    RHO(:,k) = rho;  vErr = max(vErr, abs(mean(rho)-h.vol(k)));
end
assert(vErr < 1e-12, 'mt_run:ReconstructionFailed','vol mismatch %.3e', vErr);
assert(isequaln(RHO(:,end), double(res.rho(:))), 'mt_run:DesignNotPhysical', ...
    'reconstructed design is not the physical density');

P = mt_telemetry(h, RHO, NE, tcfg);

out = struct('arm',meta.arm,'label',meta.label,'mesh',[nelx nely],'NE',NE, ...
    'nOuter',nO,'innerTotal',sum(h.nInner),'wall_s',wall,'status',res.status, ...
    'omega',double(res.omega(:)),'rhoFinal',RHO(:,end), ...
    'Mnd_final',P.Mnd(end),'gray_final',P.gray(end),'mid_final',P.mid(end), ...
    'volume_final',P.volume(end),'per',P,'log',{res.log}, ...
    'cfgHash',olhoffcurrent_config_hash(cfg),'meta',meta, ...
    'sourceTree',man.treeHash,'volReconErrMax',vErr, ...
    'prodTol',olh.config.getPath(prod,'stop.tolerance'));

if nargin>=4 && ~isempty(outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    f = fullfile(outDir, sprintf('arm%s_%dx%d.mat', meta.arm, nelx, nely));
    save(f,'out','cfg','tcfg','RHO','-v7.3');
    fprintf('[mt_run] saved %s\n', f);
end
fprintf('[mt_run] ARM %s %dx%d nOuter=%d status=%s omega1=%.10g M_nd=%.4f wall=%.0fs\n', ...
        meta.arm, nelx, nely, nO, res.status, out.omega(1), out.Mnd_final, wall);
end
