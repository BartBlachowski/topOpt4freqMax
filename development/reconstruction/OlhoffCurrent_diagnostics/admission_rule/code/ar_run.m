function out = ar_run(mode, nelx, nely, outDir)
%AR_RUN  Execute one arm of the admission-rule study and record full telemetry.
%   The solver is NOT modified.  Per-iteration density is reconstructed exactly
%   as olhoffSolve forms it and validated against hist.vol.

guard = olhoffcurrent_paths(); %#ok<NASGU>
man = olhoffcurrent_source_manifest();
cur = olhoffcurrent_currentness('Verbose', false);
assert(man.ok, 'ar_run:Integrity', 'OlhoffCurrent source integrity FAILED');
assert(~strcmp(cur.state,'LOCAL_MODIFIED'), 'ar_run:LocalModified', 'state=%s', cur.state);
maxNumCompThreads(1);

[cfg, meta] = ar_config(mode, nelx, nely);

% The ONLY differences from production must be the declared stopping-policy
% fields.  Asserted over all 80 schema fields, so no other drift is possible.
prod = olhoffcurrent_config(nelx, nely, 'MaxOuter', 600, 'Diagnostics', true);
S = olh.config.schema(); allowed = {'runtime.name','stop.tolerance','stop.toleranceRule'};
bad = {};
for k = 1:size(S,1)
    p = S{k,1};
    if any(strcmp(p, allowed)); continue; end
    if ~isequaln(olh.config.getPath(cfg,p), olh.config.getPath(prod,p)); bad{end+1}=p; end %#ok<AGROW>
end
assert(isempty(bad), 'ar_run:UnexpectedDrift', ...
    'arm differs from production outside the stopping policy: %s', strjoin(bad,', '));

fprintf('[ar_run] %-11s %dx%d cap=%d\n', meta.mode, nelx, nely, meta.maxOuter);
tS = tic; res = olhoffSolve(cfg); wall = toc(tS);

NE = nelx*nely;
rhoMin = olh.config.getPath(cfg,'design.minimum');
rho0   = olh.config.getPath(cfg,'design.initial');
h = res.hist; nO = numel(h.N);
assert(isfield(res,'diag') && numel(res.diag.drho)==nO, 'ar_run:NoDiag','recorder truncated');

rho = rho0*ones(NE,1); RHO = zeros(NE,nO); vErr = 0;
for k = 1:nO
    rho = min(1, max(rhoMin, rho + res.diag.drho{k}));
    RHO(:,k) = rho; vErr = max(vErr, abs(mean(rho)-h.vol(k)));
end
assert(vErr < 1e-12, 'ar_run:ReconstructionFailed','vol mismatch %.3e', vErr);
assert(isequaln(RHO(:,end), double(res.rho(:))), 'ar_run:DesignNotPhysical', ...
    'design density is not the physical density');

P = struct();
P.outer=(1:nO).'; P.omega1=h.omega(1,:).'; P.omega2=h.omega(2,:).';
P.gap12=h.gap12(:); P.volume=h.vol(:); P.move=h.move(:); P.stage=h.stage(:);
P.beta=h.beta(:); P.l2=h.dxNorm2(:); P.rms=h.dxNorm2(:)/sqrt(NE);
P.maxAbs=h.dxOuter(:); P.ratio=h.dxOuter(:)./h.move(:);
P.nInner=h.nInner(:); P.innerConv=h.innerConv(:); P.multN=h.N(:); P.degen=h.degen(:);
P.Mnd=zeros(nO,1); P.gray=zeros(nO,1); P.mid=zeros(nO,1);
for k=1:nO
    r=RHO(:,k);
    P.Mnd(k)=100*mean(4*r.*(1-r));
    P.gray(k)=mean(r>0.1 & r<0.9);
    P.mid(k)=mean(r>=0.4 & r<=0.6);
end
% move bookkeeping
P.descent=[false; P.move(2:end)<P.move(1:end-1)];
P.moveChanged=[true; P.move(2:end)~=P.move(1:end-1)];
sinceChange=zeros(nO,1); c=0;
for k=1:nO
    if P.moveChanged(k); c=0; else; c=c+1; end
    sinceChange(k)=c;
end
P.itersSinceMoveChange = sinceChange;   % 0 on the iteration the level changed

out = struct('mode',meta.mode,'label',meta.label,'mesh',[nelx nely],'NE',NE, ...
    'nOuter',nO,'innerTotal',sum(h.nInner),'wall_s',wall, ...
    'omega',double(res.omega(:)),'rhoFinal',RHO(:,end), ...
    'Mnd_final',P.Mnd(end),'gray_final',P.gray(end),'mid_final',P.mid(end), ...
    'volume_final',P.volume(end),'per',P,'log',{res.log}, ...
    'cfgHash',olhoffcurrent_config_hash(cfg),'meta',meta, ...
    'sourceTree',man.treeHash,'volReconErrMax',vErr, ...
    'prodTol', olh.config.getPath(prod,'stop.tolerance'));

if nargin>=4 && ~isempty(outDir)
    if ~isfolder(outDir); mkdir(outDir); end
    f = fullfile(outDir, sprintf('%s_%dx%d.mat', meta.mode, nelx, nely));
    save(f,'out','cfg','RHO','-v7.3');
    fprintf('[ar_run] saved %s\n', f);
end
fprintf('[ar_run] %-11s %dx%d nOuter=%d omega1=%.10g M_nd=%.4f wall=%.0fs\n', ...
    meta.mode, nelx, nely, nO, out.omega(1), out.Mnd_final, wall);
end
