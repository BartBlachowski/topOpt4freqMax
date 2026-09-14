function A = r240_singlefactor()
%R240_SINGLEFACTOR  Phase 3 + Phase 11 pre-run audit.  READ-ONLY, no solve.
%
%   (3)  The 240x30 candidate configuration must differ from the THREE prior
%        causal candidate configurations in the MESH and nothing else.  Every
%        field is compared; any difference outside the declared mesh-derived
%        set aborts the study.
%   (11) The three-rung counterfactual must be exact under the CURRENT effective
%        configuration, not by inheritance.  Every site that reads
%        numel(move.levels), the ladder tail, or the last-stage identity is
%        re-checked against the resolved 240x30 config.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

A = struct();
A.timestamp = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));
A.note = ['cv_config is CALLED, not copied: the 240x30 controller is the same ' ...
          'function that produced the 160x20 / 320x40 / 400x50 causal evidence.'];

% ---- the four candidate configurations, from the SAME cv_config ---------
meshes = [240 30; 160 20; 320 40; 400 50];
cfgs = cell(4,1); metas = cell(4,1);
for i = 1:4
    [cfgs{i}, metas{i}] = cv_config('C', meshes(i,1), meshes(i,2));
end
A.cap = metas{1}.maxOuter;
A.capInheritedFrom = 'cv_config.m CAP constant, identical for every candidate arm';
A.overrides = metas{1}.overrides;
A.preset = metas{1}.preset;
A.upstreamPreset = metas{1}.upstreamPreset;

% ---- mesh-derived fields that are ALLOWED to differ, declared up front --
allowed = { 'domain_mesh_nelx', 'domain_mesh_nely', 'stop_tolerance', ...
            'runtime_name', 'provenance_resolvedAt', 'provenance_configHash', ...
            'provenance_overrides' };
A.allowedDifferences = allowed;
A.allowedRationale = struct( ...
  'domain_mesh_nelx', 'THE authorized experimental factor', ...
  'domain_mesh_nely', 'THE authorized experimental factor', ...
  'stop_tolerance',   'deterministic function of NE under the inherited meshScaled law 0.05*sqrt(NE/3200); not a free choice', ...
  'runtime_name',     'a run label; enters no computation', ...
  'provenance_overrides', 'the RECORDED override list, which embeds nelx/nely/name; verified element-wise below to differ in those entries only');

% ---- provenance.overrides: verify the difference is mesh/label ONLY -----
% Declaring this "allowed" without checking would be an assumption.  Every
% element is compared; only the nelx, nely and runtime.name entries may differ.
A.overrideDiff = struct('mesh',{},'entries',{},'onlyMeshOrName',{});
for i = 2:4
    ov240 = cfgs{1}.provenance.overrides;
    ovOth = cfgs{i}.provenance.overrides;
    ent = {}; ok = (numel(ov240) == numel(ovOth));
    if ok
        for j = 1:numel(ov240)
            if ~isequaln(ov240{j}, ovOth{j})
                ent{end+1} = sprintf('[%d] %s -> %s', j, local_str(ovOth{j}), local_str(ov240{j})); %#ok<AGROW>
                % the differing element must be a mesh number or the run name
                v = ov240{j};
                isMeshNum = isnumeric(v) && isscalar(v) && any(v == [240 30]);
                isName    = ischar(v) && startsWith(v, 'CV_C_');
                ok = ok && (isMeshNum || isName);
            end
        end
    end
    A.overrideDiff(end+1) = struct('mesh', meshes(i,:), 'entries', {ent}, ...
                                   'onlyMeshOrName', ok);
end
A.overridesMeshOnly = all([A.overrideDiff.onlyMeshOrName]);

A.diffVsPriorArm = struct('mesh',{},'nDiff',{},'differences',{},'unexpected',{});
for i = 2:4
    d = local_diff(cfgs{i}, cfgs{1});
    unexpected = {};
    for j = 1:numel(d)
        key = strtok(d{j}, ':');
        if ~any(strcmp(key, allowed)); unexpected{end+1} = d{j}; end %#ok<AGROW>
    end
    A.diffVsPriorArm(end+1) = struct('mesh', meshes(i,:), 'nDiff', numel(d), ...
        'differences', {d}, 'unexpected', {unexpected});
end
A.singleFactorOk = all(cellfun(@(u) isempty(u), {A.diffVsPriorArm.unexpected})) && ...
                   A.overridesMeshOnly;

% ---- the frozen scientific lock, asserted on the 240x30 config ----------
g = @(p) olh.config.getPath(cfgs{1}, p);
NE = 240*30;
L = struct( ...
  'p',                g('material.stiffness.p'), ...
  'pContinuation',    g('material.stiffness.continuation.enabled'), ...
  'massModel',        g('material.mass.model'), ...
  'q',                g('material.mass.q'), ...
  'filterType',       g('filter.type'), ...
  'filterApplyTo',    g('filter.applyTo'), ...
  'radiusPhysical',   g('filter.radiusPhysical'), ...
  'projection',       g('projection.enabled'), ...
  'multMethod',       g('multiplicity.method'), ...
  'subspaceSize',     g('multiplicity.subspaceSize'), ...
  'diagonalOffsets',  g('multiplicity.diagonalOffsets'), ...
  'offDiagonal',      g('multiplicity.offDiagonal'), ...
  'mmaVariant',       g('optimizer.inner.variant'), ...
  'volumeFraction',   g('design.volumeFraction'), ...
  'designInitial',    g('design.initial'), ...
  'moveLevels',       g('move.levels'), ...
  'movePolicy',       g('move.policy'), ...
  'contSignal',       g('move.continuation.signal'), ...
  'stopRule',         g('stop.rule'), ...
  'stopNorm',         g('stop.norm'), ...
  'stopTolerance',    g('stop.tolerance'), ...
  'maxOuter',         g('runtime.maxOuter'));
A.lock = L;
A.lockOk = L.p == 3 && ~L.pContinuation && strcmp(L.massModel,'eq4b') && L.q == 1 && ...
    strcmp(L.filterType,'sensitivity') && strcmp(L.filterApplyTo,'all') && ...
    L.radiusPhysical == 0.06 && ~L.projection && ...
    strcmp(L.multMethod,'subspace') && L.subspaceSize == 2 && ...
    L.diagonalOffsets && L.offDiagonal && strcmp(L.mmaVariant,'published') && ...
    isequal(L.moveLevels(:).', [0.04 0.02 0.01 0.005]) && ...
    strcmp(L.movePolicy,'ladder') && strcmp(L.contSignal,'stageExhaustion') && ...
    strcmp(L.stopRule,'stageExhaustion') && ...
    abs(L.stopTolerance - 0.05*sqrt(NE/3200)) < 1e-15;
A.toleranceExpected = 0.05*sqrt(NE/3200);

% ---- Phase 11: ladder-dependence audit under THIS config ---------------
pCont = g('material.stiffness.continuation.enabled');
F = struct( ...
  'exhaustMove',  strcmp(g('move.continuation.signal'),'stageExhaustion'), ...
  'exhaustStop',  strcmp(g('stop.rule'),'stageExhaustion'), ...
  'useProj',      g('projection.enabled'), ...
  'guardLadderExhausted', g('stop.guards.ladderExhausted'), ...
  'guardMaxDesignChange', g('stop.guards.maxDesignChange'), ...
  'guardSettledMove',     g('stop.guards.settledMove'), ...
  'pContinuation', pCont);
F.anyStopGuard = F.guardLadderExhausted || F.guardMaxDesignChange;
if pCont
    F.pDriver = g('material.stiffness.continuation.driver');
    F.pOwnCounter = strcmp(F.pDriver,'ownCounter');
else
    F.pDriver = 'n/a'; F.pOwnCounter = false;
end
% the three sites that could break exactness, and why each cannot here
F.site485_ladderTail_neverRuns   = ~F.anyStopGuard || F.exhaustStop;
F.site312_pContinuation_neverRuns = ~F.pOwnCounter;
F.site_projectionBlock_neverRuns  = ~F.useProj;
F.site509_dependsOnLengthOnly = true;   % atLastLevel = stage >= numel(levels)
F.site109_dependsOnLengthOnly = true;   % descend iff declared && stage < numel(levels)
F.site122_valueLookupByStage  = true;   % mv = levels(stage): equal for stages 1..3
A.ladderDependence = F;
A.counterfactualExactPreRun = F.site485_ladderTail_neverRuns && ...
    F.site312_pContinuation_neverRuns && F.site_projectionBlock_neverRuns;

% ---- the three-rung ladder must be a LEGAL config at this mesh ---------
A.threeRungLegal = false; A.threeRungDiff = {};
try
    c3 = olh.config.resolve(A.upstreamPreset, ...
        'domain.mesh.nelx', 240, 'domain.mesh.nely', 30, ...
        'runtime.maxOuter', A.cap, 'runtime.singleThread', true, ...
        'runtime.diagnostics', true, 'runtime.verbose', false, ...
        'runtime.name', metas{1}.label, ...
        'move.levels', [0.04 0.02 0.01], A.overrides{:});
    A.threeRungLegal = isequal(olh.config.getPath(c3,'move.levels'), [0.04 0.02 0.01]);
    d = local_diff(cfgs{1}, c3);
    keep = {};
    for j = 1:numel(d)
        k2 = strtok(d{j}, ':');
        if ~any(strcmp(k2, {'runtime_name','provenance_resolvedAt','provenance_configHash'}))
            keep{end+1} = d{j}; %#ok<AGROW>
        end
    end
    A.threeRungDiff = keep;
catch ME
    A.threeRungError = ME.message;
end

A.configHash240 = olhoffcurrent_config_hash(cfgs{1});

out = fullfile(study,'evidence','single_factor.json');
fid = fopen(out,'w'); fprintf(fid,'%s',jsonencode(A,'PrettyPrint',true)); fclose(fid);

fprintf('\n%s\nC240 SINGLE-FACTOR + LADDER-DEPENDENCE AUDIT\n%s\n', repmat('=',1,72), repmat('=',1,72));
fprintf('  preset %s (upstream %s)  cap %d  cfgHash %s\n', A.preset, A.upstreamPreset, A.cap, A.configHash240);
fprintf('  overrides: %s\n', strjoin(cellfun(@(x) local_s(x), A.overrides, 'uni', false), ' '));
for i = 1:numel(A.diffVsPriorArm)
    dd = A.diffVsPriorArm(i);
    fprintf('  vs %dx%d: %d field differences, %d unexpected\n', dd.mesh(1), dd.mesh(2), ...
        dd.nDiff, numel(dd.unexpected));
    for j = 1:numel(dd.differences); fprintf('      %s\n', dd.differences{j}); end
    for j = 1:numel(dd.unexpected); fprintf('      !! UNEXPECTED %s\n', dd.unexpected{j}); end
end
for i = 1:numel(A.overrideDiff)
    od = A.overrideDiff(i);
    fprintf('  provenance.overrides vs %dx%d: %d differing entries, meshOrNameOnly=%d\n', ...
        od.mesh(1), od.mesh(2), numel(od.entries), od.onlyMeshOrName);
    for j = 1:numel(od.entries); fprintf('        %s\n', od.entries{j}); end
end
fprintf('  singleFactorOk = %d  (overridesMeshOnly=%d)\n', A.singleFactorOk, A.overridesMeshOnly);
fprintf('  lockOk = %d   tol = %.6g (expected %.6g)   levels = %s\n', ...
    A.lockOk, A.lock.stopTolerance, A.toleranceExpected, mat2str(A.lock.moveLevels));
fprintf('  anyStopGuard=%d pOwnCounter=%d useProj=%d\n', F.anyStopGuard, F.pOwnCounter, F.useProj);
fprintf('  site485 never runs=%d  site312 never runs=%d  projection block never runs=%d\n', ...
    F.site485_ladderTail_neverRuns, F.site312_pContinuation_neverRuns, F.site_projectionBlock_neverRuns);
fprintf('  counterfactual exact (pre-run, static) = %d\n', A.counterfactualExactPreRun);
fprintf('  three-rung ladder legal at 240x30 = %d ; non-metadata diffs = %d\n', ...
    A.threeRungLegal, numel(A.threeRungDiff));
for j = 1:numel(A.threeRungDiff); fprintf('      %s\n', A.threeRungDiff{j}); end
fprintf('  written %s\n', out);
end

function s = local_s(x)
if ischar(x); s = x; elseif isnumeric(x); s = mat2str(x); else; s = class(x); end
end

function d = local_diff(a, b)
d = {};
fa = local_flatten(a, ''); fb = local_flatten(b, '');
k = unique([fieldnames(fa); fieldnames(fb)]);
for i = 1:numel(k)
    va = []; vb = []; ha = isfield(fa,k{i}); hb = isfield(fb,k{i});
    if ha; va = fa.(k{i}); end
    if hb; vb = fb.(k{i}); end
    if ~ha || ~hb || ~isequaln(va, vb)
        d{end+1} = sprintf('%s: %s -> %s', k{i}, local_str(va), local_str(vb)); %#ok<AGROW>
    end
end
end

function f = local_flatten(s, pre)
f = struct();
n = fieldnames(s);
for i = 1:numel(n)
    v = s.(n{i});
    key = n{i}; if ~isempty(pre); key = [pre '_' n{i}]; end
    if isstruct(v) && isscalar(v)
        g = local_flatten(v, key); gn = fieldnames(g);
        for j = 1:numel(gn); f.(gn{j}) = g.(gn{j}); end
    else
        f.(key) = v;
    end
end
end

function s = local_str(v)
if isempty(v); s = '<absent>';
elseif ischar(v); s = v;
elseif isnumeric(v) || islogical(v); s = mat2str(v);
elseif iscell(v); s = sprintf('cell[%d]', numel(v));
else; s = class(v); end
end
