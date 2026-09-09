function A = tr3_configaudit()
%TR3_CONFIGAUDIT  Phase 3 STATIC audit: does anything in the resolved candidate
%   configuration make the computation depend on numel(move.levels) or on the
%   IDENTITY of the final rung, before the stage-3 exhaustion declaration?
%
%   Read-only.  Resolves configurations; executes no solve.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
addpath(root);
guard = olhoffcurrent_paths(); %#ok<NASGU>   % keeps +impl on the path for this scope

A = struct();
A.timestamp = char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z'''));

info = olhoffcurrent_preset();
CAP  = 1600;                      % the candidate arm's preregistered cap
armOv = {'move.continuation.signal','stageExhaustion','stop.rule','stageExhaustion'};

meshes = [160 20; 320 40; 400 50];
A.meshes = struct('mesh',{},'flags',{},'cfgHash',{});
for i = 1:size(meshes,1)
    cfg = olh.config.resolve(info.upstreamPreset, ...
        'domain.mesh.nelx', meshes(i,1), 'domain.mesh.nely', meshes(i,2), ...
        'runtime.maxOuter', CAP, 'runtime.singleThread', true, ...
        'runtime.diagnostics', true, 'runtime.verbose', false, ...
        'runtime.name', sprintf('CV_C_%dx%d', meshes(i,1), meshes(i,2)), armOv{:});
    g = @(p) olh.config.getPath(cfg,p);
    pCont = g('material.stiffness.continuation.enabled');
    f = struct( ...
      'movePolicy',      g('move.policy'), ...
      'moveLevels',      g('move.levels'), ...
      'moveInitial',     g('move.initial'), ...
      'contSignal',      g('move.continuation.signal'), ...
      'stopRule',        g('stop.rule'), ...
      'stopTolerance',   g('stop.tolerance'), ...
      'stopNorm',        g('stop.norm'), ...
      'projectionEnabled',      g('projection.enabled'), ...
      'pContinuationEnabled',   pCont, ...
      'guardLadderExhausted',   g('stop.guards.ladderExhausted'), ...
      'guardMaxDesignChange',   g('stop.guards.maxDesignChange'), ...
      'guardSettledMove',       g('stop.guards.settledMove'), ...
      'designInitial',   g('design.initial'), ...
      'p',               g('material.stiffness.p'), ...
      'q',               g('material.mass.q'), ...
      'filterType',      g('filter.type'), ...
      'radiusPhysical',  g('filter.radiusPhysical'), ...
      'subspaceSize',    g('multiplicity.subspaceSize'), ...
      'offDiagonal',     g('multiplicity.offDiagonal'), ...
      'mmaVariant',      g('optimizer.inner.variant'));
    if pCont
        f.pDriver = g('material.stiffness.continuation.driver');
        f.pBlocksStopUntilFinal = g('material.stiffness.continuation.blockStopUntilFinal');
    else
        f.pDriver = 'n/a (continuation disabled)';
        f.pBlocksStopUntilFinal = false;
    end
    % the solver-level derived flags, computed exactly as olhoffSolve does
    f.exhaustMove   = strcmp(f.contSignal,'stageExhaustion');
    f.exhaustStop   = strcmp(f.stopRule,'stageExhaustion');
    f.useExhaustion = f.exhaustMove || f.exhaustStop;
    f.anyStopGuard  = f.guardLadderExhausted || f.guardMaxDesignChange;
    f.pOwnCounter   = pCont && strcmp(f.pDriver,'ownCounter');
    f.pBlocksStop   = pCont && f.pBlocksStopUntilFinal;
    f.useProj       = f.projectionEnabled;
    % the three inert-ness conditions the counterfactual proof needs
    f.line485_inert = f.anyStopGuard && f.exhaustStop;      % guard block skipped
    f.line485_neverRuns = ~f.anyStopGuard || f.exhaustStop;
    f.line312_neverRuns = ~f.pOwnCounter;
    f.projectionBlockNeverRuns = ~f.useProj;

    A.meshes(i) = struct('mesh', meshes(i,:), 'flags', f, ...
                         'cfgHash', olhoffcurrent_config_hash(cfg));
end

% ---- the three-rung ladder must be a LEGAL configuration ----------------
% schema allows [1 Inf]; validate requires non-increasing and the paired
% signal/stop rule.  Checked by resolving it -- resolution runs validate.
A.threeRungLegal = false; A.threeRungError = '';
try
    c3 = olh.config.resolve(info.upstreamPreset, ...
        'domain.mesh.nelx', 160, 'domain.mesh.nely', 20, ...
        'move.levels', [0.04 0.02 0.01], armOv{:});
    A.threeRungLegal  = isequal(olh.config.getPath(c3,'move.levels'), [0.04 0.02 0.01]);
    A.threeRungLevels = olh.config.getPath(c3,'move.levels');
    % every non-ladder field must be identical to the four-rung resolution
    c4 = olh.config.resolve(info.upstreamPreset, ...
        'domain.mesh.nelx', 160, 'domain.mesh.nely', 20, armOv{:});
    A.threeRungDiff = local_diff(c4, c3);
catch ME
    A.threeRungError = ME.message;
end

% ---- production must be untouched ---------------------------------------
prodCfg = olhoffcurrent_config(160, 20);
A.productionLevels = olh.config.getPath(prodCfg,'move.levels');
A.productionSignal = olh.config.getPath(prodCfg,'move.continuation.signal');
A.productionUnchanged = isequal(A.productionLevels(:).', [0.04 0.02 0.01 0.005]) && ...
                        ~strcmp(A.productionSignal,'stageExhaustion');

% ---- source-level dependence inventory ----------------------------------
A.dependenceSites = { ...
 struct('file','+impl/architecture/olhoffSolve.m','line',104, ...
        'expr','moveLevels = g(''move.levels'')','role','read only','active',true, ...
        'dependsOnLength',false,'dependsOnFinalRungIdentity',false)
 struct('file','+impl/architecture/olhoffSolve.m','line',312, ...
        'expr','mvNow = moveLevels(1)','role','p-continuation stall consumption', ...
        'active',false,'dependsOnLength',false,'dependsOnFinalRungIdentity',false)
 struct('file','+impl/architecture/olhoffSolve.m','line',485, ...
        'expr','~any(moveLevels(stage+1:end) > epsRMS)','role','ladder-restoration stop guard', ...
        'active',false,'dependsOnLength',true,'dependsOnFinalRungIdentity',true)
 struct('file','+impl/architecture/olhoffSolve.m','line',509, ...
        'expr','atLastLevel = stage >= numel(moveLevels)','role','terminal admission', ...
        'active',true,'dependsOnLength',true,'dependsOnFinalRungIdentity',false)
 struct('file','+impl/architecture/+olh/+move/limit.m','line',109, ...
        'expr','state.stage < numel(cfg.move.levels)','role','descent gate', ...
        'active',true,'dependsOnLength',true,'dependsOnFinalRungIdentity',false)
 struct('file','+impl/architecture/+olh/+move/limit.m','line',122, ...
        'expr','mv = cfg.move.levels(state.stage)','role','move value at the stage', ...
        'active',true,'dependsOnLength',false,'dependsOnFinalRungIdentity',false)
 struct('file','+impl/architecture/+olh/+move/limit.m','line',148, ...
        'expr','state.stage = min(state.stage+1, numel(cfg.move.levels))', ...
        'role','beta-stall descent (other continuation branch)','active',false, ...
        'dependsOnLength',true,'dependsOnFinalRungIdentity',false)
 struct('file','+impl/architecture/+olh/+move/limit.m','line',152, ...
        'expr','mv = cfg.move.levels(state.stage)','role','beta-stall move value', ...
        'active',false,'dependsOnLength',false,'dependsOnFinalRungIdentity',false)};

out = fullfile(study,'evidence','config_audit.json');
fid = fopen(out,'w'); fprintf(fid,'%s',jsonencode(A,'PrettyPrint',true)); fclose(fid);

fprintf('\n%s\nTHREE-RUNG STATIC CONFIG AUDIT\n%s\n', repmat('=',1,72), repmat('=',1,72));
for i = 1:numel(A.meshes)
    f = A.meshes(i).flags; m = A.meshes(i).mesh;
    fprintf('  %dx%d  policy=%s levels=%s initial=%.4g tol=%.4g\n', m(1), m(2), ...
        f.movePolicy, mat2str(f.moveLevels), f.moveInitial, f.stopTolerance);
    fprintf('        exhaustMove=%d exhaustStop=%d useProj=%d pCont=%d pOwnCounter=%d pBlocksStop=%d\n', ...
        f.exhaustMove, f.exhaustStop, f.useProj, f.pContinuationEnabled, f.pOwnCounter, f.pBlocksStop);
    fprintf('        guards: ladder=%d maxChange=%d settled=%d anyStopGuard=%d\n', ...
        f.guardLadderExhausted, f.guardMaxDesignChange, f.guardSettledMove, f.anyStopGuard);
    fprintf('        line485 never runs=%d   line312 never runs=%d   projection block never runs=%d\n', ...
        f.line485_neverRuns, f.line312_neverRuns, f.projectionBlockNeverRuns);
end
fprintf('  three-rung ladder is a legal configuration: %d  (%s)\n', A.threeRungLegal, ...
    mat2str(A.threeRungLevels));
fprintf('  three-rung vs four-rung resolved-config differences: %d\n', numel(A.threeRungDiff));
for i = 1:numel(A.threeRungDiff); fprintf('      %s\n', A.threeRungDiff{i}); end
fprintf('  production levels %s  signal %s  unchanged=%d\n', ...
    mat2str(A.productionLevels), A.productionSignal, A.productionUnchanged);
fprintf('  written %s\n', out);
end

function d = local_diff(a, b)
%LOCAL_DIFF  Field-for-field differences between two resolved configs.
d = {};
fa = local_flatten(a, ''); fb = local_flatten(b, '');
k = unique([fieldnames(fa); fieldnames(fb)]);
for i = 1:numel(k)
    va = []; vb = []; ha = isfield(fa,k{i}); hb = isfield(fb,k{i});
    if ha; va = fa.(k{i}); end
    if hb; vb = fb.(k{i}); end
    if ~ha || ~hb || ~isequaln(va, vb)
        d{end+1} = sprintf('%s: %s -> %s', strrep(k{i},'_','.'), ...
            local_str(va), local_str(vb)); %#ok<AGROW>
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
else; s = class(v); end
end
