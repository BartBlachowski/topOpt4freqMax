function out = tr_policy_recovery()
%TR_POLICY_RECOVERY  Phases 7, 8 and 12 (static parts only).
%
%   Recovers the EXACT validated candidate effective configuration from the
%   completed retry -- by re-resolving the retry's own frozen recipe, never
%   from memory or from prose -- records canonical production as it stands
%   today, and enumerates the promotion delta.
%
%   It also re-verifies that canonical production already dispatches to the
%   controller code the validated run used.
%
%   PROMOTES NOTHING.  Resolves configurations and hashes files; runs no
%   optimization.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
repo  = fileparts(fileparts(root));
r1    = fullfile(root,'diagnostics','three_rung_promotion_validation_retry1');
addpath(root); addpath(here);
addpath(fullfile(r1,'scripts'));
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

NELX = 320; NELY = 40;
out = struct();

% ---- Phase 7: the validated candidate, re-resolved from retry1's recipe --
[cand, cmeta] = tr_config(NELX, NELY);     % retry1's own frozen builder
out.candCfgHash = olhoffcurrent_config_hash(cand);
M = jsondecode(fileread(fullfile(r1,'METRICS.json')));
out.recordedCandCfgHash = M.run.cfgHash;
out.candMatchesValidatedRun = strcmp(out.candCfgHash, out.recordedCandCfgHash);
out.candMeta = cmeta;

g = @(c,p) olh.config.getPath(c,p);
POLICY = {'move.policy','move.levels','move.continuation.signal', ...
          'move.continuation.window','move.continuation.tolerance', ...
          'stop.rule','stop.norm','stop.tolerance','stop.toleranceRule', ...
          'stop.guards.settledMove','stop.guards.ladderExhausted','stop.guards.maxDesignChange'};
SCI    = {'material.stiffness.p','material.stiffness.continuation.enabled', ...
          'material.mass.model','material.mass.q', ...
          'filter.type','filter.applyTo','filter.radiusPhysical', ...
          'projection.enabled','multiplicity.method','multiplicity.subspaceSize', ...
          'multiplicity.diagonalOffsets','multiplicity.offDiagonal', ...
          'optimizer.inner.type','optimizer.inner.variant','optimizer.inner.variable', ...
          'design.initial','design.minimum','volume.fraction'};

prod = olhoffcurrent_config(NELX, NELY);
out.prodCfgHash = olhoffcurrent_config_hash(prod);

out.policy = struct('path',{},'validated',{},'production',{},'same',{});
for k = 1:numel(POLICY)
    p = POLICY{k};
    a = g(cand,p); b = g(prod,p);
    out.policy(end+1) = struct('path',p,'validated',local_show(a), ...
        'production',local_show(b),'same',isequaln(a,b)); %#ok<AGROW>
end
out.science = struct('path',{},'validated',{},'production',{},'same',{});
for k = 1:numel(SCI)
    p = SCI{k};
    a = g(cand,p); b = g(prod,p);
    out.science(end+1) = struct('path',p,'validated',local_show(a), ...
        'production',local_show(b),'same',isequaln(a,b)); %#ok<AGROW>
end
out.scienceAllSame = all([out.science.same]);

% ---- Phase 8: the full schema delta, classified -------------------------
S = olh.config.schema();
HARNESS = {'runtime.name','runtime.maxOuter','runtime.diagnostics', ...
           'runtime.verbose','runtime.singleThread'};
out.delta = struct('path',{},'production',{},'validated',{},'class',{});
for k = 1:size(S,1)
    p = S{k,1};
    a = g(prod,p); b = g(cand,p);
    if ~isequaln(a,b)
        if any(strcmp(p,HARNESS)), cls = 'HARNESS_OR_LABEL';
        elseif any(strcmp(p,POLICY)), cls = 'POLICY';
        else, cls = 'UNEXPECTED'; end
        out.delta(end+1) = struct('path',p,'production',local_show(a), ...
            'validated',local_show(b),'class',cls); %#ok<AGROW>
    end
end
out.policyDelta     = out.delta(strcmp({out.delta.class},'POLICY'));
out.unexpectedDelta = out.delta(strcmp({out.delta.class},'UNEXPECTED'));
out.minimalPromotion = isempty(out.unexpectedDelta);

% ---- the frozen A/B, by hash -------------------------------------------
out.abHashes = struct('path',{},'sha256',{});
for f = {'+impl/architecture/+olh/+move/exhaustion.m'
         '+impl/architecture/+olh/+move/limit.m'
         '+impl/architecture/olhoffSolve.m'}'
    out.abHashes(end+1) = struct('path',f{1}, ...
        'sha256', olhoffcurrent_sha256_file(fullfile(root,f{1}))); %#ok<AGROW>
end
out.abConstants = struct('W',20,'P',20,'Wnp',10, ...
    'tolRule', g(cand,'stop.toleranceRule'), 'tolAt320', g(cand,'stop.tolerance'), ...
    'source','+impl/architecture/+olh/+move/exhaustion.m', ...
    'preregistration','diagnostics/two_branch_maturity_240/PREREGISTRATION.md', ...
    'preregSha256', olhoffcurrent_sha256_file(fullfile(root,'diagnostics', ...
        'two_branch_maturity_240','PREREGISTRATION.md')));

% ---- Phase 12 (static): dispatch ---------------------------------------
try
    olhoffcurrent_assert_dispatch(); out.dispatchOk = true; out.dispatchErr = '';
catch ME
    out.dispatchOk = false; out.dispatchErr = ME.message;
end
qn = {'olh.move.exhaustion','olh.move.limit','olhoffSolve'};
out.dispatch = struct('name',{},'resolvesTo',{},'underProduction',{});
for k = 1:numel(qn)
    w = which(qn{k});
    out.dispatch(end+1) = struct('name',qn{k},'resolvesTo',w, ...
        'underProduction', startsWith(w, fullfile(root,'+impl'))); %#ok<AGROW>
end
out.dispatchAllProduction = all([out.dispatch.underProduction]);
out.implTree = olhoffcurrent_source_manifest('Verify',false).treeHash;
out.validatedRunImplTree = M.run.implTree;
out.implTreeMatchesValidatedRun = strcmp(out.implTree, out.validatedRunImplTree);

% ---- Phase 13 (static): can production reach legacy accidentally? ------
% Production's DEFAULT today IS the legacy beta path -- that is precisely what
% promotion would change.  Recorded as a fact, not a defect.
out.legacy = struct( ...
  'productionSignalToday', g(prod,'move.continuation.signal'), ...
  'productionStopRuleToday', g(prod,'stop.rule'), ...
  'presetDefaultLevels', mat2str(olh.config.getPath( ...
       olh.config.resolve('duOlhoffFrozenM4'),'move.levels')), ...
  'schemaDefaultSignal', local_schemaDefault(S,'move.continuation.signal'), ...
  'schemaDefaultStopRule', local_schemaDefault(S,'stop.rule'), ...
  'schemaDefaultLevels', local_schemaDefault(S,'move.levels'));

fid = fopen(fullfile(study,'evidence','policy_recovery.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));

fprintf('\n== PHASE 7: validated candidate recovered ==\n');
fprintf('re-resolved cfgHash : %s\n', out.candCfgHash);
fprintf('recorded  in retry1 : %s\n', out.recordedCandCfgHash);
fprintf('MATCH               : %d\n', out.candMatchesValidatedRun);
fprintf('\n  policy fields (validated | production):\n');
for k=1:numel(out.policy)
    fprintf('    %-38s %-22s %-22s same=%d\n', out.policy(k).path, ...
        out.policy(k).validated, out.policy(k).production, out.policy(k).same);
end
fprintf('\n  scientific formulation identical: %d\n', out.scienceAllSame);
for k=1:numel(out.science)
    if ~out.science(k).same
        fprintf('    DIFFERS %s: validated=%s production=%s\n', out.science(k).path, ...
            out.science(k).validated, out.science(k).production);
    end
end
fprintf('\n== PHASE 8: promotion delta ==\n');
for k=1:numel(out.delta)
    fprintf('  [%-17s] %-34s production=%-46s validated=%s\n', out.delta(k).class, ...
        out.delta(k).path, out.delta(k).production, out.delta(k).validated);
end
fprintf('  POLICY changes: %d   UNEXPECTED: %d   minimal=%d\n', ...
    numel(out.policyDelta), numel(out.unexpectedDelta), out.minimalPromotion);

fprintf('\n== PHASE 12 (static): dispatch ==\n');
fprintf('assert_dispatch=%d  allUnderProduction=%d\n', out.dispatchOk, out.dispatchAllProduction);
for k=1:numel(out.dispatch)
    fprintf('  %-22s %s\n', out.dispatch(k).name, out.dispatch(k).resolvesTo);
end
fprintf('implTree now = validated-run implTree : %d  (%s)\n', ...
    out.implTreeMatchesValidatedRun, out.implTree);
end

function s = local_schemaDefault(S, path)
i = find(strcmp(S(:,1), path), 1);
if isempty(i), s = '<not in schema>'; return; end
v = S{i,3};
if ischar(v), s = v; elseif isnumeric(v), s = mat2str(v); else, s = class(v); end
end

function s = local_show(v)
if ischar(v); s = v; elseif isstring(v); s = char(v);
elseif islogical(v); s = mat2str(v);
elseif isnumeric(v); s = mat2str(v,17);
elseif iscell(v); s = ['{' strjoin(cellfun(@local_show,v,'UniformOutput',false),',') '}'];
else, s = class(v); end
end
