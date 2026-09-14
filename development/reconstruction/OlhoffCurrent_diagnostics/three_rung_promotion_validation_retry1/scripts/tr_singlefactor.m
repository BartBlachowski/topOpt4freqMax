function out = tr_singlefactor()
%TR_SINGLEFACTOR  Part C: prove the candidate differs from the oracle arm in
%   exactly ONE computational field, move.levels.
%
%   Both configurations are resolved and every schema leaf is compared.  The
%   audit does not trust tr_config's own claim; it reads the resolved structs.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

NELX = 320; NELY = 40;
[cfg3, meta, cfg4] = tr_config(NELX, NELY);

S = olh.config.schema();
diffs = struct('path',{},'four',{},'three',{},'class',{});
LABEL_ONLY = {'runtime.name'};
for k = 1:size(S,1)
    p = S{k,1};
    a = olh.config.getPath(cfg4,p); b = olh.config.getPath(cfg3,p);
    if ~isequaln(a,b)
        cls = 'COMPUTATIONAL';
        if any(strcmp(p, LABEL_ONLY)), cls = 'LABEL_ONLY'; end
        diffs(end+1) = struct('path',p,'four',local_show(a),'three',local_show(b),'class',cls); %#ok<AGROW>
    end
end

out = struct();
out.nSchemaRows = size(S,1);
out.diffs = diffs;
comp = diffs(strcmp({diffs.class},'COMPUTATIONAL'));
out.computationalDiffs = comp;
out.cfgHash4 = olhoffcurrent_config_hash(cfg4);
out.cfgHash3 = olhoffcurrent_config_hash(cfg3);

% ---- the locked scope, asserted field by field on the CANDIDATE ---------
g = @(p) olh.config.getPath(cfg3,p);
NE = NELX*NELY;
locks = { ...
 'material.stiffness.p',                    g('material.stiffness.p') == 3; ...
 'material.stiffness.continuation.enabled', ~g('material.stiffness.continuation.enabled'); ...
 'material.mass.model',                     strcmp(g('material.mass.model'),'eq4b'); ...
 'material.mass.q',                         g('material.mass.q') == 1; ...
 'filter.type',                             strcmp(g('filter.type'),'sensitivity'); ...
 'filter.applyTo',                          strcmp(g('filter.applyTo'),'all'); ...
 'filter.radiusPhysical',                   g('filter.radiusPhysical') == 0.06; ...
 'projection.enabled',                      ~g('projection.enabled'); ...
 'multiplicity.method',                     strcmp(g('multiplicity.method'),'subspace'); ...
 'multiplicity.subspaceSize',               g('multiplicity.subspaceSize') == 2; ...
 'multiplicity.diagonalOffsets',            g('multiplicity.diagonalOffsets'); ...
 'multiplicity.offDiagonal',                g('multiplicity.offDiagonal'); ...
 'optimizer.inner.variant',                 strcmp(g('optimizer.inner.variant'),'published'); ...
 'move.policy',                             strcmp(g('move.policy'),'ladder'); ...
 'move.continuation.signal',                strcmp(g('move.continuation.signal'),'stageExhaustion'); ...
 'stop.rule',                               strcmp(g('stop.rule'),'stageExhaustion'); ...
 'stop.tolerance',                          g('stop.tolerance') == 0.05*sqrt(NE/3200); ...
 'runtime.maxOuter',                        g('runtime.maxOuter') == 1600; ...
 'runtime.singleThread',                    g('runtime.singleThread'); ...
 'design.initial',                          g('design.initial') == olh.config.getPath(cfg4,'design.initial'); ...
 'move.levels',                             isequal(g('move.levels'), [0.04 0.02 0.01]); ...
};
out.locks = struct('name',locks(:,1),'ok',locks(:,2));
out.allLocksOk = all(cell2mat(locks(:,2)));

out.pass = out.allLocksOk && numel(comp) == 1 && strcmp(comp(1).path,'move.levels');
if out.pass, out.verdict = 'THREE_RUNG_SINGLE_FACTOR_PASS';
else,        out.verdict = 'THREE_RUNG_SINGLE_FACTOR_FAIL'; end

fid = fopen(fullfile(study,'evidence','singlefactor.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));

fprintf('\n== single factor: %d schema rows compared ==\n', out.nSchemaRows);
for k = 1:numel(diffs)
    fprintf('  [%s] %-28s  four=%s  three=%s\n', diffs(k).class, diffs(k).path, diffs(k).four, diffs(k).three);
end
fprintf('  cfgHash four-rung  : %s\n', out.cfgHash4);
fprintf('  cfgHash three-rung : %s\n', out.cfgHash3);
fprintf('  locks all ok       : %d\n', out.allLocksOk);
for k = 1:size(locks,1)
    if ~locks{k,2}, fprintf('    LOCK FAILED: %s\n', locks{k,1}); end
end
fprintf('VERDICT: %s\n', out.verdict);
end

function s = local_show(v)
if ischar(v); s = v; elseif isstring(v); s = char(v);
elseif islogical(v); s = mat2str(v);
elseif isnumeric(v); s = mat2str(v,17);
elseif iscell(v); s = ['{' strjoin(cellfun(@local_show,v,'UniformOutput',false),',') '}'];
else, s = class(v); end
end
