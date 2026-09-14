function out = tr_dispatch()
%TR_DISPATCH  Part J preparation: prove that canonical production ALREADY
%   dispatches to the exact controller code the candidate was tested with, and
%   record precisely which fields a future promotion would have to change.
%
%   This PROMOTES NOTHING.  It resolves production's configuration and compares
%   it with the validated candidate; it writes no configuration and touches no
%   production path.

here  = fileparts(mfilename('fullpath'));
study = fileparts(here);
root  = fileparts(fileparts(study));
addpath(root); addpath(here);
addpath(fullfile(root,'diagnostics','two_branch_controller_validation','scripts'));
guard = olhoffcurrent_paths(); %#ok<NASGU>

NELX = 320; NELY = 40;
out = struct();

% ---- 1. the dispatch audit the repository ships -------------------------
try
    olhoffcurrent_assert_dispatch();
    out.dispatchOk = true; out.dispatchErr = '';
catch ME
    out.dispatchOk = false; out.dispatchErr = ME.message;
end

% ---- 2. the controller files the candidate actually executed ------------
% Package functions must be resolved by their QUALIFIED name -- `which` on the
% bare name cannot see inside a +package and would report a false negative.
files = { ...
 '+impl/architecture/+olh/+move/exhaustion.m', 'olh.move.exhaustion'
 '+impl/architecture/+olh/+move/limit.m',      'olh.move.limit'
 '+impl/architecture/olhoffSolve.m',           'olhoffSolve'};
out.controller = struct('path',{},'sha256',{},'resolvesTo',{},'isProduction',{});
for k = 1:size(files,1)
    p = fullfile(root, files{k,1});
    w = which(files{k,2});
    out.controller(end+1) = struct('path', files{k,1}, ...
        'sha256', olhoffcurrent_sha256_file(p), ...
        'resolvesTo', w, ...
        'isProduction', startsWith(w, fullfile(root,'+impl'))); %#ok<AGROW>
end
out.controllerAllProduction = all([out.controller.isProduction]);
out.implTree = olhoffcurrent_source_manifest('Verify',false).treeHash;

% ---- 3. production as it resolves TODAY vs the validated candidate ------
prod = olhoffcurrent_config(NELX, NELY);
[cand, ~] = tr_config(NELX, NELY);
S = olh.config.schema();
diffs = struct('path',{},'production',{},'candidate',{});
for k = 1:size(S,1)
    p = S{k,1};
    a = olh.config.getPath(prod,p); b = olh.config.getPath(cand,p);
    if ~isequaln(a,b)
        diffs(end+1) = struct('path',p,'production',local_show(a), ...
                              'candidate',local_show(b)); %#ok<AGROW>
    end
end
out.productionVsCandidate = diffs;
out.prodCfgHash = olhoffcurrent_config_hash(prod);
out.candCfgHash = olhoffcurrent_config_hash(cand);

% the fields a promotion would have to change, ignoring labels and the
% study-only runtime overrides
LABEL_OR_RUNTIME = {'runtime.name','runtime.maxOuter','runtime.diagnostics', ...
                    'runtime.verbose','runtime.singleThread'};
core = diffs(~ismember({diffs.path}, LABEL_OR_RUNTIME));
out.promotionDelta = core;
out.promotionDeltaPaths = {core.path};

fid = fopen(fullfile(study,'evidence','dispatch.json'),'w');
c = onCleanup(@() fclose(fid)); fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true));

fprintf('\n== dispatch audit ==\n');
fprintf('olhoffcurrent_assert_dispatch : %d  %s\n', out.dispatchOk, out.dispatchErr);
for k = 1:numel(out.controller)
    fprintf('  %-46s prod=%d  %s\n', out.controller(k).path, ...
        out.controller(k).isProduction, out.controller(k).sha256(1:16));
end
fprintf('implTree %s\n', out.implTree);
fprintf('\nproduction cfgHash : %s\n', out.prodCfgHash);
fprintf('candidate  cfgHash : %s\n', out.candCfgHash);
fprintf('\nproduction vs validated candidate, %d schema differences:\n', numel(diffs));
for k = 1:numel(diffs)
    fprintf('  %-34s production=%-46s candidate=%s\n', diffs(k).path, ...
        diffs(k).production, diffs(k).candidate);
end
fprintf('\nfields a promotion would have to change (%d): %s\n', ...
    numel(core), strjoin(out.promotionDeltaPaths, ', '));
end

function s = local_show(v)
if ischar(v); s = v; elseif isstring(v); s = char(v);
elseif islogical(v); s = mat2str(v);
elseif isnumeric(v); s = mat2str(v,17);
elseif iscell(v); s = ['{' strjoin(cellfun(@local_show,v,'UniformOutput',false),',') '}'];
else, s = class(v); end
end
