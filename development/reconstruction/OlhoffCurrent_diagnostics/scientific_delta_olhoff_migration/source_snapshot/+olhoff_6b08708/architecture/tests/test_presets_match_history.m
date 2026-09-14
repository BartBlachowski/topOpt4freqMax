function nFail = test_presets_match_history()
%TEST_PRESETS_MATCH_HISTORY  Each preset must reproduce its historical config.
%
%   For every historical realization, resolve the corresponding preset at the
%   same mesh and compare the LEGACY rendering field by field against the config
%   that was actually run.  Only fields the legacy solver reads are compared;
%   run labels and diagnostics flags are runtime, not science.

root = '/Users/piotrek/Programming/Matlab/Olhoff';

% preset                     mesh        historical config
C = {
'duOlhoffFrozenM4',         [160 20],  fullfile(root,'audit_m4_projection_invariance','runs','REG160_160x20.mat.config.mat')
'duOlhoffMatureM4',         [160 20],  fullfile(root,'audit_m4_projection_invariance','runs','B160_160x20.mat.config.mat')
'duOlhoffMatureM4',         [240 30],  fullfile(root,'audit_m4_projection_invariance','runs','B240_240x30.mat.config.mat')
'duOlhoffMatureM4',         [320 40],  fullfile(root,'audit_m4_projection_invariance','runs','B320_320x40.mat.config.mat')
'restorationLadderGuard',   [320 40],  fullfile(root,'audit_m4_topology_restoration','runs','R1_320x40.mat.config.mat')
'duOlhoffMatureM4',         [320 40],  fullfile(root,'audit_m4_topology_restoration','runs','R2_320x40.mat.config.mat')
'pContinuationCoupled',     [320 40],  fullfile(root,'audit_p_continuation','runs','P1_320x40.mat.config.mat')
'pContinuationDecoupled',   [320 40],  fullfile(root,'audit_m4_p_continuation_decoupled','runs','PD1_320x40.mat.config.mat')
'pMassCompatible',          [320 40],  fullfile(root,'audit_pm1_printed_mass_continuation','runs','PM1_320x40.mat.config.mat')
'projectionIdentity',       [160 20],  fullfile(root,'audit_m4_projection_invariance','runs','D160_160x20.mat.config.mat')
'projected',                [160 20],  fullfile(root,'audit_m4_projection_invariance','runs','T160_160x20.mat.config.mat')
'projected',                [240 30],  fullfile(root,'audit_m4_projection_invariance','runs','T240_240x30.mat.config.mat')
'projected',                [320 40],  fullfile(root,'audit_m4_projection_invariance','runs','T320_320x40.mat.config.mat')
'projected',                [800 100], fullfile(root,'audit_projection_800x100','runs','T800_800x100.mat.config.mat')
};

% Runtime-only fields: caps and labels chosen per campaign, not formulation.
RUNTIME = {'name','diag','verbose','threads','maxOuter','mmasubPath'};

% Fields a preset emits that the historical config OMITTED.  Emitting them is
% only admissible if the legacy solver would have behaved identically without
% them, so each one carries its justification and the justification is CHECKED.
%   'fallback'  the legacy code supplies exactly this value when the field is
%               absent -- verified against the cited line.
%   'unread'    the field is read only on a code path this config never takes.
JUSTIFIED = {
's2Signal', 'fallback', 'beta',  'moveControl.m:115 -- absent or empty => ''beta'''
'tolEnter', 'unread',   [],      'multRule.m:60 -- read only by multRule ''hyst'''
'tolExit',  'unread',   [],      'multRule.m:62 -- read only by multRule ''hyst'''
};

fprintf('=== presets vs the configurations actually run ===\n');
nFail = 0;
for k = 1:size(C,1)
    preset = C{k,1};  mesh = C{k,2};  file = C{k,3};
    L = load(file);  hist = L.cfg;
    ws = warning('off','olh:config:suspicious');
    cfg = olh.config.resolve(preset, 'domain.mesh.nelx', mesh(1), ...
                                     'domain.mesh.nely', mesh(2));
    warning(ws);
    got = olh.config.toLegacy(cfg);

    hist = local_effRadius(hist);  got = local_effRadius(got);
    bad = {};
    f = setdiff(union(fieldnames(hist), fieldnames(got)), RUNTIME);
    for j = 1:numel(f)
        a = local_get(hist,f{j});  b = local_get(got,f{j});
        if isequaln(local_norm(a), local_norm(b)), continue; end
        % preset-only field: admissible only with a checked justification
        ji = find(strcmp(f{j}, JUSTIFIED(:,1)), 1);
        if ~isfield(hist, f{j}) && ~isempty(ji)
            switch JUSTIFIED{ji,2}
                case 'fallback'
                    if isequaln(local_norm(b), local_norm(JUSTIFIED{ji,3}))
                        continue    % emits exactly the legacy fallback
                    end
                case 'unread'
                    if ~strcmp(got.multRule,'hyst')
                        continue    % never reached by this configuration
                    end
            end
        end
        bad{end+1} = sprintf('%s: run=%s preset=%s', f{j}, ...
            local_show(a), local_show(b)); %#ok<AGROW>
    end
    [~,fn,~] = fileparts(file);
    if isempty(bad)
        fprintf('  ok    %-24s %4dx%-4d <- %s\n', preset, mesh(1), mesh(2), fn);
    else
        fprintf('  FAIL  %-24s %4dx%-4d <- %s\n', preset, mesh(1), mesh(2), fn);
        for j = 1:numel(bad), fprintf('          %s\n', bad{j}); end
        nFail = nFail + 1;
    end
end
fprintf('=== preset mismatches: %d ===\n', nFail);
end

function v = local_get(s,f)
if isfield(s,f), v = s.(f); else, v = '<absent>'; end
end
function c = local_effRadius(c)
if isfield(c,'rminPhys') && ~isempty(c.rminPhys) && ~isnan(c.rminPhys) && c.rminPhys > 0
    c.rminEl = c.rminPhys/(c.b/c.nely);
end
end

function v = local_norm(v)
if isnumeric(v) && isscalar(v) && isnan(v), v = []; end
if islogical(v), v = double(v); end
if isnumeric(v) && isvector(v) && ~isscalar(v), v = v(:).'; end
if isstruct(v)
    f = sort(fieldnames(v)); w = struct();
    for i=1:numel(f), w.(f{i}) = local_norm(v.(f{i})); end
    v = w;
end
end
function s = local_show(v)
if ischar(v)
    s = ['''' v ''''];
elseif isnumeric(v) && isscalar(v)
    s = num2str(v,'%.17g');
elseif islogical(v) && isscalar(v)
    s = mat2str(v);
elseif isnumeric(v)
    s = mat2str(v);
elseif isstruct(v)
    s = ['struct(' strjoin(fieldnames(v)',',') ')'];
else
    s = class(v);
end
end
