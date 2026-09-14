function D = mt_configDiff(meshes)
%MT_CONFIGDIFF  Field-level resolved-configuration diff, ARM U vs ARM P vs
%   production (brief sec. 9).
%
%   The experimental factor of this study is NOT a configuration field: it is
%   the transition controller handed to the solver.  The field-level diff
%   between ARM P and ARM U is therefore EMPTY by construction, and that is the
%   cleanest possible statement of "single factor".  The diff versus production
%   contains only the two declared stopping-policy fields that unstop the run
%   (the terminal admission is applied offline, identically for both arms) and
%   the free-text run name.

S = olh.config.schema();
D = struct('note', ['The experimental factor is the move-transition controller ' ...
                    '(mt_moveLimit), not a configuration field. ARM P and ARM U ' ...
                    'resolve to the SAME scientific configuration.'], ...
           'meshes', {{}});
for m = 1:size(meshes,1)
    nelx = meshes(m,1); nely = meshes(m,2);
    [cP, tP] = mt_config('P', nelx, nely);
    [cU, tU] = mt_config('U', nelx, nely);
    prod = olhoffcurrent_config(nelx, nely, 'MaxOuter', 600, 'Diagnostics', true);

    dPU = local_diff(S, cP, cU);
    dPprod = local_diff(S, prod, cP);
    dUprod = local_diff(S, prod, cU);

    D.meshes{end+1} = struct('mesh',[nelx nely], ...
        'cfgHash_armP', olhoffcurrent_config_hash(cP), ...
        'cfgHash_armU', olhoffcurrent_config_hash(cU), ...
        'cfgHash_production', olhoffcurrent_config_hash(prod), ...
        'nSchemaFields', size(S,1), ...
        'diff_armP_vs_armU', {dPU}, ...
        'diff_production_vs_armP', {dPprod}, ...
        'diff_production_vs_armU', {dUprod}, ...
        'transition_armP', tP, 'transition_armU', tU);
end
end

function d = local_diff(S, a, b)
d = {};
for k = 1:size(S,1)
    p = S{k,1};
    va = olh.config.getPath(a,p); vb = olh.config.getPath(b,p);
    if ~isequaln(va, vb)
        d{end+1} = struct('field',p,'a',local_str(va),'b',local_str(vb)); %#ok<AGROW>
    end
end
end

function s = local_str(v)
if ischar(v); s = v; elseif isstring(v); s = char(v);
elseif islogical(v); s = mat2str(v);
elseif isnumeric(v); s = mat2str(v,17);
else; s = class(v); end
end
