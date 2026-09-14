function [cM, cS, D, rep] = sd_m1_config()
%SD_M1_CONFIG  Resolve the preregistered M1 configuration from the snapshot and
%   compare it leaf by leaf with the stored S480x60 configuration.
%   Requires sd_use_source() to have been called.
P = sd_paths();
L  = load(P.s480, 'cfg');
cS = L.cfg;
cM = olh.config.resolve('duOlhoffAdaptiveMove', ...
    'domain.mesh.nelx', 480, 'domain.mesh.nely', 60, ...
    'runtime.name', 'M1_480x60_simp4b_adaptive', 'runtime.verbose', true, ...
    'move.initial', 0.10, 'runtime.diagnostics', true);
D = sd_cfgdiff(cS, cM);
allowed = {'material.stiffness.model','material.mass.model','runtime.name','runtime.diagnostics'};
rep = struct('path',{},'s480',{},'m1',{},'allowed',{});
for i = 1:numel(D)
    rep(end+1) = struct('path', D(i).path, 's480', sd_str(D(i).a), 'm1', sd_str(D(i).b), ...
        'allowed', any(strcmp(D(i).path, allowed)) ); %#ok<AGROW>
end
end

function s = sd_str(v)
if ischar(v) || isstring(v), s = char(v);
elseif isnumeric(v) || islogical(v), s = mat2str(v, 17);
elseif iscell(v), s = strjoin(cellfun(@sd_str, v, 'UniformOutput', false), ' | ');
else, s = class(v);
end
end
