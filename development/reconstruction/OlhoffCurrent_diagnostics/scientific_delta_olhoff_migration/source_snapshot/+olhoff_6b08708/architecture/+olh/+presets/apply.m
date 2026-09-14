function cfg = apply(name, cfg)
%APPLY  Apply a named preset to a canonical configuration.
if nargin < 2, cfg = olh.config.defaults(); end
T = olh.presets.list();
if ~any(strcmp(name, T(:,1)))
    error('olh:presets:unknown', ...
        'Unknown preset ''%s''. Known presets:\n    %s', name, ...
        strjoin(T(:,1).', sprintf('\n    ')));
end
fn = str2func(['olh.presets.' name]);
cfg = fn(cfg);
end
