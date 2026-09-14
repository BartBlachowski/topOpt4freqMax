function cfg = defaults()
%DEFAULTS  The canonical default configuration, built from olh.config.schema.
%
%   This is the ONLY source of default values in the project.  Nothing else may
%   supply a fallback: not a runner, not a preset, and not the solver.
%
%   See also OLH.CONFIG.SCHEMA, OLH.CONFIG.RESOLVE.

S = olh.config.schema();
cfg = struct();
for i = 1:size(S,1)
    cfg = olh.config.setPath(cfg, S{i,1}, S{i,3});
end
end
