function cfg = projectionIdentity(cfg)
%PROJECTIONIDENTITY  Density filter with the projection at exact identity.
%
%   Historical label: D160 (audit_m4_projection_invariance).
%   Classification: EXPERIMENT_PRESET -- a CONTROL, not a treatment.
%   Parent: duOlhoffMatureM4.
%
%   Projection is enabled at betaProj = 0, where the tanh operator returns the
%   exact identity.  What remains is the change of FORMULATION: the design
%   variable becomes z, the Sigmund SENSITIVITY filter is replaced by a DENSITY
%   filter, and sensitivities reach z through the chain rule.
%
%   This is what separates the two Class-D departures from each other:
%     D2  the filter switch          <- isolated by THIS preset
%     D1  the projection itself      <- added by olh.presets.projected
%   Without this control, any difference between the frozen realization and a
%   projected one confounds the two.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffMatureM4(cfg);
cfg = olh.config.assign(cfg, ...
    'filter.type',            'density', ...
    'projection.enabled',     true, ...
    'projection.beta.levels', 0, ...
    'projection.eta',         0.5);
end
