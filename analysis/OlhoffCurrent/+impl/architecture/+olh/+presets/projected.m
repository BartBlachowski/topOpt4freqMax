function cfg = projected(cfg)
%PROJECTED  The preregistered projection treatment.
%
%   Historical labels: T160, T240, T320, T800.
%   Classification: EXPERIMENT_PRESET.
%   Parent conceptual basis: the frozen reconstruction (via duOlhoffMatureM4).
%
%   *** POST-PUBLICATION MODIFICATION / NEW REALIZATION. ***
%   Du & Olhoff did NOT publish this.  "projection", "Heaviside" and "density
%   filter" occur ZERO times in Du & Olhoff (2007) and in Olhoff & Du (2014),
%   and the operator is absent from the Krog & Olhoff lineage.  It postdates the
%   2007 paper.  Do not present any projected result as theirs.
%
%   Formulation
%     design variable  z, box [0,1]
%     filtered density zTilde = (H z)/Hs                        DENSITY filter
%     physical density rhoPhys = rhomin + (1-rhomin)*P(zTilde;beta,eta), P tanh
%     sensitivities    carried to z by the complete chain rule, applied to EVERY
%                      generalized gradient, diagonal and off-diagonal, and to f_JJ
%     volume           (25e) evaluated EXACTLY at z+dz, value and gradient as a
%                      consistent pair
%     continuation     beta = 1 -> 2 -> 4 -> 8, each level advancing when the
%                      outer convergence event fires; no new constant, no new
%                      window, no new counter
%
%   The filter radius, the move ladder, MMA, the generalized gradients and the
%   multiplicity treatment are untouched.
%
%   NOTE the stopping-field consequence, which the architecture now makes
%   explicit: stop.field is the DESIGN variable, so under projection the outer
%   test monitors d(z), not d(rhoPhys).  The physical change is recorded as
%   hist.dxPhys2 for cross-reference.  This differs in MEANING from the frozen
%   realization, where the design variable IS the density.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffMatureM4(cfg);
cfg = olh.config.assign(cfg, ...
    'filter.type',            'density', ...
    'projection.enabled',     true, ...
    'projection.beta.levels', [1 2 4 8], ...
    'projection.eta',         0.5, ...
    'runtime.maxOuter',       1200);
end
