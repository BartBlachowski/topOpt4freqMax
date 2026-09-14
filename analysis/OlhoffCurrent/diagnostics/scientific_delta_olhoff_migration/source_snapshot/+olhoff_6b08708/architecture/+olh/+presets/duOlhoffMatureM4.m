function cfg = duOlhoffMatureM4(cfg)
%DUOLHOFFMATUREM4  The frozen realization allowed to finish.
%
%   Historical labels: Bmature, R2.
%   Classification: EXPERIMENT_PRESET.
%   Parent: duOlhoffFrozenM4.  Departure: one stopping guard.
%
%   Adds stop.guards.maxDesignChange: convergence is asserted only once
%   max|d(design)| has fallen below eps/sqrt(NE), the same RMS scale the outer
%   tolerance uses.  Without it the frozen rule can stop while the design is
%   still moving at a level the ladder has merely stopped resolving.
%
%   Provenance: CLASS D.  A preregistered safeguard from
%   audit_m4_topology_restoration; it appears in no Du-Olhoff source.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffFrozenM4(cfg);
cfg = olh.config.assign(cfg, 'stop.guards.maxDesignChange', true);
end
