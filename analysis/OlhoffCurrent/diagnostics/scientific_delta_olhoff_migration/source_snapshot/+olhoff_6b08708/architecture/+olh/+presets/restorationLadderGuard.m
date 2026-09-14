function cfg = restorationLadderGuard(cfg)
%RESTORATIONLADDERGUARD  Frozen realization with the ladder-exhaustion guard.
%
%   Historical label: R1.
%   Classification: EXPERIMENT_PRESET.
%   Parent: duOlhoffFrozenM4.  Departure: one stopping guard.
%
%   Convergence is asserted only once no REMAINING move-ladder level exceeds
%   eps/sqrt(NE) -- i.e. once no scheduled step could still produce a change the
%   tolerance would regard as significant.  This is a statement about the
%   SCHEDULE; maxDesignChange is a statement about the DESIGN.  The two are
%   different questions and are deliberately separate fields.
%
%   Provenance: CLASS D, preregistered in audit_m4_topology_restoration.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffFrozenM4(cfg);
cfg = olh.config.assign(cfg, 'stop.guards.ladderExhausted', true);
end
