function cfg = noDescentFixedMove(cfg)
%NODESCENTFIXEDMOVE  Frozen realization with the move descent removed.
%
%   Historical label: nodescent (audit_s2_final_conference_restoration).
%   Classification: EXPERIMENT_PRESET.
%   Parent: duOlhoffFrozenM4.  Departure: move policy only.
%
%   The move limit is held at the ladder's own first level, 0.04, for the whole
%   run.  No new constant is introduced: the value is the one the ladder starts
%   from.  This isolates what the descent contributes, after
%   audit_s2_design_continuation returned S2_LADDER_ITSELF_DEFECTIVE.
%
%   Note the interaction that makes this preset worth having: with a FIXED move
%   the settledMove guard is vacuously satisfied at every iteration, so the
%   frozen stopping rule is evaluated exactly as Du & Olhoff (sec. 3.5.1) write
%   it, with no schedule artefact to suppress.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffFrozenM4(cfg);
cfg = olh.config.assign(cfg, 'move.policy', 'fixed', 'move.initial', 0.04);
end
