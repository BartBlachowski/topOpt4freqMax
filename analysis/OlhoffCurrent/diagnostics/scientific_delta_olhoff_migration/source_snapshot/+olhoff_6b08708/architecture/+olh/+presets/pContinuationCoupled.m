function cfg = pContinuationCoupled(cfg)
%PCONTINUATIONCOUPLED  p continuation driven by the move-ladder stage.
%
%   Historical label: P1 (audit_p_continuation).
%   Classification: EXPERIMENT_PRESET.
%   Parent: duOlhoffMatureM4.  Departure: stiffness continuation.
%
%   p runs 1 -> 2 -> 3, INDEXED BY THE LADDER STAGE.  The transition is the
%   stall event the move controller already computes, so no new numerical
%   constant enters.  That reuse is a deliberate policy, not an accident, and it
%   is what distinguishes this realization from pContinuationDecoupled.
%
%   Provenance: sec. 2.1 states p is "normally assigned values increasing from 1
%   to 3 during the optimization process", so running a p schedule is CLASS A as
%   a practice and closer to the paper than the frozen fixed p=3.  The schedule
%   VALUES are class B (endpoints printed) and the transition RULE is class C
%   (the paper gives none).
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffMatureM4(cfg);
cfg = olh.config.assign(cfg, ...
    'material.stiffness.continuation.enabled',  true, ...
    'material.stiffness.continuation.schedule', [1 2 3], ...
    'material.stiffness.continuation.driver',   'moveLadderStage');
end
