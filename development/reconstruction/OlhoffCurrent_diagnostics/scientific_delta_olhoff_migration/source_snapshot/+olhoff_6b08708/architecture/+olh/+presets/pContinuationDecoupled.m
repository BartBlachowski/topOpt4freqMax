function cfg = pContinuationDecoupled(cfg)
%PCONTINUATIONDECOUPLED  p continuation on its own counter.
%
%   Historical label: PD1 (audit_m4_p_continuation_decoupled).
%   Classification: EXPERIMENT_PRESET.
%   Parent: pContinuationCoupled.  Departure: the continuation driver.
%
%   p still advances on the SAME stall event, but it no longer shares the
%   ladder's index: while p is below its final value the stall is consumed by
%   the p controller instead of the ladder, the move is restored to the ladder's
%   first level and the ladder's re-arm clock is reset.  Once p is final the
%   event is left to the ladder and refinement proceeds normally.
%
%   The point of the experiment: under the coupled driver, p and the move limit
%   cannot be varied independently, so a p-continuation result is confounded
%   with a move-schedule result.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.pContinuationCoupled(cfg);
cfg = olh.config.assign(cfg, 'material.stiffness.continuation.driver', 'ownCounter');
end
