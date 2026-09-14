function cfg = duOlhoffAdaptiveMove(cfg)
%DUOLHOFFADAPTIVEMOVE  The frozen formulation with a per-element move box
%   adapted by Svanberg's asymptote rule on the outer design history.
%
%   Classification: SCIENTIFIC_PRESET.
%   Provenance: RECONSTRUCTION preset (class C).  Not a published Du-Olhoff
%   realization.
%
%   Same formulation and the same genuine nested inner loop as
%   duOlhoffFrozenM4 (p = 3 fixed, mass eq. (4b), Sigmund sensitivity filter
%   on every f_sk at R = 0.06, fixed subspace N = 2 with diagonal offsets,
%   published MMA on the increment with asymptotes reset per outer iteration,
%   erratum form of (25d) live).  It replaces the move LADDER by
%
%     move.policy   = 'adaptive'   per-element box d_e; an element whose last
%                                  two outer steps agree in sign gets
%                                  d_e *= 1.2, one that reversed gets d_e *= 0.7
%                                  (Svanberg's asyincr / asydecr), clamped to
%                                  [move.minimum, move.initial]
%     move.initial  = 0.04         the ladder's own first level
%     move.minimum  = 0.002
%     stop.guards.settledMove = false        there is no schedule to settle
%
%   Why.  Under a fixed or laddered box the outer sequence has no damping:
%   the frozen sub-problem is solved to convergence at every outer iteration
%   and the design chatters at a floor set by the box, so the printed test
%   ||drho|| < eps can only be fired by a scheduled reduction of the box
%   (verdict S2_CONTINUATION_DEFECT).  In a single-call MMA architecture that
%   damping is provided by the asymptotes adapting on the design history; here
%   the SAME rule is applied to the box, so the inner loop is untouched and
%   ||drho|| decays where the design oscillates while the design keeps moving
%   at full pace where it evolves monotonically.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffFrozenM4(cfg);
cfg = olh.config.assign(cfg, ...
    'move.policy',                      'adaptive', ...
    'move.initial',                     0.04, ...
    'move.minimum',                     0.002, ...
    'move.adaptive.grow',               1.2, ...
    'move.adaptive.shrink',             0.7, ...
    'stop.guards.settledMove',          false, ...
    'stop.guards.boxInactiveFraction',  0);
end
