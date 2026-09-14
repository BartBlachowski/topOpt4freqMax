function cfg = duOlhoffOuterAsymptotes(cfg)
%DUOLHOFFOUTERASYMPTOTES  The frozen formulation with Svanberg's asymptote
%   memory carried across the outer loop, and no move ladder.
%
%   Classification: SCIENTIFIC_PRESET.
%   Provenance: RECONSTRUCTION preset (class C).  Not a published Du-Olhoff
%   realization.
%
%   Same formulation as duOlhoffFrozenM4 (p = 3 fixed, mass eq. (4b),
%   Sigmund sensitivity filter on every f_sk at R = 0.06, fixed subspace N = 2
%   with diagonal offsets, published MMA, the genuine nested inner loop with
%   the erratum form of (25d) live).  It differs in HOW the nested loop is
%   joined to MMA and in what may stop it:
%
%     optimizer.inner.variable        = 'design'   MMA acts on rho, not drho
%     optimizer.inner.asymptoteHistory= 'outer'    asymptotes formed from
%                                                  rho_k, rho_k-1, rho_k-2 and
%                                                  HELD during the inner loop
%     move.policy                     = 'fixed'    ONE move limit, no ladder;
%     move.initial                    = 0.04       Inf = the box (25f) alone
%     stop.guards.settledMove         = false      no ladder, nothing to settle
%     stop.guards.boxInactiveFraction = 0.5        the eps test may fire only
%                                                  when max|drho| <= 0.5*move
%
%   Why.  The frozen realization resets the MMA state every outer iteration,
%   so the outer sequence has no damping: ||drho|| chatters at a floor set by
%   the move box and the printed test ||drho|| < eps can only be fired by a
%   scheduled reduction of the box (audit_termination_mesh_admission, verdict
%   S2_CONTINUATION_DEFECT).  With the asymptotes adapted on the outer design
%   history, elements that reverse direction between outer iterations get
%   their asymptotes contracted (asydecr) and the step decays, which is the
%   convergence mechanism of Svanberg (1987) and is what lets the printed
%   test fire on its own.
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffFrozenM4(cfg);
cfg = olh.config.assign(cfg, ...
    'optimizer.inner.variable',         'design', ...
    'optimizer.inner.asymptoteHistory', 'outer', ...
    'move.policy',                      'fixed', ...
    'move.initial',                     0.04, ...
    'stop.guards.settledMove',          false, ...
    'stop.guards.settledWindow',        1, ...
    'stop.guards.boxInactiveFraction',  0.5);
end
