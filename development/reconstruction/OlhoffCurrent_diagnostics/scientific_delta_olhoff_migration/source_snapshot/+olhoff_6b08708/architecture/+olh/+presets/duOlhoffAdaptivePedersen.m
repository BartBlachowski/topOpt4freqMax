function cfg = duOlhoffAdaptivePedersen(cfg)
%DUOLHOFFADAPTIVEPEDERSEN  Adaptive move box + Pedersen (2000) stiffness
%   linearization + linear mass, fixed PHYSICAL filter radius 0.06.
%
%   Classification: SCIENTIFIC_PRESET.  Provenance: RECONSTRUCTION (class C
%   in the box policy and the radius; class B in the Pedersen scheme, which
%   Du & Olhoff sec. 2.2 name as their alternative to the mass cut-off).
%
%   Parent duOlhoffAdaptiveMove.  Departures:
%     material.stiffness.model       = 'pedersen', linearBelow = 0.1 (printed)
%     material.mass.model            = 'eq2'  (linear, as Pedersen uses it)
%     move.initial                   = 0.10
%     filter.radiusPhysical          = 0.06  (= 1.2 el at 160x20, 6 el at 800x100)
%
%   Why.  NOTES sec. 32: the Pedersen scheme removes every localized-mode
%   spike that (4)/(4a)/(4b) leave, which is what lets the printed test
%   ||drho|| < eps with the mesh-scaled eps fire on a mature, black-and-white
%   design at 800x100 (run H1) instead of on a grey one (C1) or inside a
%   spike (E2).  The radius is fixed physically so that one realization is
%   run at every mesh of a sweep; 0.06 is the smallest value that is still a
%   filter (>= 1.2 elements) at the 160x20 floor.  Validated at 240x30 and
%   800x100 with R = 0.0433 (P1, H1); R = 0.06 is validated by the sweep
%   itself (repro/results/S*_*).
if nargin < 1, cfg = olh.config.defaults(); end
cfg = olh.presets.duOlhoffAdaptiveMove(cfg);
cfg = olh.config.assign(cfg, ...
    'material.stiffness.model',        'pedersen', ...
    'material.stiffness.linearBelow',  0.1, ...
    'material.mass.model',             'eq2', ...
    'move.initial',                    0.10, ...
    'filter.radiusPhysical',           0.06, ...
    'filter.radiusElements',           []);
end
