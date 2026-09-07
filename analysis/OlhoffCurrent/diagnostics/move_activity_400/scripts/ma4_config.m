function [cfg, meta] = ma4_config(arm, nelx, nely)
%MA4_CONFIG  Configuration for one arm of the 400x50 third-mesh measurement.
%
%   arm = 'P'  production: the production preset, unchanged
%   arm = 'F'  fixed-move counterfactual: move.policy='fixed', move.initial=0.04
%
%   The 'F' override pair is EXACTLY the pair move_stop used for its fixedmove
%   arm at 160x20 and 320x40 (ms_config.m), so the third mesh is comparable to
%   the first two by construction rather than by resemblance.
%
%   Caps are preregistered: 400 for P (the production default; production stops
%   far earlier) and 600 for F (generous, because 400x50 is a new mesh and
%   160x20 hit move_stop's 400).  A larger cap cannot change a trajectory, only
%   when it stops.

info = olhoffcurrent_preset();

switch upper(char(arm))
    case 'P', armOv = {};                                          cap = 400;
              label = 'ARM P400 -- production move ladder';
    case 'F', armOv = {'move.policy','fixed','move.initial',0.04};  cap = 600;
              label = 'ARM F400 -- fixed move 0.04 counterfactual';
    otherwise, error('ma4_config:UnknownArm','arm must be P or F, got %s', arm);
end

common = { 'domain.mesh.nelx',     nelx, ...
           'domain.mesh.nely',     nely, ...
           'runtime.maxOuter',     cap, ...
           'runtime.singleThread', true, ...
           'runtime.diagnostics',  true, ...
           'runtime.verbose',      false, ...
           'runtime.name',         sprintf('MA4_%s_%dx%d', upper(char(arm)), nelx, nely) };

cfg = olh.config.resolve(info.upstreamPreset, common{:}, armOv{:});
cfg.provenance.productionPreset = info.name;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = struct('arm', upper(char(arm)), 'label', label, 'nelx', nelx, 'nely', nely, ...
              'preset', info.name, 'upstreamPreset', info.upstreamPreset, ...
              'overrides', {armOv}, 'maxOuter', cap);
end
