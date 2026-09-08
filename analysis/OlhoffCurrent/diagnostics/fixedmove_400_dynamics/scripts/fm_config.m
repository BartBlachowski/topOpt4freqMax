function [cfg, meta] = fm_config(nelx, nely)
%FM_CONFIG  RUN C -- the single authorized run: 400x50, fixed move = 0.04.
%
%   Canonical production scientific formulation with exactly two conceptual
%   changes, both preregistered:
%     move.policy='fixed', move.initial=0.04   the mechanism-measurement arm
%     stop.tolerance=0, stop.toleranceRule='explicit'
%                                              the MINIMAL stop override, so
%                                              native termination cannot
%                                              truncate the measurement
%   Nothing else differs.  No solver copy is used.

info = olhoffcurrent_preset();
cap  = 1200;                                   % preregistered safety cap
overrides = { 'move.policy', 'fixed', 'move.initial', 0.04, ...
              'stop.tolerance', 0, 'stop.toleranceRule', 'explicit' };

cfg = olh.config.resolve(info.upstreamPreset, ...
        'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
        'runtime.maxOuter', cap, 'runtime.singleThread', true, ...
        'runtime.diagnostics', true, 'runtime.verbose', false, ...
        'runtime.name', sprintf('FM_RUNC_FIXEDMOVE_%dx%d', nelx, nely), ...
        overrides{:});

cfg.provenance.productionPreset = info.name;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = struct('run','C','label','RUN C -- 400x50 fixed move 0.04 (unstopped)', ...
              'nelx',nelx,'nely',nely,'preset',info.name, ...
              'upstreamPreset',info.upstreamPreset,'maxOuter',cap, ...
              'overrides',{overrides});
end
