function [cfg, meta] = dr_config(run, nelx, nely)
%DR_CONFIG  Configuration for one of the two preregistered runs.
%
%   run = 'A'  400x50 PRODUCTION.  The canonical production entry point,
%              unmodified: production ladder, production beta-stall transition,
%              production admission/stopping rule.  NO scientific override.
%   run = 'B'  320x40 EXTENDED FIXED MOVE.  Production formulation with
%              move.policy = 'fixed', move.initial = 0.04 (the experimental
%              factor) and the stopping policy suppressed so the run reaches a
%              regime classification (PREREGISTRATION sec. 1).
%
%   No solver copy is used by either run: both go through olhoffSolve.

info = olhoffcurrent_preset();

switch upper(run)
    case 'A'
        cap = 400;                       % production runtime.maxOuter default
        [cfg, ~] = olhoffcurrent_config(nelx, nely, 'MaxOuter', cap, ...
            'Diagnostics', true, 'Name', sprintf('DR_RUNA_PROD_%dx%d', nelx, nely));
        overrides = {};
        label = 'RUN A -- 400x50 production (ladder + beta-stall + production stop)';
    case 'B'
        cap = 1200;                      % preregistered generous cap
        overrides = { 'move.policy', 'fixed', 'move.initial', 0.04, ...
                      'stop.tolerance', 0, 'stop.toleranceRule', 'explicit' };
        cfg = olh.config.resolve(info.upstreamPreset, ...
            'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
            'runtime.maxOuter', cap, 'runtime.singleThread', true, ...
            'runtime.diagnostics', true, 'runtime.verbose', false, ...
            'runtime.name', sprintf('DR_RUNB_FIXEDMOVE_%dx%d', nelx, nely), ...
            overrides{:});
        label = 'RUN B -- 320x40 extended fixed move 0.04 (unstopped)';
    otherwise
        error('dr_config:UnknownRun','run must be A or B, got %s', run);
end

cfg.provenance.productionPreset = info.name;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = struct('run', upper(run), 'label', label, 'nelx', nelx, 'nely', nely, ...
              'preset', info.name, 'upstreamPreset', info.upstreamPreset, ...
              'maxOuter', cap, 'overrides', {overrides});
end
