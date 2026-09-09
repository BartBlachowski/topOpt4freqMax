function [cfg, meta] = cv_config(arm, nelx, nely)
%CV_CONFIG  Configuration for one arm of the causal controller validation.
%
%   arm = 'P'  the production baseline, exactly olhoffcurrent_config
%   arm = 'C'  the candidate: the frozen two-branch stage-exhaustion controller
%
%   The 'C' override list is EXACTLY the two declared switches plus the
%   preregistered safety cap and the recorder.  Nothing else.  Any other
%   difference is a single-factor failure and cv_singlefactor stops the study.
%
%   Cap 1600 and diagnostics ON are preregistered
%   (PREREGISTRATION.md sections 5 and 7); the cap is common to all three meshes
%   and is not raised after seeing a trajectory.

CAP = 1600;
info = olhoffcurrent_preset();

switch upper(char(arm))
    case 'P'
        armOv = {};
        label = 'production move ladder, beta-stall continuation';
    case 'C'
        armOv = {'move.continuation.signal', 'stageExhaustion', ...
                 'stop.rule',                'stageExhaustion'};
        label = 'candidate: frozen two-branch stage exhaustion, E = A OR B';
    otherwise
        error('cv_config:UnknownArm','arm must be P or C, got %s', arm);
end

common = { 'domain.mesh.nelx',     nelx, ...
           'domain.mesh.nely',     nely, ...
           'runtime.maxOuter',     CAP, ...
           'runtime.singleThread', true, ...
           'runtime.diagnostics',  true, ...
           'runtime.verbose',      false, ...
           'runtime.name',         sprintf('CV_%s_%dx%d', upper(char(arm)), nelx, nely) };

cfg = olh.config.resolve(info.upstreamPreset, common{:}, armOv{:});
cfg.provenance.productionPreset = info.name;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = struct('arm', upper(char(arm)), 'label', label, 'nelx', nelx, 'nely', nely, ...
              'preset', info.name, 'upstreamPreset', info.upstreamPreset, ...
              'overrides', {armOv}, 'maxOuter', CAP);
end
