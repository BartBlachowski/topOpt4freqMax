function [cfg, meta] = ms_config(arm, nelx, nely)
%MS_CONFIG  Configuration for one diagnostic arm.  ONE conceptual factor differs.
%
%   arm = 'baseline'  : the production preset, unchanged
%   arm = 'fixedmove' : identical except move.policy = fixed, move.initial = 0.04
%
%   Both arms are built through the sanctioned route -- olh.config.resolve on the
%   production preset -- so both are validated, and the ONLY difference between
%   the two override lists is the move policy.  ms_run additionally asserts that
%   the baseline arm is field-for-field identical to what the production entry
%   point olhoffcurrent_config produces, so "baseline" really is production.
%
%   Cap and recorder are preregistered: 400 outer (the production
%   runtime.maxOuter, deliberately not a new number) and diagnostics ON.

info = olhoffcurrent_preset();

common = { 'domain.mesh.nelx',     nelx, ...
           'domain.mesh.nely',     nely, ...
           'runtime.maxOuter',     400, ...
           'runtime.singleThread', true, ...
           'runtime.diagnostics',  true, ...
           'runtime.verbose',      false, ...
           'runtime.name',         sprintf('MS_%s_%dx%d', upper(arm), nelx, nely) };

switch lower(arm)
    case 'baseline'
        armOv = {};
        label = 'Production: move ladder';
    case 'fixedmove'
        armOv = { 'move.policy', 'fixed', 'move.initial', 0.04 };
        label = 'Diagnostic: fixed move 0.04';
    otherwise
        error('ms_config:UnknownArm', 'arm must be baseline or fixedmove, got %s', arm);
end

cfg = olh.config.resolve(info.upstreamPreset, common{:}, armOv{:});
cfg.provenance.productionPreset = info.name;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = struct('arm', lower(arm), 'label', label, 'nelx', nelx, 'nely', nely, ...
              'preset', info.name, 'upstreamPreset', info.upstreamPreset, ...
              'overrides', {armOv}, 'maxOuter', 400);
end
