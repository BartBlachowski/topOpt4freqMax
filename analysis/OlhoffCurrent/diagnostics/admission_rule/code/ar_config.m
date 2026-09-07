function [cfg, meta] = ar_config(mode, nelx, nely)
%AR_CONFIG  Configuration for the admission-rule study.
%
%   mode = 'production' : the production preset, unchanged (regression anchor)
%   mode = 'unstopped'  : identical, except the ADMISSION POLICY is disabled so
%                         the loop runs to the cap and exposes the full
%                         trajectory.  Two fields only:
%                             stop.tolerance    = 0
%                             stop.toleranceRule= 'explicit'
%                         Both are stopping-policy fields, which the brief
%                         places under explicit experimental control.  Nothing
%                         else differs -- asserted field-by-field in ar_run.
%
%   Why an unstopped run is enough.  In this configuration `convOuter` in
%   olhoffSolve is computed AFTER olh.move.limit, is never written back into the
%   move-controller state, and is read only by the settledMove guard, the stop
%   guards, the projection trigger (projection off -> dead), the p-continuation
%   stop-block (p continuation off -> dead) and the `break`.  The trajectory is
%   therefore independent of when convergence is admitted, so ONE unstopped run
%   per mesh contains the exact trajectory every candidate rule would follow.

info = olhoffcurrent_preset();
common = { 'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
           'runtime.maxOuter', 600, 'runtime.singleThread', true, ...
           'runtime.diagnostics', true, 'runtime.verbose', false, ...
           'runtime.name', sprintf('AR_%s_%dx%d', upper(mode), nelx, nely) };

switch lower(mode)
    case 'production'
        ov = {};  label = 'Production admission (L2 + settledMove)';
    case 'unstopped'
        ov = { 'stop.tolerance', 0, 'stop.toleranceRule', 'explicit' };
        label = 'Unstopped trajectory (admission disabled)';
    otherwise
        error('ar_config:UnknownMode','mode must be production or unstopped');
end

cfg = olh.config.resolve(info.upstreamPreset, common{:}, ov{:});
cfg.provenance.productionPreset = info.name;
meta = struct('mode', lower(mode), 'label', label, 'nelx', nelx, 'nely', nely, ...
              'overrides', {ov}, 'maxOuter', 600, 'preset', info.name);
end
