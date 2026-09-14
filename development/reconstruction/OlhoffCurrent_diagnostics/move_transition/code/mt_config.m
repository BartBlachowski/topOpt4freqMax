function [cfg, tcfg, meta] = mt_config(arm, nelx, nely)
%MT_CONFIG  Configuration for one arm of the move-transition study.
%
%   arm = 'P' : PRODUCTION.  Move descent on the existing bound-variable stall
%               detector.  tcfg.metric = 'boundVariableStall', which makes
%               mt_moveLimit delegate verbatim to olh.move.limit.
%   arm = 'U' : UTILIZATION-GATED TRANSITION.  Identical configuration; the ONLY
%               difference is the move-stage transition signal.
%
%   THE SCIENTIFIC CONFIGURATION IS THE SAME OBJECT FOR BOTH ARMS.  The
%   experimental factor is not a cfg field at all -- it is the transition
%   controller handed to the solver -- so the field-level cfg diff between the
%   arms is EMPTY by construction, and mt_run asserts it.
%
%   Both arms are run UNSTOPPED (stop.tolerance = 0, stop.toleranceRule =
%   'explicit' -- two stopping-policy fields, nothing else) to the preregistered
%   safety cap of 600, and the terminal admission rule is evaluated offline.
%   This is sound because, with projection and p-continuation off, convOuter is
%   computed after olh.move.limit, is never written back into hist or into the
%   move-controller state, and only drives the break; the trajectory is
%   therefore independent of when convergence is admitted.  Verified bitwise at
%   both meshes by the preceding admission-rule study.

info = olhoffcurrent_preset();

common = { 'domain.mesh.nelx',     nelx, ...
           'domain.mesh.nely',     nely, ...
           'runtime.maxOuter',     600, ...
           'runtime.singleThread', true, ...
           'runtime.diagnostics',  true, ...
           'runtime.verbose',      false, ...
           'runtime.name',         sprintf('MT_ARM%s_%dx%d', upper(arm), nelx, nely) };

% Unstopped: the admission decision is made offline, identically for both arms.
unstop = { 'stop.tolerance', 0, 'stop.toleranceRule', 'explicit' };

switch upper(arm)
    case 'P'
        tcfg  = struct('metric','boundVariableStall');
        label = 'ARM P -- production bound-variable-stall move transition';
    case 'U'
        tcfg  = struct('metric','maxUtilization','threshold',0.5,'persistence',10);
        label = 'ARM U -- utilization-gated move transition (r_rho < 0.5 x 10)';
    otherwise
        error('mt_config:UnknownArm','arm must be P or U, got %s', arm);
end

cfg = olh.config.resolve(info.upstreamPreset, common{:}, unstop{:});
cfg.provenance.productionPreset = info.name;
cfg.provenance.implementation   = 'analysis/OlhoffCurrent';

meta = struct('arm', upper(arm), 'label', label, 'nelx', nelx, 'nely', nely, ...
              'preset', info.name, 'upstreamPreset', info.upstreamPreset, ...
              'maxOuter', 600, 'unstopped', {unstop}, 'tcfg', tcfg);
end
