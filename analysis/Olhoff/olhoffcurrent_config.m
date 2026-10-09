function [cfg, info] = olhoffcurrent_config(nelx, nely, varargin)
%OLHOFFCURRENT_CONFIG  Effective configuration of a NAMED preset at one mesh.
%
%   [cfg, info] = OLHOFFCURRENT_CONFIG(nelx, nely, 'Preset', name) returns the
%   CANONICAL configuration that preset resolves to at that mesh, built through
%   the only sanctioned route:
%
%       olh.config.defaults -> upstream preset -> preset overrides -> mesh and
%       runtime overrides -> derived rules -> validate
%
%   The mesh is applied as an OVERRIDE, before the derived rules run, so the
%   mesh-scaled outer tolerance eps = 0.05*sqrt(NE/3200) is re-derived for this
%   mesh automatically rather than retyped.
%
%   Name/value options:
%     'Preset'    REQUIRED.  A canonical OlhoffCurrent preset name or a declared
%                 compatibility alias (OLHOFFCURRENT_PRESETS).  There is no
%                 unnamed default, so no call can change formulation silently
%                 when the production choice changes.
%     'MaxOuter'  (default: the preset's runtime default -- 400, or 1600 for the
%                 historical stage-exhaustion diagnostic)  outer iteration cap.
%                 A run that reaches it is CAP_HIT and is NOT converged.  A
%                 RUNTIME override, not a different preset.
%     'StopToleranceFactor' (default [] = the preset's rule)  c in the outer
%                 stop test ||drho||_2 < c*sqrt(NE/3200).  The preset's
%                 meshScaled rule is the same law with c = 0.05
%                 (olh.config.epsilonForMesh); the paper gives no value.  A
%                 value here resolves stop.toleranceRule = 'explicit', so the
%                 configuration hash records the change.  A RUNTIME override,
%                 not a different preset.
%     'StopMaxChangeTolerance' (default [] = off)  replaces the outer stop by
%                 the Proposed method's rule: stop at the first outer
%                 iteration with max|drho| <= tol, on the design variable, with
%                 no mesh scaling and no guards (stop.norm = 'max',
%                 stop.rule = 'designChange', every stop.guards.* off).  The
%                 solver tests max|drho| < stop.tolerance, so stop.tolerance is
%                 set to the next double above tol, which makes the test
%                 exactly max|drho| <= tol.  Cannot be combined with
%                 'StopToleranceFactor'.  A RUNTIME override, not a different
%                 preset.
%     'StopRelativeChangeTolerance' (default [] = off)  replaces the outer stop
%                 by the relative-increment rule: stop at the first outer
%                 iteration with ||drho||_2/||rho||_2 < tol, rho the design
%                 variable BEFORE the update, with no mesh scaling and no guards
%                 (stop.norm = 'relativeL2', stop.rule = 'designChange', every
%                 stop.guards.* off).  At most one of the three Stop* options may
%                 be given.  A RUNTIME override, not a different preset.
%     'Diagnostics' (default false)  per-iteration recorder.  Purely additive
%                 and proved bitwise inert, but it costs measurable time per
%                 outer iteration, so benchmarks leave it off.
%     'Name'      (default sprintf('OLHOFF_CURRENT_%dx%d', nelx, nely))
%
%   Requires the production path to be installed (olhoffcurrent_paths), because
%   olh.config.* lives inside +impl/ and is unreachable otherwise.
%
%   See also OLHOFFCURRENT_PRESET, OLHOFFCURRENT_RUN, OLH.CONFIG.DESCRIBE.

p = inputParser();
p.addRequired('nelx', @(v) isnumeric(v) && isscalar(v) && v > 0 && mod(v,1) == 0);
p.addRequired('nely', @(v) isnumeric(v) && isscalar(v) && v > 0 && mod(v,1) == 0);
p.addParameter('Preset', '', @(v) ischar(v) || isstring(v));
p.addParameter('MaxOuter', [], @(v) isnumeric(v) && isscalar(v) && v >= 1);
p.addParameter('StopToleranceFactor', [], ...
    @(v) isempty(v) || (isnumeric(v) && isscalar(v) && isfinite(v) && v > 0));
p.addParameter('StopMaxChangeTolerance', [], ...
    @(v) isempty(v) || (isnumeric(v) && isscalar(v) && isfinite(v) && v > 0));
p.addParameter('StopRelativeChangeTolerance', [], ...
    @(v) isempty(v) || (isnumeric(v) && isscalar(v) && isfinite(v) && v > 0));
p.addParameter('Diagnostics', false, @(v) islogical(v) && isscalar(v));
p.addParameter('Name', '', @(v) ischar(v) || isstring(v));
p.parse(nelx, nely, varargin{:});
opt = p.Results;

nelx = double(opt.nelx);
nely = double(opt.nely);

% The simply-supported idealization pins the supports at MID HEIGHT, which has
% no node to sit on when nely is odd.  Refused here rather than silently moved.
assert(mod(nely, 2) == 0, 'olhoffcurrent_config:OddNely', ...
    ['The simply-supported idealization pins the supports at MID HEIGHT, ' ...
     'which requires an even nely (got %d).'], nely);

presetName = char(string(opt.Preset));
if isempty(presetName)
    error('olhoffcurrent_config:PresetRequired', ...
        ['olhoffcurrent_config requires ''Preset'', name. Registered presets: %s. ' ...
         'The current production choice is olhoffcurrent_production_preset().name.'], ...
        strjoin({olhoffcurrent_presets().name}, ', '));
end
info = olhoffcurrent_preset(presetName);

name = char(string(opt.Name));
if isempty(name); name = sprintf('OLHOFF_CURRENT_%dx%d', nelx, nely); end

maxOuter = info.runtimeDefaults.maxOuter;
if ~isempty(opt.MaxOuter); maxOuter = double(opt.MaxOuter); end

% Applied after the preset's own overrides, so it wins over them.
stopArgs = {};
if (~isempty(opt.StopToleranceFactor) + ~isempty(opt.StopMaxChangeTolerance) ...
        + ~isempty(opt.StopRelativeChangeTolerance)) > 1
    error('olhoffcurrent_config:StopRuleConflict', ...
        ['Give at most one of ''StopToleranceFactor'', ''StopMaxChangeTolerance'' ' ...
         'and ''StopRelativeChangeTolerance''.']);
end
if ~isempty(opt.StopToleranceFactor)
    stopArgs = {'stop.toleranceRule', 'explicit', ...
                'stop.tolerance', double(opt.StopToleranceFactor)*sqrt(nelx*nely/3200)};
end
if ~isempty(opt.StopMaxChangeTolerance)
    % olhoffSolve tests max|drho| < stop.tolerance.  For a positive double tol,
    % x < tol + eps(tol) holds exactly when x <= tol, which is the Proposed
    % method's test (topopt_freq: stop when max|x - x_old| <= conv_tol).
    tolMax = double(opt.StopMaxChangeTolerance);
    stopArgs = {'stop.rule',                      'designChange', ...
                'stop.norm',                      'max', ...
                'stop.toleranceRule',             'explicit', ...
                'stop.tolerance',                 tolMax + eps(tolMax), ...
                'stop.guards.settledMove',        false, ...
                'stop.guards.boxInactiveFraction', 0, ...
                'stop.guards.ladderExhausted',    false, ...
                'stop.guards.maxDesignChange',    false};
end
if ~isempty(opt.StopRelativeChangeTolerance)
    % olhoffSolve tests ||drho||_2/||rho||_2 < stop.tolerance (strict).
    stopArgs = {'stop.rule',                      'designChange', ...
                'stop.norm',                      'relativeL2', ...
                'stop.toleranceRule',             'explicit', ...
                'stop.tolerance',                 double(opt.StopRelativeChangeTolerance), ...
                'stop.guards.settledMove',        false, ...
                'stop.guards.boxInactiveFraction', 0, ...
                'stop.guards.ladderExhausted',    false, ...
                'stop.guards.maxDesignChange',    false};
end

cfg = olh.config.resolve(info.upstreamPreset, ...
    'domain.mesh.nelx',    nelx, ...
    'domain.mesh.nely',    nely, ...
    info.overrides{:}, ...
    stopArgs{:}, ...
    'runtime.maxOuter',    maxOuter, ...
    'runtime.singleThread', true, ...
    'runtime.diagnostics', logical(opt.Diagnostics), ...
    'runtime.verbose',     false, ...
    'runtime.name',        name);

% OlhoffCurrent identity, stamped on top of the resolved configuration.  Purely
% additive metadata outside the schema: the solver reads scientific fields only,
% and olhoffcurrent_config_hash iterates schema rows, so none of this enters the
% configuration hash.
cfg.provenance.olhoffCurrentPreset  = info.name;
cfg.provenance.requestedPresetName  = info.requestedName;
cfg.provenance.presetResolvedVia    = info.resolvedVia;
cfg.provenance.presetRole           = info.role;
cfg.provenance.upstreamPreset       = info.upstreamPreset;
cfg.provenance.upstreamCommit       = info.upstreamCommit;
cfg.provenance.compatibilityAliases = info.compatibilityAliases;
cfg.provenance.historicalAliases    = info.historicalAliases;
cfg.provenance.implementation       = 'analysis/Olhoff';
end
