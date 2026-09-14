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

cfg = olh.config.resolve(info.upstreamPreset, ...
    'domain.mesh.nelx',    nelx, ...
    'domain.mesh.nely',    nely, ...
    info.overrides{:}, ...
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
cfg.provenance.implementation       = 'analysis/OlhoffCurrent';
end
