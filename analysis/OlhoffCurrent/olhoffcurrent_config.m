function [cfg, info] = olhoffcurrent_config(nelx, nely, varargin)
%OLHOFFCURRENT_CONFIG  Effective production configuration for one mesh.
%
%   [cfg, info] = OLHOFFCURRENT_CONFIG(nelx, nely) returns the CANONICAL
%   configuration the production preset resolves to at that mesh, built through
%   the only sanctioned route:
%
%       olh.config.defaults -> preset -> overrides -> derived rules -> validate
%
%   The mesh is applied as an OVERRIDE, before the derived rules run, so the
%   mesh-scaled outer tolerance eps = 0.05*sqrt(NE/3200) is re-derived for this
%   mesh automatically rather than retyped.
%
%   Name/value options:
%     'MaxOuter'  (default 400)  outer iteration cap.  A run that reaches it is
%                 CAP_HIT and is NOT converged.  Raising or lowering it is a
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
p.addParameter('MaxOuter', 400, @(v) isnumeric(v) && isscalar(v) && v >= 1);
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

name = char(string(opt.Name));
if isempty(name); name = sprintf('OLHOFF_CURRENT_%dx%d', nelx, nely); end

info = olhoffcurrent_preset();

cfg = olh.config.resolve(info.upstreamPreset, ...
    'domain.mesh.nelx',    nelx, ...
    'domain.mesh.nely',    nely, ...
    'runtime.maxOuter',    double(opt.MaxOuter), ...
    'runtime.singleThread', true, ...
    'runtime.diagnostics', logical(opt.Diagnostics), ...
    'runtime.verbose',     false, ...
    'runtime.name',        name);

% Production identity, stamped on top of the resolved configuration.  Purely
% additive metadata: the solver reads scientific fields only, never these.
% The upstream preset name is kept as a PROVENANCE ALIAS, not as the API name.
cfg.provenance.productionPreset  = info.name;
cfg.provenance.upstreamPreset    = info.upstreamPreset;
cfg.provenance.historicalAliases = info.historicalAliases;
cfg.provenance.implementation    = 'analysis/OlhoffCurrent';
end
