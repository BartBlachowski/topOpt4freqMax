function cfg = resolve(preset, varargin)
%RESOLVE  Build an effective configuration.  THE ONLY sanctioned way.
%
%   cfg = OLH.CONFIG.RESOLVE(preset, 'path.to.field', value, ...)
%
%   Order, and there is no other:
%
%       canonical defaults -> preset -> explicit overrides -> derived rules
%           -> validation -> effective cfg (recorded in cfg.provenance)
%
%   `preset` is a name from olh.presets.list, or '' for bare canonical defaults.
%   Passing '' is legitimate but is recorded as such, so that no run can claim a
%   realization it did not name.
%
%   Example
%       cfg = olh.config.resolve('duOlhoffFrozenM4', ...
%                                'domain.mesh.nelx', 320, 'domain.mesh.nely', 40);
%   The mesh override is applied BEFORE the mesh-scaled tolerance is derived, so
%   stop.tolerance becomes 0.1 automatically.

if nargin < 1, preset = ''; end
cfg = olh.config.defaults();

if ~isempty(preset)
    cfg = olh.presets.apply(preset, cfg);
end

% ---- explicit overrides --------------------------------------------------
if mod(numel(varargin),2) ~= 0
    error('olh:config:resolvePairs','resolve expects path/value pairs after the preset name.');
end
known = olh.config.schema();
for k = 1:2:numel(varargin)
    path = varargin{k};
    if ~any(strcmp(path, known(:,1)))
        error('olh:config:unknownOverride', ...
            'Override targets unknown field ''%s''. See olh.config.schema.', path);
    end
    cfg = olh.config.setPath(cfg, path, varargin{k+1});
end
overrides = varargin;

% ---- derived rules, applied AFTER overrides ------------------------------
if strcmp(olh.config.getPath(cfg,'stop.toleranceRule'),'meshScaled')
    cfg = olh.config.setPath(cfg, 'stop.tolerance', ...
        olh.config.epsilonForMesh(olh.config.getPath(cfg,'domain.mesh.nelx'), ...
                                  olh.config.getPath(cfg,'domain.mesh.nely')));
end

% ---- validation ----------------------------------------------------------
[cfg, warnings] = olh.config.validate(cfg);
for k = 1:numel(warnings)
    warning('olh:config:suspicious','%s', warnings{k});
end

% ---- provenance ----------------------------------------------------------
if isempty(preset), presetName = '(none - bare canonical defaults)';
else,               presetName = preset; end
cfg.provenance = struct( ...
    'preset',     presetName, ...
    'overrides',  {overrides}, ...
    'warnings',   {warnings}, ...
    'resolvedAt', char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss')), ...
    'schemaRows', size(known,1));
end
