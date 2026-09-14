function info = olhoffcurrent_preset(name)
%OLHOFFCURRENT_PRESET  One named OlhoffCurrent preset, looked up by NAME.
%
%   info = OLHOFFCURRENT_PRESET(name) returns the registry entry (see
%   OLHOFFCURRENT_PRESETS) for a canonical preset name or one of its declared
%   compatibility aliases, plus
%
%     info.requestedName   the name that was passed
%     info.resolvedVia     'canonical' | 'compatibilityAlias'
%
%   A NAME IS REQUIRED.  There is no unnamed default: an unnamed lookup would
%   silently change formulation whenever the production preset changes.  The
%   production choice is read explicitly with OLHOFFCURRENT_PRODUCTION_PRESET.
%
%   Historical audit codes and run labels (M4, TMA, S160x20, ...) are provenance
%   aliases, not API, and are refused with a pointer to the preset they belong to.
%
%   Canonical presets:
%     duOlhoffSimpEq4bBetaStallLadderSensitivityFiltered   (compatibility alias
%                                          duOlhoffFixedPenaltySensitivityFiltered)
%     duOlhoffSimpEq4bThreeRungStageExhaustionSensitivityFiltered
%     duOlhoffPedersenAdaptiveBoxSensitivityFiltered
%
%   Full formulation of a resolved configuration, in scientific terms:
%       olh.config.describe(olhoffcurrent_config(nelx, nely, 'Preset', name))
%
%   See also OLHOFFCURRENT_PRESETS, OLHOFFCURRENT_CONFIG, OLHOFFCURRENT_PRODUCTION_PRESET.

R = olhoffcurrent_presets();
canonical = {R.name};

if nargin < 1 || isempty(name)
    error('olhoffcurrent_preset:NameRequired', ...
        ['An OlhoffCurrent preset must be named explicitly; there is no unnamed ' ...
         'default. Registered presets: %s. The current production choice is ' ...
         'olhoffcurrent_production_preset().name.'], strjoin(canonical, ', '));
end
name = char(string(name));

i = find(strcmp(name, canonical), 1);
via = 'canonical';
if isempty(i)
    for k = 1:numel(R)
        if any(strcmp(name, R(k).compatibilityAliases)), i = k; via = 'compatibilityAlias'; break; end
    end
end
if isempty(i)
    owner = '';
    for k = 1:numel(R)
        if any(strcmp(name, R(k).historicalAliases)), owner = R(k).name; break; end
    end
    if ~isempty(owner)
        error('olhoffcurrent_preset:ProvenanceAlias', ...
            ['"%s" is a historical PROVENANCE alias of %s, not a preset name. ' ...
             'Name the canonical preset.'], name, owner);
    end
    error('olhoffcurrent_preset:Unknown', ...
        'Unknown OlhoffCurrent preset "%s". Registered presets: %s.', name, strjoin(canonical, ', '));
end

info = R(i);
info.requestedName = name;
info.resolvedVia   = via;
end
