function [info, event] = olhoffcurrent_production_preset()
%OLHOFFCURRENT_PRODUCTION_PRESET  The preset currently selected for production.
%
%   [info, event] = OLHOFFCURRENT_PRODUCTION_PRESET() reads the LAST entry of
%   PROVENANCE.json production_preset_events and returns the registry entry of
%   the preset it names, plus the event itself.
%
%   The production choice is a recorded provenance event, not a constant in code:
%   changing it means appending an event (date, old and new preset, upstream
%   commits, configuration hashes, rationale), never editing an earlier one.
%   Selecting production does not make any other preset unavailable; every
%   registered preset stays resolvable by name.
%
%   See also OLHOFFCURRENT_PRESETS, OLHOFFCURRENT_PROVENANCE.

prov = jsondecode(fileread(fullfile(olhoffcurrent_root(), 'PROVENANCE.json')));
assert(isfield(prov, 'production_preset_events') && ~isempty(prov.production_preset_events), ...
    'olhoffcurrent_production_preset:NoEvent', ...
    'PROVENANCE.json records no production_preset_events.');
ev = prov.production_preset_events;
if iscell(ev); event = ev{end}; else; event = ev(end); end
info = olhoffcurrent_preset(event.new_preset);
assert(strcmp(info.resolvedVia, 'canonical'), 'olhoffcurrent_production_preset:NotCanonical', ...
    'The production event must name a canonical preset, not an alias (%s).', event.new_preset);
assert(info.productionEligible, 'olhoffcurrent_production_preset:NotEligible', ...
    'Preset %s is not production-eligible.', info.name);
end
