function [names, dirs] = olhoffm4_owned_names()
%OLHOFFM4_OWNED_NAMES  Every function this import owns, and the folders holding them.
%
%   The dispatch gate asserts that each of these resolves INSIDE the import
%   root before any scientific execution.  Adding a file to +frozen/ without
%   adding it here would leave it unchecked, so the list is derived from the
%   directory rather than restated.
%
%   See also OLHOFFM4_PATHS.

root = olhoffm4_root();
core = fullfile(root, '+frozen');
dirs = struct( ...
    'algo',          fullfile(core, 'algo'), ...
    'fem',           fullfile(core, 'fem'), ...
    'filter',        fullfile(core, 'filter'), ...
    'mma_published', fullfile(core, 'mma_published'), ...
    'mma',           fullfile(core, 'mma'));

names = {};
fn = fieldnames(dirs);
for i = 1:numel(fn)
    d = dirs.(fn{i});
    if exist(d, 'dir') ~= 7
        error('olhoffm4_owned_names:MissingDirectory', ...
            'The import is incomplete: %s is missing.', d);
    end
    listing = dir(fullfile(d, '*.m'));
    for k = 1:numel(listing)
        [~, base] = fileparts(listing(k).name);
        names{end+1} = base; %#ok<AGROW>
    end
end
names = unique(names);
end
