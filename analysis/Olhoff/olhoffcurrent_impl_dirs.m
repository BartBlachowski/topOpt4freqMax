function dirs = olhoffcurrent_impl_dirs()
%OLHOFFCURRENT_IMPL_DIRS  The executable folders of the promoted implementation.
%
%   The core lives under +impl/ for the same reason the frozen reconstruction's
%   lives under +frozen/: GENPATH SKIPS FOLDERS WHOSE NAME BEGINS WITH '+', AND
%   EVERY FOLDER BENEATH THEM.  Six scripts under examples/Revision_v1/ call
%   addpath(genpath(<repo>/analysis)).  A plain subfolder would put this copy of
%   olhoffOpt.m, innerLoop.m, mmasub.m and 60 more onto the path for the rest of
%   the MATLAB session, where they would compete with Matlab/reproduction2007
%   (49 shared bare names -- see analysis/OLHOFF_SOURCE_LINEAGE_AUDIT.md).
%
%   Under +impl/ the core is invisible to genpath and reachable ONLY through
%   olhoffcurrent_paths(), which asserts its identity before anything runs.
%
%   The sibling layout of algo/ fem/ filter/ mma/ mma_published/ architecture/
%   is LOAD-BEARING and is preserved exactly as upstream: algo/useMMA.m locates
%   the MMA variants as fileparts(fileparts(mfilename('fullpath')))/mma*, so
%   moving algo/ relative to mma_published/ would break variant selection.
%
%   See also OLHOFFCURRENT_PATHS, OLHOFFCURRENT_OWNED_NAMES.

core = fullfile(olhoffcurrent_root(), '+impl');
dirs = struct( ...
    'algo',          fullfile(core, 'algo'), ...
    'fem',           fullfile(core, 'fem'), ...
    'filter',        fullfile(core, 'filter'), ...
    'architecture',  fullfile(core, 'architecture'), ...
    'mma_published', fullfile(core, 'mma_published'), ...
    'mma',           fullfile(core, 'mma'));

fn = fieldnames(dirs);
for i = 1:numel(fn)
    if exist(dirs.(fn{i}), 'dir') ~= 7
        error('olhoffcurrent_impl_dirs:MissingDirectory', ...
            'The promoted implementation is incomplete: %s is missing.', dirs.(fn{i}));
    end
end
end
