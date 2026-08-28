function [treeHash, entries] = repro2007_tree_hash()
%REPRO2007_TREE_HASH  Content hash of the FROZEN numerical source that executes.
%
%   [treeHash, entries] = REPRO2007_TREE_HASH()
%
%   Hashes every .m file under the reproduction's algo/, fem/, filter/ and mma/
%   directories -- the four that contain the numerics OLHOFFOPT runs -- and
%   folds them into one SHA-256 over a canonical `relpath  sha256` listing,
%   sorted by path.
%
%   runner/ is deliberately EXCLUDED.  It is integration code written for this
%   repository (PROVENANCE.md, "Import integrity"), it is not part of the
%   clean-room import, and it is expected to change; folding it in would
%   invalidate the equivalence gate every time a report field was added.  The
%   gate pins runner/ separately, through the source commit.
%
%   ENTRIES is an n x 2 cell array {relpath, sha256} for the audit trail.
%
%   See also SHA256_HEX, REPRO2007_ROOT, VERIFY_REPRO2007_BENCHMARK_EQUIVALENCE.

root = repro2007_root();
subdirs = {'algo', 'fem', 'filter', 'mma'};

entries = cell(0, 2);
for i = 1:numel(subdirs)
    d = fullfile(root, subdirs{i});
    listing = dir(fullfile(d, '*.m'));
    for k = 1:numel(listing)
        rel = [subdirs{i} '/' listing(k).name];
        fid = fopen(fullfile(d, listing(k).name), 'r');
        bytes = fread(fid, Inf, '*uint8');
        fclose(fid);
        entries(end+1, :) = {rel, sha256_hex(bytes)}; %#ok<AGROW>
    end
end

if isempty(entries)
    error('repro2007_tree_hash:NoSource', ...
        'No .m files found under %s -- the implementation is missing.', root);
end

[~, order] = sort(entries(:, 1));
entries = entries(order, :);

lines = cell(size(entries, 1), 1);
for i = 1:numel(lines)
    lines{i} = sprintf('%s  %s', entries{i, 1}, entries{i, 2});
end
treeHash = sha256_hex(strjoin(lines, newline));
end
