function [h, entries] = olhoff_benchmark_path_hash()
%OLHOFF_BENCHMARK_PATH_HASH  Content hash of the code that defines the Olhoff
%   benchmark execution path and the proof made about it.
%
%   [h, entries] = OLHOFF_BENCHMARK_PATH_HASH()
%
%   Hashes, in a fixed sorted order, every file that can change either what the
%   benchmark-dispatched Olhoff path does or what the equivalence proof means:
%   the dispatcher, the reproduction's runner/, the benchmark profile and its
%   task JSON, the independent configuration builder, the normalizer, and the
%   harness itself.
%
%   WHY NOT JUST THE COMMIT
%   -----------------------
%   The obvious binding for a preflight proof is the repository HEAD.  It does
%   not work here, for a structural reason: recording the proof adds files to
%   the repository, so committing the evidence moves HEAD and would invalidate
%   the very proof just committed.  Binding to the *code* instead is both
%   stable under that and strictly tighter -- a commit that edits an unrelated
%   file no longer invalidates the proof, and an uncommitted edit to the
%   dispatcher does.
%
%   The frozen clean-room implementation is NOT folded in here; it has its own
%   hash (REPRO2007_TREE_HASH) so that the two can be cited apart.  The source
%   commit is still recorded in every artifact as provenance.
%
%   ENTRIES is an n x 2 cell array {relpath, sha256}.
%
%   See also OLHOFF_EQUIVALENCE_GATE, REPRO2007_TREE_HASH, SHA256_HEX.

here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));

explicit = { ...
    'tools/Matlab/run_topopt_from_json.m'
    'examples/Performance/performance_comparison.json'
    'examples/Performance/performance_benchmark_profile.m'
    'examples/Performance/repro2007_direct_cfg.m'
    'examples/Performance/repro2007_normalized_config.m'
    'examples/Performance/repro2007_tree_hash.m'
    'examples/Performance/sha256_hex.m'
    'examples/Performance/verify_repro2007_benchmark_equivalence.m'};

rel = explicit;
runnerDir = fullfile('Matlab', 'reproduction2007', 'runner');
listing = dir(fullfile(repoRoot, runnerDir, '*.m'));
for i = 1:numel(listing)
    rel{end+1} = strrep(fullfile(runnerDir, listing(i).name), filesep, '/'); %#ok<AGROW>
end

rel = sort(rel);
entries = cell(numel(rel), 2);
for i = 1:numel(rel)
    f = fullfile(repoRoot, strrep(rel{i}, '/', filesep));
    if exist(f, 'file') ~= 2
        error('olhoff_benchmark_path_hash:MissingFile', ...
            ['Cannot hash the benchmark path: %s is missing.  The gate ' ...
             'refuses to certify a path it cannot read in full.'], rel{i});
    end
    fid = fopen(f, 'r');
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
    entries(i, :) = {rel{i}, sha256_hex(bytes)};
end

lines = cell(size(entries, 1), 1);
for i = 1:numel(lines)
    lines{i} = sprintf('%s  %s', entries{i, 1}, entries{i, 2});
end
h = sha256_hex(strjoin(lines, newline));
end
