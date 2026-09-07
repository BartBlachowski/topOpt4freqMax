function removed = olhoffcurrent_scrub_forbidden_paths(repoRoot)
%OLHOFFCURRENT_SCRUB_FORBIDDEN_PATHS  Take every non-production Olhoff tree off
%   the MATLAB path.
%
%   removed = OLHOFFCURRENT_SCRUB_FORBIDDEN_PATHS(repoRoot) removes every path
%   entry lying inside a directory named by OLHOFFCURRENT_FORBIDDEN_PATHS --
%   repository-relative trees and the external development repository alike --
%   and returns the sorted list of what it removed.
%
%   WHY A PRODUCTION SCRIPT SHOULD CALL THIS
%   ----------------------------------------
%   Declining to add a competing implementation is not enough, because the
%   MATLAB path is SESSION state.  Six scripts under examples/Revision_v1/ call
%   addpath(genpath(<repo>/analysis)), which leaves analysis/Olhoff* and the
%   audit trees on the path for the rest of the session.  A driver that merely
%   does not add them still INHERITS them, and olhoffOpt then resolves to
%   whichever realization came first -- a run that looks fine and is
%   scientifically void.
%
%   So a production driver removes what the session handed it, and then lets
%   olhoffcurrent_paths re-check the result independently.  Scrubbing and
%   checking use the same list, so they cannot disagree about what is
%   forbidden; and because the check is a separate step, a scrub that missed
%   something still fails closed rather than passing quietly.
%
%   This never touches analysis/OlhoffCurrent itself, which is not on the
%   forbidden list and whose core under +impl/ is invisible to genpath anyway.
%
%   See also OLHOFFCURRENT_FORBIDDEN_PATHS, OLHOFFCURRENT_PATHS.

if nargin < 1 || isempty(repoRoot)
    repoRoot = fileparts(fileparts(olhoffcurrent_root()));
end

% Matches are collected across every forbidden root BEFORE anything is removed,
% then removed in one call.  The roots overlap by prefix (analysis/OlhoffApproach
% also matches analysis/OlhoffApproachExact/...), so removing per root would try
% to remove entries a previous root already took and fill the console with
% "not found in path" warnings.
[repoRel, absolute] = olhoffcurrent_forbidden_paths();
roots = absolute(:).';
for k = 1:numel(repoRel)
    roots{end+1} = fullfile(repoRoot, repoRel{k}); %#ok<AGROW>
end

onPath = strsplit(path, pathsep);
hit = false(size(onPath));
for i = 1:numel(roots)
    hit = hit | strncmp(onPath, roots{i}, numel(roots{i}));
end

removed = unique(onPath(hit));
if ~isempty(removed)
    rmpath(removed{:});
end
end
