function removed = olhoffm4_scrub_forbidden_paths(repoRoot)
%OLHOFFM4_SCRUB_FORBIDDEN_PATHS  Take every superseded Olhoff tree off the path.
%
%   removed = OLHOFFM4_SCRUB_FORBIDDEN_PATHS(repoRoot) removes from the MATLAB
%   path every entry lying inside a directory named by OLHOFFM4_FORBIDDEN_PATHS,
%   and returns the sorted list of what it removed.
%
%   WHY A SCRIPT SHOULD CALL THIS
%   -----------------------------
%   Not adding a superseded implementation is not enough, because the MATLAB
%   path is SESSION state.  Other scripts in this repository -- every
%   examples/Revision_v1/*.m, and repro2007_verify_isolation.m -- call
%   addpath(genpath(<repo>/analysis)), which leaves analysis/Olhoff* and
%   Matlab/reproduction2007 on the path for the rest of the session.  A driver
%   that merely declines to add them still inherits them, olhoffOpt then
%   resolves to whichever of the seven realizations came first, and the result
%   is a run that looks fine and is scientifically void.
%
%   A driver that curates its own path should REMOVE what the session handed it
%   and then let its preflight re-check the result independently.  The prefix
%   match here is the same one confbench_preflight uses to verify it, so the
%   scrub and the check cannot disagree about what counts as forbidden.
%
%   This never touches analysis/OlhoffM4Reconstruction: the conference-active
%   import is not on the forbidden list, and its solver core under +frozen/ is
%   invisible to genpath in any case.
%
%   See also OLHOFFM4_FORBIDDEN_PATHS, OLHOFFM4_PATHS, CONFBENCH_PREFLIGHT.

if nargin < 1 || isempty(repoRoot)
    repoRoot = fileparts(fileparts(olhoffm4_root()));
end

% Matches are collected across every forbidden root BEFORE anything is
% removed, then removed in one call.  The roots overlap by prefix
% (analysis/OlhoffApproach also matches analysis/OlhoffApproachExact/...), so
% removing per root would try to remove entries a previous root already took
% and fill the console with "not found in path" warnings.
forbidden = olhoffm4_forbidden_paths();
onPath = strsplit(path, pathsep);
hit = false(size(onPath));
for i = 1:numel(forbidden)
    p = fullfile(repoRoot, forbidden{i});
    hit = hit | strncmp(onPath, p, numel(p));
end

removed = unique(onPath(hit));
if ~isempty(removed)
    rmpath(removed{:});
end
end
