function mig_use_snapshot(root, withAnchors)
%MIG_USE_SNAPSHOT  Put the read-only 253069 snapshot (and nothing else Olhoff) on the path.
%   Mirrors the snapshot's own setpaths.m (fem, filter, algo, mma, architecture)
%   after restoredefaultpath and proves every solver symbol resolves inside it.
%   withAnchors (default false) also adds architecture/anchors/code, used only to
%   compute anchor records/digests from already-saved results.
if nargin < 2, withAnchors = false; end
here = fileparts(mfilename('fullpath'));
restoredefaultpath;
addpath(here);
addpath(fullfile(root,'fem'), fullfile(root,'filter'), fullfile(root,'algo'), fullfile(root,'mma'));
addpath(fullfile(root,'architecture'));
if withAnchors, addpath(fullfile(root,'architecture','anchors','code')); end
names = {'olhoffSolve','olhoffOpt','innerLoop','mmasub','subsolv','genGrad','assemble2D', ...
         'eigSolve','applyFilter','prepFilter','deltaLambda','massScale','model2D','useMMA'};
for i = 1:numel(names)
    w = which(names{i}, '-all'); if ischar(w), w = {w}; end
    assert(~isempty(w) && strncmp(w{1}, root, numel(root)), 'mig:path', '%s -> %s', names{i}, strjoin(w, ' | '));
    other = w(~strncmp(w, root, numel(root)));
    other = other(~contains(other, [filesep '@']) & ~contains(other, [filesep '+']));
    assert(isempty(other), 'mig:path', '%s has a foreign candidate %s', names{i}, strjoin(other, ' | '));
end
assert(strncmp(which('olh.config.schema'), root, numel(root)), 'mig:path', 'olh package not from snapshot');
entries = strsplit(path, pathsep);
foreign = entries(startsWith(entries, '/Users/piotrek/Programming/topOpt4freqMax') & ~strcmp(entries, here));
assert(isempty(foreign), 'mig:path', 'target-repository directories on the snapshot path: %s', strjoin(foreign, ' | '));
end
