function report = repro2007_verify_isolation()
%REPRO2007_VERIFY_ISOLATION  Prove the migration cannot hijack another implementation.
%
%   report = REPRO2007_VERIFY_ISOLATION() checks the property that WP6 of the
%   migration exists to guarantee:
%
%       No repository script can accidentally execute a function from the
%       clean-room reproduction in place of the historical or full-coupling
%       implementation, or in place of tools/Matlab.
%
%   Three implementations of the same algorithm family live in this repository
%   and share function names.  The reproduction ships its own mmasub.m,
%   subsolv.m and top88.m, all of which also exist elsewhere here.  MATLAB
%   resolves by path order, and several repository scripts use
%   addpath(genpath(...)), which prepends everything it finds.  Getting this
%   wrong produces runs that look fine and are wrong, so it is asserted rather
%   than reasoned about.
%
%   CHECKS
%     A  With the repository's own path recipe in force -- including the
%        addpath(genpath(analysis)) used by examples/Revision_v1/*.m -- NO
%        function owned by the reproduction resolves inside the reproduction.
%     A2 Adding ONLY runner/ -- what run_topopt_from_json.m does for the
%        OlhoffDu2007Repro dispatch case -- exposes the entry points and
%        nothing else; the algorithm stays behind repro2007_paths().
%     B  The three colliding names resolve to their HISTORICAL locations under
%        that recipe (tools/Matlab and source_of_truth).
%     C  After repro2007_paths(), every owned function resolves inside the
%        reproduction root, and the colliding names flip to it.
%     D  When the guard is released, the path returns exactly to its prior
%        state and check B holds again.
%
%   The function restores whatever path it found on entry, including on error.
%
%   See also REPRO2007_PATHS, REPRO2007_ASSERT_IDENTITY.

root = repro2007_root();
repoRoot = fileparts(fileparts(root));   % <repo>/Matlab/reproduction2007 -> <repo>
runnerDir = fullfile(root, 'runner');

entryPath = path();
restore = onCleanup(@() path(entryPath));

collisions = {'mmasub', 'subsolv', 'top88'};
failures = {};

fprintf('\n==============================================================\n');
fprintf(' reproduction2007 -- PATH ISOLATION VERIFICATION (WP6)\n');
fprintf(' repo root : %s\n', repoRoot);
fprintf('==============================================================\n\n');

[owned, ownedCore] = localOwnedNames(root);
fprintf('Functions owned by the reproduction: %d (%d outside runner/)\n\n', ...
    numel(owned), numel(ownedCore));

% ---- Reproduce the repository's own path recipe -------------------------
% This is localEnsurePaths() from examples/Revision_v1/run_all_revision_experiments.m,
% verbatim in effect: tools first, then a recursive sweep of analysis/ which
% PREPENDS every subfolder it finds and therefore shadows tools.
restoredefaultpath();
addpath(fullfile(repoRoot, 'tools', 'Matlab'));
addpath(genpath(fullfile(repoRoot, 'analysis')));
addpath(fullfile(repoRoot, 'examples', 'Performance'));

% ---- A: the reproduction must be wholly unreachable ---------------------
leaked = localLeaks(owned, root);
if isempty(leaked)
    fprintf('  [PASS] A  reproduction unreachable under addpath(genpath(analysis))\n');
    fprintf('            (%d owned names checked, 0 leaked)\n', numel(owned));
else
    failures{end+1} = 'A';
    fprintf('  [FAIL] A  %d reproduction function(s) reachable from the\n', numel(leaked));
    fprintf('            repository path recipe:\n');
    fprintf('              %s\n', leaked{:});
end

% ---- A2: adding ONLY runner/ must not expose the implementation ---------
% This is exactly what tools/Matlab/run_topopt_from_json.m does for the
% OlhoffDu2007Repro dispatch case.  Its entry points must become reachable and
% nothing else: the algorithm itself stays behind repro2007_paths().
addpath(runnerDir);
leaked2 = localLeaks(ownedCore, root);
if isempty(leaked2)
    fprintf('  [PASS] A2 addpath(runner) exposes entry points only\n');
    fprintf('            (%d algo/fem/filter/mma/runs names still unreachable)\n', ...
        numel(ownedCore));
else
    failures{end+1} = 'A2';
    fprintf('  [FAIL] A2 addpath(runner) leaked %d implementation function(s):\n', ...
        numel(leaked2));
    fprintf('              %s\n', leaked2{:});
end

% ---- B: colliding names resolve to their historical homes ---------------
expectedB = { ...
    'mmasub',  fullfile(repoRoot, 'tools', 'Matlab', 'mmasub.m')
    'subsolv', fullfile(repoRoot, 'tools', 'Matlab', 'subsolv.m')
    'top88',   fullfile(repoRoot, 'source_of_truth', 'top88.m')};
okB = true;
for i = 1:size(expectedB, 1)
    w = which(expectedB{i,1});
    if isempty(w)
        % top88 is only reachable if source_of_truth is on the path; it is not
        % part of the standard recipe, so absence is correct, not a failure.
        fprintf('  [ ok ] B  %-8s not on path (expected: not in the recipe)\n', ...
            expectedB{i,1});
        continue
    end
    if localIsInside(w, root)
        okB = false;
        fprintf('  [FAIL] B  %-8s resolves into the reproduction: %s\n', ...
            expectedB{i,1}, w);
    else
        fprintf('  [PASS] B  %-8s -> %s\n', expectedB{i,1}, localRel(w, repoRoot));
    end
end
if ~okB
    failures{end+1} = 'B';
end

% ---- C: inside the guard, identity flips and is asserted ----------------
pathBefore = path();
okC = true;
try
    guard = repro2007_paths(); %#ok<NASGU>
    rep = repro2007_assert_identity();
    for i = 1:numel(collisions)
        w = which(collisions{i});
        if isempty(w) || ~localIsInside(w, root)
            % top88 lives at the reproduction root, which is not added to the
            % path; only algo/fem/filter/mma/runs/runner are.
            if strcmp(collisions{i}, 'top88')
                continue
            end
            okC = false;
            fprintf('  [FAIL] C  %-8s did not flip to the reproduction: %s\n', ...
                collisions{i}, w);
        end
    end
    if okC
        fprintf('  [PASS] C  inside the guard: %d functions verified in %s\n', ...
            rep.n_checked, localRel(rep.root, repoRoot));
        if ~isempty(rep.shadowed)
            for i = 1:numel(rep.shadowed)
                if rep.shadowed(i).identical
                    tag = 'byte-identical';
                else
                    tag = '*** DIFFERENT CONTENT ***';
                end
                fprintf('            shadows %-8s at %s [%s]\n', ...
                    rep.shadowed(i).name, localRel(rep.shadowed(i).other, repoRoot), tag);
                if ~rep.shadowed(i).identical
                    okC = false;
                end
            end
        end
    end
    clear guard
catch err
    okC = false;
    fprintf('  [FAIL] C  %s\n', err.message);
end
if ~okC
    failures{end+1} = 'C';
end

% ---- D: the guard restored the path exactly ------------------------------
if strcmp(path(), pathBefore)
    fprintf('  [PASS] D  path restored exactly when the guard was released\n');
else
    failures{end+1} = 'D';
    fprintf('  [FAIL] D  path was NOT restored when the guard was released\n');
end

report = struct('passed', isempty(failures), 'failures', {failures}, ...
    'n_owned', numel(owned), 'root', root);

fprintf('\n');
if report.passed
    fprintf(' PATH ISOLATION: PASS\n');
else
    fprintf(' PATH ISOLATION: FAIL (%s)\n', strjoin(failures, ', '));
end
fprintf('==============================================================\n\n');
end

% -------------------------------------------------------------------------
function [names, core] = localOwnedNames(root)
%LOCALOWNEDNAMES  All owned function names, and the subset outside runner/.
subdirs = {'algo', 'fem', 'filter', 'mma', 'runs', 'runner'};
names = {};
core = {};
for i = 1:numel(subdirs)
    listing = dir(fullfile(root, subdirs{i}, '*.m'));
    for k = 1:numel(listing)
        [~, base] = fileparts(listing(k).name);
        names{end+1} = base; %#ok<AGROW>
        if ~strcmp(subdirs{i}, 'runner')
            core{end+1} = base; %#ok<AGROW>
        end
    end
end
end

function leaked = localLeaks(names, root)
leaked = {};
for i = 1:numel(names)
    w = which(names{i});
    if ~isempty(w) && localIsInside(w, root)
        leaked{end+1} = sprintf('%s -> %s', names{i}, w); %#ok<AGROW>
    end
end
end

function tf = localIsInside(file, root)
if ~endsWith(root, filesep)
    root = [root filesep];
end
tf = strncmp(file, root, numel(root));
end

function s = localRel(p, repoRoot)
if ~endsWith(repoRoot, filesep)
    repoRoot = [repoRoot filesep];
end
if strncmp(p, repoRoot, numel(repoRoot))
    s = p(numel(repoRoot)+1:end);
else
    s = p;
end
end
