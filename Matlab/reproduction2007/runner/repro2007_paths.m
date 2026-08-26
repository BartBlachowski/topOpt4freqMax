function guard = repro2007_paths()
%REPRO2007_PATHS  Install an ISOLATED MATLAB path for the clean-room reproduction.
%
%   guard = REPRO2007_PATHS() prepends this implementation's directories to the
%   MATLAB path, verifies that every function this implementation owns resolves
%   inside its own root, and returns an onCleanup object.  When GUARD is
%   cleared -- when the calling function returns, including via an error -- the
%   previous path is restored exactly.
%
%   Hold the return value:
%
%       guard = repro2007_paths();   %#ok<NASGU>   keep it alive for the scope
%
%   Discarding it (calling REPRO2007_PATHS with no output) restores the path
%   immediately and is therefore useless; that case errors rather than
%   silently doing nothing.
%
%   WHY THIS EXISTS
%   ---------------
%   This repository holds three independent realizations of the Du & Olhoff
%   algorithm family (see Matlab/README.md).  They contain functions with
%   identical names, and several repository scripts -- notably
%   examples/Revision_v1/*.m -- call
%
%       addpath(genpath(fullfile(repoRoot,'analysis')))
%
%   which is precisely the arrangement that lets MATLAB execute a function
%   from the wrong implementation.  Two defences are in force:
%
%     1. This implementation lives at Matlab/reproduction2007/, OUTSIDE
%        analysis/, so no genpath(analysis) sweep can ever reach it.
%     2. This function adds only the six directories it owns, never with
%        genpath, and asserts the resulting resolution before returning.
%
%   The assertion FAILS CLOSED: if any owned function resolves outside this
%   root, the path is restored and an error is raised.  A run that starts is a
%   run whose implementation identity has been proved.
%
%   KNOWN BENIGN COLLISIONS
%   -----------------------
%   mmasub.m, subsolv.m and top88.m exist both here and elsewhere in this
%   repository (tools/Matlab/ and source_of_truth/).  At migration time all
%   three pairs were byte-identical (SHA256 verified, see SOURCE_SHA256.txt),
%   so the shadowing is currently harmless.  It is asserted anyway, because
%   "currently harmless" is not a property that survives an edit.
%
%   See also REPRO2007_ROOT, REPRO2007_ASSERT_IDENTITY, RUN_REPRO2007.

if nargout < 1
    error('repro2007_paths:GuardDiscarded', ...
        ['repro2007_paths must be called with an output argument that the ' ...
         'caller keeps alive:  guard = repro2007_paths();  Discarding the ' ...
         'guard restores the path immediately and leaves the implementation ' ...
         'unreachable.']);
end

root = repro2007_root();

subdirs = {'algo', 'fem', 'filter', 'mma', 'runs', 'runner'};
dirs = cell(1, numel(subdirs));
for i = 1:numel(subdirs)
    d = fullfile(root, subdirs{i});
    if exist(d, 'dir') ~= 7
        error('repro2007_paths:MissingDirectory', ...
            'Reproduction implementation is incomplete: missing %s', d);
    end
    dirs{i} = d;
end

oldPath = path();
guard = onCleanup(@() path(oldPath));

% addpath with several arguments prepends them all, in order.
addpath(dirs{:});

% Fail closed.  If this throws, GUARD is destroyed as the frame unwinds and the
% previous path is restored by its cleanup before the error reaches the caller.
repro2007_assert_identity();
end
