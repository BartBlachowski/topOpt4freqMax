function [guard, report] = olhoffcurrent_paths()
%OLHOFFCURRENT_PATHS  Install the ISOLATED, fail-closed production Olhoff path.
%
%   [guard, report] = OLHOFFCURRENT_PATHS() prepends the folders this
%   implementation owns, PROVES that exactly one Olhoff implementation is
%   visible to MATLAB, and returns an onCleanup object.  When GUARD is cleared
%   -- when the calling function returns, including via an error -- the previous
%   path is restored exactly.
%
%   Hold the return value:
%
%       guard = olhoffcurrent_paths();   %#ok<NASGU>
%
%   Discarding it restores the path immediately and leaves the solver
%   unreachable, so the function refuses to be called without an output.
%
%   THE INVARIANT
%   -------------
%       exactly one executable Olhoff implementation is visible:
%       analysis/OlhoffCurrent
%
%   Enforced by olhoffcurrent_assert_dispatch, which checks every owned symbol
%   with which(name,'-all') -- so helper shadowing is caught even when the
%   top-level solver resolves correctly.  A run that starts is a run whose
%   implementation has been proved.
%
%   NOTE ON mma/.  Only mma_published/ is added.  The production preset uses the
%   published Svanberg constants, and algo/useMMA reasserts that choice at solve
%   time; adding mma/ as well would place a second mmasub.m on the path for no
%   reason.  Leaving it off means mmasub is proved to be the published copy
%   BEFORE the expensive solve starts rather than after it.
%
%   See also OLHOFFCURRENT_ASSERT_DISPATCH, OLHOFFCURRENT_RUN.

if nargout < 1
    error('olhoffcurrent_paths:GuardDiscarded', ...
        ['olhoffcurrent_paths must be called with an output argument the ' ...
         'caller keeps alive:  guard = olhoffcurrent_paths();  Discarding ' ...
         'the guard restores the path immediately and leaves the solver ' ...
         'unreachable.']);
end

dirs = olhoffcurrent_impl_dirs();

oldPath = path();
guard = onCleanup(@() path(oldPath));

addpath(dirs.algo, dirs.fem, dirs.filter, dirs.architecture, dirs.mma_published);

report = olhoffcurrent_assert_dispatch();

% Beyond "inside production", mmasub must be the PUBLISHED Svanberg copy: the
% two variants differ in their default move and asyinit, so resolving to the
% wrong one changes the nested sub-optimization silently and produces numbers
% that look entirely reasonable.
mmaFile = which('mmasub');
if ~strncmp(mmaFile, [dirs.mma_published filesep], numel(dirs.mma_published)+1)
    error('olhoffcurrent_paths:WrongMMAVariant', ...
        ['mmasub resolves to %s; the production preset requires the ' ...
         'published Svanberg copy under %s.'], mmaFile, dirs.mma_published);
end
end
