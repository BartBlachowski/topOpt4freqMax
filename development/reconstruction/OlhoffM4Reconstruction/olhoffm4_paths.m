function [guard, resolved] = olhoffm4_paths()
%OLHOFFM4_PATHS  Install an ISOLATED, fail-closed path for the imported M4 solver.
%
%   [guard, resolved] = OLHOFFM4_PATHS() prepends the three implementation
%   folders this import owns, proves that every owned function resolves inside
%   the import root and outside every superseded Olhoff implementation, and
%   returns an onCleanup object.  When GUARD is cleared -- when the calling
%   function returns, including via an error -- the previous path is restored
%   exactly.
%
%   Hold the return value:
%
%       guard = olhoffm4_paths();   %#ok<NASGU>
%
%   WHY THIS EXISTS
%   ---------------
%   This repository contains SIX independent realizations of the Du & Olhoff
%   algorithm family (analysis/OLHOFF_IMPLEMENTATION_STATUS.md).  Several share
%   function names -- olhoffOpt, model2D, assemble2D, eigSolve, genGrad,
%   innerLoop, prepFilter, applyFilter, mmasub, subsolv -- and MATLAB resolves
%   by path order.  Executing the wrong one produces a run that looks fine and
%   is scientifically void.  So identity is ASSERTED, never assumed: a run that
%   starts is a run whose implementation has been proved.
%
%   WHY THE CORE LIVES UNDER +frozen/
%   ---------------------------------
%   genpath skips folders whose name begins with '+', and every folder beneath
%   them.  Repository scripts (examples/Revision_v1/*.m) call
%   addpath(genpath(<repo>/analysis)); putting the imported core in a plain
%   subfolder would place THIS copy of mmasub.m, subsolv.m, olhoffOpt.m and the
%   rest ahead of tools/Matlab and would break the existing isolation guarantee
%   that Matlab/reproduction2007/runner/repro2007_verify_isolation.m checks.
%   Under +frozen/ the core is invisible to genpath and reachable only through
%   this function, which is exactly the property wanted.
%
%   See also OLHOFFM4_ROOT, OLHOFFM4_OWNED_NAMES, OLHOFFM4_FORBIDDEN_PATHS.

if nargout < 1
    error('olhoffm4_paths:GuardDiscarded', ...
        ['olhoffm4_paths must be called with an output argument the caller ' ...
         'keeps alive:  guard = olhoffm4_paths();  Discarding the guard ' ...
         'restores the path immediately and leaves the solver unreachable.']);
end

root = olhoffm4_root();
[names, dirs] = olhoffm4_owned_names();

oldPath = path();
guard = onCleanup(@() path(oldPath));

% The three implementation folders, plus the MMA variant the frozen
% realization uses.  +frozen/algo/useMMA.m is the audited variant selector and
% will re-add mma_published itself and assert its own resolution; adding it
% here as well means mmasub is PROVED to be this copy BEFORE the expensive
% solve starts rather than after it.  mma/ (the 'asfound' variant) is
% deliberately left off the path: the frozen realization is 'published'.
addpath(dirs.algo, dirs.fem, dirs.filter, dirs.mma_published);

resolved = olhoffm4_assert_dispatch(names, root);

% Beyond "inside the import", mmasub must be the PUBLISHED Svanberg copy.  The
% two variants differ in their default move and asyinit, so resolving to the
% wrong one changes the nested sub-optimization silently.
mmaFile = which('mmasub');
if ~strncmp(mmaFile, dirs.mma_published, numel(dirs.mma_published))
    error('olhoffm4_paths:WrongMMAVariant', ...
        ['mmasub resolves to %s; the frozen realization requires the ' ...
         'published Svanberg copy under %s.'], mmaFile, dirs.mma_published);
end
end
