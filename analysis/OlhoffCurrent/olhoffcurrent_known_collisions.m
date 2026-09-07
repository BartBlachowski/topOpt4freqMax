function C = olhoffcurrent_known_collisions()
%OLHOFFCURRENT_KNOWN_COLLISIONS  Declared non-Olhoff files that share a name.
%
%   Some files elsewhere in this repository legitimately carry a name that
%   analysis/OlhoffCurrent also owns, without being an Olhoff implementation.
%   They are declared HERE, one entry each, with a reason -- never skipped
%   silently by the dispatch gate.
%
%   A declared collision is NOT a permission to lose.  For every entry the gate
%   still requires that the production copy WINS the resolution; a declared file
%   that becomes the winner is a hard blocker exactly as an undeclared one is.
%   Declaring a file only downgrades "a second candidate exists" from a blocker
%   to a recorded warning.
%
%   THE ONE THAT MATTERS
%   --------------------
%   tools/Matlab/mmasub.m is byte-identical to the 'asfound' MMA variant
%   (sha256 54c1680036e6...), NOT to the 'published' Svanberg copy the
%   production preset requires (4507b73e3e44...).  The two differ in their
%   default move and asyinit, so resolving to it would change the nested
%   sub-optimization silently and produce numbers that look entirely
%   reasonable.  It cannot be removed: performance_comparison.m adds
%   tools/Matlab for the Proposed and Yuksel methods, which need it.
%
%   Three independent mechanisms keep the production copy in front:
%     1. olhoffcurrent_paths prepends +impl/mma_published;
%     2. olhoffcurrent_paths asserts which('mmasub') is under mma_published;
%     3. algo/useMMA re-prepends it at solve time and asserts again.
%   This registry adds a fourth: the resolution is checked, and the declared
%   file's current hash is reported, so a change to shared tooling is visible in
%   the run artifact rather than discovered later.
%
%   Fields: symbol, relativePath (repository-relative), sha256 (expected, or ''
%   to record only), reason.
%
%   See also OLHOFFCURRENT_ASSERT_DISPATCH.

C = struct('symbol', {}, 'relativePath', {}, 'sha256', {}, 'reason', {});

C(end+1) = struct('symbol','mmasub', ...
    'relativePath', fullfile('tools','Matlab','mmasub.m'), ...
    'sha256','54c1680036e6effdb70bc32524b0c11c366cebed64d5a508c82553e871ff7660', ...
    'reason',['shared repository tooling for the Proposed and Yuksel methods. ' ...
              'Byte-identical to the ''asfound'' MMA variant, NOT the ' ...
              '''published'' copy the production preset requires. Must never win.']);

C(end+1) = struct('symbol','subsolv', ...
    'relativePath', fullfile('tools','Matlab','subsolv.m'), ...
    'sha256','130033335ac5a21f31437a196bf8f37c5b97880a00e93cf2962a8a54bac2b98a', ...
    'reason',['shared repository tooling for the Proposed and Yuksel methods. ' ...
              'Byte-identical to BOTH MMA variants'' subsolv, so the collision ' ...
              'is benign today; declared so that a future divergence is not.']);
end
