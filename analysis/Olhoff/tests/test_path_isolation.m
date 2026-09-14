function nFail = test_path_isolation()
%TEST_PATH_ISOLATION  Negative tests for the production path gate.
%
%   The gate is the thing standing between a scientifically void run and a
%   published number, so it is tested DELIBERATELY -- including the cases it is
%   supposed to refuse.  A gate that has only ever been observed to pass has
%   not been tested.
%
%   TEST A   clean path + OlhoffCurrent                            -> PASS
%   TEST B   OlhoffCurrent + analysis/OlhoffM4Reconstruction       -> BLOCK
%   TEST C   OlhoffCurrent + external /Users/.../Matlab/Olhoff     -> BLOCK
%   TEST D   OlhoffCurrent + Matlab/reproduction2007               -> BLOCK
%   TEST E   helper shadowing only, top-level solver still correct -> BLOCK
%   TEST F   a DECLARED benign collision that WINS the resolution   -> BLOCK
%
%   In every BLOCK case the assertion is not merely "an error was raised" but
%   "the error was raised by the gate, and NO OPTIMIZATION STARTED".
%
%   Run:  nFail = test_path_isolation()

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root);
repo = fileparts(fileparts(root));
ext  = '/Users/piotrek/Programming/Matlab/Olhoff';

nFail = 0;
fprintf('\n%s\nTEST_PATH_ISOLATION\n%s\n', repmat('=',1,72), repmat('=',1,72));

% ---------------------------------------------------------------- TEST A
nFail = nFail + check('A  clean path + OlhoffCurrent -> PASS', @() local_A(), true);

% ---------------------------------------------------------------- TEST B
nFail = nFail + check('B  + analysis/OlhoffM4Reconstruction -> BLOCK', ...
    @() local_contaminate({fullfile(repo,'analysis','OlhoffM4Reconstruction')}), false);

% ---------------------------------------------------------------- TEST C
if exist(ext,'dir') == 7
    nFail = nFail + check('C  + external /Matlab/Olhoff -> BLOCK', ...
        @() local_contaminate({fullfile(ext,'algo'), fullfile(ext,'fem')}), false);
else
    fprintf('  [SKIP] C  external development repository not present on this machine\n');
end

% ---------------------------------------------------------------- TEST D
nFail = nFail + check('D  + Matlab/reproduction2007 -> BLOCK', ...
    @() local_contaminate({fullfile(repo,'Matlab','reproduction2007','algo'), ...
                           fullfile(repo,'Matlab','reproduction2007','fem')}), false);

% ---------------------------------------------------------------- TEST E
nFail = nFail + check('E  helper shadowing only, olhoffSolve still ours -> BLOCK', ...
    @() local_E(repo), false);

% ---------------------------------------------------------------- TEST F
nFail = nFail + check('F  a declared collision that WINS -> BLOCK', ...
    @() local_F(repo), false);

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

% =========================================================================
function ok = local_A()
%LOCAL_A  The gate must pass and every owned symbol must resolve into +impl.
old = path(); c = onCleanup(@() path(old));
restoredefaultpath(); addpath(fileparts(fileparts(mfilename('fullpath'))));
[guard, rep] = olhoffcurrent_paths(); %#ok<ASGLU>
core = fullfile(olhoffcurrent_root(), '+impl');
ok = rep.ok && ~isempty(rep.resolved);
for k = 1:numel(rep.resolved)
    if ~strncmp(rep.resolved(k).file, [core filesep], numel(core)+1)
        ok = false;
    end
end
end

function ok = local_contaminate(dirs)
%LOCAL_CONTAMINATE  Put a competing Olhoff tree on the path, then demand refusal.
old = path(); c = onCleanup(@() path(old));
for k = 1:numel(dirs)
    if exist(dirs{k},'dir') == 7, addpath(dirs{k}); end
end
ok = local_expectBlock();
end

function ok = local_E(repo)
%LOCAL_E  The hard case: the TOP-LEVEL solver still resolves to OlhoffCurrent,
%   but a HELPER is shadowed by a competing tree.  A gate that checks only the
%   entry point passes this; a gate that checks helper resolution must not.
old = path(); c = onCleanup(@() path(old));
dirs = {fullfile(repo,'Matlab','reproduction2007','algo'), ...
        fullfile(repo,'Matlab','reproduction2007','fem')};
for k = 1:numel(dirs)
    if exist(dirs{k},'dir') == 7, addpath(dirs{k}, '-begin'); end
end
% Install ours normally, then verify the premise of the test really holds:
% olhoffSolve is OURS (it exists nowhere else) while innerLoop is NOT.
try
    [guard, rep] = olhoffcurrent_paths(); %#ok<ASGLU>
    fprintf('      (gate returned ok=%d -- expected a refusal)\n', rep.ok);
    ok = false;
    return
catch ME
    if ~strcmp(ME.identifier, 'olhoffcurrent_assert_dispatch:PathContaminated')
        fprintf('      (wrong error id: %s)\n', ME.identifier);
        ok = false; return
    end
end
% Premise check, with the contamination re-established the same way.
path(old);
for k = 1:numel(dirs)
    if exist(dirs{k},'dir') == 7, addpath(dirs{k}, '-begin'); end
end
d = olhoffcurrent_impl_dirs();
addpath(d.algo, d.fem, d.filter, d.architecture, d.mma_published);
% Now put the competing tree back IN FRONT so a helper really is shadowed.
for k = 1:numel(dirs)
    if exist(dirs{k},'dir') == 7, addpath(dirs{k}, '-begin'); end
end
topOk    = strncmp(which('olhoffSolve'), olhoffcurrent_root(), numel(olhoffcurrent_root()));
helperNo = ~strncmp(which('innerLoop'),  olhoffcurrent_root(), numel(olhoffcurrent_root()));
rep2 = olhoffcurrent_assert_dispatch('Throw', false);
ok = topOk && helperNo && ~rep2.ok;
fprintf('      premise: olhoffSolve is ours = %d, innerLoop shadowed = %d, gate refuses = %d\n', ...
    topOk, helperNo, ~rep2.ok);
end

function ok = local_F(repo)
%LOCAL_F  Declaring a collision permits it to EXIST, never to WIN.
%   tools/Matlab/mmasub.m is byte-identical to the 'asfound' MMA variant, not
%   the 'published' copy the production preset requires.  It is declared in
%   olhoffcurrent_known_collisions so its mere presence is a warning rather
%   than a blocker -- but if it ever gets in FRONT of the production copy, that
%   is the silent-wrong-variant failure the registry exists to prevent.
old = path(); c = onCleanup(@() path(old));
tools = fullfile(repo, 'tools', 'Matlab');
if exist(fullfile(tools,'mmasub.m'), 'file') ~= 2
    fprintf('      (skipped: tools/Matlab/mmasub.m absent)\n');
    ok = true; return
end
d = olhoffcurrent_impl_dirs();
addpath(d.algo, d.fem, d.filter, d.architecture, d.mma_published);
addpath(tools, '-begin');          % put the declared copy in FRONT

won = strncmp(which('mmasub'), tools, numel(tools));
rep = olhoffcurrent_assert_dispatch('Throw', false);
ok = won && ~rep.ok;
fprintf('      premise: declared copy won = %d, gate refuses = %d\n', won, ~rep.ok);
end

function ok = local_expectBlock()
%LOCAL_EXPECTBLOCK  The gate must throw, and it must throw for the right reason.
try
    guard = olhoffcurrent_paths(); %#ok<NASGU>
    ok = false;                      % returned -> it did NOT block
catch ME
    ok = strcmp(ME.identifier, 'olhoffcurrent_assert_dispatch:PathContaminated') || ...
         strcmp(ME.identifier, 'olhoffcurrent_paths:WrongMMAVariant');
    if ~ok, fprintf('      (unexpected error id: %s)\n', ME.identifier); end
end
end

function n = check(label, fn, ~)
try
    ok = fn();
catch ME
    fprintf('  [FAIL] %-58s (threw %s)\n', label, ME.identifier);
    n = 1; return
end
if ok, fprintf('  [PASS] %s\n', label); n = 0;
else,  fprintf('  [FAIL] %s\n', label); n = 1;
end
end
