function test_acceptance_gates()
%TEST_ACCEPTANCE_GATES  Lightweight tests for the revision acceptance gates.
%
%   test_acceptance_gates()
%
%   Pure struct fixtures.  No solver, no MATLAB optimization run, no file I/O
%   beyond a temporary directory.  Runs in well under one second.
%
%   Covers CHECK_REVISION_RUN, which is the single implementation of the
%   declared revision acceptance rule used by the active stage gates
%   (localAccept_S1, localAccept_Exp2Authoritative, localAccept_Exp2b,
%   localAccept_Exp3Authoritative):
%
%       PASS  valid converged run
%       FAIL  capped run
%       FAIL  success = false
%       FAIL  missing termination metadata
%       FAIL  missing convergence metadata
%       FAIL  invalid schema
%       FAIL  design change above the declared tolerance

fprintf('\n=== test_acceptance_gates ===\n');
nPass = 0; nFail = 0;

% Baseline: a valid, converged run (cap 2000, tol 2e-3).
good = struct('success', true, 'iterations', 1052, 'cap', 2000, ...
              'design_change', 1.66e-4, 'tol', 2e-3);

% ---- PASS: valid converged result --------------------------------------
[nPass, nFail] = expectPass('valid converged run', good, nPass, nFail);

% ---- FAIL: capped run ---------------------------------------------------
r = good; r.iterations = 2000;                       % iterations == cap
[nPass, nFail] = expectFail('capped run (iterations == cap)', r, ...
    'iteration cap', nPass, nFail);

r = good; r.iterations = 2001;                       % iterations > cap
[nPass, nFail] = expectFail('capped run (iterations > cap)', r, ...
    'iteration cap', nPass, nFail);

% A capped run must be rejected EVEN IF its design change looks converged.
r = good; r.iterations = 2000; r.design_change = 1e-9;
[nPass, nFail] = expectFail('capped run with small design change', r, ...
    'iteration cap', nPass, nFail);

% ---- FAIL: success = false ---------------------------------------------
r = good; r.success = false;
[nPass, nFail] = expectFail('success = false', r, 'success=false', nPass, nFail);

% ---- FAIL: missing termination metadata --------------------------------
r = good; r.iterations = NaN;
[nPass, nFail] = expectFail('missing termination metadata (iterations NaN)', r, ...
    'missing termination metadata', nPass, nFail);

r = good; r.cap = NaN;
[nPass, nFail] = expectFail('missing termination metadata (cap NaN)', r, ...
    'missing termination metadata', nPass, nFail);

r = good; r.cap = 0;
[nPass, nFail] = expectFail('missing termination metadata (cap = 0)', r, ...
    'missing termination metadata', nPass, nFail);

% ---- FAIL: missing convergence metadata --------------------------------
% This is the Olhoff/Yuksel case: the comparator reports no design change.
r = good; r.design_change = NaN;
[nPass, nFail] = expectFail('missing convergence metadata (design change NaN)', r, ...
    'missing convergence metadata', nPass, nFail);

r = good; r.design_change = [];
[nPass, nFail] = expectFail('missing convergence metadata (design change empty)', r, ...
    'missing convergence metadata', nPass, nFail);

% ---- FAIL: invalid schema ----------------------------------------------
[nPass, nFail] = expectFail('invalid schema (not a struct)', 42, ...
    'invalid result schema', nPass, nFail);

r = rmfield(good, 'design_change');
[nPass, nFail] = expectFail('invalid schema (field removed)', r, ...
    'invalid result schema', nPass, nFail);

r = rmfield(good, 'cap');
[nPass, nFail] = expectFail('invalid schema (cap removed)', r, ...
    'invalid result schema', nPass, nFail);

% ---- FAIL: design change above declared tolerance ----------------------
r = good; r.design_change = 5.24e-3;                 % the CR2 Variant A value
[nPass, nFail] = expectFail('design change above declared tolerance', r, ...
    'exceeds the declared', nPass, nFail);

r = good; r.design_change = 2.0e-3 + eps(2.0e-3);    % just above tol
[nPass, nFail] = expectFail('design change marginally above tolerance', r, ...
    'exceeds the declared', nPass, nFail);

% Boundary: design_change exactly == tol is ACCEPTED (rule is <=, as declared).
r = good; r.design_change = 2e-3;
[nPass, nFail] = expectPass('design change exactly at tolerance', r, nPass, nFail);

% ---- Regression: the two historically mis-accepted runs ----------------
% EXP2b alpha=1.00 and alpha=0.75 terminated at 2000/2000 and were ACCEPTED by
% the old gate.  They must now be rejected.
r = struct('success', true, 'iterations', 2000, 'cap', 2000, ...
           'design_change', 0.15522, 'tol', 2e-3);
[nPass, nFail] = expectFail('regression: historically accepted capped EXP2b run', r, ...
    'iteration cap', nPass, nFail);

% Historical regression (from the retired EXP1): a comparator that ran to its
% cap at every mesh must still be rejected by the shared rule.  EXP1 itself is
% retired; this fixture is kept only to pin the capped-run behaviour.
r = struct('success', true, 'iterations', 10000, 'cap', 10000, ...
           'design_change', NaN, 'tol', 3e-3);
[nPass, nFail] = expectFail('regression: capped comparator run', r, ...
    'iteration cap', nPass, nFail);

fprintf('\n  passed: %d   failed: %d\n', nPass, nFail);
if nFail > 0
    error('test_acceptance_gates:Failed', ...
        '%d acceptance-gate test(s) failed.', nFail);
end
fprintf('  ALL ACCEPTANCE-GATE TESTS PASSED\n\n');
end

% =========================================================================
function [nPass, nFail] = expectPass(name, run, nPass, nFail)
[ok, reason] = check_revision_run('T', run);
if ok
    fprintf('  [PASS] %s\n', name);
    nPass = nPass + 1;
else
    fprintf(2, '  [FAIL] %s -- expected accept, got reject: %s\n', name, reason);
    nFail = nFail + 1;
end
end

function [nPass, nFail] = expectFail(name, run, expectSubstring, nPass, nFail)
[ok, reason] = check_revision_run('T', run);
if ok
    fprintf(2, '  [FAIL] %s -- expected reject, got ACCEPT\n', name);
    nFail = nFail + 1;
elseif ~contains(lower(reason), lower(expectSubstring))
    fprintf(2, '  [FAIL] %s -- rejected for the wrong reason:\n         got: %s\n         want substring: %s\n', ...
        name, reason, expectSubstring);
    nFail = nFail + 1;
else
    fprintf('  [PASS] %s  (rejected: %s)\n', name, reason);
    nPass = nPass + 1;
end
end
