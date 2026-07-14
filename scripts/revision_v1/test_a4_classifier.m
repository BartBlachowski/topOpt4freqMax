function test_a4_classifier()
%TEST_A4_CLASSIFIER  Synthetic-fixture tests for check_a4_run (spec V3 Part 5).
%
%   Pure struct fixtures. No solver, no optimization, no file I/O.
%
%   The central test is the EXP4 REGRESSION (spec §7.2): the historical
%   frozen-vs-refresh comparison that produced -62% MUST classify as B3
%   (spurious-mode contamination, disqualified as an accuracy reference) and
%   MUST NOT be readable as accuracy evidence. That is the failure this whole
%   specification exists to prevent.
%
%   Also asserts the inversion of the campaign-wide rule: for A4, a capped run
%   and a lost mode are RESULTS (Class C), not rejections.

fprintf('\n=== test_a4_classifier ===\n');
nPass = 0; nFail = 0;

% Baseline: a clean, converged, mode-retaining arm.
good = struct('success', true, 'iterations', 1200, 'cap', 2000, ...
    'final_design_change', 8e-4, 'tol', 1e-3, 'feasibility', 0, ...
    'omega1_tracked', 159.3, 'omega1_min', 159.3, ...
    'mac_to_phi0', 0.9998, 'mode_index_jstar', 1);

[nPass, nFail] = expect('clean converged arm -> ACCEPTED', good, 'ACCEPTED', '', nPass, nFail);

% ---- Class A: REJECTED (the machinery broke) ----------------------------
r = good; r.success = false;
[nPass, nFail] = expect('success=false -> REJECTED', r, 'REJECTED', '', nPass, nFail);

r = good; r.exception_id = 'MATLAB:eigs:AminusBSingular';
[nPass, nFail] = expect('solver exception -> REJECTED', r, 'REJECTED', '', nPass, nFail);

r = good; r.omega1_tracked = NaN;
[nPass, nFail] = expect('non-finite endpoint frequency -> REJECTED', r, 'REJECTED', '', nPass, nFail);

% A BROKEN OC: 5% volume error. (Normal OC bisection residual is ~1e-5 relative,
% which must NOT be rejected -- see the feasibility-tolerance note in check_a4_run.)
r = good; r.feasibility = 0.05;
[nPass, nFail] = expect('volume constraint violated 5% (broken OC) -> REJECTED', r, ...
    'REJECTED', '', nPass, nFail);

r = good; r.feasibility = 1e-5;
[nPass, nFail] = expect('normal OC bisection residual 1e-5 -> still ACCEPTED', r, ...
    'ACCEPTED', '', nPass, nFail);

r = good; r.factor_drift = true;
[nPass, nFail] = expect('factor drift -> REJECTED', r, 'REJECTED', '', nPass, nFail);

r = good; r.nondeterministic = true;
[nPass, nFail] = expect('non-deterministic replay -> REJECTED', r, 'REJECTED', '', nPass, nFail);

r = rmfield(good, 'mac_to_phi0');
[nPass, nFail] = expect('missing schema field -> REJECTED', r, 'REJECTED', '', nPass, nFail);

% ---- Class C: approximation failure is a RESULT, not a rejection --------
r = good; r.mode_index_jstar = 3;
[nPass, nFail] = expect('mode migration (j*=3, MAC high) -> B1', r, ...
    'ACCEPTED_WITH_BREAKDOWN', 'B1', nPass, nFail);

r = good; r.mac_to_phi0 = 0.62;
[nPass, nFail] = expect('frozen-mode breakdown (MAC 0.62) -> B2', r, ...
    'ACCEPTED_WITH_BREAKDOWN', 'B2', nPass, nFail);

r = good; r.refresh_inadmissible = true;
[nPass, nFail] = expect('refresh reference inadmissible -> B3', r, ...
    'ACCEPTED_WITH_BREAKDOWN', 'B3', nPass, nFail);

r = good; r.omega1_min = 49.8; r.omega1_tracked = 131.2;
[nPass, nFail] = expect('omega1_min << omega1_tracked -> B3', r, ...
    'ACCEPTED_WITH_BREAKDOWN', 'B3', nPass, nFail);

r = good; r.iterations = 2000; r.final_design_change = 0.1; ...
    r.limit_cycle = true; r.omitted_term_ratio = 0.713;
[nPass, nFail] = expect('limit cycle + omitted-term 0.713 -> B4', r, ...
    'ACCEPTED_WITH_BREAKDOWN', 'B4', nPass, nFail);

r = good; r.iterations = 2000; r.final_design_change = 0.1;
[nPass, nFail] = expect('capped run -> Class C (NOT rejected)', r, ...
    'ACCEPTED_WITH_BREAKDOWN', 'B4', nPass, nFail);

% ---- THE EXP4 REGRESSION (spec §7.2) ------------------------------------
% Historical: frozen omega1 = 131.24; refresh-every-50 omega1 = 49.84 (-62%).
% The refreshed arm locked onto a spurious mode. It must be B3 -- disqualified
% as an accuracy reference -- and must NOT be presented as "refresh is worse".
exp4Refreshed = struct('success', true, 'iterations', 400, 'cap', 400, ...
    'final_design_change', 6e-3, 'tol', 1e-3, 'feasibility', 0, ...
    'omega1_tracked', 131.24, 'omega1_min', 49.84, ...
    'mac_to_phi0', 0.55, 'mode_index_jstar', 1, ...
    'refresh_inadmissible', false);
[cls, bd, why] = check_a4_run(exp4Refreshed);
ok = strcmp(cls, 'ACCEPTED_WITH_BREAKDOWN') && strcmp(bd, 'B3');
[nPass, nFail] = report( ...
    'EXP4 REGRESSION: the -62% refreshed arm classifies as B3, not as evidence', ...
    ok, cls, bd, why, nPass, nFail);

ok2 = ~strcmp(cls, 'ACCEPTED');
[nPass, nFail] = report( ...
    'EXP4 REGRESSION: the -62% arm is NOT eligible as an accuracy reference', ...
    ok2, cls, bd, '', nPass, nFail);

fprintf('\n  passed: %d   failed: %d\n', nPass, nFail);
if nFail > 0
    error('test_a4_classifier:Failed', '%d A4 classifier test(s) failed.', nFail);
end
fprintf('  ALL A4 CLASSIFIER TESTS PASSED\n\n');
end

% =========================================================================

function [nPass, nFail] = expect(name, r, wantCls, wantBd, nPass, nFail)
[cls, bd, why] = check_a4_run(r);
ok = strcmp(cls, wantCls) && strcmp(bd, wantBd);
[nPass, nFail] = report(name, ok, cls, bd, why, nPass, nFail);
end

function [nPass, nFail] = report(name, ok, cls, bd, why, nPass, nFail)
if ok
    if isempty(bd)
        fprintf('  [PASS] %s  (%s)\n', name, cls);
    else
        fprintf('  [PASS] %s  (%s/%s)\n', name, cls, bd);
    end
    nPass = nPass + 1;
else
    fprintf(2, '  [FAIL] %s  -> got %s/%s : %s\n', name, cls, bd, why);
    nFail = nFail + 1;
end
end
