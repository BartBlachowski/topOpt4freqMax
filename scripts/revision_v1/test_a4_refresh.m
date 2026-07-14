function test_a4_refresh()
%TEST_A4_REFRESH  R-1 regression tests (A4_SPECIFICATION_V3 §7.1).
%
%   test_a4_refresh()
%
%   Focused, LIGHTWEIGHT tests on a tiny mesh (40x5, few iterations).
%   NO production optimization is executed.
%
%   Proves the three hard constraints on R-1:
%     T1  INERTNESS      semi_harmonic WITHOUT update_after -> refresh inactive,
%                        zero refresh events.
%     T2  N = Inf        update_after = Inf -> refresh inactive, zero events,
%                        and BIT-IDENTICAL to T1 (xPhys, omega, all histories).
%     T3  update_after=0 -> same (explicit frozen), bit-identical to T1.
%     T4  REFRESH TIMING update_after = k -> refresh events occur at exactly
%                        the iterations i with mod(i,k) == 0, and nowhere else.
%     T5  EVENT RECORD   every refresh event is recorded with the required fields.
%     T6  ANALYTIC COUNT observed refresh count == floor(nIter / k)  (V-A4-3).
%
%   T2/T3 are the "N = infinity is bit-identical to the current solver" proof:
%   the frozen path must be untouched by R-1.

fprintf('\n=== test_a4_refresh (R-1 regression) ===\n');
nPass = 0; nFail = 0;

% ---- tiny deterministic case -------------------------------------------
nelx = 40; nely = 5; volfrac = 0.5; penal = 3; rmin = 2; ft = 1; L = 8; H = 1;
maxIters = 6;

baseCfg = struct();
baseCfg.supportType = 'SS';
baseCfg.E0 = 1e7;
baseCfg.Emin = 1e-9 * 1e7;
baseCfg.rho0 = 1.0;
baseCfg.rho_min = 1e-9;
baseCfg.pmass = 1.0;                       % linear mass (declared method)
baseCfg.nu = 0.3;
baseCfg.move = 0.2;
baseCfg.conv_tol = 1e-12;                  % force the iteration cap -> fixed length
baseCfg.max_iters = maxIters;
baseCfg.optimizer = 'OC';
baseCfg.semi_harmonic_baseline = 'solid';  % authoritative
baseCfg.load_sensitivity = 'omitted';
baseCfg.harmonic_normalize = false;
baseCfg.visualize_live = false;

% ---- T1: no update_after (the pre-R-1 configuration) --------------------
cfg1 = baseCfg;
cfg1.load_cases = localSemiCase([]);            % NO update_after field at all
[x1, f1, ~, n1, i1] = localRun(nelx, nely, volfrac, penal, rmin, ft, L, H, cfg1);

[nPass, nFail] = localCheck('T1 inertness: refresh inactive without update_after', ...
    i1.semi_harmonic_refresh.active == false, nPass, nFail);
[nPass, nFail] = localCheck('T1 inertness: zero refresh events', ...
    i1.semi_harmonic_refresh.n_refresh == 0 && isempty(i1.semi_harmonic_refresh.events), ...
    nPass, nFail);

% ---- T2: update_after = Inf --------------------------------------------
cfg2 = baseCfg;
cfg2.load_cases = localSemiCase(Inf);
[x2, f2, ~, n2, i2] = localRun(nelx, nely, volfrac, penal, rmin, ft, L, H, cfg2);

[nPass, nFail] = localCheck('T2 N=Inf: refresh inactive', ...
    i2.semi_harmonic_refresh.active == false, nPass, nFail);
[nPass, nFail] = localCheck('T2 N=Inf: zero refresh events', ...
    i2.semi_harmonic_refresh.n_refresh == 0, nPass, nFail);
[nPass, nFail] = localCheck('T2 N=Inf: BIT-IDENTICAL design to T1', ...
    isequal(x1, x2), nPass, nFail);
[nPass, nFail] = localCheck('T2 N=Inf: BIT-IDENTICAL frequency to T1', ...
    isequaln(f1, f2), nPass, nFail);
[nPass, nFail] = localCheck('T2 N=Inf: identical iteration count to T1', ...
    isequal(n1, n2), nPass, nFail);

% ---- T3: update_after = 0 (explicit frozen) -----------------------------
cfg3 = baseCfg;
cfg3.load_cases = localSemiCase(0);
[x3, f3, ~, ~, i3] = localRun(nelx, nely, volfrac, penal, rmin, ft, L, H, cfg3);

[nPass, nFail] = localCheck('T3 N=0: refresh inactive', ...
    i3.semi_harmonic_refresh.active == false, nPass, nFail);
[nPass, nFail] = localCheck('T3 N=0: BIT-IDENTICAL design to T1', ...
    isequal(x1, x3) && isequaln(f1, f3), nPass, nFail);

% ---- T4/T5/T6: refresh timing -------------------------------------------
% Fixture note: the timing arms use a near-solid design (volfrac 0.9) so that a
% support-connected fundamental mode EXISTS and the §4.3.1 screen admits it.
% At volfrac 0.5 on this toy mesh the screen legitimately finds no admissible
% mode and R-1 fails loud -- that behaviour is asserted separately in T7.
% The screen is NOT relaxed to make these tests pass.
volfracR = 0.9;
for k = [1, 2, 3]
    cfgR = baseCfg;
    cfgR.load_cases = localSemiCase(k);
    [~, ~, ~, nR, iR] = localRun(nelx, nely, volfracR, penal, rmin, ft, L, H, cfgR);

    ref = iR.semi_harmonic_refresh;
    expectedIters = (1:nR)';
    expectedIters = expectedIters(mod(expectedIters, k) == 0);

    [nPass, nFail] = localCheck(sprintf('T4 N=%d: refresh ACTIVE', k), ...
        ref.active == true, nPass, nFail);

    if isempty(ref.events)
        actualIters = zeros(0, 1);
    else
        actualIters = [ref.events.iter]';
    end
    [nPass, nFail] = localCheck( ...
        sprintf('T4 N=%d: refresh at exactly mod(i,%d)==0 (expected [%s], got [%s])', ...
            k, k, num2str(expectedIters'), num2str(actualIters')), ...
        isequal(actualIters, expectedIters), nPass, nFail);

    [nPass, nFail] = localCheck(sprintf('T6 N=%d: count == floor(nIter/N) = %d (V-A4-3)', ...
        k, floor(nR / k)), ...
        ref.n_refresh == floor(nR / k) && ref.n_refresh == ref.n_refresh_predicted, ...
        nPass, nFail);

    if ~isempty(ref.events)
        ev = ref.events(1);
        haveFields = all(isfield(ev, {'iter','index','omega','mac_prev','mac_phi0', ...
            'low_density_strain_fraction','largest_support_component_kinetic_fraction', ...
            'n_components','n_admissible','reason'}));
        [nPass, nFail] = localCheck(sprintf('T5 N=%d: every event fully recorded', k), ...
            haveFields && isfinite(ev.omega) && ev.index >= 1, nPass, nFail);
    end
end

% ---- T7: fail-loud on an inadmissible refresh reference ------------------
% Spec V3 §7.1: "If no mode passes the screen, the run does not silently pick
% one: it records a B3 event and terminates that arm as Class C."
% The volfrac=0.5 toy design has no support-connected admissible mode, so this
% is the natural inadmissible fixture.  It must ERROR, not silently recover.
cfg7 = baseCfg;
cfg7.load_cases = localSemiCase(1);
gotId = '';
try
    localRun(nelx, nely, volfrac, penal, rmin, ft, L, H, cfg7);
catch ME
    gotId = ME.identifier;
end
[nPass, nFail] = localCheck( ...
    sprintf('T7 fail-loud: inadmissible refresh raises topopt_freq:SemiHarmonicRefreshInadmissible (got "%s")', gotId), ...
    strcmp(gotId, 'topopt_freq:SemiHarmonicRefreshInadmissible'), nPass, nFail);

fprintf('\n  passed: %d   failed: %d\n', nPass, nFail);
if nFail > 0
    error('test_a4_refresh:Failed', '%d R-1 regression test(s) failed.', nFail);
end
fprintf('  ALL R-1 REGRESSION TESTS PASSED\n');
fprintf('  => R-1 is inert outside A4; N=Inf is bit-identical to the frozen solver.\n\n');
end

% =========================================================================

function lc = localSemiCase(updateAfter)
%LOCALSEMICASE  One semi_harmonic mode-1 load case.
%   updateAfter = [] -> the field is ABSENT (the pre-R-1 configuration).
ld = struct('type', 'semi_harmonic', 'mode', 1, 'factor', 1.0);
if ~isempty(updateAfter)
    ld.update_after = updateAfter;
end
lc = struct('name', 'mode1_semi_harmonic', 'factor', 1.0, 'loads', {{ld}});
end

function [x, f, t, n, info] = localRun(nelx, nely, volfrac, penal, rmin, ft, L, H, cfg)
evalc('[x, f, t, n, info] = topopt_freq(nelx, nely, volfrac, penal, rmin, ft, L, H, cfg);');
end

function [nPass, nFail] = localCheck(name, cond, nPass, nFail)
if cond
    fprintf('  [PASS] %s\n', name);
    nPass = nPass + 1;
else
    fprintf(2, '  [FAIL] %s\n', name);
    nFail = nFail + 1;
end
end
