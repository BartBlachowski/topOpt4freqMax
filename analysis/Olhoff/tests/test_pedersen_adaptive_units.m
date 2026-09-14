function nFail = test_pedersen_adaptive_units()
%TEST_PEDERSEN_ADAPTIVE_UNITS  The adaptive move box and the Pedersen stiffness
%   law, exercised directly through the promoted +impl code on synthetic inputs.
%
%   No solve.  The configuration is the RESOLVED
%   duOlhoffPedersenAdaptiveBoxSensitivityFiltered preset at 160x20, so the
%   constants tested are the ones production uses, not retyped ones.
%
%   ADAPTIVE BOX (olh.move.limit, policy 'adaptive')
%     A1  outer 1 and 2: every element gets the initial box 0.10
%     A2  from outer 3: same-sign last two steps -> x1.2, reversal -> x0.7,
%         a zero step -> unchanged; clamped to [0.002, 0.10]
%     A3  repeated reversals contract geometrically to the 0.002 floor and stay
%     A4  a monotone element grows back but never exceeds 0.10
%     A5  refuses to run without the current design
%   PEDERSEN LAW (olh.material.stiffnessInterpolation)
%     P1  rho >= 0.1: g = rho^3, dg = 3 rho^2 (identical to SIMP)
%     P2  rho <  0.1: g = rho*rho0^(p-1) (= rho/100 to rounding), dg = rho0^(p-1)
%     P3  C0 continuity at rho0 = 0.1
%     P4  the SIMP model is unchanged by the Pedersen code path
%     P5  mass/stiffness ratio at rho_min = 1e-3 with linear mass is 100

here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(root);
guard = olhoffcurrent_paths(); %#ok<NASGU>
nFail = 0;
fprintf('\n%s\nTEST_PEDERSEN_ADAPTIVE_UNITS\n%s\n', repmat('=',1,72), repmat('=',1,72));

cfg = olhoffcurrent_config(160, 20, 'Preset', 'duOlhoffPedersenAdaptiveBoxSensitivityFiltered');
d0 = cfg.move.initial; dmin = cfg.move.minimum; up = cfg.move.adaptive.grow; dn = cfg.move.adaptive.shrink;
nFail = nFail + chk('resolved constants: initial 0.10, floor 0.002, grow 1.2, shrink 0.7', ...
    d0 == 0.10 && dmin == 0.002 && up == 1.2 && dn == 0.7 && strcmp(cfg.move.policy, 'adaptive'));

hist = struct('N', [], 'beta', [], 'omega', []);
% four elements: monotone up, reversing, stalled (zero step), monotone down
r = {[0.5;0.5;0.5;0.5], [0.6;0.6;0.5;0.4], [0.7;0.5;0.5;0.3], [0.8;0.6;0.5;0.2]};
st = [];
[m1, st] = olh.move.limit(cfg, 1, hist, st, r{1});
[m2, st] = olh.move.limit(cfg, 2, hist, st, r{2});
nFail = nFail + chk('A1 outer 1-2: every element at the initial box', isequal(m1, d0*ones(4,1)) && isequal(m2, d0*ones(4,1)));
[m3, st] = olh.move.limit(cfg, 3, hist, st, r{3});
% steps: e1 +0.1,+0.1 -> grow (clamped at 0.10); e2 +0.1,-0.1 -> shrink; e3 0,0 -> keep; e4 -0.1,-0.1 -> grow (clamped)
nFail = nFail + chk('A2 outer 3: grow clamped at 0.10, reversal x0.7, zero step unchanged', ...
    isequal(m3, [min(d0, up*d0); dn*d0; d0; min(d0, up*d0)]));
[m4, st] = olh.move.limit(cfg, 4, hist, st, r{4});
% e2: last steps -0.2 (0.5-0.7)... then +0.1 (0.6-0.5): reversal again -> x0.7
nFail = nFail + chk('A2 outer 4: second reversal contracts again (0.049)', abs(m4(2) - dn*dn*d0) < 1e-15 && m4(1) == d0);

% A3: persistent oscillation of one element contracts to the floor and stays there
st = []; x = 0.5; mv = [];
for k = 1:40
    x = 0.5 + 0.01*(-1)^k;
    [mv, st] = olh.move.limit(cfg, k, hist, st, x);
end
nFail = nFail + chk(sprintf('A3 oscillating element contracts to the floor (%.4g)', mv), mv == dmin);
% A4: then a monotone run grows it back, never above 0.10
for k = 41:80
    x = x + 0.001;
    [mv, st] = olh.move.limit(cfg, k, hist, st, x);
end
nFail = nFail + chk(sprintf('A4 monotone element grows back to the ceiling (%.4g), never above', mv), mv == d0);
nFail = nFail + chk('A5 refuses without the current design', ...
    throwsId(@() olh.move.limit(cfg, 1, hist, [], []), 'olh:move:adaptiveNeedsRho'));

% ---- Pedersen stiffness ------------------------------------------------------
s = cfg.material.stiffness;
rhoHi = [0.1; 0.25; 0.5; 1]; rhoLo = [1e-3; 0.01; 0.05; 0.0999];
[gH, dgH] = olh.material.stiffnessInterpolation(rhoHi, s);
[gL, dgL] = olh.material.stiffnessInterpolation(rhoLo, s);
nFail = nFail + chk('P1 rho >= 0.1: g = rho^3, dg = 3 rho^2', isequal(gH, rhoHi.^3) && isequal(dgH, 3*rhoHi.^2));
% exactly the documented branch g = rho*rho0^(p-1) (0.1^2 is 0.010000000000000002 in
% double precision, so "rho/100" holds to rounding, not bit for bit)
c = s.linearBelow^(s.p - 1);
nFail = nFail + chk('P2 rho < 0.1: g = rho*rho0^(p-1) = rho/100, dg = rho0^(p-1)', ...
    isequal(gL, c*rhoLo) && all(dgL == c) && max(abs(gL - rhoLo/100)) < 1e-17 && abs(c - 0.01) < 1e-17);
[ga, ~] = olh.material.stiffnessInterpolation(0.1 - 1e-12, s);
[gb, ~] = olh.material.stiffnessInterpolation(0.1, s);
nFail = nFail + chk('P3 C0 continuous at rho0 = 0.1', abs(ga - gb) < 1e-12);
[gS, dgS] = olh.material.stiffnessInterpolation(rhoLo, struct('model', 'simp', 'p', 3));
nFail = nFail + chk('P4 SIMP branch unchanged (rho^3)', isequal(gS, rhoLo.^3) && isequal(dgS, 3*rhoLo.^2));
[g0, ~] = olh.material.stiffnessInterpolation(cfg.design.minimum, s);
nFail = nFail + chk(sprintf('P5 linear mass / Pedersen stiffness at rho_min = %.4g (want 100)', cfg.design.minimum/g0), ...
    strcmp(cfg.material.mass.model, 'eq2') && abs(cfg.design.minimum/g0 - 100) < 1e-9);

fprintf('%s\n  failures: %d\n\n', repmat('-',1,72), nFail);
end

function tf = throwsId(fn, id)
try, fn(); tf = false; catch ME, tf = strcmp(ME.identifier, id); end
end

function n = chk(label, ok)
if ok, fprintf('  [PASS] %s\n', label); n = 0;
else,  fprintf('  [FAIL] %s\n', label); n = 1; end
end
