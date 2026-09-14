function T = fp_testpoints(P, ev)
%FP_TESTPOINTS  The preregistered deterministic admissible test points.
%   T3 (M5000) and T15 (reference) are appended only if their files exist.
NE = P.NE; lo = P.xmin(1:NE); hi = P.xmax(1:NE);
clip = @(d) min(hi, max(lo, d));
T = struct('name',{},'drho',{},'source',{});
add = @(T,nm,d,src) [T, struct('name',nm,'drho',d,'source',src)];
L = load(fullfile(ev,'frozen_ctx.mat'),'xP19','xM500');
T = add(T,'T0_zero', zeros(NE,1), 'drho = 0');
T = add(T,'T1_P19', L.xP19(1:NE), 'production DRHO(:,386)');
T = add(T,'T2_M500', L.xM500(1:NE), 'prior audit stC.xFinal');
f = fullfile(ev,'mma_replay.mat');
if isfile(f), R = load(f,'xM5000'); T = add(T,'T3_M5000', R.xM5000(1:NE), 'fp_replay 5000'); end
rng(20260912,'twister');
for k = 1:5
    T = add(T,sprintf('T%d_rand',3+k), lo + (hi-lo).*rand(NE,1), 'uniform in box, seeded');
end
T = add(T,'T9_plusmove', hi, '+move clipped');
T = add(T,'T10_minusmove', lo, '-move clipped');
T = add(T,'T11_signF11', clip(P.move*sign(P.F11)), 'move*sign(F11) clipped');
for k = 1:3
    u = 2*rand(NE,1)-1; u = u - mean(u);
    ratio = zeros(NE,1); pos = u > 0; ratio(pos) = hi(pos)./u(pos); ratio(~pos) = lo(~pos)./u(~pos);
    s = 0.9*min(ratio(u ~= 0));
    T = add(T,sprintf('T%d_volneutral',11+k), clip(s*u), 'volume-neutral, seeded');
end
f = fullfile(ev,'conic_reference.mat');
if isfile(f), R = load(f,'xRef'); T = add(T,'T15_reference', R.xRef(1:NE), 'coneprog reference'); end
end
