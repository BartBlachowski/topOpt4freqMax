function ok = v_multiplicity()
% V_MULTIPLICITY  Phase 3: multiplicity detection with hysteresis.
%
%   Tests DETECT_MULTIPLICITY and DETECT_MULTIPLICITY_BELOW against the
%   behaviour required by Olhoff & Du (2014) Fig. 1 step 1 and Eq. (19b)/(20e).
%
%   T1  simple eigenvalue    -> N = 1, J = n+1, cluster = {n}
%   T2  exact double         -> N = 2, J = n+2
%   T3  the tolerance that matters: a cluster 0.8 % wide on omega (1.6 % on
%       lambda) is what natural coalescence looks like in these beams.  A
%       tolerance of 1e-3 on lambda MUST miss it (this is the historical
%       failure); the default join tolerance MUST catch it.
%   T4  hysteresis: with N_prev = 2 a member that has drifted between
%       tol_join and tol_leave is RETAINED; beyond tol_leave it is DROPPED.
%       Without this the detected N chatters and a structurally different
%       subproblem is posed every outer iteration.
%   T5  J = 0 when the cluster runs off the end of the computed spectrum
%       (constraint (19b) cannot be formed and must be omitted, not faked).
%   T6  downward detection for the gap problem: R, the cluster n-R..n-1, and
%       the j = n-R-1 guard index of (20e).
%   T7  N_max capping.

fprintf('\n=== Phase 3: multiplicity detection ===\n');
ok = true;
tj = 2e-2; tl = 5e-2;                   % defaults from topopt_freq_exact (on lambda)

%% T1 simple
lam = [1.0 2.0 3.0 4.0]';
[N, J, cl] = detect_multiplicity(lam, 1, tj, tl, 1);
ok = check('T1 simple                ', N==1 && J==2 && isequal(cl,1), ok, ...
           sprintf('N=%d J=%d', N, J));

%% T2 exact double
lam = [1.0 1.0 3.0 4.0]';
[N, J, cl] = detect_multiplicity(lam, 1, tj, tl, 1);
ok = check('T2 exact double          ', N==2 && J==3 && isequal(cl,[1 2]), ok, ...
           sprintf('N=%d J=%d', N, J));

%% T3 realistic coalescence width
% omega2/omega1 = 1.008  ->  lam2/lam1 = 1.016
lam = [1.0 1.016 3.0 4.0]';
[N1e3, ~, ~] = detect_multiplicity(lam, 1, 1e-3, 1e-3, 1);
[Ndef, ~, ~] = detect_multiplicity(lam, 1, tj,   tl,   1);
ok = check('T3 tol=1e-3 misses 0.8%% ', N1e3==1, ok, sprintf('N=%d (want 1)', N1e3));
ok = check('T3 default catches it    ', Ndef==2, ok, sprintf('N=%d (want 2)', Ndef));

%% T4 hysteresis
lam = [1.0 1.035 3.0 4.0]';          % between tol_join(0.02) and tol_leave(0.05)
[Nfresh,~,~] = detect_multiplicity(lam, 1, tj, tl, 1);
[Nheld ,~,~] = detect_multiplicity(lam, 1, tj, tl, 2);
ok = check('T4 fresh -> N=1          ', Nfresh==1, ok, sprintf('N=%d', Nfresh));
ok = check('T4 held  -> N=2          ', Nheld ==2, ok, sprintf('N=%d', Nheld));
lam = [1.0 1.09 3.0 4.0]';           % beyond tol_leave
[Ndrop,~,~] = detect_multiplicity(lam, 1, tj, tl, 2);
ok = check('T4 beyond leave -> N=1   ', Ndrop==1, ok, sprintf('N=%d', Ndrop));

%% T5 cluster hits the end of the spectrum
lam = [1.0 1.0 1.0]';
[N, J, ~] = detect_multiplicity(lam, 1, tj, tl, 1);
ok = check('T5 J = 0 at spectrum end ', N==3 && J==0, ok, sprintf('N=%d J=%d', N, J));

%% T6 downward detection (gap problem)
%  n = 3; modes 1 and 2 coalesced -> R = 2, cluster = {1,2}, guard j = 0
lam = [1.0 1.0 5.0 9.0 12.0]';
[R, Jm, cl] = detect_multiplicity_below(lam, 3, tj, tl, 1);
ok = check('T6 R=2, cluster {1,2}    ', R==2 && isequal(cl,[1 2]) && Jm==0, ok, ...
           sprintf('R=%d Jm=%d', R, Jm));
%  n = 4; modes 2 and 3 coalesced -> R = 2, cluster = {2,3}, guard j = 1
lam = [0.5 1.0 1.0 9.0 12.0]';
[R, Jm, cl] = detect_multiplicity_below(lam, 4, tj, tl, 1);
ok = check('T6 R=2 with guard j=1    ', R==2 && isequal(cl,[2 3]) && Jm==1, ok, ...
           sprintf('R=%d Jm=%d', R, Jm));

%% T7 N_max capping happens in the caller; verify raw detection first
lam = [1 1 1 1 5]';
[N, ~, ~] = detect_multiplicity(lam, 1, tj, tl, 1);
ok = check('T7 raw N = 4             ', N==4, ok, sprintf('N=%d', N));

fprintf('\n=== v_multiplicity: %s ===\n\n', tf2s(ok));
end

function ok = check(name, cond, ok, extra)
    if cond, s = 'PASS'; else, s = 'FAIL'; end
    fprintf('  %s  %s   %s\n', name, s, extra);
    ok = ok && cond;
end

function s = tf2s(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
