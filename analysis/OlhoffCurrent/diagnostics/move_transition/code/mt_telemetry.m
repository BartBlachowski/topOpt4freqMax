function P = mt_telemetry(h, RHO, NE, tcfg)
%MT_TELEMETRY  Per-iteration telemetry required by the study brief, sec. 13.
%
%   Everything here is POST-HOC: it is computed from the recorded trajectory and
%   nothing is fed back into the solver.

nO = numel(h.N);
P = struct();
P.outer   = (1:nO).';
P.omega1  = h.omega(1,:).';
P.omega2  = h.omega(2,:).';
P.gap12   = h.gap12(:);
P.volume  = h.vol(:);
P.move    = h.move(:);
P.stage   = h.stage(:);
P.beta    = h.beta(:);
P.l2      = h.dxNorm2(:);
P.rms     = h.dxNorm2(:)/sqrt(NE);
P.maxAbs  = h.dxOuter(:);
P.ratio   = h.dxOuter(:)./h.move(:);          % r_rho
P.nInner  = h.nInner(:);
P.innerConv = h.innerConv(:);
P.multN   = h.N(:);
P.degen   = h.degen(:);
P.multJ   = h.multJ(:);

% ---- discreteness -------------------------------------------------------
P.Mnd = zeros(nO,1); P.gray = zeros(nO,1); P.mid = zeros(nO,1);
for k = 1:nO
    r = RHO(:,k);
    P.Mnd(k)  = 100*mean(4*r.*(1-r));
    P.gray(k) = mean(r>0.1 & r<0.9);
    P.mid(k)  = mean(r>=0.4 & r<=0.6);
end

% ---- move bookkeeping ---------------------------------------------------
P.descent     = [false; P.move(2:end) <  P.move(1:end-1)];
P.moveChanged = [true;  P.move(2:end) ~= P.move(1:end-1)];
c = 0; since = zeros(nO,1);
for k = 1:nO
    if P.moveChanged(k); c = 0; else; c = c+1; end
    since(k) = c;
end
P.itersSinceMoveChange = since;   % 0 on the iteration the level changed

% ---- the utilization persistence counter, re-derived offline ------------
% Re-derived with mt_utilCount -- the SAME function the live controller used --
% so the counter reported here cannot disagree with the counter that acted.
% The value at iteration k is the counter that was in force when the controller
% was called AT k, i.e. formed from completed iterations 1..k-1.
thr = 0.5; if isfield(tcfg,'threshold'); thr = tcfg.threshold; end
P.utilThreshold = thr*ones(nO,1);
P.utilCount = zeros(nO,1);
lastStage = 0;
for k = 1:nO
    startJ = max(lastStage,1);
    if k >= 2
        P.utilCount(k) = mt_utilCount(P.ratio(1:k-1), startJ, k-1, thr);
    end
    if P.moveChanged(k) && k >= 2; lastStage = k; end
end
P.utilBelow = P.ratio < thr;

% ---- the EXISTING production bound-variable stall signal, for comparison -
% mean of the last W of beta against the mean of the W before it, exactly as
% olh.move.limit forms it.  Recorded for both arms; it ACTS only in ARM P.
W = 10; tol = 5e-3;
P.betaStallRel = nan(nO,1); P.betaStallFires = false(nO,1);
for k = 1:nO
    b = P.beta(1:k-1);
    if numel(b) >= 2*W
        w2 = mean(b(end-W+1:end)); w1 = mean(b(end-2*W+1:end-W));
        P.betaStallRel(k) = (w2-w1)/max(abs(w1),eps);
        P.betaStallFires(k) = P.betaStallRel(k) < tol;
    end
end

% ---- omega1 relative stability over W ----------------------------------
P.objRelRange = nan(nO,1);
for k = W:nO
    v = P.omega1(k-W+1:k);
    P.objRelRange(k) = (max(v)-min(v))/mean(v);
end
end
