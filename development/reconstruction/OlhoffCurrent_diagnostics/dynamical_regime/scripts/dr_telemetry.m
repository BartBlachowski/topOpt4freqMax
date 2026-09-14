function P = dr_telemetry(h, RHO, NE, rho0)
%DR_TELEMETRY  Per-iteration telemetry + the preregistered dynamical quantities.
%
%   Everything is POST HOC: computed from the recorded trajectory, fed back
%   into nothing.  Definitions are exactly PREREGISTRATION sec. 2-4.
%
%   Indexing: RHO(:,k) is the design AFTER outer iteration k.  rho_0 (the
%   uniform initial design) is a real state and is used to form drho_1.

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
P.ratio   = h.dxOuter(:)./h.move(:);
P.nInner  = h.nInner(:);
P.innerConv = h.innerConv(:);
P.multN   = h.N(:);
P.degen   = h.degen(:);
P.multJ   = h.multJ(:);

% ---- discreteness (identical formulas to the preceding studies) ---------
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

% ---- production beta-stall signal, re-derived exactly as olh.move.limit --
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

% ======================= DYNAMICAL QUANTITIES ===========================
% Norms: primary L2/sqrt(NE); secondary L1/NE.  Ratios are invariant to both.
n2 = @(v) norm(v)/sqrt(NE);
n1 = @(v) sum(abs(v))/NE;

X  = [rho0*ones(NE,1), RHO];        % column j+1 = rho_j, so X(:,1) = rho_0
D  = diff(X,1,2);                   % D(:,k) = drho_k, k = 1..nO

P.d1   = nan(nO,1); P.d2   = nan(nO,1); P.q2   = nan(nO,1);
P.d1_1 = nan(nO,1); P.d2_1 = nan(nO,1); P.q2_1 = nan(nO,1);
P.cosT = nan(nO,1); P.cos2 = nan(nO,1);
P.r2   = nan(nO,1); P.r4   = nan(nO,1);
P.stepNorm = nan(nO,1); P.stepNorm1 = nan(nO,1);
P.boundFrac = nan(nO,1); P.revFrac = nan(nO,1);
P.undefQ2 = false(nO,1); P.undefCos = false(nO,1);

for k = 1:nO
    P.stepNorm(k)  = n2(D(:,k));
    P.stepNorm1(k) = n1(D(:,k));
    P.boundFrac(k) = mean(abs(D(:,k)) > 0.9*P.move(k));
end
for k = 2:nO
    P.d1(k)   = n2(X(:,k+1)-X(:,k));      % ||rho_k - rho_{k-1}||
    P.d1_1(k) = n1(X(:,k+1)-X(:,k));
    P.revFrac(k) = mean(D(:,k).*D(:,k-1) < 0);
    a = D(:,k); b = D(:,k-1); na = norm(a); nb = norm(b);
    if na > 0 && nb > 0
        P.cosT(k) = (a.'*b)/(na*nb);
    else
        P.undefCos(k) = true;             % recorded as NaN, never as 0
    end
end
for k = 3:nO
    P.d2(k)   = n2(X(:,k+1)-X(:,k-1));    % ||rho_k - rho_{k-2}||
    P.d2_1(k) = n1(X(:,k+1)-X(:,k-1));
    P.r2(k)   = P.d2(k);
    if P.d1(k) > 0
        P.q2(k)   = P.d2(k)/P.d1(k);
        P.q2_1(k) = P.d2_1(k)/P.d1_1(k);
    else
        P.undefQ2(k) = true;
    end
    a = D(:,k); b = D(:,k-2); na = norm(a); nb = norm(b);
    if na > 0 && nb > 0, P.cos2(k) = (a.'*b)/(na*nb); end
end
for k = 5:nO
    P.r4(k) = n2(X(:,k+1)-X(:,k-3));
end

% ---- net progress vs path length, W = 10 (inherited, NOT tuned) ---------
P.W = W;
P.path_W = nan(nO,1); P.net_W = nan(nO,1); P.net_ratio = nan(nO,1);
P.path_W1 = nan(nO,1); P.net_W1 = nan(nO,1); P.net_ratio1 = nan(nO,1);
for k = W:nO
    pw  = sum(arrayfun(@(j) n2(D(:,j)), (k-W+1):k));
    pw1 = sum(arrayfun(@(j) n1(D(:,j)), (k-W+1):k));
    nw  = n2(X(:,k+1)-X(:,k-W+1));
    nw1 = n1(X(:,k+1)-X(:,k-W+1));
    P.path_W(k)=pw; P.net_W(k)=nw;   if pw >0, P.net_ratio(k)  = nw/pw;   end
    P.path_W1(k)=pw1;P.net_W1(k)=nw1;if pw1>0, P.net_ratio1(k) = nw1/pw1; end
end
P.cancellation = 1 - P.net_ratio;

% ---- robustness: same diagnostics with bound-saturated elements removed -
P.q2_unsat = nan(nO,1); P.cosT_unsat = nan(nO,1); P.net_ratio_unsat = nan(nO,1);
for k = 3:nO
    sat = abs(D(:,k)) > 0.9*P.move(k) | abs(D(:,k-1)) > 0.9*P.move(k-1);
    m = ~sat;
    if nnz(m) > 0
        d1u = norm(X(m,k+1)-X(m,k)); d2u = norm(X(m,k+1)-X(m,k-1));
        if d1u > 0, P.q2_unsat(k) = d2u/d1u; end
        a = D(m,k); b = D(m,k-1);
        if norm(a)>0 && norm(b)>0, P.cosT_unsat(k) = (a.'*b)/(norm(a)*norm(b)); end
    end
end
for k = W:nO
    sat = any(abs(D(:,(k-W+1):k)) > 0.9*max(P.move((k-W+1):k)), 2);
    m = ~sat;
    if nnz(m) > 0
        pw = sum(arrayfun(@(j) norm(D(m,j)), (k-W+1):k));
        nw = norm(X(m,k+1)-X(m,k-W+1));
        if pw > 0, P.net_ratio_unsat(k) = nw/pw; end
    end
end
end
