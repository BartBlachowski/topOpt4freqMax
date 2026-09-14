function ex = exhaustion(ex, NE, tol, outer, rho, drho, rho0)
%EXHAUSTION  The frozen two-branch stage-exhaustion detector, evaluated online.
%
%   ex = EXHAUSTION(ex, NE, tol, outer, rho, drho, rho0) advances the detector
%   by one outer iteration.  Call it ONCE per outer iteration, immediately
%   after the design update, with:
%
%       rho    the design variable AFTER the update at iteration `outer`
%       drho   the increment the inner sub-problem returned at that iteration
%       rho0   the uniform initial design (a real state; it forms drho_1)
%       tol    the convergence scale, = cfg.stop.tolerance
%
%   PROVENANCE.  This is not a new signal.  It is the rule frozen in
%   analysis/OlhoffCurrent/diagnostics/two_branch_maturity_240/PREREGISTRATION.md
%   (SHA-256 6274822525...) sections 2-7, whose executable form is that study's
%   scripts/tb_branches.m, evaluated forwards in time instead of retrospectively.
%   No threshold, window, persistence length, normalization or tolerance is
%   introduced or altered here.
%
%   THE RULE
%   --------
%       cos(k)      = <d_k, d_{k-1}> / (||d_k|| ||d_{k-1}||)     d_k = rho_k - rho_{k-1}
%       net(k)      = n2(rho_k - rho_{k-10}) / sum_{j=k-9}^{k} n2(d_j)   n2 = ||.||/sqrt(NE)
%       amp(k)      = ||drho_k||_2                               the INHERITED native measure
%       med20 x (k) = median over [k-19, k], 'omitnan'
%
%       A(k) = med20 cos(k) < 0  AND  med20 net(k) < 0.5  AND  amp(k) >= tol
%       B(k) = amp(k) < tol      AND  med20 cos(k) > 0
%       E(k) = A(k) OR B(k)
%
%   A and B are mutually exclusive at any single k (amp >= tol vs amp < tol), so
%   at most one persistence counter is ever running.
%
%   Branch B is the inherited native design-change stop criterion PLUS a
%   coherence guard PLUS persistence.  It is not presented as novel physics.
%
%   DECLARATION.  E is DECLARED at the first iteration t at which either branch
%   has held for P = 20 consecutive iterations.  That is the same event
%   tb_branches reports as beginning at t - 19; scanning t upwards finds the
%   smallest window start over both branches, so within one move stage the
%   online declaration reproduces tb_branches exactly.
%
%   STAGE LOCALITY.  Let s = ex.stageStart be the first outer iteration executed
%   at the current move level.  Every quantity entering a predicate is formed
%   only from iterations j >= s:
%
%       amp(j)   j >= s          cos(j)   j >= s+1        net(j)  j >= s+9
%       medians  defined only once the trailing 20-window is entirely inside the
%                stage, i.e. j >= s+19 -- which is exactly the k >= W condition
%                tb_branches applies at s = 1
%
%   Under a move change ||drho|| is bounded by a different constant, so a window
%   straddling the change reports the SCHEDULE rather than the design -- the
%   defect stop.guards.settledMove exists to suppress, applied consistently to
%   the whole window.  Carrying a window across a transition would let a
%   mechanically halved amplitude satisfy Branch B and trigger a spurious
%   descent.  The reset can only delay a descent, never advance one.
%
%   ANCHOR.  net(j) at j = s+9 uses rho_{s-1}, the design at the moment the
%   stage began -- exactly as the first stage uses rho_0.  Only the anchor is
%   pre-stage; every STEP in the window is stage-local.
%
%   See also OLH.MOVE.LIMIT, OLHOFFSOLVE.

W = 20; P = 20; Wnp = 10;

if isempty(ex)
    ex = struct( ...
        'W', W, 'P', P, 'Wnp', Wnp, 'tol', tol, 'NE', NE, ...
        'stageStart', 1, ...
        'X',    rho0*ones(NE,1), ...   % rolling designs, X(:,end) = rho_{k}, up to Wnp+1 columns
        'dPrev', [], ...               % d_{k-1}
        'dn',   [], ...                % n2 norms of the last <= Wnp steps
        'amp',  [], 'cos', [], 'net', [], 'medcos', [], 'mednet', [], ...
        'A', [], 'B', [], 'E', [], 'nA', [], 'nB', [], ...
        'cntA', 0, 'cntB', 0, ...
        'declared', false, 'declIter', NaN, 'declBranch', '', 'declBegin', NaN, ...
        'events', zeros(0,3), 'eventBranch', {{}});
end

k = outer;
s = ex.stageStart;
n2 = @(v) norm(v)/sqrt(NE);

d = rho - ex.X(:,end);                       % d_k, from the designs actually visited

% ---- amplitude: the inherited native measure, hist.dxNorm2 --------------
ex.amp(k,1) = norm(drho);

% ---- directional coherence ---------------------------------------------
c = NaN;
if (k-1) >= s && ~isempty(ex.dPrev)
    na = norm(d); nb = norm(ex.dPrev);
    if na > 0 && nb > 0, c = (d.'*ex.dPrev)/(na*nb); end
end
ex.cos(k,1) = c;

% ---- rolling buffers ----------------------------------------------------
ex.X = [ex.X, rho];
if size(ex.X,2) > Wnp+1, ex.X = ex.X(:, end-Wnp:end); end
ex.dn = [ex.dn, n2(d)];
if numel(ex.dn) > Wnp, ex.dn = ex.dn(end-Wnp+1:end); end
ex.dPrev = d;

% ---- net progress against path length ----------------------------------
np = NaN;
if (k-Wnp+1) >= s && size(ex.X,2) == Wnp+1
    pw = sum(ex.dn);
    if pw > 0, np = n2(ex.X(:,end) - ex.X(:,1))/pw; end
end
ex.net(k,1) = np;

% ---- trailing medians, only once the window is wholly inside the stage --
if k >= s + W - 1
    mc = median(ex.cos(k-W+1:k), 'omitnan');
    mn = median(ex.net(k-W+1:k), 'omitnan');
else
    mc = NaN; mn = NaN;
end
ex.medcos(k,1) = mc;
ex.mednet(k,1) = mn;

% ---- the two branches ---------------------------------------------------
a = ~isnan(mc) && ~isnan(mn) && mc < 0 && mn < 0.5 && ex.amp(k) >= tol;
b = ~isnan(mc)                && mc > 0 && ex.amp(k) <  tol;
ex.A(k,1) = a;  ex.B(k,1) = b;  ex.E(k,1) = a || b;

% ---- persistence --------------------------------------------------------
if a, ex.cntA = ex.cntA + 1; else, ex.cntA = 0; end
if b, ex.cntB = ex.cntB + 1; else, ex.cntB = 0; end
ex.nA(k,1) = ex.cntA;  ex.nB(k,1) = ex.cntB;

if ~ex.declared
    if ex.cntA >= P
        ex.declared = true; ex.declBranch = 'A';
        ex.declIter = k;    ex.declBegin  = k - P + 1;
    elseif ex.cntB >= P
        ex.declared = true; ex.declBranch = 'B';
        ex.declIter = k;    ex.declBegin  = k - P + 1;
    end
end
end
