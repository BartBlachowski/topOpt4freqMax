function E = ar_predicate(P, cand)
%AR_PREDICATE  Evaluate an admission rule on a recorded trajectory.
%
%   E = AR_PREDICATE(P, cand) returns every component of the predicate at every
%   iteration, plus the first iteration at which admission would occur.
%
%   cand fields: name, tauAbs, tauRel, D (dwell), W (objective window), tauObj.
%
%   The rule (see PREREGISTRATION.md sec.3):
%
%     A  move-settled dwell   move level unchanged for the last D iterations
%     B  local design change  max|drho| < tauAbs  AND  max|drho|/move < tauRel
%     C  objective stability  relative range of omega1 over W < tauObj
%
%   B's second part is what enforces the invariant: max|drho|/move is invariant
%   under a change of the move limit, so a descent cannot reduce it.

n = numel(P.outer);
E = struct();
E.name = cand.name;
E.cand = cand;

% ---- A: move-settled dwell ---------------------------------------------
% itersSinceMoveChange is 0 on the iteration the level changed, so requiring
% >= D means the level has been held for D full iterations after the change.
E.A = P.itersSinceMoveChange >= cand.D;

% ---- B: local design change, absolute AND dimensionless ----------------
E.Babs = P.maxAbs < cand.tauAbs;
E.Brel = P.ratio  < cand.tauRel;
E.B    = E.Babs & E.Brel;

% ---- C: objective stability over a window ------------------------------
E.C = false(n,1);
E.objRelRange = nan(n,1);
for k = 1:n
    if k < cand.W; continue; end
    w = (k-cand.W+1):k;
    v = P.omega1(w);
    E.objRelRange(k) = (max(v)-min(v))/mean(v);
    E.C(k) = E.objRelRange(k) < cand.tauObj;
end

E.admit = E.A & E.B & E.C;
E.stopIter = find(E.admit, 1);          % empty => never admitted (CAP_HIT)

% ---- distance from the nearest preceding move descent -------------------
if isempty(E.stopIter)
    E.status = 'CAP_HIT';
    E.gapToLastDescent = NaN;
    E.lastDescentBefore = NaN;
else
    E.status = 'ADMITTED';
    d = find(P.descent & (P.outer <= E.stopIter));
    if isempty(d)
        E.lastDescentBefore = NaN; E.gapToLastDescent = Inf;
    else
        E.lastDescentBefore = d(end);
        E.gapToLastDescent  = E.stopIter - d(end);
    end
end
end
