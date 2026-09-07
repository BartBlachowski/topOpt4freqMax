function S = mt_spatial(RHO, P, rho0, rhoMin)
%MT_SPATIAL  How the per-element utilization |drho_e|/move is DISTRIBUTED.
%
%   The brief (sec. 12) anticipates one specific failure mode: a SMALL NUMBER of
%   elements keeping max|drho|/move >= 0.5 while the bulk topology matures.
%   Answering report question 7 honestly requires measuring whether the elements
%   at the bound are few or many -- so this function reports the whole
%   distribution, not just its maximum.
%
%   MEASUREMENT ONLY.  Nothing here defines an alternative transition statistic,
%   and none is proposed: sec. 12 places that in a FUTURE task.  These numbers
%   exist to say what the max is hiding, not to replace it.
%
%   The applied increment is recovered exactly:  drho_k = RHO(:,k) - RHO(:,k-1).
%   The box in innerLoop already guarantees rho+drho lies in [rho_min, 1], so the
%   update's own min/max clamp is a no-op and the difference IS the increment.

nO = numel(P.outer);
NE = size(RHO,1);
q  = [0.50 0.90 0.99 1.00];
S = struct();
S.quantileLevels = q;
S.utilQuantiles  = nan(nO, numel(q));   % quantiles of |drho_e|/move
S.fracAtBound    = nan(nO,1);           % |drho_e| >= 0.99*move
S.fracAbove50    = nan(nO,1);           % |drho_e| >= 0.50*move
S.nAtBound       = nan(nO,1);
S.NE = NE;

prev = rho0*ones(NE,1);
for k = 1:nO
    d = abs(RHO(:,k) - prev) / P.move(k);
    prev = RHO(:,k);
    S.utilQuantiles(k,:) = quantile(d, q);
    S.fracAtBound(k) = mean(d >= 0.99);
    S.fracAbove50(k) = mean(d >= 0.50);
    S.nAtBound(k)    = sum(d >= 0.99);
end
S.rhoMin = rhoMin;
end
