function B = tb_branches(per, dyn, NE)
%TB_BRANCHES  The FROZEN two-branch predicates (PREREGISTRATION sections 3-7).
%   Nothing here may be altered after 2026-09-08T12:04:53Z.
%
%   tol(NE) = 0.05*sqrt(NE/3200)          the inherited meshScaled scale,
%             equivalently RMS(drho) < 8.838835e-04 (mesh-independent).
%   W = 20 median window, P = 20 persistence, both inherited.
%
%   BRANCH A  med20 cos < 0  AND  med20 net_path < 0.5  AND  ||drho||_2 >= tol
%   BRANCH B  ||drho||_2 < tol  AND  med20 cos > 0
%   EVENT     first iteration at which either branch's 20-iteration window BEGINS

W = 20; P = 20;
tol = 0.05*sqrt(NE/3200);
n = numel(per.move);

medcos = nan(n,1); mednet = nan(n,1);
for k = W:n
    medcos(k) = median(dyn.cosT(k-W+1:k),      'omitnan');
    mednet(k) = median(dyn.net_ratio(k-W+1:k), 'omitnan');
end
amp = per.l2;                                   % the inherited L2 design change

Apred = (medcos < 0) & (mednet < 0.5) & (amp >= tol);
Bpred = (amp  < tol) & (medcos > 0);
Apred(isnan(medcos) | isnan(mednet)) = false;
Bpred(isnan(medcos)) = false;

kA = local_sustain(Apred, P);
kB = local_sustain(Bpred, P);

if     isnan(kA) && isnan(kB), kEv = NaN;  branch = 'neither';
elseif isnan(kB),              kEv = kA;   branch = 'A';
elseif isnan(kA),              kEv = kB;   branch = 'B';
elseif kA <= kB,               kEv = kA;   branch = 'A';
else,                          kEv = kB;   branch = 'B';
end

B = struct('W',W,'P',P,'tol',tol,'rmsConstant',0.05/sqrt(3200), ...
    'medcos',medcos,'mednet',mednet,'amp',amp, ...
    'Apred',Apred,'Bpred',Bpred,'kA',kA,'kB',kB, ...
    'event',kEv,'branch',branch,'bothFired',~isnan(kA)&&~isnan(kB));
end

function k = local_sustain(v, P)
k = NaN; n = numel(v);
for j = 1:(n-P+1)
    if all(v(j:j+P-1)), k = j; return; end
end
end
