function C = dr_classify(P, Pdef)
%DR_CLASSIFY  Preregistered regime classifier (PREREGISTRATION sec. 5).
%
%   Trailing-window medians over P = 20 outer iterations:
%     PERIOD2   median q2 < 1    AND median cos_theta < 0
%     COHERENT  median q2 > 1.5  AND median cos_theta > 0.25
%     OTHER     otherwise
%   Onset of a label = first k for which the label holds for P consecutive
%   iterations.  A single dip never counts.
%
%   Constants are diagnostic-classifier parameters, NOT controller parameters.

if nargin < 2, Pdef = 20; end
Pw = Pdef;
n  = numel(P.q2);
lab = repmat("OTHER", n, 1);
mq  = nan(n,1); mc = nan(n,1);
for k = Pw:n
    w = (k-Pw+1):k;
    mq(k) = median(P.q2(w),   'omitnan');
    mc(k) = median(P.cosT(w), 'omitnan');
    if isnan(mq(k)) || isnan(mc(k)); continue; end
    if     mq(k) < 1.0 && mc(k) < 0.00,  lab(k) = "PERIOD2";
    elseif mq(k) > 1.5 && mc(k) > 0.25,  lab(k) = "COHERENT";
    end
end

onset = struct('PERIOD2',NaN,'COHERENT',NaN);
for nm = ["PERIOD2","COHERENT"]
    for k = 1:(n-Pw+1)
        if all(lab(k:k+Pw-1) == nm), onset.(char(nm)) = k; break; end
    end
end

C = struct('P',Pw,'label',lab,'medq2',mq,'medcos',mc,'onset',onset, ...
           'labelFinal',char(lab(end)), ...
           'fracPERIOD2',mean(lab=="PERIOD2"),'fracCOHERENT',mean(lab=="COHERENT"));
end
