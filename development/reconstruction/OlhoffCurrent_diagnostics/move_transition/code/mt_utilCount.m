function c = mt_utilCount(rrho, startJ, upTo, threshold)
%MT_UTILCOUNT  The literal "N consecutive" utilization persistence counter.
%
%   c = MT_UTILCOUNT(rrho, startJ, upTo, threshold) counts the TRAILING run of
%   consecutive completed iterations j, walking backwards from j = upTo, for
%   which
%
%       rrho(j) < threshold        and        j >= startJ ,
%
%   where startJ is the first iteration at the CURRENT, unchanged move level.
%   The walk stops at the first j with rrho(j) >= threshold (counter reset by a
%   violation) or at j < startJ (counter reset by a move transition).
%
%   This is the ONLY definition of the counter in this study.  The live
%   controller (mt_moveLimit) and the offline analysis both call this function,
%   so they cannot disagree about what "10 consecutive" means.
%
%   DELIBERATELY NOT IMPLEMENTED, per the preregistration: mean over the window,
%   median, 9-of-10, endpoint comparison, cumulative average, percentile, RMS,
%   or any exception clause.  The hypothesis under test is one specific rule.

c = 0;
for j = upTo:-1:max(startJ,1)
    if rrho(j) < threshold
        c = c + 1;
    else
        break
    end
end
end
