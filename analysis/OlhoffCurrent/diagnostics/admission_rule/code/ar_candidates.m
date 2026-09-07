function C = ar_candidates()
%AR_CANDIDATES  The preregistered candidate set.  Exactly two.  No sweep.
%
%   Thresholds and their origins are fixed in PREREGISTRATION.md sec.4:
%     tauAbs 0.01   Bendsoe & Sigmund 99-line / top88_reference.m in this tree
%     tauRel 0.50   the move ladder's own halving factor
%     D  10, W 10   move.continuation.window
%     tauObj 5e-3   move.continuation.tolerance
C = struct('name',{},'tauAbs',{},'tauRel',{},'D',{},'W',{},'tauObj',{},'role',{});
C(1) = struct('name','C1_settledLocalObjective', 'tauAbs',0.01, 'tauRel',0.50, ...
              'D',10,'W',10,'tauObj',5e-3, 'role','primary');
C(2) = struct('name','C2_strictRatio',          'tauAbs',0.01, 'tauRel',0.25, ...
              'D',10,'W',10,'tauObj',5e-3, 'role','sensitivity');
end
