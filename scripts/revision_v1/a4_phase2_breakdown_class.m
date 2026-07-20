function [cls, breakdown, reason] = a4_phase2_breakdown_class(arm)
%A4_PHASE2_BREAKDOWN_CLASS  In-scope legacy B-classes with B3 retired (§7).
% Phase 2 does not resolve B4 or implement a limit-cycle detector. It retains
% B1/B2 and the existing unattributed-cap B4 behavior solely as the explicitly
% out-of-scope companion classification described by §7.6.

cls = 'REJECTED'; breakdown = ''; reason = '';
if ~arm.success
    reason = sprintf('exception during run: %s', arm.exception_id); return;
end
if any(~isfinite([arm.omega1_tracked, arm.omega1_min, arm.mac_to_phi0]))
    reason = 'non-finite endpoint'; return;
end
if arm.mac_to_phi0 < 0.8
    cls = 'ACCEPTED_WITH_BREAKDOWN'; breakdown = 'B2';
    reason = sprintf('B2: MAC %.4f < 0.8', arm.mac_to_phi0); return;
end
if arm.mode_index_jstar ~= 1
    cls = 'ACCEPTED_WITH_BREAKDOWN'; breakdown = 'B1';
    reason = sprintf('B1: tracked mode index is %d', arm.mode_index_jstar); return;
end
if arm.iterations >= arm.cap
    cls = 'ACCEPTED_WITH_BREAKDOWN'; breakdown = 'B4';
    reason = ['B4 (unattributed): iteration cap reached; M-2 remains open and ' ...
        'Phase 2 makes no campaign-level decision.'];
    return;
end
if isfinite(arm.final_design_change) && arm.final_design_change <= arm.tol
    cls = 'ACCEPTED'; reason = 'clean converged legacy B-class';
else
    reason = 'unconverged without a named in-scope breakdown';
end
end
