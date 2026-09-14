function out = measurement_budget(B0, bRef, P, BRef)
%MEASUREMENT_BUDGET Apply the frozen, non-extendable measurement-budget rule.
arguments
    B0 (1,1) double {mustBeInteger,mustBePositive}
    bRef (1,1) double {mustBeInteger,mustBePositive}
    P (1,1) double {mustBeInteger,mustBePositive}
    BRef (1,1) double {mustBeInteger,mustBePositive}
end
assert(bRef <= BRef, 'ie2a:InvalidReferenceEndpoint', 'b_ref exceeds B_ref.');
out.B0 = B0;
out.b_ref = bRef;
out.P = P;
out.B_ref = BRef;
out.requested_end = bRef + P - 1;
out.B_meas = min(max(B0, out.requested_end), BRef);
out.certification_tail_truncated = out.requested_end > BRef;
out.tail_truncation = max(0, out.requested_end - BRef);
end
