function [search, proof] = a4_readonly_diagnostic(Kf, Mf, free, ndof, xPhys, ctx, phiPrev, phi0)
%A4_READONLY_DIAGNOSTIC  Non-perturbing wrapper for Phase-2 diagnostics.
% The adaptive search receives MATLAB value arguments only. This wrapper also
% verifies bit identity of all optimization-relevant inputs and the random
% generator state across the call (Phase-2 specification §4.4).

before = struct('Kf', Kf, 'Mf', Mf, 'xPhys', xPhys, ...
    'phiPrev', phiPrev, 'phi0', phi0);
rngBefore = rng;
search = a4_adaptive_mode_search(Kf, Mf, free, ndof, xPhys, ctx, phiPrev, phi0);
rngAfter = rng;

proof = struct();
proof.design_bit_identical = isequaln(before.xPhys, xPhys);
proof.reference_bit_identical = isequaln(before.phiPrev, phiPrev) && isequaln(before.phi0, phi0);
proof.matrices_bit_identical = isequaln(before.Kf, Kf) && isequaln(before.Mf, Mf);
proof.rng_bit_identical = isequaln(rngBefore, rngAfter);
proof.read_only = proof.design_bit_identical && proof.reference_bit_identical && ...
    proof.matrices_bit_identical && proof.rng_bit_identical;
if ~proof.rng_bit_identical
    rng(rngBefore);
end
if ~proof.read_only
    error('a4_readonly_diagnostic:StateMutation', ...
        'Diagnostic screening modified optimization input or RNG state.');
end
end
