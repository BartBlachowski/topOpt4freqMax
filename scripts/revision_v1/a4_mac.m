function v = a4_mac(a, b, M)
%A4_MAC  Mass-weighted Modal Assurance Criterion (Gate A0-F4 convention).
%
%   v = A4_MAC(a, b, M)
%
%   v = |a' M b|^2 / ((a' M a) * (b' M b))
%
%   Mass-weighted, and therefore invariant to eigenvector scaling and sign.
%   This is the definition declared by Gate A0-F4 and used by manuscript Eq. 9;
%   A4 uses it for both the tracked-mode identification (spec V3 §4.1) and the
%   refresh continuity test (§4.3.1).  Returns NaN if either vector is null.

a = a(:);
b = b(:);
if isempty(M)
    num = abs(a' * b)^2;
    den = real(a' * a) * real(b' * b);
else
    Ma = M * a;
    Mb = M * b;
    num = abs(a' * Mb)^2;
    den = real(a' * Ma) * real(b' * Mb);
end
if ~isfinite(den) || den <= 0
    v = NaN;
else
    v = real(num) / den;
end
end
