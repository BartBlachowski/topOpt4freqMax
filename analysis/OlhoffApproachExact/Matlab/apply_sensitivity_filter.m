function s_hat = apply_sensitivity_filter(s, rho, h, Hs, nely, nelx)
% APPLY_SENSITIVITY_FILTER  Sigmund (1997) mesh-independent sensitivity filter.
%
%   s_hat = apply_sensitivity_filter(s, rho, h, Hs, nely, nelx)
%
%   Formula (Du & Olhoff 2007, citing Sigmund 1997/1998):
%
%       s_hat_e = [sum_i H_ei * rho_i * s_i] / [rho_e * sum_i H_ei]
%
%   The filter spreads sensitivities over the neighborhood of each element,
%   weighted by density.  It is applied to raw eigenvalue sensitivities before
%   the MMA subproblem.  K and M are assembled from unfiltered rho (no density
%   filtering in the exact algorithm).
%
%   Inputs:
%     s      nEl x 1   raw element sensitivity vector
%     rho    nEl x 1   current physical density, bounded below by rho_min
%     h                2D filter kernel from build_filter
%     Hs     nely x nelx  per-element kernel weight sums from build_filter
%     nely, nelx        mesh dimensions
%
%   The denominator uses rho_e * Hs_e.  Since rho >= rho_min = 1e-3 and
%   Hs > 0 everywhere, no clamping is needed provided the caller enforces
%   rho >= rho_min before calling.
%
%   Note: the volume constraint sensitivity (d vol / d rho_e = V_e / V0) is
%   NOT filtered; only eigenvalue/gradient sensitivities pass through here.
%
%   Reference: Sigmund (1997), Mech Struct Mach 25(4):493-524.
%              Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

rho_mat = reshape(rho, nely, nelx);
s_mat   = reshape(s,   nely, nelx);

num   = imfilter(rho_mat .* s_mat, h, 'symmetric');
denom = rho_mat .* Hs;

s_hat = reshape(num ./ denom, [], 1);
end
