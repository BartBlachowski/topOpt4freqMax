function [h, Hs] = build_filter(nelx, nely, rmin_elem)
% BUILD_FILTER  Mesh-independent hat-function filter kernel and weight sums.
%
%   [h, Hs] = build_filter(nelx, nely, rmin_elem)
%
%   rmin_elem: filter radius in element units (= rmin_physical / element_size).
%              The paper uses rmin_elem >= 1.5 to suppress checkerboard.
%
%   h:   2D convolution kernel (hat function, values >= 0).
%        H_ei = max(0, rmin_elem - dist(e, i))  where dist is in element units.
%
%   Hs:  nely x nelx matrix; Hs(e) = sum_i H_ei  (sum of kernel weights for
%        each element e).  Used as the denominator in both filters.
%
%   Usage
%   -----
%   Sensitivity filter (this phase):
%       s_hat = apply_sensitivity_filter(s, rho, h, Hs, nely, nelx)
%
%   Reference: Sigmund (1997), Mech Struct Mach 25(4):493-524.

r  = ceil(rmin_elem) - 1;
[dxf, dyf] = meshgrid(-r:r, -r:r);
h  = max(0, rmin_elem - sqrt(dxf.^2 + dyf.^2));
Hs = imfilter(ones(nely, nelx), h, 'symmetric');
end
