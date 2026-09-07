function [m, dm] = mass_interp(rho_e, mode, q)
% MASS_INTERP  Element mass interpolation coefficient and derivative.
%
%   [m, dm] = mass_interp(rho_e, mode)
%   [m, dm] = mass_interp(rho_e, mode, q)
%
%   Vectorized.  Returns m(rho_e) such that M_e = m * Me_star, and its
%   derivative dm = d m / d rho_e.
%
%   Modes
%   -----
%   'olhoff2014_pow'   [E]  Olhoff & Du (2014) Eq. (5), literal:
%                               m  = rho_e^q
%                               dm = q * rho_e^(q-1)
%                      q defaults to 1 (Du & Olhoff 2007 Eq. 2, "q = 1
%                      normally").  This is the ONLY mass model stated in
%                      Olhoff2014; the chapter defers the interpolation to
%                      Olhoff & Du (2013A), which is not in references/.
%
%   'linear'           [E]  identical to olhoff2014_pow with q = 1.
%
%   'du2007_step'      [D]  Du & Olhoff (2007) Eq. (4), r = 6:
%                               m = rho_e     , rho_e >  0.1
%                               m = rho_e^6   , rho_e <= 0.1
%                      Derivative discontinuous at 0.1.
%
%   'du2007_c0'        [D]  Du & Olhoff (2007) Eq. (4a), c0 = 1e5:
%                               m = c0 * rho_e^6, rho_e <= 0.1
%                      C0 at 0.1: 1e5*(0.1)^6 = 0.1.
%
%   'du2007_c1'        [D]  Du & Olhoff (2007) Eq. (4b), c1 = 6e5, c2 = -5e6:
%                               m = c1*rho_e^6 + c2*rho_e^7, rho_e <= 0.1
%                      C0 at 0.1: 0.6 - 0.5 = 0.1.
%                      C1 at 0.1: 36 - 35 = 1.
%
%   Tags [E]/[D] refer to the exactness contract in
%   analysis/OlhoffApproachExact/PLAN_Olhoff2014_exact.md section 1.

if nargin < 3 || isempty(q), q = 1; end

rho_e = rho_e(:);
n     = numel(rho_e);
m     = zeros(n,1);
dm    = zeros(n,1);

switch lower(strtrim(mode))

    case {'olhoff2014_pow', 'pow'}
        if q == 1
            m(:)  = rho_e;
            dm(:) = 1;
        else
            m(:)  = rho_e .^ q;
            dm(:) = q * rho_e .^ (q - 1);
        end

    case 'linear'
        m(:)  = rho_e;
        dm(:) = 1;

    case 'du2007_step'
        hi = rho_e > 0.1;
        lo = ~hi;
        m(hi)  = rho_e(hi);
        m(lo)  = rho_e(lo).^6;
        dm(hi) = 1;
        dm(lo) = 6 * rho_e(lo).^5;

    case 'du2007_c0'
        c0 = 1e5;
        hi = rho_e > 0.1;
        lo = ~hi;
        m(hi)  = rho_e(hi);
        m(lo)  = c0 * rho_e(lo).^6;
        dm(hi) = 1;
        dm(lo) = 6*c0 * rho_e(lo).^5;

    case 'du2007_c1'
        c1 =  6e5;
        c2 = -5e6;
        hi = rho_e > 0.1;
        lo = ~hi;
        m(hi)  = rho_e(hi);
        m(lo)  = c1*rho_e(lo).^6 + c2*rho_e(lo).^7;
        dm(hi) = 1;
        dm(lo) = 6*c1*rho_e(lo).^5 + 7*c2*rho_e(lo).^6;

    otherwise
        error('mass_interp:UnknownMode', ...
            ['mass_interp: unknown mode ''%s''. Use olhoff2014_pow, linear, ' ...
             'du2007_step, du2007_c0, or du2007_c1.'], mode);
end
end
