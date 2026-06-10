function [m, dm] = mass_interp(rho_e, mode)
% MASS_INTERP  Element mass interpolation coefficient and derivative.
%
%   [m, dm] = mass_interp(rho_e, mode)
%
%   Vectorized.  Returns scalar m(rho_e) such that M_e = m * Me_star, and
%   its derivative dm = d m / d rho_e.
%
%   Modes  (Du & Olhoff 2007):
%
%   'linear'
%       Paper Eq. 2 with q = 1.
%           m  = rho_e
%           dm = 1
%
%   'du2007_step'
%       Paper Eq. 4 (discontinuous at rho_e = 0.1, penalization power r = 6):
%           m  = rho_e       ,  rho_e >  0.1
%           m  = rho_e^6     ,  rho_e <= 0.1
%       Derivative is discontinuous at 0.1; use the C0/C1 variants for
%       smooth gradient descent.
%
%   'du2007_c0'
%       Paper Eq. 4a  (C0 continuous, c0 = 1e5):
%           m  = rho_e             ,  rho_e >  0.1
%           m  = c0 * rho_e^6      ,  rho_e <= 0.1
%       C0 check at rho_e = 0.1:  1e5 * (0.1)^6 = 0.1  OK.
%
%   'du2007_c1'
%       Paper Eq. 4b  (C1 continuous, c1 = 6e5, c2 = -5e6):
%           m  = rho_e                          ,  rho_e >  0.1
%           m  = c1*rho_e^6 + c2*rho_e^7        ,  rho_e <= 0.1
%       C0 check: 6e5*(0.1)^6 + (-5e6)*(0.1)^7 = 0.6 - 0.5 = 0.1  OK.
%       C1 check: 6*6e5*(0.1)^5 + 7*(-5e6)*(0.1)^6 = 36 - 35 = 1   OK.
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110, Eq. 4-4b.

rho_e = rho_e(:);
n     = numel(rho_e);
m     = zeros(n,1);
dm    = zeros(n,1);

switch lower(strtrim(mode))

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
        error('mass_interp: unknown mode ''%s''. Use linear, du2007_step, du2007_c0, or du2007_c1.', mode);
end
end
