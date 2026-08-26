function [g, dg] = massScale(rho, massInterp)
%MASSSCALE  Mass interpolation factor g(rho) and its derivative, eqs (2),(4),(4a),(4b).
rho = rho(:);
g  = zeros(size(rho));
dg = zeros(size(rho));
lo = rho <= 0.1;   hi = ~lo;

g(hi)  = rho(hi);
dg(hi) = 1;

switch lower(massInterp)
    case 'lin'                                   % eq. (2) with q = 1
        g(lo)  = rho(lo);        dg(lo) = 1;
    case '4'                                     % eq. (4), r = 6
        g(lo)  = rho(lo).^6;     dg(lo) = 6*rho(lo).^5;
    case '4a'                                    % eq. (4a), c0 = 1e5  (C^0)
        c0 = 1e5;
        g(lo)  = c0*rho(lo).^6;  dg(lo) = 6*c0*rho(lo).^5;
    case '4b'                                    % eq. (4b), c1=6e5 c2=-5e6 (C^1)
        c1 = 6e5; c2 = -5e6;
        g(lo)  = c1*rho(lo).^6 + c2*rho(lo).^7;
        dg(lo) = 6*c1*rho(lo).^5 + 7*c2*rho(lo).^6;
    otherwise
        error('massScale:model','unknown mass interpolation %s',massInterp);
end
end
