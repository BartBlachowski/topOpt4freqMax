function [g, dg] = stiffnessInterpolation(rho, stiff)
%STIFFNESSINTERPOLATION  Stiffness interpolation factor g(rho) and its derivative.
%
%   [g,dg] = OLH.MATERIAL.STIFFNESSINTERPOLATION(rho, cfg.material.stiffness)
%   [g,dg] = OLH.MATERIAL.STIFFNESSINTERPOLATION(rho, p)     plain SIMP, numeric p
%
%   Models:
%     'simp'      g = rho^p                                Du & Olhoff eq. (1)
%     'pedersen'  g = rho^p                (rho >= rho0)   Pedersen (2000) eq. (5)
%                 g = rho * rho0^(p-1)     (rho <  rho0)
%                 C0-continuous at rho0; with the printed rho0 = 0.1 and p = 3
%                 the low branch is rho/100, "the penalization of the stiffness
%                 is one hundredth of the penalization of the mass", which caps
%                 the mass/stiffness ratio at 1/rho0^2 = 100.  Du & Olhoff
%                 sec. 2.2 name this scheme as their alternative to the mass
%                 cut-off of eq. (4).  Pedersen's second ingredient -- ignoring
%                 the nodes of the eigenvector convergence test in regions
%                 below 1 % density -- concerns his inverse-iteration solver and
%                 has no counterpart in a Lanczos/LAPACK solve; it is not
%                 implemented.
%   The threshold cfg.material.stiffness.linearBelow is the printed 0.1; any
%   other value is a reconstruction sweep and must be recorded as such.

rho = rho(:);
if isnumeric(stiff)
    p = stiff;  model = 'simp';
else
    p = stiff.p;  model = stiff.model;
end
switch model
    case 'simp'
        g  = rho.^p;
        dg = p*rho.^(p-1);
    case 'pedersen'
        r0 = stiff.linearBelow;
        lo = rho < r0;
        g  = rho.^p;            dg = p*rho.^(p-1);
        c  = r0^(p-1);
        g(lo)  = c*rho(lo);     dg(lo) = c;
    otherwise
        error('olh:material:stiffnessModel','unknown stiffness model ''%s''', model);
end
end
