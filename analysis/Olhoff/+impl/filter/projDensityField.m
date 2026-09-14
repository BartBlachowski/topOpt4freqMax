function [rhoPhys, sChain, zTilde] = projDensityField(H, Hs, z, betaProj, eta, rhomin)
%PROJDENSITYFIELD  The three-field map  z -> z_tilde -> rho_phys.
%
%   [rhoPhys,sChain,zTilde] = PROJDENSITYFIELD(H,Hs,z,betaProj,eta,rhomin)
%
%       zTilde  = (H*z)./Hs                                   density filter (F)
%       rhoPhys = rhomin + (1-rhomin)*P(zTilde;betaProj,eta)   projection   (P)
%       sChain  = d rhoPhys / d zTilde = (1-rhomin)*P'(zTilde)
%
%   H is the SAME Sigmund (1997) weight matrix the frozen realization builds
%   with prepFilter at the frozen physical radius; the radius is not changed.
%
%   The affine floor rhomin + (1-rhomin)*P reproduces the frozen density range
%   [rhomin,1] exactly and is smooth, unlike a clamp.  See WP2 sec.1.1.
%
%   CLASS D.  Default-off: nothing calls this unless cfg.projection.on.

z = z(:);
zTilde = (H*z)./Hs;
[P, dP] = projectDensity(zTilde, betaProj, eta);
rhoPhys = rhomin + (1-rhomin)*P;
sChain  = (1-rhomin)*dP;
end
