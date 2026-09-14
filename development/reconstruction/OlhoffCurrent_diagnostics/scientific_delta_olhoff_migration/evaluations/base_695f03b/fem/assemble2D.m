function [K, M] = assemble2D(mdl, rho, p, massInterp)
%ASSEMBLE2D  SIMP-scaled global stiffness and mass, reduced to the free dofs.
%
%   [K,M] = ASSEMBLE2D(mdl,rho,p,massInterp)
%
%   Stiffness   K = sum rho_e^p Ke                                   eq. (3)
%   Mass        massInterp selects Du & Olhoff eq. (4) / (4a) / (4b):
%     '4'   rho_e     (rho>0.1) ;  rho_e^6            (rho<=0.1)     eq. (4)
%     '4a'  rho_e     (rho>0.1) ;  c0*rho_e^6         (rho<=0.1)     eq. (4a)
%     '4b'  rho_e     (rho>0.1) ;  c1*rho^6+c2*rho^7  (rho<=0.1)     eq. (4b)
%     'lin' rho_e  everywhere  (q=1, no low-density cut-off)         eq. (2)
%
%   Returns K,M restricted to mdl.free (symmetric, sparse).

if nargin < 4 || isempty(massInterp), massInterp = '4'; end

rho = rho(:);
sK  = mdl.K0(:) * (rho.^p)';
gm  = massScale(rho, massInterp);
sM  = mdl.M0(:) * gm';

K = sparse(mdl.iK, mdl.jK, sK(:), mdl.ndof, mdl.ndof);
M = sparse(mdl.iK, mdl.jK, sM(:), mdl.ndof, mdl.ndof);
K = (K+K')/2;  M = (M+M')/2;

f = mdl.free;
K = K(f,f);  M = M(f,f);
end
