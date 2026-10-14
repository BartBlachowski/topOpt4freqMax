function [K, M] = assemble2D(mdl, rho, p, massInterp)
%ASSEMBLE2D  SIMP-scaled global stiffness and mass, reduced to the free dofs.
%
%   [K,M] = ASSEMBLE2D(mdl,rho,p,massInterp)
%
%   Stiffness   K = sum g_K(rho_e) Ke                                eq. (3)
%     p numeric  -> g_K = rho^p (SIMP, eq. 1)
%     p a struct -> cfg.material.stiffness: 'simp' or 'pedersen' (2000) eq. (5)
%   Mass        massInterp selects Du & Olhoff eq. (4) / (4a) / (4b):
%     '4'   rho_e     (rho>0.1) ;  rho_e^6            (rho<=0.1)     eq. (4)
%     '4a'  rho_e     (rho>0.1) ;  c0*rho_e^6         (rho<=0.1)     eq. (4a)
%     '4b'  rho_e     (rho>0.1) ;  c1*rho^6+c2*rho^7  (rho<=0.1)     eq. (4b)
%     'lin' rho_e  everywhere  (q=1, no low-density cut-off)         eq. (2)
%
%   Returns K,M restricted to mdl.free (symmetric, sparse).

if nargin < 4 || isempty(massInterp), massInterp = '4'; end

rho = rho(:);
gK  = olh.material.stiffnessInterpolation(rho, p);
gm  = massScale(rho, massInterp);

% Lower triangle assembled, then mirrored: exactly symmetric, and half the
% triplets of a full assembly (see model2D).  Same matrices as the former
% full assembly followed by (K+K')/2, up to round-off.
sK  = mdl.K0L * gK';
sM  = mdl.M0L * gm';
K = sparse(mdl.iL, mdl.jL, sK(:), mdl.ndof, mdl.ndof);
M = sparse(mdl.iL, mdl.jL, sM(:), mdl.ndof, mdl.ndof);
K = K + tril(K,-1)';  M = M + tril(M,-1)';

f = mdl.free;
K = K(f,f);  M = M(f,f);
end
