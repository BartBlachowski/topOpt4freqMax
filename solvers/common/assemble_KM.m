function [K, M, Ee, re] = assemble_KM(mdl, x, E0, Emin, rho0, rhoMin, p, xLin)
%ASSEMBLE_KM  Global stiffness and mass (all dofs) for SIMP stiffness
%   E(x) = Emin + x^p (E0 - Emin) and linear mass rho(x) = rhoMin + x (rho0 - rhoMin).
%   With xLin > 0 the stiffness law is linear below xLin, x*xLin^(p-1)
%   (Pedersen 2000; matched settings of Table 4 only).
if nargin < 7 || isempty(p), p = 3; end
if nargin < 8, xLin = 0; end
x = x(:);
if xLin > 0
    Ee = Emin + ((x >= xLin).*x.^p + (x < xLin).*x*xLin^(p-1)) * (E0 - Emin);
else
    Ee = Emin + x.^p * (E0 - Emin);
end
re = rhoMin + x * (rho0 - rhoMin);
K = sparse(mdl.iK, mdl.jK, reshape(mdl.KE(:) * Ee', [], 1), mdl.ndof, mdl.ndof);
M = sparse(mdl.iK, mdl.jK, reshape(mdl.ME(:) * re', [], 1), mdl.ndof, mdl.ndof);
K = (K + K')/2;  M = (M + M')/2;
end
