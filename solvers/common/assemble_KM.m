function [K, M, Ee, re] = assemble_KM(mdl, x, E0, Emin, rho0, rhoMin, p, xLin)
%ASSEMBLE_KM  Global stiffness and mass (all dofs) for SIMP stiffness
%   E(x) = Emin + x^p (E0 - Emin) and linear mass rho(x) = rhoMin + x (rho0 - rhoMin).
%   With xLin > 0 the stiffness law is linear below xLin, x*xLin^(p-1)
%   (Pedersen 2000; matched settings of Table 4 only).
%   K and M are assembled from the lower triangles of the element matrices (36 instead
%   of 64 triplets per element, as in top99neo and in the Yuksel-Yilmaz port) and then
%   completed by symmetry.  The matrices equal those of the full 64-triplet assembly to
%   round-off; MATLAB sparse() is markedly faster with the shorter triplet list on fine
%   meshes (about 5x per K+M assembly at 720x90 and 800x100 on Apple silicon).
if nargin < 7 || isempty(p), p = 3; end
if nargin < 8, xLin = 0; end
persistent key I J m
k = [mdl.nelx mdl.nely mdl.ndof size(mdl.edofMat, 1)];
if isempty(key) || ~isequal(key, k)
    [jj, ii] = meshgrid(1:8);  m = ii >= jj;  E = mdl.edofMat;
    iL = reshape(E(:, ii(m))', [], 1);  jL = reshape(E(:, jj(m))', [], 1);
    I = max(iL, jL);  J = min(iL, jL);  key = k;
end
x = x(:);
if xLin > 0
    Ee = Emin + ((x >= xLin).*x.^p + (x < xLin).*x*xLin^(p-1)) * (E0 - Emin);
else
    Ee = Emin + x.^p * (E0 - Emin);
end
re = rhoMin + x * (rho0 - rhoMin);
n = mdl.ndof;
KL = sparse(I, J, reshape(mdl.KE(m) * Ee', [], 1), n, n);
ML = sparse(I, J, reshape(mdl.ME(m) * re', [], 1), n, n);
K = KL + KL' - diag(diag(KL));  M = ML + ML' - diag(diag(ML));
end
