function F = genGrad(mdl, rho, p, massInterp, Phi, lamTilde, idx)
%GENGRAD  Generalized gradient vectors f_sk of Du & Olhoff (2007) eq. (19).
%
%   F = GENGRAD(mdl,rho,p,massInterp,Phi,lamTilde,idx)
%
%   f_sk,e = phi_s^T ( dK/drho_e  -  lamTilde * dM/drho_e ) phi_k
%          = p*rho_e^(p-1) * (phi_s^e)' K0 (phi_k^e)
%            - lamTilde * g'(rho_e) * (phi_s^e)' M0 (phi_k^e)
%
%   Phi      : reduced-dof modes (columns), M-orthonormal
%   idx      : vector of mode indices (columns of Phi) taking part
%   lamTilde : the eigenvalue to use in (19).  For an N-fold eigenvalue this
%              is the common value; for a simple mode it is that mode's own
%              eigenvalue.  May be scalar or one entry per pair (see below).
%
%   Returns F: NE x numel(idx) x numel(idx) array with F(:,s,k) = f_sk.
%   f_sk = f_ks is enforced by construction.

nI = numel(idx);
NE = mdl.nele;

% expand reduced modes to full dof vectors, then to element dof blocks
Ufull = zeros(mdl.ndof, nI);
Ufull(mdl.free,:) = Phi(:, idx);

[~, dg] = massScale(rho, massInterp);
sK = p * rho(:).^(p-1);
sM = lamTilde * dg(:);

F = zeros(NE, nI, nI);
Ue = cell(nI,1);
for s = 1:nI
    Ue{s} = reshape(Ufull(mdl.edofMat(:), s), NE, 8);   % NE x 8
end
K0 = mdl.K0;  M0 = mdl.M0;
for s = 1:nI
    UsK = Ue{s}*K0;
    UsM = Ue{s}*M0;
    for k = s:nI
        qK = sum(UsK .* Ue{k}, 2);
        qM = sum(UsM .* Ue{k}, 2);
        f  = sK.*qK - sM.*qM;
        F(:,s,k) = f;
        F(:,k,s) = f;
    end
end
end
