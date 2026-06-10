function [K, M] = assemble_KM_exact(rho, Ke_star_l, Me_star_l, iK, jK, nDof, penal, mass_mode)
% ASSEMBLE_KM_EXACT  Paper-faithful global stiffness and mass assembly.
%
%   [K, M] = assemble_KM_exact(rho, Ke_star_l, Me_star_l, iK, jK, nDof, penal, mass_mode)
%
%   Implements Du & Olhoff (2007) material interpolation exactly:
%       K_e = rho_e^p        * Ke_star     (Eq. 1, no additive E_min)
%       M_e = m(rho_e, mode) * Me_star     (Eq. 2 / 4 / 4a / 4b)
%
%   Inputs:
%     rho         nEl x 1  physical density, clamped to [rho_min, 1] by caller
%     Ke_star_l   lower-triangular entries of the unit stiffness matrix Ke_star
%     Me_star_l   lower-triangular entries of the unit mass matrix Me_star
%     iK, jK      sparse row/col pattern for symmetric assembly
%     nDof        total number of degrees of freedom
%     penal       SIMP stiffness penalization power p  (paper Eq. 1)
%     mass_mode   string passed to mass_interp
%                 ('linear', 'du2007_step', 'du2007_c0', 'du2007_c1')
%
%   Note: the lower-triangular entries and sparse pattern (iK, jK) must be
%   computed once from the mesh connectivity cMat and passed to every call:
%
%       [Il, Jl] = find(tril(ones(8)));
%       Ke_star_l = Ke_star(sub2ind([8,8], Il, Jl));
%       Me_star_l = Me_star(sub2ind([8,8], Il, Jl));
%       iK = reshape(cMat(:, Il)', [], 1);
%       jK = reshape(cMat(:, Jl)', [], 1);
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110.

rho = rho(:);
Ee  = rho .^ penal;                       % paper Eq. 1: K_e = rho^p Ke*
re  = mass_interp(rho, mass_mode);        % paper Eq. 2/4/4a/4b

K = sparse(iK, jK, kron(Ee, Ke_star_l), nDof, nDof);
M = sparse(iK, jK, kron(re, Me_star_l), nDof, nDof);
K = K + K' - diag(diag(K));
M = M + M' - diag(diag(M));
end
