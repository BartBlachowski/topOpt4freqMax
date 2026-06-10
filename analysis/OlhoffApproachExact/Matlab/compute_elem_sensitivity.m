function dlam = compute_elem_sensitivity(rho, lam_j, phi_j, ...
                                         cMat, Ke_phys, Me_phys, ...
                                         free, nDof, penal, mass_mode)
% COMPUTE_ELEM_SENSITIVITY  Raw element eigenvalue sensitivities.
%
%   dlam = compute_elem_sensitivity(rho, lam_j, phi_j, cMat, Ke_phys,
%                                   Me_phys, free, nDof, penal, mass_mode)
%
%   Computes  d lambda_j / d rho_e  for all elements simultaneously.
%   Formula from Du & Olhoff (2007) Eq. 10/11, with paper interpolation:
%
%       d lambda_j       d K_e              d M_e
%       ----------  =  phi_j^T -----  phi_j - lambda_j * phi_j^T ----- phi_j
%        d rho_e          d rho_e              d rho_e
%
%   With paper Eq. 1 and Eq. 2/4/4a/4b:
%       d K_e / d rho_e = p * rho_e^(p-1) * Ke_phys
%       d M_e / d rho_e = dm(rho_e)        * Me_phys
%
%   In vectorized form (pe = phi_j(cMat)):
%       dK_term_e = p * rho_e^(p-1) * sum((pe * Ke_phys) .* pe, 2)
%       dM_term_e = dm_e             * sum((pe * Me_phys) .* pe, 2)
%       dlam_e    = dK_term_e - lambda_j * dM_term_e
%
%   Inputs:
%     rho       nEl x 1   physical density (clamped to [rho_min, 1])
%     lam_j     scalar    eigenvalue lambda_j = omega_j^2
%     phi_j     nDof x 1  M-normalized eigenvector (phi^T M phi = 1).
%                         Must be defined over ALL DOFs (zeros at fixed DOFs).
%     cMat      nEl x 8   element DOF connectivity (global, 1-based)
%     Ke_phys   8 x 8     unit stiffness matrix scaled by E0 (= E0 * Ke_star)
%     Me_phys   8 x 8     unit mass matrix scaled by rho0 (= rho0 * Me_star)
%     free      nFree x 1 free DOF indices (1-based)
%     nDof      scalar    total DOFs
%     penal     scalar    SIMP stiffness penalization power p
%     mass_mode string    mass interpolation mode (see mass_interp)
%
%   Output:
%     dlam      nEl x 1   d lambda_j / d rho_e  (unfiltered)
%
%   The caller should pass through apply_sensitivity_filter before using dlam
%   in the MMA subproblem.
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110, Eq. 10/14.

rho = rho(:);
nEl = numel(rho);

% Element DOF values of phi_j  (nEl x 8).
pe = phi_j(cMat);

% Stiffness term: phi_e^T Ke_phys phi_e  for each element.
Kphi2 = sum((pe * Ke_phys) .* pe, 2);   % nEl x 1

% Mass term: phi_e^T Me_phys phi_e  for each element.
Mphi2 = sum((pe * Me_phys) .* pe, 2);   % nEl x 1

% d K_e / d rho_e  scalar coefficient: p * rho_e^(p-1).
dke = penal * rho .^ (penal - 1);

% d M_e / d rho_e  scalar coefficient from mass interpolation derivative.
[~, dme] = mass_interp(rho, mass_mode);

dlam = dke .* Kphi2 - lam_j * dme .* Mphi2;
end
