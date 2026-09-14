function fsk = compute_generalized_gradients(rho, lambda_bar, Phi_cluster, ...
                                              cMat, Ke_phys, Me_phys, penal, mass_mode)
% COMPUTE_GENERALIZED_GRADIENTS  Generalized gradient array for an N-fold cluster.
%
%   fsk = compute_generalized_gradients(rho, lambda_bar, Phi_cluster,
%                                       cMat, Ke_phys, Me_phys, penal, mass_mode)
%
%   For an N-fold cluster of eigenfrequencies with common eigenvalue
%   lambda_bar = omega_cluster^2, computes the N x N family of
%   NE-dimensional gradient vectors defined by Du & Olhoff (2007) Eq. 19:
%
%       f_sk[e] = phi_s^T (K'_e - lambda_bar * M'_e) phi_k
%
%   where K'_e = dK_e/drho_e = penal*rho_e^(penal-1)*Ke_phys  (Eq. 1)
%         M'_e = dM_e/drho_e = dm(rho_e)*Me_phys              (Eq. 2/4a/4b)
%
%   For N = 1 the result reduces to the usual eigenvalue sensitivity:
%       fsk(:,1,1) == compute_elem_sensitivity(rho, lambda_bar, Phi_cluster, ...)
%
%   By symmetry of K and M: fsk(:,s,k) == fsk(:,k,s).
%
%   In the outer loop of the paper's algorithm (Fig. 1), these vectors are
%   assembled into the N x N matrix
%
%       F(s,k) = sum_e fsk(e,s,k) * Delta_rho_e  (Eq. 25 inner-loop input)
%
%   which is then solved as a generalized eigenproblem to obtain the
%   constraint normals for the MMA subproblem.
%
%   Inputs:
%     rho          nEl x 1     physical density (must satisfy rho >= rho_min)
%     lambda_bar   scalar      cluster eigenvalue (average of cluster lam_j)
%     Phi_cluster  nDof x N    M-orthonormalized cluster eigenvectors
%                              (phi_i^T M phi_j = delta_ij; zeros at fixed DOFs)
%     cMat         nEl x 8     element DOF connectivity (1-based global indices)
%     Ke_phys      8 x 8       physical stiffness kernel (= E0 * Ke_star)
%     Me_phys      8 x 8       physical mass kernel (= rho0 * Me_star)
%     penal        scalar      SIMP penalization exponent p
%     mass_mode    string      mass interpolation mode (see mass_interp)
%
%   Output:
%     fsk   nEl x N x N   3D array;  fsk(e,s,k) = f_{sk} for element e
%
%   To build F(s,k) for a perturbation Delta_rho (nEl x 1):
%       fsk2D = reshape(fsk, nEl, N*N);
%       F     = reshape(fsk2D' * Delta_rho, N, N);
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110, Eq. 18-19.

rho = rho(:);
nEl = numel(rho);
N   = size(Phi_cluster, 2);

% Stiffness derivative coefficient: p * rho_e^(p-1)  (nEl x 1)
dke = penal * rho .^ (penal - 1);

% Mass derivative coefficient: dm(rho_e, mode)  (nEl x 1)
[~, dme] = mass_interp(rho, mass_mode);

fsk = zeros(nEl, N, N);

for s = 1:N
    phi_s = Phi_cluster(:, s);   % nDof x 1
    pe_s  = phi_s(cMat);         % nEl  x 8  (element DOF values)

    % Pre-multiply once: pe_s * Ke_phys  (nEl x 8),  pe_s * Me_phys  (nEl x 8)
    peKe  = pe_s * Ke_phys;      % nEl x 8
    peMe  = pe_s * Me_phys;      % nEl x 8

    for k = 1:N
        phi_k = Phi_cluster(:, k);   % nDof x 1
        pe_k  = phi_k(cMat);         % nEl  x 8

        % phi_s^T K'_e phi_k  = dke_e * (pe_s * Ke_phys) .* pe_k  summed over 8 DOFs
        Kterm = sum(peKe .* pe_k, 2);    % nEl x 1

        % phi_s^T M'_e phi_k  = dme_e * (pe_s * Me_phys) .* pe_k  summed over 8 DOFs
        Mterm = sum(peMe .* pe_k, 2);    % nEl x 1

        fsk(:, s, k) = dke .* Kterm - lambda_bar * dme .* Mterm;
    end
end
end
