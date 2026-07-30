function Fe = generalized_gradients(rho, lam_ref, Phi_c, cMat, Ke_phys, Me_phys, ...
                                    penal, mass_mode, mass_q)
% GENERALIZED_GRADIENTS  Generalized gradient vectors f_sk of an N-fold cluster.
%
%   Fe = generalized_gradients(rho, lam_ref, Phi_c, cMat, Ke_phys, Me_phys, ...
%                              penal, mass_mode, mass_q)
%
%   Olhoff & Du (2014) Eq. (13):
%
%       f_sk = { phi_s' (K'_rho1 - lam~ M'_rho1) phi_k , ... ,
%                phi_s' (K'_rhoNE - lam~ M'_rhoNE) phi_k }'
%
%   with, from Eq. (5),
%       K'_e = dK_e/drho_e = p * rho_e^(p-1) * Ke_phys
%       M'_e = dM_e/drho_e = dm(rho_e)       * Me_phys
%
%   Output is returned as the element-indexed 3D array
%
%       Fe(e, s, k) = f_sk[e]
%
%   which is the natural layout for the sub-eigenvalue problem (12): for a
%   design increment Delta_rho the N x N matrix of Eq. (12) is
%
%       F(Delta_rho)(s,k) = f_sk' * Delta_rho
%                         = sum_e Fe(e,s,k) * Delta_rho(e)
%
%   Properties enforced/relied upon
%     * Fe(:,s,k) == Fe(:,k,s) exactly (K and M are symmetric, Eq. (13) note).
%       The array is filled symmetrically so that F is symmetric to the last
%       bit, which the sub-eigenvalue solvers require.
%     * For N = 1, Fe(:,1,1) == compute_elem_sensitivity(...) with lam_j = lam_ref
%       (Eqs. (14), (15): f_nn = grad lam_n).
%     * The eigenvalues of F are invariant under an orthogonal change of basis
%       Phi_c -> Phi_c*Q (F -> Q'FQ), so the whole increment subproblem is
%       independent of which basis of the invariant subspace the eigensolver
%       happened to return.  verify/v_basis_invariance.m tests this.
%
%   Inputs
%     rho        nEl x 1     physical density in [rho_min, 1]
%     lam_ref    scalar      lam~, the cluster reference eigenvalue
%     Phi_c      nDof x N    M-orthonormalized cluster eigenvectors, defined on
%                            ALL DOFs (zeros at fixed DOFs)
%     cMat       nEl x 8     element DOF connectivity, 1-based
%     Ke_phys    8 x 8       E0 * Ke_star
%     Me_phys    8 x 8       rho0 * Me_star
%     penal      scalar      SIMP exponent p
%     mass_mode  char        mass interpolation mode (see mass_interp)
%     mass_q     scalar      mass exponent q for 'olhoff2014_pow' (default 1)
%
%   Output
%     Fe         nEl x N x N
%
%   Reference: Olhoff & Du (2014), Eqs. (5), (12), (13), (18).

if nargin < 9 || isempty(mass_q), mass_q = 1; end

rho = rho(:);
nEl = numel(rho);
N   = size(Phi_c, 2);

dke      = penal * rho .^ (penal - 1);              % dK_e coefficient
[~, dme] = mass_interp(rho, mass_mode, mass_q);     % dM_e coefficient

Fe = zeros(nEl, N, N);

% Pre-extract element DOF values and the Ke/Me products once per mode.
peK = cell(N,1);
peM = cell(N,1);
pe  = cell(N,1);
for s = 1:N
    pe{s}  = Phi_c(cMat(:), s);
    pe{s}  = reshape(pe{s}, nEl, 8);
    peK{s} = pe{s} * Ke_phys;
    peM{s} = pe{s} * Me_phys;
end

for s = 1:N
    for k = s:N
        Kterm = sum(peK{s} .* pe{k}, 2);            % phi_s' K'_e phi_k / dke
        Mterm = sum(peM{s} .* pe{k}, 2);            % phi_s' M'_e phi_k / dme
        f     = dke .* Kterm - lam_ref * dme .* Mterm;
        Fe(:, s, k) = f;
        Fe(:, k, s) = f;                            % enforce exact symmetry
    end
end
end
