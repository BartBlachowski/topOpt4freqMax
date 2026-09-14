% VERIFY_MULTIPLICITY  Phase 3 acceptance test for OlhoffApproachExact.
%
% Checks four things:
%
%   A. detect_multiplicity  (unit tests on synthetic omega arrays)
%      Three cases: simple eigenvalue, bimodal cluster, cluster at last mode.
%
%   B. compute_generalized_gradients correctness via FD on K and M
%      For a 40x5 CC mesh with random rho, compute fsk analytically and via
%      central-difference perturbation of K and M (no eigensolver re-solve).
%      Tested for s,k in {1,2} (both diagonal and off-diagonal).
%      Relative errors must be < 0.1 %.
%
%   C. N = 1 reduction
%      fsk(:,1,1) must equal compute_elem_sensitivity to machine precision.
%
%   D. Symmetry and off-diagonal check on a symmetric domain
%      Use a square 20x20 domain with 4-corner pin supports (4-fold symmetry).
%      With uniform rho = 0.5, search for a degenerate pair using
%      detect_multiplicity.  If found, verify fsk(:,s,k) = fsk(:,k,s) and
%      report off-diagonal magnitudes to confirm they are non-trivially non-zero.

fprintf('==========================================================\n');
fprintf(' OlhoffApproachExact — Phase 3 verification\n');
fprintf('==========================================================\n\n');

all_passed = true;

%% ------------------------------------------------------------------
%  A.  detect_multiplicity unit tests
% -------------------------------------------------------------------
fprintf('--- A. detect_multiplicity unit tests ---\n');

mult_tol = 1e-3;   % 0.1 % relative tolerance

% Case 1: simple eigenvalue at position 2.
omega1 = [100; 200; 300; 400];
[N1, J1, ci1] = detect_multiplicity(omega1, 2, mult_tol);
pass_A1 = (N1 == 1) && (J1 == 3) && isequal(ci1, 2);
fprintf('  Case 1 (simple at n=2): N=%d J=%d cluster=[%s]  %s\n', ...
    N1, J1, num2str(ci1), yesno(pass_A1));

% Case 2: bimodal cluster at positions 2-3 (0.05 % apart < mult_tol).
omega2 = [100; 200; 200*(1+5e-4); 300];
[N2, J2, ci2] = detect_multiplicity(omega2, 2, mult_tol);
pass_A2 = (N2 == 2) && (J2 == 4) && isequal(ci2, 2:3);
fprintf('  Case 2 (bimodal n=2-3): N=%d J=%d cluster=[%s]  %s\n', ...
    N2, J2, num2str(ci2), yesno(pass_A2));

% Case 3: cluster extends to last mode (J_idx should be 0).
omega3 = [100; 200; 200*(1+5e-4)];
[N3, J3, ci3] = detect_multiplicity(omega3, 2, mult_tol);
pass_A3 = (N3 == 2) && (J3 == 0) && isequal(ci3, 2:3);
fprintf('  Case 3 (cluster at end): N=%d J=%d cluster=[%s]  %s\n', ...
    N3, J3, num2str(ci3), yesno(pass_A3));

pass_A = pass_A1 && pass_A2 && pass_A3;
if ~pass_A, all_passed = false; end
fprintf('%s  detect_multiplicity unit tests\n\n', yesno(pass_A));

%% ------------------------------------------------------------------
%  Problem setup: 40x5 CC beam, random rho, 4 modes
% -------------------------------------------------------------------
E0   = 1e7;  nu = 0.3;  rho0 = 1.0;  t = 1.0;
L    = 8.0;  H  = 1.0;
nelx = 40;   nely = 5;
dx   = L/nelx;  dy = H/nely;
nEl  = nelx*nely;
nDof = 2*(nelx+1)*(nely+1);
penal    = 3.0;
rho_min  = 1e-3;
mass_mode = 'du2007_c1';

rng(42);
rho = rho_min + (1 - rho_min) * rand(nEl, 1);

[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
Ke_phys = E0  * Ke_star;
Me_phys = rho0 * Me_star;

nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec    = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat    = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
           cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

fixed = build_supports_exact('CC', nodeNrs);
free  = setdiff(1:nDof, fixed);

[K0, M0] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
Kf = K0(free, free);
Mf = M0(free, free);

opts_eig.tol   = 1e-12;
opts_eig.maxit = 800;
[V, D] = eigs(Kf, Mf, 4, 'SM', opts_eig);
[lam_sorted, idx] = sort(real(diag(D)));
V = real(V(:, idx));

% M-orthonormalize (eigs should already do this, but enforce numerically).
for jj = 1:4
    v = V(:, jj);
    v = v / sqrt(abs(v' * (Mf * v)));
    V(:, jj) = v;
end

Phi = zeros(nDof, 4);
for jj = 1:4
    Phi(free, jj) = V(:, jj);
end

%% ------------------------------------------------------------------
%  B.  FD check of f_sk via K and M perturbation
% -------------------------------------------------------------------
fprintf('--- B. FD check of f_sk (perturb K and M, no eigensolver) ---\n');

% Use modes 1 and 2 as a synthetic N=2 cluster.
% lambda_bar = average of the two eigenvalues (or just use lam_1 for N=1 path).
lambda_bar = 0.5 * (lam_sorted(1) + lam_sorted(2));
Phi12      = Phi(:, 1:2);   % nDof x 2

fsk_anal = compute_generalized_gradients(rho, lambda_bar, Phi12, ...
               cMat, Ke_phys, Me_phys, penal, mass_mode);  % nEl x 2 x 2

h_fd     = 1e-6;
n_test   = 10;
rng(7);
test_elems = sort(randperm(nEl, n_test));

fprintf('%-6s  %-4s  %-4s  %14s  %14s  %12s  %6s\n', ...
    'elem', 's', 'k', 'analytical', 'FD', 'rel err', 'pass?');

pass_B = true;
for ei = 1:n_test
    e = test_elems(ei);

    % Perturb rho at element e.
    rho_p = rho;  rho_p(e) = rho_p(e) + h_fd;
    rho_m = rho;  rho_m(e) = rho_m(e) - h_fd;

    [Kp, Mp] = assemble_KM_exact(rho_p, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
    [Km, Mm] = assemble_KM_exact(rho_m, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);

    for s = 1:2
        phi_s = Phi12(:, s);
        for k = 1:2
            phi_k = Phi12(:, k);

            % Central-difference inner product.
            dK_val = (phi_s' * Kp * phi_k - phi_s' * Km * phi_k) / (2*h_fd);
            dM_val = (phi_s' * Mp * phi_k - phi_s' * Mm * phi_k) / (2*h_fd);
            fsk_fd = dK_val - lambda_bar * dM_val;

            fsk_an = fsk_anal(e, s, k);
            abs_err = abs(fsk_fd - fsk_an);
            if abs(fsk_an) > 1e-12
                rel_err = abs_err / abs(fsk_an);
            else
                rel_err = abs_err;
            end

            % Absolute fallback: 1e-4 * max over all elements
            atol = 1e-4 * max(abs(fsk_anal(:, s, k)));
            pass_sk = rel_err < 1e-3 || abs_err < atol;
            if ~pass_sk, pass_B = false; all_passed = false; end

            fprintf('%-6d  %-4d  %-4d  %14.6g  %14.6g  %12.2e  %s\n', ...
                e, s, k, fsk_an, fsk_fd, rel_err, yesno(pass_sk));
        end
    end
end
fprintf('%s  f_sk FD check\n\n', yesno(pass_B));

%% ------------------------------------------------------------------
%  C.  N = 1 reduction: fsk(:,1,1) == compute_elem_sensitivity
% -------------------------------------------------------------------
fprintf('--- C. N=1 reduction: fsk(:,1,1) vs compute_elem_sensitivity ---\n');

lam_1  = lam_sorted(1);
dlam_cs = compute_elem_sensitivity(rho, lam_1, Phi(:,1), ...
              cMat, Ke_phys, Me_phys, free, nDof, penal, mass_mode);

fsk_N1 = compute_generalized_gradients(rho, lam_1, Phi(:,1), ...
              cMat, Ke_phys, Me_phys, penal, mass_mode);   % nEl x 1 x 1

max_diff_C = max(abs(fsk_N1(:,1,1) - dlam_cs));
pass_C = max_diff_C < 1e-14 * max(abs(dlam_cs));
if ~pass_C, all_passed = false; end

fprintf('  max |fsk(:,1,1) - compute_elem_sensitivity| = %.2e\n', max_diff_C);
fprintf('%s  N=1 reduction to compute_elem_sensitivity\n\n', yesno(pass_C));

%% ------------------------------------------------------------------
%  D.  Symmetry check and off-diagonal terms on a symmetric square domain
% -------------------------------------------------------------------
fprintf('--- D. Symmetry fsk(:,s,k)=fsk(:,k,s), square 20x20 domain ---\n');

% ---- Symmetry check on the existing 40x5 data (always available) --------
fsk_anal_full = compute_generalized_gradients(rho, lambda_bar, Phi12, ...
                    cMat, Ke_phys, Me_phys, penal, mass_mode);

sym_diff = max(abs(fsk_anal_full(:,1,2) - fsk_anal_full(:,2,1)));
pass_sym = sym_diff < 1e-11 * max(abs(fsk_anal_full(:,1,2)));
if ~pass_sym, all_passed = false; end
fprintf('  max |fsk(:,1,2) - fsk(:,2,1)| = %.2e   %s\n', sym_diff, yesno(pass_sym));

% ---- Off-diagonal terms are non-trivially non-zero ---------------------
offdiag_max = max(abs(fsk_anal_full(:,1,2)));
diag_max    = max(abs(fsk_anal_full(:,1,1)));
pass_offdiag = offdiag_max > 1e-10 * diag_max;
if ~pass_offdiag, all_passed = false; end
fprintf('  max|fsk(:,1,2)| = %.4g,  max|fsk(:,1,1)| = %.4g  (ratio = %.2e)  %s\n', ...
    offdiag_max, diag_max, offdiag_max/diag_max, yesno(pass_offdiag));

% ---- Square domain: search for genuine multiplicity --------------------
fprintf('\n  Searching for genuine multiplicity on 20x20 square domain ...\n');

nelx_sq = 20; nely_sq = 20;
L_sq = 1.0; H_sq = 1.0;
dx_sq = L_sq/nelx_sq; dy_sq = H_sq/nely_sq;
nEl_sq  = nelx_sq*nely_sq;
nDof_sq = 2*(nelx_sq+1)*(nely_sq+1);

[Ke_star_sq, Me_star_sq] = fe_q4_exact(nu, t, dx_sq, dy_sq);
Ke_phys_sq = E0  * Ke_star_sq;
Me_phys_sq = rho0 * Me_star_sq;

nodeNrs_sq = reshape(1:(nelx_sq+1)*(nely_sq+1), nely_sq+1, nelx_sq+1);
cVec_sq    = reshape(2*nodeNrs_sq(1:nely_sq,1:nelx_sq)+1, nEl_sq, 1);
cMat_sq    = [cVec_sq, cVec_sq+1, cVec_sq+2*nely_sq+2, cVec_sq+2*nely_sq+3, ...
              cVec_sq+2*nely_sq, cVec_sq+2*nely_sq+1, cVec_sq-2, cVec_sq-1];

[Il_sq, Jl_sq] = find(tril(ones(8)));
iK_sq = reshape(cMat_sq(:,Il_sq)', [], 1);
jK_sq = reshape(cMat_sq(:,Jl_sq)', [], 1);
Ke_l_sq = Ke_phys_sq(sub2ind([8,8], Il_sq, Jl_sq));
Me_l_sq = Me_phys_sq(sub2ind([8,8], Il_sq, Jl_sq));

% 4-corner pin supports: fix ux and uy at each corner node.
corners_sq = [nodeNrs_sq(1,1), nodeNrs_sq(1,end), ...
              nodeNrs_sq(end,1), nodeNrs_sq(end,end)];
fixed_sq   = sort([2*corners_sq-1, 2*corners_sq]);
free_sq    = setdiff(1:nDof_sq, fixed_sq);

rho_sq = 0.5 * ones(nEl_sq, 1);
[K_sq, M_sq] = assemble_KM_exact(rho_sq, Ke_l_sq, Me_l_sq, iK_sq, jK_sq, ...
                                  nDof_sq, penal, mass_mode);
Kf_sq = K_sq(free_sq, free_sq);
Mf_sq = M_sq(free_sq, free_sq);

opts_sq.tol = 1e-12; opts_sq.maxit = 800;
[V_sq, D_sq] = eigs(Kf_sq, Mf_sq, 8, 'SM', opts_sq);
[lam_sq, idx_sq] = sort(real(diag(D_sq)));
V_sq = real(V_sq(:, idx_sq));
omega_sq = sqrt(max(lam_sq, 0));

fprintf('  First 8 eigenfrequencies (rad/s):\n  ');
fprintf('  %8.3f', omega_sq);
fprintf('\n');

% Detect multiplicity of mode 1 with increasing tolerances.
mult_tol_sq = 1e-2;   % 1 % tolerance for near-degenerate detection
[N_sq, J_sq, ci_sq] = detect_multiplicity(omega_sq, 1, mult_tol_sq);
fprintf('  detect_multiplicity(omega, 1, %.0e): N=%d, J_idx=%d, cluster=[%s]\n', ...
    mult_tol_sq, N_sq, J_sq, num2str(ci_sq));

if N_sq >= 2
    % M-orthonormalize cluster eigenvectors.
    for jj = 1:N_sq
        v = V_sq(:, jj);
        v = v / sqrt(abs(v' * (Mf_sq * v)));
        V_sq(:, jj) = v;
    end
    Phi_sq = zeros(nDof_sq, N_sq);
    for jj = 1:N_sq
        Phi_sq(free_sq, jj) = V_sq(:, jj);
    end

    lam_bar_sq = mean(lam_sq(ci_sq));
    fsk_sq = compute_generalized_gradients(rho_sq, lam_bar_sq, Phi_sq(:,1:N_sq), ...
                 cMat_sq, Ke_phys_sq, Me_phys_sq, penal, mass_mode);

    fprintf('  N=%d cluster confirmed.  Off-diagonal fsk norms:\n', N_sq);
    for s = 1:N_sq
        for k = 1:N_sq
            v = fsk_sq(:, s, k);
            fprintf('    |fsk(:,%d,%d)| = %g\n', s, k, norm(v));
        end
    end
    % Symmetry for the square case.
    if N_sq >= 2
        sym_sq = max(abs(fsk_sq(:,1,2) - fsk_sq(:,2,1)));
        fprintf('  max|fsk(:,1,2) - fsk(:,2,1)| = %.2e\n', sym_sq);
    end
else
    fprintf('  No multiplicity found with tol=%.0e on 20x20 domain.\n', mult_tol_sq);
    fprintf('  (Off-diagonal fsk is verified analytically in Part B above.)\n');
end

%% ------------------------------------------------------------------
%  Summary
% -------------------------------------------------------------------
fprintf('\n==========================================================\n');
fprintf(' Summary\n');
fprintf('==========================================================\n');
fprintf('  A. detect_multiplicity unit tests:         %s\n', yesno(pass_A));
fprintf('  B. f_sk FD check (perturb K,M):            %s\n', yesno(pass_B));
fprintf('  C. N=1 reduction to elem sensitivity:      %s\n', yesno(pass_C));
fprintf('  D. fsk symmetry fsk(:,s,k)=fsk(:,k,s):    %s\n', yesno(pass_sym));
fprintf('  D. Off-diagonal terms non-trivially ≠ 0:  %s\n', yesno(pass_offdiag));

if all_passed
    fprintf('\nPHASE 3 PASSED\n');
else
    fprintf('\nPHASE 3 FAILED — inspect output above\n');
end
fprintf('==========================================================\n');

%% ------------------------------------------------------------------
function s = yesno(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
