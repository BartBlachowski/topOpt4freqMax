% VERIFY_INNER_LOOP  Phase 4 acceptance test for OlhoffApproachExact.
%
% Checks three things:
%
%   A. N = 1 reduction
%      Run inner_loop_mma with N=1 and verify that the resulting drho
%      maximises beta = lambda_1 + fsk(:,1,1)' * drho, which is exactly
%      the simple-eigenvalue increment subproblem.
%      Specifically: at convergence the single cluster constraint is active
%      (beta ≈ lambda_1 + fsk(:,1,1)' * drho) and beta <= lambda_J.
%
%   B. Bimodal case (N = 2): both cluster constraints active at convergence
%      Use the 20x20 square domain with 4-corner pins and uniform rho = 0.5
%      where detect_multiplicity found N = 2 (Phase 3 result).
%      At inner-loop convergence both fval(1) and fval(2) must be within
%      1 % of |lambda_bar| (both constraints effectively active).
%
%   C. No trial eigensolve inside the inner loop
%      The only eigenproblem in inner_loop_mma is the tiny N x N
%      subeigenproblem of F (solved via eig, not eigs/arpack).
%      Verified by inspection: eigs is not called inside inner_loop_mma.

fprintf('==========================================================\n');
fprintf(' OlhoffApproachExact — Phase 4 verification\n');
fprintf('==========================================================\n\n');

all_passed = true;

%% ----------------------------------------------------------------
%  Shared: 40x5 CC beam, random rho, Phase-2 setup
% ----------------------------------------------------------------
E0   = 1e7;  nu = 0.3;  rho0 = 1.0;  t = 1.0;
L    = 8.0;  H  = 1.0;
nelx = 40;   nely = 5;
dx   = L/nelx;  dy = H/nely;
nEl  = nelx*nely;
nDof = 2*(nelx+1)*(nely+1);
penal    = 3.0;
rho_min  = 1e-3;
volfrac  = 0.5;
mass_mode = 'du2007_c1';

rng(42);
rho_beam = rho_min + (1 - rho_min) * rand(nEl, 1);

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

[K0, M0] = assemble_KM_exact(rho_beam, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
Kf = K0(free, free);
Mf = M0(free, free);

opts_eig.tol = 1e-12; opts_eig.maxit = 800;
[V, D] = eigs(Kf, Mf, 3, 'SM', opts_eig);
[lam_sorted, idx] = sort(real(diag(D)));
V = real(V(:, idx));
for jj = 1:3
    v = V(:, jj);
    V(:, jj) = v / sqrt(abs(v' * (Mf * v)));
end
Phi = zeros(nDof, 3);
for jj = 1:3, Phi(free, jj) = V(:, jj); end

rmin_elem = 2.5;
[h_filt, Hs_filt] = build_filter(nelx, nely, rmin_elem);

%% ----------------------------------------------------------------
%  A.  N = 1 reduction
% ----------------------------------------------------------------
fprintf('--- A. N=1 reduction (40x5 CC beam, mode 1) ---\n');

lambda_1   = lam_sorted(1);
lambda_2   = lam_sorted(2);   % J-mode for this single-mode test

% fsk(:,1,1) == dlam_1 (verified in Phase 3 Part C).
fsk1 = compute_generalized_gradients(rho_beam, lambda_1, Phi(:,1), ...
           cMat, Ke_phys, Me_phys, penal, mass_mode);   % nEl x 1 x 1

% Sensitivity-filter each vector before passing to inner loop.
fsk1_filt = apply_sensitivity_filter(fsk1(:,1,1), rho_beam, h_filt, Hs_filt, nely, nelx);
dlam2_filt = apply_sensitivity_filter( ...
    compute_elem_sensitivity(rho_beam, lambda_2, Phi(:,2), ...
        cMat, Ke_phys, Me_phys, free, nDof, penal, mass_mode), ...
    rho_beam, h_filt, Hs_filt, nely, nelx);

fsk1_3d = reshape(fsk1_filt, nEl, 1, 1);

inner_max = 60;
inner_tol = 1e-4;
move_lim  = 0.2;

[drho_N1, beta_N1, hist_N1] = inner_loop_mma(rho_beam, lambda_1, fsk1_3d, ...
    lambda_2, dlam2_filt, volfrac, rho_min, inner_max, inner_tol, move_lim);

% Verify cluster constraint is active at convergence.
active_val_N1 = beta_N1 - lambda_1 - fsk1_filt' * drho_N1;
rho_new_N1    = rho_beam + drho_N1;
vol_frac_N1   = sum(rho_new_N1) / nEl;
% J-mode slack (should be >= 0: beta <= lambda_J-mode).
J_slack_N1    = lambda_2 + dlam2_filt' * drho_N1 - beta_N1;

pass_A1 = abs(active_val_N1) < 1e-3 * lambda_1;
pass_A2 = vol_frac_N1 <= volfrac + 1e-6;
pass_A3 = J_slack_N1 >= -1e-4 * lambda_2;   % J-mode must not be violated

fprintf('  Inner iterations: %d\n', hist_N1.n_iters);
fprintf('  beta_final = %.6g,  lambda_1 = %.6g\n', beta_N1, lambda_1);
fprintf('  Cluster constraint value (active ≈ 0): %.3e   %s\n', ...
    active_val_N1, yesno(pass_A1));
fprintf('  Volume fraction: %.5f <= %.5f   %s\n', vol_frac_N1, volfrac, yesno(pass_A2));
fprintf('  J-mode slack >= 0: %.3e   %s\n', J_slack_N1, yesno(pass_A3));

pass_A = pass_A1 && pass_A2 && pass_A3;
if ~pass_A, all_passed = false; end
fprintf('%s  N=1 reduction\n\n', yesno(pass_A));

%% ----------------------------------------------------------------
%  B.  N = 2 bimodal case: both cluster constraints active
% ----------------------------------------------------------------
fprintf('--- B. N=2 bimodal case (20x20 square, 4-corner pins) ---\n');

nelx_sq = 20;  nely_sq = 20;
L_sq = 1.0;  H_sq = 1.0;
dx_sq = L_sq/nelx_sq;  dy_sq = H_sq/nely_sq;
nEl_sq  = nelx_sq * nely_sq;
nDof_sq = 2*(nelx_sq+1)*(nely_sq+1);

[Ke_s, Me_s] = fe_q4_exact(nu, t, dx_sq, dy_sq);
Ke_sq = E0 * Ke_s;
Me_sq = rho0 * Me_s;

nodeNrs_sq = reshape(1:(nelx_sq+1)*(nely_sq+1), nely_sq+1, nelx_sq+1);
cVec_sq    = reshape(2*nodeNrs_sq(1:nely_sq,1:nelx_sq)+1, nEl_sq, 1);
cMat_sq    = [cVec_sq, cVec_sq+1, cVec_sq+2*nely_sq+2, cVec_sq+2*nely_sq+3, ...
              cVec_sq+2*nely_sq, cVec_sq+2*nely_sq+1, cVec_sq-2, cVec_sq-1];
[Il_sq, Jl_sq] = find(tril(ones(8)));
iK_sq = reshape(cMat_sq(:,Il_sq)', [], 1);
jK_sq = reshape(cMat_sq(:,Jl_sq)', [], 1);
Ke_l_sq = Ke_sq(sub2ind([8,8], Il_sq, Jl_sq));
Me_l_sq = Me_sq(sub2ind([8,8], Il_sq, Jl_sq));

corners_sq = [nodeNrs_sq(1,1), nodeNrs_sq(1,end), ...
              nodeNrs_sq(end,1), nodeNrs_sq(end,end)];
fixed_sq   = sort([2*corners_sq-1, 2*corners_sq]);
free_sq    = setdiff(1:nDof_sq, fixed_sq);

rho_sq   = 0.5 * ones(nEl_sq, 1);
volfrac_sq = 0.5;

[K_sq, M_sq] = assemble_KM_exact(rho_sq, Ke_l_sq, Me_l_sq, iK_sq, jK_sq, ...
                                   nDof_sq, penal, mass_mode);
Kf_sq = K_sq(free_sq, free_sq);
Mf_sq = M_sq(free_sq, free_sq);

opts_sq.tol = 1e-12;  opts_sq.maxit = 800;
[V_sq, D_sq] = eigs(Kf_sq, Mf_sq, 4, 'SM', opts_sq);
[lam_sq, idx_sq] = sort(real(diag(D_sq)));
V_sq = real(V_sq(:, idx_sq));
for jj = 1:4
    v = V_sq(:, jj);
    V_sq(:, jj) = v / sqrt(abs(v' * (Mf_sq * v)));
end
Phi_sq = zeros(nDof_sq, 4);
for jj = 1:4, Phi_sq(free_sq, jj) = V_sq(:, jj); end

% Detect N=2 cluster.
omega_sq = sqrt(max(lam_sq, 0));
mult_tol = 1e-2;
[N_sq, J_sq_idx, ci_sq] = detect_multiplicity(omega_sq, 1, mult_tol);
fprintf('  N = %d (cluster modes %d-%d),  J_idx = %d\n', N_sq, ci_sq(1), ci_sq(end), J_sq_idx);

lambda_bar_sq = mean(lam_sq(ci_sq));
lambda_J_sq   = lam_sq(J_sq_idx);

% Compute and filter fsk for the 2-mode cluster.
Phi_cl_sq = Phi_sq(:, 1:N_sq);   % nDof x N
fsk_sq_raw = compute_generalized_gradients(rho_sq, lambda_bar_sq, Phi_cl_sq, ...
                 cMat_sq, Ke_sq, Me_sq, penal, mass_mode);   % nEl x N x N

rmin_sq = 2.5;
[h_sq, Hs_sq] = build_filter(nelx_sq, nely_sq, rmin_sq);

fsk_sq_filt = zeros(size(fsk_sq_raw));
for s = 1:N_sq
    for k = 1:N_sq
        fsk_sq_filt(:, s, k) = apply_sensitivity_filter( ...
            fsk_sq_raw(:,s,k), rho_sq, h_sq, Hs_sq, nely_sq, nelx_sq);
    end
end

% dlam_J for mode J_sq_idx.
phi_J = Phi_sq(:, J_sq_idx);
dlam_J_sq_raw = compute_elem_sensitivity(rho_sq, lambda_J_sq, phi_J, ...
    cMat_sq, Ke_sq, Me_sq, free_sq, nDof_sq, penal, mass_mode);
dlam_J_sq = apply_sensitivity_filter(dlam_J_sq_raw, rho_sq, h_sq, Hs_sq, ...
    nely_sq, nelx_sq);

inner_max_sq = 200;
inner_tol_sq = 1e-4;
move_lim_sq  = 0.2;

[drho_sq, beta_sq, hist_sq] = inner_loop_mma(rho_sq, lambda_bar_sq, fsk_sq_filt, ...
    lambda_J_sq, dlam_J_sq, volfrac_sq, rho_min, ...
    inner_max_sq, inner_tol_sq, move_lim_sq);

% Rebuild F at convergence to evaluate constraint residuals.
fsk2D_sq = reshape(fsk_sq_filt, nEl_sq, N_sq^2);
F_final  = reshape(fsk2D_sq' * drho_sq, N_sq, N_sq);
mu_final = sort(real(eig(F_final)), 'ascend');

fval1_final = beta_sq - lambda_bar_sq - mu_final(1);
fval2_final = beta_sq - lambda_bar_sq - mu_final(2);
vol_frac_sq = (sum(rho_sq) + sum(drho_sq)) / nEl_sq;

% Both cluster constraints should be active (small non-positive slack).
activity_tol = 0.01 * abs(lambda_bar_sq);   % 1 % of lambda_bar
pass_B1 = abs(fval1_final) < activity_tol;
pass_B2 = abs(fval2_final) < activity_tol;
pass_B3 = vol_frac_sq <= volfrac_sq + 1e-4;
pass_B4 = hist_sq.n_iters < inner_max_sq;   % must have converged

fprintf('  Inner iterations: %d / %d\n', hist_sq.n_iters, inner_max_sq);
fprintf('  beta_final = %.6g,  lambda_bar = %.6g\n', beta_sq, lambda_bar_sq);
fprintf('  mu_1 = %.4g,  mu_2 = %.4g\n', mu_final(1), mu_final(2));
fprintf('  fval cluster 1 (active ≈ 0): %+.3e   %s\n', fval1_final, yesno(pass_B1));
fprintf('  fval cluster 2 (active ≈ 0): %+.3e   %s\n', fval2_final, yesno(pass_B2));
fprintf('  Volume fraction: %.5f <= %.5f   %s\n', vol_frac_sq, volfrac_sq, yesno(pass_B3));
fprintf('  Convergence before max iter:          %s\n', yesno(pass_B4));

pass_B = pass_B1 && pass_B2 && pass_B3 && pass_B4;
if ~pass_B, all_passed = false; end
fprintf('%s  N=2 bimodal: both constraints active\n\n', yesno(pass_B));

%% ----------------------------------------------------------------
%  C.  No trial eigensolve inside inner loop (verified by inspection)
% ----------------------------------------------------------------
fprintf('--- C. No trial eigensolve (eigs) inside inner_loop_mma ---\n');
fprintf('  inner_loop_mma solves only a %d x %d dense subeigenproblem\n', N_sq, N_sq);
fprintf('  via eig(), NOT eigs(). No ARPACK/Krylov call inside the loop.\n');
fprintf('  Verified by code inspection: eigs does not appear in inner_loop_mma.m\n');

% Automated check: grep-equivalent in Matlab.
fid = fopen(which('inner_loop_mma'), 'r');
src = fread(fid, '*char')';
fclose(fid);
has_eigs = ~isempty(strfind(src, 'eigs('));
pass_C = ~has_eigs;
if ~pass_C, all_passed = false; end
fprintf('%s  No eigs() call found in inner_loop_mma.m\n\n', yesno(pass_C));

%% ----------------------------------------------------------------
%  Summary
% ----------------------------------------------------------------
fprintf('==========================================================\n');
fprintf(' Summary\n');
fprintf('==========================================================\n');
fprintf('  A. N=1: cluster constraint active at convergence:  %s\n', yesno(pass_A));
fprintf('  A. N=1: volume constraint satisfied:               %s\n', yesno(pass_A2));
fprintf('  A. N=1: J-mode constraint not violated:            %s\n', yesno(pass_A3));
fprintf('  B. N=2: both cluster constraints active:           %s\n', yesno(pass_B1 && pass_B2));
fprintf('  B. N=2: volume satisfied:                          %s\n', yesno(pass_B3));
fprintf('  B. N=2: converged before max iter:                 %s\n', yesno(pass_B4));
fprintf('  C. No eigs() inside inner loop:                    %s\n', yesno(pass_C));

if all_passed
    fprintf('\nPHASE 4 PASSED\n');
else
    fprintf('\nPHASE 4 FAILED — inspect output above\n');
end
fprintf('==========================================================\n');

%% ----------------------------------------------------------------
function s = yesno(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
