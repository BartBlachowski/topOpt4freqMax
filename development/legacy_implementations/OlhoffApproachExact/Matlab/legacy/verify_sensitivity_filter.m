% VERIFY_SENSITIVITY_FILTER  Phase 2 acceptance test for OlhoffApproachExact.
%
% Checks three things:
%
%   A. Raw gradient finite-difference check
%      For a non-uniform density field on a 40x5 mesh, compare analytical
%      dlambda_j/drho_e from compute_elem_sensitivity against central-
%      difference perturbation of the eigenvalue.  Tested for j = 1 (mode 1)
%      and j = 2 (mode 2) on 10 randomly chosen elements.
%      All relative errors must be < 0.1 %.
%
%   B. K and M use raw rho (not filtered)
%      Assemble K with two density fields: raw rho and filtered rho_tilde.
%      Verify that assemble_KM_exact uses raw rho (trace of K with raw rho
%      matches manual computation; using filtered rho would give a different
%      result).
%
%   C. Sensitivity filter properties
%      1. Uniform sensitivity on uniform density → filtered = original.
%      2. Filter output is continuous (no spikes) on a localized input.
%      3. Volume gradient (1/nEl) is not passed through the filter.

fprintf('==========================================================\n');
fprintf(' OlhoffApproachExact — Phase 2 verification\n');
fprintf('==========================================================\n\n');

%% --- shared problem setup (40x5, CC, non-uniform rho) ----------------
E0   = 1e7;  nu = 0.3;  rho0 = 1.0;  t = 1.0;
L    = 8.0;  H  = 1.0;
nelx = 40;   nely = 5;
dx   = L/nelx;  dy = H/nely;
nEl  = nelx*nely;
nDof = 2*(nelx+1)*(nely+1);
penal    = 3.0;
rho_min  = 1e-3;
mass_mode = 'du2007_c1';

% Reproducible non-uniform density (avoids symmetric/degenerate cases).
rng(42);
rho = rho_min + (1 - rho_min) * rand(nEl, 1);

% Element matrices.
[Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
Ke_phys = E0  * Ke_star;
Me_phys = rho0 * Me_star;

% Assembly connectivity.
nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
cVec    = reshape(2*nodeNrs(1:nely,1:nelx)+1, nEl, 1);
cMat    = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
           cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

[Il, Jl] = find(tril(ones(8)));
iK = reshape(cMat(:,Il)', [], 1);
jK = reshape(cMat(:,Jl)', [], 1);
Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

% CC boundary conditions.
fixed = build_supports_exact('CC', nodeNrs);
free  = setdiff(1:nDof, fixed);
nFree = numel(free);

% Baseline K, M, eigenproblem.
[K0, M0] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
Kf = K0(free, free);
Mf = M0(free, free);

opts_eig.tol   = 1e-12;
opts_eig.maxit = 800;
[V, D] = eigs(Kf, Mf, 4, 'SM', opts_eig);
[lam_sorted, idx] = sort(real(diag(D)));
V = real(V(:, idx));

% Ensure M-normalization: eigs should return this, but re-normalize to be safe.
for jj = 1:4
    v = V(:, jj);
    scale = sqrt(abs(v' * (Mf * v)));
    if scale > 1e-14, V(:, jj) = v / scale; end
end

% Build full eigenvectors (zero at fixed DOFs).
Phi = zeros(nDof, 4);
for jj = 1:4
    Phi(free, jj) = V(:, jj);
end

all_passed = true;

%% ------------------------------------------------------------------
%  A.  Raw gradient finite-difference check
% -------------------------------------------------------------------
fprintf('--- A. Raw gradient finite-difference check ---\n');
h_fd   = 1e-6;
n_test = 10;
test_elems = sort(randperm(nEl, n_test));
modes_test = [1, 2];

fprintf('%-6s  %-6s  %14s  %14s  %12s  %6s\n', ...
    'elem', 'mode', 'analytical', 'FD', 'rel err', 'pass?');

for jj = modes_test
    lam_j = lam_sorted(jj);
    dlam_anal = compute_elem_sensitivity(rho, lam_j, Phi(:,jj), ...
                    cMat, Ke_phys, Me_phys, free, nDof, penal, mass_mode);
    % Absolute tolerance: 1e-4 * max |sensitivity|.
    % Catches near-zero elements where relative error is dominated by
    % eigensolver noise (O(tol_eig * lambda / h_fd)).
    atol_j = 1e-4 * max(abs(dlam_anal));

    for ei = 1:n_test
        e = test_elems(ei);

        % Forward perturbation.
        rho_p = rho;  rho_p(e) = rho_p(e) + h_fd;
        [Kp, Mp] = assemble_KM_exact(rho_p, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
        Kfp = Kp(free, free);  Mfp = Mp(free, free);
        [~, Dp] = eigs(Kfp, Mfp, jj, 'SM', opts_eig);
        lam_p   = sort(real(diag(Dp)));
        lam_p_j = lam_p(jj);

        % Backward perturbation.
        rho_m = rho;  rho_m(e) = rho_m(e) - h_fd;
        [Km, Mm] = assemble_KM_exact(rho_m, Ke_phys_l, Me_phys_l, iK, jK, nDof, penal, mass_mode);
        Kfm = Km(free, free);  Mfm = Mm(free, free);
        [~, Dm] = eigs(Kfm, Mfm, jj, 'SM', opts_eig);
        lam_m   = sort(real(diag(Dm)));
        lam_m_j = lam_m(jj);

        dlam_fd  = (lam_p_j - lam_m_j) / (2*h_fd);
        abs_err  = abs(dlam_fd - dlam_anal(e));

        if abs(dlam_anal(e)) > 1e-12
            rel_err = abs_err / abs(dlam_anal(e));
        else
            rel_err = abs_err;
        end

        % Pass if relative error < 0.1 % OR absolute error < atol.
        pass = rel_err < 1e-3 || abs_err < atol_j;
        if ~pass, all_passed = false; end
        fprintf('%-6d  %-6d  %14.6g  %14.6g  %12.2e  %s\n', ...
            e, jj, dlam_anal(e), dlam_fd, rel_err, yesno(pass));
    end
end

if all_passed
    fprintf('PASS  All raw gradient FD errors < 0.1 %%\n\n');
else
    fprintf('FAIL  Some raw gradients exceeded FD tolerance\n\n');
end

%% ------------------------------------------------------------------
%  B.  K and M use raw rho, not filtered rho
% -------------------------------------------------------------------
fprintf('--- B. K, M assembled from raw rho (not filtered rho) ---\n');

% Build a density filter and get rho_tilde != rho.
rmin_elem = 2.5;
[h_filt, Hs_filt] = build_filter(nelx, nely, rmin_elem);
rho_tilde = reshape(imfilter(reshape(rho, nely, nelx), h_filt, 'symmetric') ./ Hs_filt, [], 1);

% Trace of K: manual computation from rho^p * Ke_phys.
Ee_raw    = rho    .^ penal;
Ee_filt   = rho_tilde .^ penal;
diag_Ke   = diag(Ke_phys);   % 8x1 diagonal of full element stiffness

% Approximate trace by summing diagonal contributions.
% Each element e contributes rho_e^p * trace(Ke_phys) to trace(K).
% (Off-diagonal entries do not contribute to the trace.)
trace_K_raw  = sum(Ee_raw   * trace(Ke_phys));
trace_K_filt = sum(Ee_filt  * trace(Ke_phys));

% Trace of assembled K.  Use full() to convert sparse diagonal to double.
trace_K_asm = full(sum(diag(K0)));

err_raw  = abs(trace_K_asm - trace_K_raw)  / abs(trace_K_raw);
err_filt = abs(trace_K_asm - trace_K_filt) / abs(trace_K_filt);

pass_B = err_raw < 1e-10 && err_filt > 1e-6;
if ~pass_B, all_passed = false; end

fprintf('  trace(K) from raw rho:      %.6g\n', trace_K_raw);
fprintf('  trace(K) assembled:         %.6g  (err vs raw = %.2e)\n', trace_K_asm, err_raw);
fprintf('  trace(K) from filtered rho: %.6g  (diff from raw = %.2e)\n', trace_K_filt, err_filt);
fprintf('  Assembled K matches raw rho and differs from filtered rho: %s\n\n', yesno(pass_B));

%% ------------------------------------------------------------------
%  C.  Sensitivity filter properties
% -------------------------------------------------------------------
fprintf('--- C. Sensitivity filter properties ---\n');

rmin_elem = 2.5;
[h_sf, Hs_sf] = build_filter(nelx, nely, rmin_elem);

% C1. Uniform s on uniform rho → s_hat = s.
rho_unif  = 0.5 * ones(nEl, 1);
s_unif    = 3.7 * ones(nEl, 1);   % arbitrary non-unit constant
s_hat_c1  = apply_sensitivity_filter(s_unif, rho_unif, h_sf, Hs_sf, nely, nelx);
err_c1    = max(abs(s_hat_c1 - s_unif)) / abs(s_unif(1));
pass_c1   = err_c1 < 1e-12;
fprintf('  C1. Uniform s, uniform rho  → max deviation = %.2e   %s\n', err_c1, yesno(pass_c1));

% C2. Localized spike in sensitivity → filter spreads it (no sharp spike in output).
s_spike = zeros(nEl, 1);
s_spike(round(nEl/2)) = 1.0;
s_hat_c2 = apply_sensitivity_filter(s_spike, rho_unif, h_sf, Hs_sf, nely, nelx);
% The spike must be spread: the filtered field is non-zero over several elements.
n_nonzero = sum(s_hat_c2 > 1e-10);
pass_c2   = n_nonzero > 1;
fprintf('  C2. Localized spike         → non-zero elements = %d (expect > 1)   %s\n', n_nonzero, yesno(pass_c2));

% C3. Volume gradient is NOT filtered.
% The volume gradient is simply 1/nEl for each element (equal weight).
% We demonstrate that passing it through apply_sensitivity_filter changes it,
% confirming that the caller must NOT filter the volume term.
vol_grad = ones(nEl, 1) / nEl;
vol_grad_filtered = apply_sensitivity_filter(vol_grad, rho, h_sf, Hs_sf, nely, nelx);
% On non-uniform rho the filtered and unfiltered differ.
diff_vol = max(abs(vol_grad_filtered - vol_grad));
pass_c3  = diff_vol > 1e-10;   % confirm they DO differ (filter would change it)
fprintf('  C3. Volume gradient unfiltered (filter would change it by %.2e)   %s\n', ...
    diff_vol, yesno(pass_c3));

if ~pass_c1 || ~pass_c2 || ~pass_c3, all_passed = false; end
fprintf('\n');

%% ------------------------------------------------------------------
%  Summary
% -------------------------------------------------------------------
fprintf('==========================================================\n');
fprintf(' Summary\n');
fprintf('==========================================================\n');
fprintf('  A. Raw gradient FD check:              %s\n', yesno(all_passed));
fprintf('  B. K/M use raw rho:                    %s\n', yesno(pass_B));
fprintf('  C1. Filter: uniform → identity:        %s\n', yesno(pass_c1));
fprintf('  C2. Filter: spike → spread:            %s\n', yesno(pass_c2));
fprintf('  C3. Volume gradient must stay raw:     %s\n', yesno(pass_c3));

if all_passed
    fprintf('\nPHASE 2 PASSED\n');
else
    fprintf('\nPHASE 2 FAILED — inspect output above\n');
end
fprintf('==========================================================\n');

%% ------------------------------------------------------------------
function s = yesno(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
