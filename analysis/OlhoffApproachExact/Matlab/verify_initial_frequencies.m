% VERIFY_INITIAL_FREQUENCIES  Phase 1 acceptance test for OlhoffApproachExact.
%
% Checks two things:
%
%   A. Initial-frequency benchmark (Du & Olhoff 2007, Fig. 2 caption)
%      Uniform rho = 0.5 with E=1e7, nu=0.3, rho0=1, alpha=0.5.
%      Paper values:  SS = 68.7,  CS = 104.1,  CC = 146.1  rad/s.
%      Test is run on three mesh resolutions (40x5, 80x10, 160x20) to show
%      convergence; the finest mesh must match within 1 %.
%
%   B. Mass derivative finite-difference check
%      All four mass modes tested at rho_e = 0.05 and rho_e = 0.5 (both
%      sides of the 0.1 kink).  Relative FD error must be < 1e-5.
%
% Run from the repository root or any directory that has this folder on the
% Matlab path.  No external dependencies beyond the four files in this
% folder.

fprintf('==========================================================\n');
fprintf(' OlhoffApproachExact — Phase 1 verification\n');
fprintf('==========================================================\n\n');

%% ------------------------------------------------------------------
%  A.  Mass-derivative finite-difference check
% -------------------------------------------------------------------
fprintf('--- A. Mass derivative finite-difference check ---\n');

modes = {'linear','du2007_step','du2007_c0','du2007_c1'};
test_rhos = [0.05, 0.15, 0.5];   % below kink, just above kink, mid-range
h_fd = 1e-7;
all_passed = true;

fprintf('%-15s  %6s   %12s   %12s   %6s\n', ...
    'mode', 'rho', 'm(rho)', 'FD error', 'pass?');
for mi = 1:numel(modes)
    for ri = 1:numel(test_rhos)
        rho0 = test_rhos(ri);
        [~, dm_anal] = mass_interp(rho0, modes{mi});
        m_fwd = mass_interp(rho0 + h_fd, modes{mi});
        m_bwd = mass_interp(rho0 - h_fd, modes{mi});
        dm_fd = (m_fwd - m_bwd) / (2*h_fd);

        if abs(dm_anal) > 1e-14
            rel_err = abs(dm_fd - dm_anal) / abs(dm_anal);
        else
            rel_err = abs(dm_fd - dm_anal);
        end

        % du2007_step has a true discontinuity at 0.1; the FD straddles it
        % for rho0 = 0.1 +/- h, so we skip only that exact point.  The
        % test points [0.05, 0.15, 0.5] never straddle the kink.
        tol  = 1e-5;
        pass = rel_err < tol;
        if ~pass, all_passed = false; end

        [m_val, ~] = mass_interp(rho0, modes{mi});
        fprintf('%-15s  %6.2f   %12.6g   %12.2e   %s\n', ...
            modes{mi}, rho0, m_val, rel_err, yesno(pass));
    end
end

if all_passed
    fprintf('PASS  All mass derivatives match FD to < 1e-5\n\n');
else
    fprintf('FAIL  Some mass derivatives exceeded FD tolerance\n\n');
end

%% ------------------------------------------------------------------
%  B.  Initial-frequency benchmark
% -------------------------------------------------------------------
fprintf('--- B. Initial-frequency benchmark ---\n\n');

% Paper material parameters (Section 4.1).
E0    = 1e7;
nu    = 0.3;
rho0  = 1.0;
t     = 1.0;
L     = 8.0;
H     = 1.0;
alpha = 0.5;          % volume fraction → rho = 0.5 uniformly
penal = 3.0;
rho_min = 1e-3;       % lower bound per paper (Section 3, after Eq. 7e)
mass_mode = 'du2007_c1';   % most general; paper says differences are negligible

% Paper benchmark values (Fig. 2 caption).
omega_paper = struct('SS', 68.7, 'CS', 104.1, 'CC', 146.1);
bc_cases    = {'SS','CS','CC'};

% Unit element matrices (E=1, rho=1); scale to physical E0, rho0 below.
% Ke_star with E=1: physical Ke_e = rho^p * E0 * Ke_star
% Me_star with rho=1: physical Me_e = m(rho) * rho0 * Me_star
% So to use assemble_KM_exact directly we bake E0 into Ke_star and rho0
% into Me_star once, then the assembly scaling is rho^p and m(rho).

mesh_configs = [40, 5; 80, 10; 160, 20];
tol_finest   = 0.01;   % 1 % agreement required at finest mesh

freq_table = zeros(size(mesh_configs,1), 3);   % rows: meshes, cols: BC cases

fprintf('Mesh resolutions tested: ');
for mi = 1:size(mesh_configs,1)
    fprintf('%dx%d', mesh_configs(mi,1), mesh_configs(mi,2));
    if mi < size(mesh_configs,1), fprintf(', '); end
end
fprintf('\nTolerance at finest mesh: %.0f %%\n\n', tol_finest*100);

final_pass = true;

for ci = 1:3
    bc = bc_cases{ci};
    fprintf('  BC = %s  (paper omega1 = %.1f rad/s)\n', bc, omega_paper.(bc));
    fprintf('  %-10s  %10s  %10s\n', 'mesh', 'omega1', 'err %');

    for mi = 1:size(mesh_configs,1)
        nelx = mesh_configs(mi,1);
        nely = mesh_configs(mi,2);
        dx   = L/nelx;
        dy   = H/nely;
        nEl  = nelx*nely;
        nDof = 2*(nelx+1)*(nely+1);

        % Element matrices (physical E0 and rho0 baked in).
        [Ke_star, Me_star] = fe_q4_exact(nu, t, dx, dy);
        Ke_phys = E0  * Ke_star;   % K_e = rho^p * Ke_phys
        Me_phys = rho0 * Me_star;  % M_e = m(rho) * Me_phys

        % Assembly connectivity.
        nodeNrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);
        cVec    = reshape(2*nodeNrs(1:nely, 1:nelx)+1, nEl, 1);
        cMat    = [cVec, cVec+1, cVec+2*nely+2, cVec+2*nely+3, ...
                   cVec+2*nely, cVec+2*nely+1, cVec-2, cVec-1];

        [Il, Jl] = find(tril(ones(8)));
        iK = reshape(cMat(:,Il)', [], 1);
        jK = reshape(cMat(:,Jl)', [], 1);
        Ke_phys_l = Ke_phys(sub2ind([8,8], Il, Jl));
        Me_phys_l = Me_phys(sub2ind([8,8], Il, Jl));

        % Boundary conditions.
        fixed = build_supports_exact(bc, nodeNrs);
        free  = setdiff(1:nDof, fixed);

        % Uniform density rho = alpha = 0.5, clamped to [rho_min, 1].
        rho = max(rho_min, alpha) * ones(nEl, 1);

        % Assemble K and M.
        [K, M] = assemble_KM_exact(rho, Ke_phys_l, Me_phys_l, ...
                                   iK, jK, nDof, penal, mass_mode);
        Kf = K(free, free);
        Mf = M(free, free);

        % Eigensolve (shift-invert avoids singular factorisation).
        opts_eig.tol   = 1e-10;
        opts_eig.maxit = 600;
        [~, D] = eigs(Kf, Mf, 3, 'SM', opts_eig);
        lam    = sort(real(diag(D)));
        omega1 = sqrt(max(lam(1), 0));

        err_pct = 100 * abs(omega1 - omega_paper.(bc)) / omega_paper.(bc);
        freq_table(mi, ci) = omega1;
        fprintf('  %4dx%-4d   %10.3f  %10.2f\n', nelx, nely, omega1, err_pct);
    end

    % Finest-mesh check.
    omega_fine = freq_table(end, ci);
    err_fine   = abs(omega_fine - omega_paper.(bc)) / omega_paper.(bc);
    if err_fine > tol_finest
        fprintf('  FAIL  Finest mesh error %.2f %% > tolerance %.0f %%\n', ...
            err_fine*100, tol_finest*100);
        final_pass = false;
    else
        fprintf('  PASS  Finest mesh error %.2f %% within tolerance\n', err_fine*100);
    end
    fprintf('\n');
end

%% ------------------------------------------------------------------
%  Summary
% -------------------------------------------------------------------
fprintf('==========================================================\n');
fprintf(' Summary\n');
fprintf('==========================================================\n');
fprintf('  Mass derivative FD check: %s\n', yesno(all_passed));
fprintf('  Initial frequency benchmark:\n');
for ci = 1:3
    bc      = bc_cases{ci};
    omega1  = freq_table(end, ci);
    err_pct = 100*abs(omega1 - omega_paper.(bc)) / omega_paper.(bc);
    fprintf('    %s: omega1 = %.3f  (paper %.1f)  err = %.2f %%\n', ...
        bc, omega1, omega_paper.(bc), err_pct);
end
if all_passed && final_pass
    fprintf('\nPHASE 1 PASSED\n');
else
    fprintf('\nPHASE 1 FAILED — inspect output above\n');
end
fprintf('==========================================================\n');

%% ------------------------------------------------------------------
function s = yesno(tf)
    if tf, s = 'PASS'; else, s = 'FAIL'; end
end
