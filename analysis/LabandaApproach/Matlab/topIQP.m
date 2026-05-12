function result = topIQP(nelx, nely, volfrac, penal, rmin, opts)
%TOPIQP  Minimum compliance topology optimisation using IQP-only SQP loop.
%
%   Implements the TopIQP variant of the TopSQP method from:
%     Rojas-Labanda & Stolpe (2016), "An efficient second-order SQP method
%     for structural topology optimization", Struct Multidisc Optim 53:1315-1333.
%
%   The IQP sub-problem is solved in its dual form (Section 5.2 of the paper)
%   using Matlab's quadprog.  No EQP phase is included here; see topSQP.m.
%
%   Problem (P^c):
%     min   u(t)^T K(t) u(t)
%     s.t.  (1/n) sum(t) <= volfrac,   0 <= t <= 1
%           K(t) u = f   (SIMP: E_e = Ev + (E1-Ev)*t_e^p)
%
%   Usage
%   -----
%     result = topIQP(nelx, nely, volfrac, penal, rmin)
%     result = topIQP(nelx, nely, volfrac, penal, rmin, opts)
%
%   Inputs
%   ------
%     nelx, nely  : element grid dimensions
%     volfrac     : volume fraction target (e.g. 0.3)
%     penal       : SIMP penalty exponent (e.g. 3)
%     rmin        : density filter radius in element units (e.g. 1.5)
%     opts        : optional struct
%       qpFormulation : 'dual' (default, paper), 'primal', or 'auto'
%       practicalStop : stop stalled quadprog tails using eps4-level KKT
%       maxIter       : outer SQP/IQP iteration cap
%
%   Output
%   ------
%     result : struct with fields
%       xPhys        — final physical density (nely x nelx)
%       compliance   — final compliance value
%       nIter        — number of outer iterations
%       kktHistory   — [statErr, feasErr, compErr] per iteration
%       compHistory  — compliance per iteration

% =========================================================================
% Parameters (Table 2 & 5 in Rojas-Labanda & Stolpe 2016)
% =========================================================================
E1      = 1e2;      % solid Young's modulus
Ev      = 1e-1;     % void  Young's modulus
nu      = 0.3;
stat_tol = 1e-6;    % KKT stationarity  epsilon_1
feas_tol = 1e-8;    % KKT feasibility   epsilon_2
comp_tol = 1e-6;    % KKT complementarity epsilon_3
max_iter = 1000;
sigma    = 1e-4;    % sufficient decrease constant (Armijo)
kappa    = 0.5;     % backtrack factor
max_ls   = 20;      % maximum Armijo backtracking reductions
eps4     = 1e-4;    % working-set tolerance
eps5     = 1e-6;    % relaxed merit reduction tolerance
max_forced_iq = 5;  % paper Section 7: force full IQP step at most 5 times
min_accepted_alpha = kappa^5;  % avoid accepting numerically stagnant backtracks
practical_stat_tol = eps4;      % quadprog/IQP-only safeguard for stalled tails
practical_comp_tol = 10*comp_tol;
min_practical_iter = 50;

if nargin < 6 || isempty(opts)
    opts = struct();
end
qp_formulation = lower(string(getOption(opts, 'qpFormulation', 'dual')));
use_practical_stop = getOption(opts, 'practicalStop', true);
max_iter = getOption(opts, 'maxIter', max_iter);

% =========================================================================
% Mesh and DOF bookkeeping
% =========================================================================
nEl    = nelx * nely;
nNodes = (nelx + 1) * (nely + 1);
nDof   = 2 * nNodes;
hx     = 1.0;       % unit element size (domain scaled to nelx x nely)
hy     = 1.0;

if qp_formulation == "auto"
    qp_formulation = "dual";
end
if ~any(qp_formulation == ["dual", "primal"])
    error('TopIQP:InvalidQPFormulation', ...
          'opts.qpFormulation must be ''auto'', ''dual'', or ''primal''.');
end

% Element connectivity: LL -> LR -> UR -> UL (1-based DOFs) — vectorised
elxV = floor((0:nEl-1)' / nely);          % column index per element (0-based)
elyV = mod  ((0:nEl-1)', nely);            % row    index per element (0-based)
n1 = (nely+1)* elxV      + elyV + 1;      % LL node
n2 = (nely+1)*(elxV + 1) + elyV + 1;      % LR node
n3 = n2 + 1;                               % UR node
n4 = n1 + 1;                               % UL node
edofMat = int32([2*n1-1, 2*n1, 2*n2-1, 2*n2, 2*n3-1, 2*n3, 2*n4-1, 2*n4]);
iK = reshape(kron(edofMat, ones(8,1,'int32'))', 64*nEl, 1);
jK = reshape(kron(edofMat, ones(1,8,'int32'))', 64*nEl, 1);

% =========================================================================
% Element stiffness matrix (E=1, hx=hy=1)
% =========================================================================
KE = keElastic(hx, hy, nu);

% =========================================================================
% MBB boundary conditions and load
%   - symmetry half-beam: left edge ux=0, bottom-right corner uy=0
%   - point load fy=-1 at top-left node
%
%   Node numbering: n(elx,ely) = (nely+1)*elx + ely + 1  (1-based)
%     ely=0 = bottom row,  ely=nely = top row
%     elx=0 = left col,    elx=nelx = right col
%   DOFs: ux(n) = 2n-1,  uy(n) = 2n
% =========================================================================
% Fixed DOFs (1-based)
leftNodes    = 1:(nely+1);                    % elx=0, ely=0..nely
botRightNode = (nely+1)*nelx + 1;             % elx=nelx, ely=0
fixedDofs    = [2*leftNodes-1, ...            % ux=0 on left edge (symmetry)
                2*botRightNode];              % uy=0 at bottom-right corner
fixedDofs    = unique(fixedDofs(:));
freeDofs     = setdiff((1:nDof)', fixedDofs);
nFree        = numel(freeDofs);

% Load vector: fy=-1 at top-left node (elx=0, ely=nely)
topLeftNode = nely + 1;
F      = zeros(nDof, 1);
F(2*topLeftNode) = -1.0;

% =========================================================================
% Density filter (Table 5: rmin = 0.04*Lx in element units for unit hx).
% xTilde is the design variable constrained by volume/bounds; xPhys is the
% filtered density used in SIMP stiffness interpolation.
% =========================================================================
if rmin > 0
    filterRadius = ceil(rmin) - 1;
    maxFilterEntries = nEl * (2*filterRadius + 1)^2;
    iH = zeros(maxFilterEntries, 1);
    jH = zeros(maxFilterEntries, 1);
    sH = zeros(maxFilterEntries, 1);
    kH = 0;
    for ix1 = 1:nelx
        for iy1 = 1:nely
            e1 = (ix1 - 1) * nely + iy1;
            ixMin = max(ix1 - filterRadius, 1);
            ixMax = min(ix1 + filterRadius, nelx);
            iyMin = max(iy1 - filterRadius, 1);
            iyMax = min(iy1 + filterRadius, nely);
            for ix2 = ixMin:ixMax
                for iy2 = iyMin:iyMax
                    e2 = (ix2 - 1) * nely + iy2;
                    weight = max(0, rmin - sqrt((ix1 - ix2)^2 + (iy1 - iy2)^2));
                    if weight > 0
                        kH = kH + 1;
                        iH(kH) = e1;
                        jH(kH) = e2;
                        sH(kH) = weight;
                    end
                end
            end
        end
    end
    Hfilt = sparse(iH(1:kH), jH(1:kH), sH(1:kH), nEl, nEl);
    Hs    = full(sum(Hfilt, 2));
else
    Hfilt = speye(nEl);
    Hs    = ones(nEl, 1);
end
Hnorm = spdiags(1 ./ Hs, 0, nEl, nEl) * Hfilt;

% =========================================================================
% Precompute F-matrix sparsity pattern (constant across all iterations)
%   Ffree is nFree x nEl;  sparsity depends only on edofMat and freeDofs.
% =========================================================================
freeMap          = zeros(nDof, 1, 'int32');
freeMap(freeDofs) = int32(1:nFree);
edofFull         = double(edofMat);           % nEl x 8
iF_cols          = repmat((1:nEl)', 1, 8);    % nEl x 8, element indices
iF_rows_all      = freeMap(edofFull);         % nEl x 8, free-DOF row indices (0=fixed)
fMask            = iF_rows_all > 0;           % nEl x 8 logical mask

% =========================================================================
% Initialisation
%   t0 = volfrac * ones(n)  (paper Table 2)
%   Lagrange multipliers: lambda (volume), xi (upper bound), eta (lower bound)
% =========================================================================
xTilde = volfrac * ones(nEl, 1);    % unfiltered design variable
xPhys  = full(Hnorm * xTilde);

lambda = 0.0;                        % scalar (single volume constraint)
xi     = zeros(nEl, 1);              % upper-bound multipliers
eta_m  = zeros(nEl, 1);             % lower-bound multipliers  (eta_m avoids clash with Matlab eta)
pi_pen = 1.0;                        % merit penalty parameter

a = ones(nEl, 1) / nEl;             % volume constraint coefficients a_i = 1/n
forced_iq_count = 0;

n_th = nFree;
n_nu = 2*nEl + 1;
Ak   = [a, speye(nEl), -speye(nEl)];       % nEl x (2*nEl+1)
lb_dual = [-inf(n_th, 1); zeros(n_nu, 1)];

quadprog_opt = optimoptions('quadprog', ...
    'Algorithm',              'interior-point-convex', ...
    'Display',                'off', ...
    'OptimalityTolerance',    1e-9,  ...
    'ConstraintTolerance',    1e-9,  ...
    'MaxIterations',          1000);

compHistory = zeros(max_iter, 1);
kktHistory  = zeros(max_iter, 3);

fprintf('TopIQP — minimum compliance\n');
fprintf('  mesh %d x %d,  volfrac %.3f,  penal %.1f,  rmin %.2f\n', ...
        nelx, nely, volfrac, penal, rmin);
fprintf('  QP formulation: %s via quadprog\n', char(qp_formulation));
fprintf('  %5s  %14s  %8s  %6s  %8s  %10s  %10s  %10s\n', ...
        'Iter', 'Compliance', 'Vol', 'alpha', '|d_iq|', 'statErr', 'feasErr', 'compErr');

% =========================================================================
% Main optimisation loop
% =========================================================================
for iter = 1:max_iter

    % ---------------------------------------------------------------------
    % 1. Assemble stiffness matrix and Cholesky factorisation
    % ---------------------------------------------------------------------
    Ee  = Ev + (E1 - Ev) * xPhys(:).^penal;    % SIMP interpolation (nEl x 1)
    sK  = reshape(KE(:) * Ee', 64*nEl, 1);
    K   = sparse(double(iK), double(jK), sK, nDof, nDof);
    K   = (K + K') / 2;
    Kf  = K(freeDofs, freeDofs);
    R   = chol(Kf);                             % Kf = R' * R  (upper triangular)

    % ---------------------------------------------------------------------
    % 2. Static solve: K u = F
    % ---------------------------------------------------------------------
    uFree = R \ (R' \ F(freeDofs));
    u     = zeros(nDof, 1);
    u(freeDofs) = uFree;

    % ---------------------------------------------------------------------
    % 3. Compliance and gradient
    % ---------------------------------------------------------------------
    ue  = u(edofMat);                           % nEl x 8
    ce  = sum((ue * KE) .* ue, 2);              % element strain energies
    f_val = Ee' * ce;                           % compliance = sum(Ee .* ce)

    df_dxPhys = -penal * (E1 - Ev) * xPhys(:).^(penal-1) .* ce;

    % Chain rule through density filter
    df_dx = Hnorm' * df_dxPhys;

    % Volume constraint on the unfiltered design variable.
    g_val = a' * xTilde - volfrac;

    % ---------------------------------------------------------------------
    % 4. F-matrix: F(:,e) = dK_e/dt_e * u_e  (paper eq. after eq. 17)
    %    = penal*(E1-Ev)*xPhys_e^(p-1) * KE * u_e   at 8 DOFs of element e
    %    Stored sparse, size nFree x nEl (restricted to free DOFs)
    %    Sparsity pattern is fixed; only values change each iteration.
    % ---------------------------------------------------------------------
    se     = penal * (E1 - Ev) * xPhys(:).^(penal-1);  % nEl x 1
    KEue   = KE * ue';                                   % 8 x nEl
    iF_vals = se(:) .* KEue';                            % nEl x 8
    Fphys   = sparse(double(iF_rows_all(fMask)), iF_cols(fMask), ...
                     iF_vals(fMask), nFree, nEl);
    Ffree   = Fphys * Hnorm;

    % uFree as column vector is already available
    % F_k' * u_free  (nEl x 1) — used as RHS in IQP dual
    Ftu = full(Ffree' * uFree);                      % nEl x 1

    % ---------------------------------------------------------------------
    % 5. Solve IQP dual problem (Section 5.2 of paper)
    %
    %   min   (1/4) theta' * Kf * theta + nu' * bk
    %   s.t.  Ak' * nu - Ffree' * theta = Ftu      (nEl equalities)
    %         nu >= 0
    %
    %   Variables: y = [theta (nFree); nu (2*nEl+1)]
    %   Ak  (nEl x (2*nEl+1)): volume + upper-bound + lower-bound constraints
    %   bk  (2*nEl+1):         RHS margins
    % ---------------------------------------------------------------------
    bk = [volfrac - a'*xTilde; ...       % volume slack  (scalar)
          1 - xTilde(:);       ...        % upper-bound margins  (nEl)
          xTilde(:)];                     % lower-bound margins  (nEl)

    switch qp_formulation
        case "dual"
            % Quadratic term: (1/2) y' P y,  P = blkdiag(Kf/2, 0)
            % quadprog uses (1/2) x' H x + f' x, so Kf/2 gives
            % the paper's (1/4) theta' K theta.
            P = [Kf/2,              sparse(n_th, n_nu);
                 sparse(n_nu, n_th), sparse(n_nu, n_nu)];

            q_qp = [zeros(n_th, 1); bk];

            % Equality constraints: -Ffree'*theta + Ak'*nu = Ftu
            Ceq  = [-Ffree', Ak];          % nEl x (nFree+2*nEl+1)
            beq  = Ftu;                    % nEl x 1

            [y_sol, ~, qp_flag, ~, qp_mults] = ...
                quadprog(P, q_qp, [], [], Ceq, beq, lb_dual, [], [], quadprog_opt);

            if qp_flag <= 0
                warning('TopIQP: quadprog returned flag %d at iter %d. Using zero step.', qp_flag, iter);
                d_iq = zeros(nEl, 1);
                lam_iq = lambda;  xi_iq = xi;  eta_iq = eta_m;
            else
                % Primal direction: d = -chi  where chi = qp_mults.eqlin
                d_iq = -qp_mults.eqlin;      % nEl x 1

                % Extract Lagrange multipliers from dual solution
                nu_sol = y_sol(n_th+1:end);  % 2*nEl+1 x 1
                lam_iq = nu_sol(1);          % volume constraint multiplier
                xi_iq  = nu_sol(2:nEl+1);    % upper-bound multipliers
                eta_iq = nu_sol(nEl+2:end);  % lower-bound multipliers
            end

        case "primal"
            % Equivalent primal IQP:
            %   min  df'*d + 0.5*d'*B*d
            %   s.t. a'*d <= volume slack,  -x <= d <= 1-x
            % where B = 2*Ffree'*Kf^{-1}*Ffree.
            KinvF = R \ (R' \ Ffree);
            Bq    = 2 * (Ffree' * KinvF);
            Bq    = (Bq + Bq') / 2;

            [d_iq, ~, qp_flag, ~, qp_mults] = ...
                quadprog(full(Bq), df_dx, a', bk(1), [], [], ...
                         -xTilde, 1 - xTilde, [], quadprog_opt);

            if qp_flag <= 0
                warning('TopIQP: quadprog returned flag %d at iter %d. Using zero step.', qp_flag, iter);
                d_iq = zeros(nEl, 1);
                lam_iq = lambda;  xi_iq = xi;  eta_iq = eta_m;
            else
                lam_iq = 0;
                if isfield(qp_mults, 'ineqlin') && ~isempty(qp_mults.ineqlin)
                    lam_iq = qp_mults.ineqlin(1);
                end
                xi_iq  = qp_mults.upper;
                eta_iq = qp_mults.lower;
            end
    end

    % ---------------------------------------------------------------------
    % 6. Merit function and model reduction
    %    phi_pi(t) = f(t) + pi * max(0, g(t))
    %    qred_pi(d) = -(df'*d + 0.5*d'*B*d) + pi*max(0,g)
    %      where d'*B*d = 2 * ||R^{-T} * Ffree * d_full||^2
    %      and d_full is d_iq projected back to nDof (zeros at fixed DOFs)
    % ---------------------------------------------------------------------
    Ftu_d  = Ffree * d_iq;                  % nFree x 1
    Rinv_d = R' \ Ftu_d;                    % nFree x 1  (R'^{-1} * Ffree*d)
    dBd    = 2 * (Rinv_d' * Rinv_d);        % scalar: d'*B*d

    phi_old = f_val + pi_pen * max(0, g_val);
    qred    = -(df_dx' * d_iq + 0.5 * dBd) + pi_pen * max(0, g_val);
    qred_req = max(qred, 0);
    step_inf = norm(d_iq, Inf);
    small_iq_step = step_inf <= eps4;

    % ---------------------------------------------------------------------
    % 7. Contraction parameter beta: largest beta in (0,1] such that
    %    linearised constraints of IQP remain feasible at d_iq (no EQP here)
    %    For TopIQP beta=1 always (no d_eq to mix in).
    % ---------------------------------------------------------------------

    % ---------------------------------------------------------------------
    % 8. Line search: test full step alpha=1, then backtrack
    %    x_{k+1} = clip(xTilde + alpha * d_iq, 0, 1)
    % ---------------------------------------------------------------------
    alpha = 1.0;
    accepted = false;
    line_search_failed = false;

    if small_iq_step
        alpha      = 0.0;
        accepted   = true;
        xTilde_new = xTilde;
        xPhys_new  = xPhys;
        f_new      = f_val;
        g_new      = g_val;
        ce_new     = ce;
    else
        for ls = 0:max_ls
            xTilde_new = max(0, min(1, xTilde + alpha * d_iq));
            xPhys_new  = full(Hnorm * xTilde_new);

            % Re-evaluate compliance at trial point
            Ee_new  = Ev + (E1 - Ev) * xPhys_new(:).^penal;
            sK_new  = reshape(KE(:) * Ee_new', 64*nEl, 1);
            K_new   = sparse(double(iK), double(jK), sK_new, nDof, nDof);
            K_new   = (K_new + K_new') / 2;
            Kf_new  = K_new(freeDofs, freeDofs);
            uFree_new = Kf_new \ F(freeDofs);
            ue_new  = zeros(nDof,1);  ue_new(freeDofs) = uFree_new;
            ue_new  = ue_new(edofMat);
            ce_new  = sum((ue_new * KE) .* ue_new, 2);
            f_new   = Ee_new' * ce_new;
            g_new   = a' * xTilde_new - volfrac;

            phi_new = f_new + pi_pen * max(0, g_new);

            required_reduction = max(sigma * alpha * qred_req - eps5, 0);
            if phi_new <= phi_old - required_reduction
                accepted = true;
                break;
            end
            alpha = kappa * alpha;
        end
    end

    if accepted && alpha > 0 && alpha < min_accepted_alpha
        accepted = false;
    end

    if ~accepted
        if forced_iq_count < max_forced_iq
            forced_iq_count = forced_iq_count + 1;
            accepted        = true;
            alpha           = 1.0;
            xTilde_new      = max(0, min(1, xTilde + d_iq));
            xPhys_new       = full(Hnorm * xTilde_new);

            Ee_new  = Ev + (E1 - Ev) * xPhys_new(:).^penal;
            sK_new  = reshape(KE(:) * Ee_new', 64*nEl, 1);
            K_new   = sparse(double(iK), double(jK), sK_new, nDof, nDof);
            K_new   = (K_new + K_new') / 2;
            Kf_new  = K_new(freeDofs, freeDofs);
            uFree_new = Kf_new \ F(freeDofs);
            ue_new  = zeros(nDof,1);  ue_new(freeDofs) = uFree_new;
            ue_new  = ue_new(edofMat);
            ce_new  = sum((ue_new * KE) .* ue_new, 2);
            f_new   = Ee_new' * ce_new;
            g_new   = a' * xTilde_new - volfrac;
        else
            warning('TopIQP: no acceptable Armijo step at iter %d. Returning current iterate.', iter);
            alpha      = 0.0;
            xTilde_new = xTilde;
            xPhys_new  = xPhys;
            f_new      = f_val;
            g_new      = g_val;
            ce_new     = ce;
            line_search_failed = true;
        end
    else
        forced_iq_count = 0;
    end

    % ---------------------------------------------------------------------
    % 9. Update design variables and Lagrange multipliers
    % ---------------------------------------------------------------------
    xTilde = xTilde_new;
    xPhys  = xPhys_new;

    % Multiplier update (paper Section 2.5, IQP-only branch)
    if small_iq_step
        lambda = lam_iq;
        xi     = xi_iq;
        eta_m  = eta_iq;
    else
        lambda = (1 - alpha) * lambda + alpha * lam_iq;
        xi     = (1 - alpha) * xi     + alpha * xi_iq;
        eta_m  = (1 - alpha) * eta_m  + alpha * eta_iq;
    end

    % Penalty parameter update: pi = ||lambda||_inf
    pi_pen = max(abs(lambda), 1e-3);

    % ---------------------------------------------------------------------
    % 10. KKT error (paper Section 2.1, eqs. 1-9)
    %     stationarity:     ||df + J'*lambda + xi - eta||_inf
    %     feasibility:      ||max(0, g)||_inf
    %     complementarity:  ||[g*lambda; (t-1).*xi; t.*eta]||_inf
    % ---------------------------------------------------------------------
    % Gradient of volume constraint w.r.t. xTilde (through filter)
    dg_dx = a;

    df_dxPhys_new = -penal * (E1 - Ev) * xPhys(:).^(penal-1) .* ce_new;
    df_dx_new     = Hnorm' * df_dxPhys_new;

    stat_err = norm(df_dx_new + lambda * dg_dx + xi - eta_m, Inf);
    feas_err = max(0, g_new);
    comp_err = norm([g_new * lambda; ...
                     (xTilde - 1) .* xi; ...
                     xTilde .* eta_m], Inf);

    compHistory(iter) = f_new;
    kktHistory(iter,:) = [stat_err, feas_err, comp_err];

    fprintf('  %5d  %14.6g  %8.4f  %6.4f  %8.3e  %10.3e  %10.3e  %10.3e\n', ...
            iter, f_new, mean(xTilde), alpha, norm(d_iq), stat_err, feas_err, comp_err);

    % ---------------------------------------------------------------------
    % 11. Convergence check
    % ---------------------------------------------------------------------
    strict_converged = stat_err < stat_tol && feas_err < feas_tol && comp_err < comp_tol;
    relaxed_converged = small_iq_step && ...
                        stat_err < max(stat_tol, 10*eps5) && ...
                        feas_err < feas_tol && ...
                        comp_err < 10*max(comp_tol, eps5);
    practical_converged = use_practical_stop && ...
                          iter >= min_practical_iter && ...
                          stat_err < practical_stat_tol && ...
                          feas_err < feas_tol && ...
                          comp_err < practical_comp_tol;

    if strict_converged || relaxed_converged || practical_converged
        fprintf('  Converged at iteration %d.\n', iter);
        break;
    end

    if small_iq_step
        warning(['TopIQP: IQP step below working-set tolerance at iter %d ', ...
                 'but relaxed KKT test was not met. Returning current iterate.'], iter);
        break;
    end

    if line_search_failed
        break;
    end

    f_val  = f_new;
    g_val  = g_new;
end

% =========================================================================
% Pack result
% =========================================================================
result.xPhys       = reshape(xPhys, nely, nelx);
result.xTilde      = reshape(xTilde, nely, nelx);
result.compliance  = f_new;
result.nIter       = iter;
result.compHistory = compHistory(1:iter);
result.kktHistory  = kktHistory(1:iter,:);
result.qpFormulation = char(qp_formulation);
end


% =========================================================================
% Local functions
% =========================================================================

function value = getOption(opts, name, defaultValue)
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = defaultValue;
    end
end

function KE = keElastic(hx, hy, nu)
% Q4 plane-stress stiffness (E=1, LL->LR->UR->UL node order, 2x2 Gauss).
    D  = (1/(1-nu^2)) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
    iJ = diag([2/hx, 2/hy]);
    dJ = 0.25 * hx * hy;
    gp = 1/sqrt(3);
    KE = zeros(8,8);
    for xi = [-gp, gp]
        for eta = [-gp, gp]
            dNdxi  = 0.25*[-(1-eta),  (1-eta),  (1+eta), -(1+eta)];
            dNdeta = 0.25*[-(1-xi),  -(1+xi),   (1+xi),  (1-xi) ];
            dNxy = iJ * [dNdxi; dNdeta];
            dNdx = dNxy(1,:);  dNdy = dNxy(2,:);
            B = zeros(3,8);
            B(1,1:2:end) = dNdx;
            B(2,2:2:end) = dNdy;
            B(3,1:2:end) = dNdy;
            B(3,2:2:end) = dNdx;
            KE = KE + (B'*D*B)*dJ;
        end
    end
end
