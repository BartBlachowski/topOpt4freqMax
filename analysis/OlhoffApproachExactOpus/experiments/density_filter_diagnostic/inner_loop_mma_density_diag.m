function [dx, beta_final, hist] = inner_loop_mma_density_diag(x, lambda_bar, fsk, ...
                                                     lambda_J, dlam_J, ...
                                                     volfrac, rho_min, ...
                                                     inner_max_iter, inner_tol, ...
                                                     move_lim, outer_move, ...
                                                     volume_current, dvol_dx)
% INNER_LOOP_MMA  Inner MMA subproblem for Du & Olhoff (2007) bound formulation.
%
%   [drho, beta_final, hist] = inner_loop_mma(rho, lambda_bar, fsk, ...
%       lambda_J, dlam_J, volfrac, rho_min, inner_max_iter, inner_tol, move_lim)
%
%   Solves for the density increment Delta_rho that maximizes a lower bound
%   beta on the updated eigenvalue cluster.  The outer-loop eigenpairs and
%   generalized gradients are held fixed throughout.
%
%   Variables: x = [beta; Delta_rho]  (nEl+1 x 1)
%   Objective: maximize beta (minimize -beta)
%
%   Constraints (m = N + 1 + has_J, numbered consecutively):
%     [1..N]   Cluster:   beta - lambda_bar - mu_i(F(Delta_rho)) <= 0
%     [N+1]    J-mode:    beta - lambda_J - dlam_J' * Delta_rho  <= 0
%                         (omitted when lambda_J = Inf)
%     [last]   Volume:    sum(rho + Delta_rho)/nEl - volfrac      <= 0
%
%   where F(Delta_rho) is the N x N matrix with
%       F(s,k) = fsk(:,s,k)' * Delta_rho = sum_e fsk(e,s,k) * Delta_rho(e)
%   and mu_i are the eigenvalues of F, sorted ascending.
%
%   The gradient of mu_i (subeigenvalue of the small N x N matrix F):
%       d(mu_i)/d(Delta_rho_e) = q_i^T F_e q_i
%                               = sum_{s,k} q_i(s) * fsk(e,s,k) * q_i(k)
%                               = (fsk2D * kron(q_i, q_i))(e)
%   where q_i is the ith (ascending) eigenvector of F and fsk2D is the
%   nEl x N^2 reshape of fsk with column (k-1)*N+s holding fsk(:,s,k).
%
%   For N=1 this reduces to the simple-eigenvalue increment formulation:
%       F = fsk(:,1,1)' * Delta_rho  (scalar),  mu_1 = F
%   and the cluster constraint becomes:
%       beta - lambda_bar - fsk(:,1,1)' * Delta_rho <= 0
%   which is identical to the standard single-eigenvalue subproblem.
%
%   IMPORTANT: fsk and dlam_J must be sensitivity-filtered (via
%   apply_sensitivity_filter) BEFORE calling this function.  The outer loop
%   is responsible for filtering; this function uses the arrays as given.
%
%   Inputs:
%     rho           nEl x 1    current physical density
%     lambda_bar    scalar     cluster eigenvalue (avg of cluster lambdas)
%     fsk           nEl x N x N  filtered generalized gradient array
%     lambda_J      scalar     eigenvalue of first mode above cluster
%                              (use Inf if J_idx = 0, i.e. cluster hits end)
%     dlam_J        nEl x 1   filtered sensitivity of lambda_J
%                              (use [] when lambda_J = Inf)
%     volfrac       scalar     volume fraction upper bound
%     rho_min       scalar     minimum density (lower bound on rho+drho)
%     inner_max_iter  scalar   max inner MMA iterations
%     inner_tol       scalar   convergence on norm(drho_change)/sqrt(nEl)
%     move_lim        scalar   trust-region radius on Delta_rho per iter
%                              (use Inf for no trust region beyond box bounds)
%
%   Outputs:
%     drho          nEl x 1   converged density increment
%     beta_final    scalar    converged beta lower bound
%     hist          struct    convergence history with fields:
%                               .n_iters       - iterations to convergence
%                               .drho_change   - norm of Delta_rho change
%                               .beta          - beta at each iteration
%                               .fval_cluster  - max cluster constraint value
%                               .fval_vol      - volume constraint value
%
%   Reference: Du & Olhoff (2007), Struct Multidisc Optim 34:91-110,
%              Eq. 15-19, 25.  Svanberg (1987) MMA.

x    = x(:);
nEl  = numel(x);
N    = size(fsk, 2);         % cluster multiplicity
if nargin < 13 || isempty(volume_current), volume_current = mean(x); end
if nargin < 14 || isempty(dvol_dx), dvol_dx = ones(nEl, 1) / nEl; end
dvol_dx = dvol_dx(:);

has_J  = isfinite(lambda_J) && ~isempty(dlam_J);
if has_J, dlam_J = dlam_J(:); end

% Reshape fsk to 2D: fsk2D(e, (k-1)*N+s) = fsk(e,s,k)  (column-major dims 2,3).
fsk2D = reshape(fsk, nEl, N*N);

% -----------------------------------------------------------------
% Variable count and constraint count.
% -----------------------------------------------------------------
n_var = nEl + 1;            % [beta; Delta_rho_1; ...; Delta_rho_nEl]
m     = N + 1 + has_J;     % N cluster + volume + optional J-mode

% -----------------------------------------------------------------
% Normalisation.
%   beta is O(lambda_bar) ~ 10^6 while Delta_rho is O(1).
%   We expose MMA to a dimensionless beta_hat = beta / lambda_ref so
%   all MMA variables and constraint values are O(1), avoiding the
%   ill-conditioning that arises from the 10^6 scale gap.
% -----------------------------------------------------------------
lambda_ref = lambda_bar;   % normalization reference (= cluster eigenvalue)

% -----------------------------------------------------------------
% MMA variable bounds (dimensionless beta_hat = beta / lambda_ref).
%   beta_hat:  [0, beta_hat_max]
%   Delta_rho: bounded by both absolute density limits and outer_move
%              (outer trust region)
% -----------------------------------------------------------------
if nargin < 11 || isempty(outer_move), outer_move = 0.2; end

% PHASE A (paper-exact): no upper bound on the bound variable beta is imposed
% in the paper formulation (25a) -- beta is limited solely by the cluster (25c)
% and J-mode (25b) constraints.  We therefore set the cap to an inactive large
% value so it never binds.  (A finite value is still required by mmasub; beta
% enters the objective and all constraints linearly, so its asymptote-based
% approximation is exact regardless of this bound's magnitude.)
beta_max_hat = 1e6;

% Outer trust region: restrict total Delta_rho to [-outer_move, +outer_move]
% (intersected with the absolute bounds [rho_min-rho, 1-rho]).
% This prevents the inner loop from accumulating changes that invalidate
% the linear approximation used in the outer loop.
drho_lb = max(rho_min - x, -outer_move * ones(nEl, 1));
drho_ub = min(1       - x, +outer_move * ones(nEl, 1));

xmin = [0;            drho_lb];
xmax = [beta_max_hat; drho_ub];

% Start beta_hat just below 1 (tight feasibility at drho=0).
xval  = [(1 - 1e-6); zeros(nEl, 1)];
xold1 = xval;
xold2 = xval;
low   = xmin;
upp   = xmax;

% MMA penalty setup.
% d > 0 is required for numerical stability: with d=0 and active constraints,
% diaglamyi -> epsi/lam^2 -> 0 inside subsolv, making Alam nearly singular.
% With d=1, active-constraint diagy is bounded (~2), keeping Alam well-conditioned.
a0 = 1;
a  = zeros(m, 1);
c  = 1e3 * ones(m, 1);
d  = ones(m, 1);

% -----------------------------------------------------------------
% History.
% -----------------------------------------------------------------
hist.drho_change  = nan(inner_max_iter, 1);
hist.beta         = nan(inner_max_iter, 1);
hist.fval_cluster = nan(inner_max_iter, 1);
hist.fval_vol     = nan(inner_max_iter, 1);
hist.n_iters      = 0;

% -----------------------------------------------------------------
% Inner MMA iterations.
% -----------------------------------------------------------------
for inner_it = 1:inner_max_iter

    beta_hat  = xval(1);             % dimensionless: beta / lambda_ref
    Delta_rho = xval(2:end);

    % ---- Build N x N matrix F(Delta_rho) and solve sub-eigenproblem. ----
    F_vec = fsk2D' * Delta_rho;           % N^2 x 1  (units: lambda)
    F_mat = reshape(F_vec, N, N);         % N x N

    [Q, Mu_D]   = eig(F_mat);            % Q: N x N; Mu_D: diagonal
    mu_raw      = real(diag(Mu_D));       % units: lambda
    [mu, si]    = sort(mu_raw, 'ascend');
    Q           = real(Q(:, si));

    % ---- Objective: minimize -beta_hat. ----
    f0     = -beta_hat;
    df0    = zeros(n_var, 1);
    df0(1) = -1;

    % ---- Constraint values and gradients (all normalised by lambda_ref). ----
    %
    % All eigenvalue constraint values are divided by lambda_ref so that
    % fval is O(1) and MMA's internal linear system is well conditioned.
    %
    fval = zeros(m, 1);
    dfdx = zeros(m, n_var);

    % Cluster constraints i = 1,...,N:
    %   (beta - lambda_bar - mu_i) / lambda_ref <= 0
    %   ≡  beta_hat - 1 - mu_i/lambda_ref       <= 0
    for i = 1:N
        fval(i)     = beta_hat - 1 - mu(i) / lambda_ref;
        dfdx(i, 1)  = 1;
        q_i         = Q(:, i);
        dmu_i       = fsk2D * kron(q_i, q_i);          % nEl x 1 (units: lambda)
        dfdx(i, 2:end) = -dmu_i' / lambda_ref;
    end

    row = N + 1;

    % J-mode constraint (optional):
    %   (beta - lambda_J - dlam_J' * Delta_rho) / lambda_ref <= 0
    if has_J
        fval(row)       = beta_hat - lambda_J/lambda_ref ...
                          - (dlam_J' * Delta_rho) / lambda_ref;
        dfdx(row, 1)    = 1;
        dfdx(row, 2:end)= -dlam_J' / lambda_ref;
        row = row + 1;
    end

    % Volume constraint (already dimensionless — no scaling needed):
    %   mean(rho_phys) + dmean(rho_phys)/dx * Delta_x - volfrac <= 0
    fval(row)        = volume_current + dvol_dx' * Delta_rho - volfrac;
    dfdx(row, 1)     = 0;
    dfdx(row, 2:end) = dvol_dx';

    % ---- MMA step. ----
    [xnew, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = ...
        mmasub(m, n_var, inner_it, xval, xmin, xmax, xold1, xold2, ...
               f0, df0, fval, dfdx, low, upp, a0, a, c, d);

    % Apply trust-region move limit on Delta_rho (not on beta).
    if isfinite(move_lim) && move_lim > 0
        xnew(2:end) = min(max(xnew(2:end), xval(2:end) - move_lim), ...
                          xval(2:end) + move_lim);
    end
    % Enforce box bounds.
    xnew = max(xmin, min(xmax, xnew));

    % ---- Convergence criterion: normalized norm of Delta_rho change. ----
    drho_change = norm(xnew(2:end) - xval(2:end));

    hist.drho_change(inner_it)  = drho_change;
    hist.beta(inner_it)         = beta_hat * lambda_ref;  % store in physical units
    hist.fval_cluster(inner_it) = max(fval(1:N));
    hist.fval_vol(inner_it)     = fval(end);
    hist.n_iters                = inner_it;

    xold2 = xold1;
    xold1 = xval;
    xval  = xnew;

    if drho_change < inner_tol * sqrt(nEl)
        break
    end
end

% Trim history to actual iterations used.
fn = fieldnames(hist);
for fi = 1:numel(fn)
    v = hist.(fn{fi});
    if numel(v) == inner_max_iter
        hist.(fn{fi}) = v(1:hist.n_iters);
    end
end

dx         = xval(2:end);
beta_final = xval(1) * lambda_ref;   % convert back to physical units (rad/s)^2
end
