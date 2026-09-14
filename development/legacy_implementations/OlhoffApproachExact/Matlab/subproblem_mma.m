function out = subproblem_mma(sp, opts)
% SUBPROBLEM_MMA  MMA solution of the Olhoff & Du (2014) increment subproblem.
%
%   out = subproblem_mma(sp)
%   out = subproblem_mma(sp, opts)
%
%   The paper offers two solvers for subproblems (19)/(20): "the MMA method
%   (Svanberg 1987) or a linear programming algorithm" (text after Eq. (20i)).
%   This is the MMA path; SUBPROBLEM_LP is the LP/SDP path and solves the same
%   problem exactly.  Both take the identical subproblem struct sp -- see
%   SUBPROBLEM_LP for the field documentation.
%
%   Formulation used here (the "N smooth constraints" embedding)
%   -----------------------------------------------------------
%   Variables:  x = [beta_2 ; (beta_1) ; Delta_rho]      (beta_1: gap mode only)
%   Objective:  minimize -beta_2            (nth)
%               minimize -(beta_2 - beta_1) (gap)
%   Constraints, with G(Delta_rho) = diag(L) + sum_e Delta_rho_e F_e :
%       upper cluster i = 1..N   beta_2 - mu_i(G_up)  <= 0        (19c)+(19d)
%       upper guard J = n+N      beta_2 - lam_J - f_JJ'Delta_rho <= 0   (19b)
%       lower cluster i = 1..R   mu_i(G_lo) - beta_1  <= 0        (20d)+(20g)
%       lower guard j = n-R-1    lam_j + f_jj'Delta_rho - beta_1 <= 0   (20e)
%       volume                   mean(rho + Delta_rho) - volfrac <= 0   (19e)
%   with the sub-eigenvalue gradients
%       d mu_i / d Delta_rho_e = q_i' F_e q_i
%   where q_i is the i-th orthonormal eigenvector of the SYMMETRIC matrix G.
%
%   Differences from the historical inner_loop_mma.m (all defects, now fixed)
%     * G is symmetrized before eig, and eig is used on a symmetric matrix, so
%       mu is real and Q orthonormal by construction (was: general eig + real()).
%     * Near-degenerate mu_i have meaningless individual gradients; when
%       |mu_i - mu_j| < degen_tol only the mu_min cut is imposed, which is the
%       mathematically correct constraint (see SUBPROBLEM_LP).
%     * The bound-variable cap is a computed over-estimate, not 1e6.  A cap of
%       1e6 on the normalized beta was NOT inert: it drove a beta asymptote span
%       of ~9e3 and a P/Q dynamic range of ~5.7e13 inside mmasub.
%     * The stopping test is reported.  out.stop_reason is always one of
%       'converged' | 'max_iter' | 'stalled'.  The historical code used
%       norm(d_new - d_old) < inner_tol*sqrt(nEl), which was never met in 300
%       outer iterations at any mesh -- the truncation at inner_max_iter was
%       silently acting as the step control.
%
%   OPTIONS
%     .max_iter    inner MMA iteration budget                  (default 200)
%     .tol         convergence tol on the scaled iterate change (default 1e-7)
%     .feas_tol    max allowed constraint violation at the stop (default 1e-6)
%     .degen_tol   relative cluster degeneracy tolerance         (default 1e-8)
%     .asyinit     MMA initial asymptote factor                  (default 0.5)
%     .c_mma       MMA constraint penalty                        (default 1e3)
%     .verbose     print inner iterations                        (default false)
%
%   OUTPUT  (same fields as SUBPROBLEM_LP where they overlap, plus)
%     .n_iters       inner MMA iterations used
%     .max_violation max constraint value at the returned point (<=0 is feasible)
%     .hist          per-iteration struct: obj, max_viol, step
%
%   Reference: Olhoff & Du (2014), Eqs. (19), (20); Svanberg (1987).

if nargin < 2, opts = struct(); end
opts = defarg(opts, 'max_iter',  200);
opts = defarg(opts, 'tol',       1e-7);
opts = defarg(opts, 'feas_tol',  1e-6);
opts = defarg(opts, 'degen_tol', 1e-8);
opts = defarg(opts, 'asyinit',   0.5);
opts = defarg(opts, 'c_mma',     1e3);
opts = defarg(opts, 'verbose',   false);

is_gap = strcmpi(sp.mode, 'gap');

rho  = sp.rho(:);
nEl  = numel(rho);
mlim = sp.move;
if isempty(mlim), mlim = Inf; end

lam_ref = max(abs(sp.up.L));
if ~(lam_ref > 0), lam_ref = 1; end

N     = numel(sp.up.L);
Lu    = sp.up.L(:) / lam_ref;
Feu2D = reshape(sp.up.Fe, nEl, N*N) / lam_ref;
has_gu = isfield(sp.up,'guard') && ~isempty(sp.up.guard);
if has_gu
    lamJ  = sp.up.guard.lam / lam_ref;
    gradJ = sp.up.guard.grad(:) / lam_ref;
end

R = 0; has_gl = false;
if is_gap
    R     = numel(sp.lo.L);
    Ll    = sp.lo.L(:) / lam_ref;
    Fel2D = reshape(sp.lo.Fe, nEl, R*R) / lam_ref;
    has_gl = isfield(sp.lo,'guard') && ~isempty(sp.lo.guard);
    if has_gl
        lamj  = sp.lo.guard.lam / lam_ref;
        gradj = sp.lo.guard.grad(:) / lam_ref;
    end
end

nb   = 1 + double(is_gap);
i_b2 = 1;  i_b1 = 2;
i_d  = nb + (1:nEl);
nvar = nb + nEl;

lb_d = max(sp.rho_min - rho, -mlim * ones(nEl,1));
ub_d = min(1          - rho, +mlim * ones(nEl,1));
lb_d = min(lb_d, 0);
ub_d = max(ub_d, 0);
span = max(abs(lb_d), abs(ub_d));

b2cap = max(Lu) + N * (max(abs(Feu2D), [], 2)' * span) + 1;
xmin  = [zeros(nb,1); lb_d];
xmax  = [b2cap;       ub_d];
if is_gap
    b1cap = max(b2cap, max(Ll) + R * (max(abs(Fel2D), [], 2)' * span) + 1);
    xmax  = [b2cap; b1cap; ub_d];
end

% Number of constraints.
m_con = N + double(has_gu) + 1 + double(is_gap)*(R + double(has_gl));

% Start at Delta_rho = 0 with the bound variables at their feasible values.
xval        = zeros(nvar,1);
xval(i_b2)  = max(0, min(Lu)) * (1 - 1e-9);
if is_gap, xval(i_b1) = max(Ll) * (1 + 1e-9); end
xold1 = xval;  xold2 = xval;
low   = xmin;  upp = xmax;

a0 = 1;
a  = zeros(m_con,1);
cc = opts.c_mma * ones(m_con,1);
dd = ones(m_con,1);

hist.obj      = nan(opts.max_iter,1);
hist.max_viol = nan(opts.max_iter,1);
hist.step     = nan(opts.max_iter,1);

stop    = 'max_iter';
n_iters = 0;

for it = 1:opts.max_iter
    n_iters = it;

    [f0, df0, fval, dfdx] = eval_sub(xval);

    hist.obj(it)      = -f0 * lam_ref;
    hist.max_viol(it) = max(fval);

    [xnew, ~, ~, ~, ~, ~, ~, ~, ~, low, upp] = ...
        mmasub(m_con, nvar, it, xval, xmin, xmax, xold1, xold2, ...
               f0, df0, fval, dfdx, low, upp, a0, a, cc, dd);

    xnew  = max(xmin, min(xmax, xnew));
    step  = norm(xnew(i_d) - xval(i_d)) / sqrt(nEl);
    hist.step(it) = step;

    xold2 = xold1;  xold1 = xval;  xval = xnew;

    if opts.verbose
        fprintf('   inner %3d: obj = %12.4f  maxviol = %9.2e  step = %9.2e\n', ...
            it, hist.obj(it), hist.max_viol(it), step);
    end

    if step < opts.tol
        [~, ~, fv_now] = eval_sub(xval);
        if max(fv_now) <= opts.feas_tol
            stop = 'converged';
        else
            stop = 'stalled';
        end
        break
    end
end

[~, ~, fval_end] = eval_sub(xval);

out.drho          = xval(i_d);
out.beta          = xval(i_b2) * lam_ref;
if is_gap
    out.beta1 = xval(i_b1) * lam_ref;
    out.obj   = (xval(i_b2) - xval(i_b1)) * lam_ref;
else
    out.beta1 = NaN;
    out.obj   = out.beta;
end
at_bound          = (out.drho <= lb_d + 1e-12) | (out.drho >= ub_d - 1e-12);
out.n_iters       = n_iters;
out.n_cuts        = 0;
out.n_lp          = 0;
out.exitflag      = 1;
out.stop_reason   = stop;
out.max_violation = max(fval_end);
out.lmi_violation = max(fval_end);
out.frac_at_bound = mean(at_bound);
out.beta_cap_active = xval(i_b2) >= b2cap - 1e-9*max(1,b2cap);
out.lam_ref       = lam_ref;
fn = {'obj','max_viol','step'};
for k = 1:numel(fn), hist.(fn{k}) = hist.(fn{k})(1:n_iters); end
out.hist = hist;

% =====================================================================
    function [f0, df0, fval, dfdx] = eval_sub(x)
        b2 = x(i_b2);
        d  = x(i_d);

        if is_gap
            b1  = x(i_b1);
            f0  = -(b2 - b1);
            df0 = zeros(nvar,1);
            df0(i_b2) = -1;  df0(i_b1) = +1;
        else
            f0  = -b2;
            df0 = zeros(nvar,1);
            df0(i_b2) = -1;
        end

        fval = zeros(m_con,1);
        dfdx = zeros(m_con,nvar);
        r    = 0;

        % ---- Upper cluster: beta_2 - mu_i(G_up) <= 0 -------------------
        Gu = diag(Lu) + reshape(Feu2D' * d, N, N);
        Gu = (Gu + Gu')/2;
        [Qu, muu] = eig_sorted(Gu);
        use_min_only = N > 1 && (muu(2) - muu(1)) < opts.degen_tol * max(1, abs(muu(1)));
        for i = 1:N
            r = r + 1;
            if use_min_only && i > 1
                % Degenerate sub-eigenvalues: individual gradients are not
                % defined.  Repeat the mu_min constraint (which is the correct
                % one) rather than differentiate a non-smooth branch.
                q = Qu(:,1);  mu = muu(1);
            else
                q = Qu(:,i);  mu = muu(i);
            end
            fval(r)        = b2 - mu;
            dfdx(r, i_b2)  = 1;
            dfdx(r, i_d)   = -(Feu2D * kron(q,q))';
        end

        % ---- Upper guard J = n+N : beta_2 - lam_J - f_JJ'd <= 0 --------
        if has_gu
            r = r + 1;
            fval(r)       = b2 - lamJ - gradJ' * d;
            dfdx(r, i_b2) = 1;
            dfdx(r, i_d)  = -gradJ';
        end

        % ---- Lower cluster (gap): mu_i(G_lo) - beta_1 <= 0 -------------
        if is_gap
            Gl = diag(Ll) + reshape(Fel2D' * d, R, R);
            Gl = (Gl + Gl')/2;
            [Ql, mul] = eig_sorted(Gl);
            use_max_only = R > 1 && (mul(R) - mul(R-1)) < opts.degen_tol * max(1, abs(mul(R)));
            for i = 1:R
                r = r + 1;
                if use_max_only && i < R
                    q = Ql(:,R);  mu = mul(R);
                else
                    q = Ql(:,i);  mu = mul(i);
                end
                fval(r)        = mu - b1;
                dfdx(r, i_b1)  = -1;
                dfdx(r, i_d)   = (Fel2D * kron(q,q))';
            end
            if has_gl
                r = r + 1;
                fval(r)       = lamj + gradj' * d - b1;
                dfdx(r, i_b1) = -1;
                dfdx(r, i_d)  = gradj';
            end
        end

        % ---- Volume (19e) ---------------------------------------------
        r = r + 1;
        fval(r)      = (sum(rho) + sum(d))/nEl - sp.volfrac;
        dfdx(r, i_d) = 1/nEl;
    end
end

% =========================================================================
function [Q, mu] = eig_sorted(G)
[Q, D] = eig(G);
[mu, s] = sort(real(diag(D)), 'ascend');
Q = real(Q(:, s));
for j = 1:size(Q,2)
    nj = norm(Q(:,j));
    if nj > 0, Q(:,j) = Q(:,j)/nj; end
end
end

% =========================================================================
function s = defarg(s, f, v)
if ~isfield(s, f) || isempty(s.(f)), s.(f) = v; end
end
