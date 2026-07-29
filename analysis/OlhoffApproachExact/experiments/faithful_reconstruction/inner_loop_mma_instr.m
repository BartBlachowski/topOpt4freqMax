function [drho, beta_final, hist] = inner_loop_mma_instr(rho, lambda_bar, fsk, ...
                                                         lambda_J, dlam_J, ...
                                                         volfrac, rho_min, ...
                                                         inner_max_iter, inner_tol, ...
                                                         move_lim, outer_move)
% INNER_LOOP_MMA_INSTR  Instrumented copy of production inner_loop_mma.m.
%
%   ALGORITHMICALLY IDENTICAL to
%   analysis/OlhoffApproachExact/Matlab/inner_loop_mma.m.
%   Every executable statement that influences `drho` or `beta_final` is
%   identical.  The ONLY additions are recording statements that write into
%   `hist`; they read state but never write it.  Equivalence is proven
%   numerically by tests/test_inner_equivalence.m (bit-identical drho).
%
%   Additional recorded fields (all diagnostic):
%     hist.beta_hat       inner_it x 1  dimensionless bound variable
%     hist.fval           inner_it x m  every constraint value
%     hist.fval_J         inner_it x 1  J-mode constraint (NaN if absent)
%     hist.asym_low_w     inner_it x 1  min (xval - low) over Delta_rho vars
%     hist.asym_upp_w     inner_it x 1  min (upp - xval) over Delta_rho vars
%     hist.asym_width_*                 (upp - low) over Delta_rho vars
%     hist.asym_beta_low/upp            asymptotes of the bound variable beta
%     hist.n_at_lb/n_at_ub              # Delta_rho vars at their box bounds
%     hist.frac_at_bound                fraction of Delta_rho vars at a bound
%     hist.mma_lam_max                  max MMA constraint multiplier
%     hist.mma_ymma_max                 max MMA artificial variable y_i
%     hist.mma_zmma                     MMA z variable
%     hist.warn_id        inner_it x 1  cell, warning id raised inside mmasub
%     hist.n_singular_warn              # inner iterations raising a singular /
%                                       RCOND warning inside mmasub/subsolv
%     hist.drho_inf       inner_it x 1  ||Delta_rho||_inf of the iterate
%     hist.pred_dlambda   inner_it x 1  mu_1 of the iterate (predicted increment)
%     hist.pred_vol       inner_it x 1  mean(rho + Delta_rho)
%
%   See inner_loop_mma.m for the full formulation documentation.

rho  = rho(:);
nEl  = numel(rho);
N    = size(fsk, 2);         % cluster multiplicity

has_J  = isfinite(lambda_J) && ~isempty(dlam_J);
if has_J, dlam_J = dlam_J(:); end

fsk2D = reshape(fsk, nEl, N*N);

n_var = nEl + 1;
m     = N + 1 + has_J;

lambda_ref = lambda_bar;

if nargin < 11 || isempty(outer_move), outer_move = 0.2; end

beta_max_hat = 1e6;

drho_lb = max(rho_min - rho, -outer_move * ones(nEl, 1));
drho_ub = min(1       - rho, +outer_move * ones(nEl, 1));

xmin = [0;            drho_lb];
xmax = [beta_max_hat; drho_ub];

xval  = [(1 - 1e-6); zeros(nEl, 1)];
xold1 = xval;
xold2 = xval;
low   = xmin;
upp   = xmax;

a0 = 1;
a  = zeros(m, 1);
c  = 1e3 * ones(m, 1);
d  = ones(m, 1);

% ----- history (production fields) -----
hist.drho_change  = nan(inner_max_iter, 1);
hist.beta         = nan(inner_max_iter, 1);
hist.fval_cluster = nan(inner_max_iter, 1);
hist.fval_vol     = nan(inner_max_iter, 1);
hist.n_iters      = 0;
hist.converged    = false;
hist.hit_max_iter = false;
hist.termination_reason = 'not_started';

% ----- history (instrumentation only) -----
hist.beta_hat          = nan(inner_max_iter, 1);
hist.fval              = nan(inner_max_iter, m);
hist.fval_J            = nan(inner_max_iter, 1);
hist.asym_low_w        = nan(inner_max_iter, 1);
hist.asym_upp_w        = nan(inner_max_iter, 1);
hist.asym_width_min    = nan(inner_max_iter, 1);
hist.asym_width_mean   = nan(inner_max_iter, 1);
hist.asym_width_max    = nan(inner_max_iter, 1);
hist.asym_beta_low     = nan(inner_max_iter, 1);
hist.asym_beta_upp     = nan(inner_max_iter, 1);
hist.n_at_lb           = nan(inner_max_iter, 1);
hist.n_at_ub           = nan(inner_max_iter, 1);
hist.frac_at_bound     = nan(inner_max_iter, 1);
hist.frac_near_bound   = nan(inner_max_iter, 1);
hist.mma_lam_max       = nan(inner_max_iter, 1);
hist.mma_ymma_max      = nan(inner_max_iter, 1);
hist.mma_zmma          = nan(inner_max_iter, 1);
hist.drho_inf          = nan(inner_max_iter, 1);
hist.pred_dlambda      = nan(inner_max_iter, 1);
hist.pred_vol          = nan(inner_max_iter, 1);
hist.warn_id           = repmat({''}, inner_max_iter, 1);
hist.n_singular_warn   = 0;
hist.nEl               = nEl;
hist.m                 = m;
hist.N                 = N;
hist.lambda_ref        = lambda_ref;
hist.box_width_min     = min(drho_ub - drho_lb);
hist.box_width_max     = max(drho_ub - drho_lb);

for inner_it = 1:inner_max_iter

    beta_hat  = xval(1);
    Delta_rho = xval(2:end);

    F_vec = fsk2D' * Delta_rho;
    F_mat = reshape(F_vec, N, N);

    [Q, Mu_D]   = eig(F_mat);
    mu_raw      = real(diag(Mu_D));
    [mu, si]    = sort(mu_raw, 'ascend');
    Q           = real(Q(:, si));

    f0     = -beta_hat;
    df0    = zeros(n_var, 1);
    df0(1) = -1;

    fval = zeros(m, 1);
    dfdx = zeros(m, n_var);

    for i = 1:N
        fval(i)     = beta_hat - 1 - mu(i) / lambda_ref;
        dfdx(i, 1)  = 1;
        q_i         = Q(:, i);
        dmu_i       = fsk2D * kron(q_i, q_i);
        dfdx(i, 2:end) = -dmu_i' / lambda_ref;
    end

    row = N + 1;

    if has_J
        fval(row)       = beta_hat - lambda_J/lambda_ref ...
                          - (dlam_J' * Delta_rho) / lambda_ref;
        dfdx(row, 1)    = 1;
        dfdx(row, 2:end)= -dlam_J' / lambda_ref;
        row = row + 1;
    end

    fval(row)        = (sum(rho) + sum(Delta_rho)) / nEl - volfrac;
    dfdx(row, 1)     = 0;
    dfdx(row, 2:end) = 1 / nEl;

    % ---- MMA step (warning capture is inert w.r.t. the returned values) ----
    [prev_msg, prev_id] = lastwarn();
    lastwarn('', '');
    [xnew, ymma, zmma, mma_lam, ~, ~, ~, ~, ~, low, upp] = ...
        mmasub(m, n_var, inner_it, xval, xmin, xmax, xold1, xold2, ...
               f0, df0, fval, dfdx, low, upp, a0, a, c, d);
    [~, this_id] = lastwarn();
    if isempty(this_id), lastwarn(prev_msg, prev_id); end

    % ---- instrumentation: read-only snapshot of the MMA internals ----
    hist.warn_id{inner_it} = this_id;
    lid = lower(this_id);
    if ~isempty(lid) && (contains(lid, 'singular') || contains(lid, 'rankdef') ...
                         || contains(lid, 'illcond') || contains(lid, 'nearlysingular'))
        hist.n_singular_warn = hist.n_singular_warn + 1;
    end
    lw = low(2:end);  up = upp(2:end);
    hist.asym_low_w(inner_it)      = min(Delta_rho - lw);
    hist.asym_upp_w(inner_it)      = min(up - Delta_rho);
    hist.asym_width_min(inner_it)  = min(up - lw);
    hist.asym_width_mean(inner_it) = mean(up - lw);
    hist.asym_width_max(inner_it)  = max(up - lw);
    hist.asym_beta_low(inner_it)   = low(1);
    hist.asym_beta_upp(inner_it)   = upp(1);
    hist.mma_lam_max(inner_it)     = max(mma_lam);
    hist.mma_ymma_max(inner_it)    = max(ymma);
    hist.mma_zmma(inner_it)        = zmma;
    hist.fval(inner_it, :)         = fval(:)';
    if has_J, hist.fval_J(inner_it) = fval(N+1); end
    hist.beta_hat(inner_it)        = beta_hat;
    hist.pred_dlambda(inner_it)    = mu(1);
    hist.pred_vol(inner_it)        = (sum(rho) + sum(Delta_rho)) / nEl;

    % ---- production algorithm resumes, unmodified ----
    if isfinite(move_lim) && move_lim > 0
        xnew(2:end) = min(max(xnew(2:end), xval(2:end) - move_lim), ...
                          xval(2:end) + move_lim);
    end
    xnew = max(xmin, min(xmax, xnew));

    drho_change = norm(xnew(2:end) - xval(2:end));

    hist.drho_change(inner_it)  = drho_change;
    hist.beta(inner_it)         = beta_hat * lambda_ref;
    hist.fval_cluster(inner_it) = max(fval(1:N));
    hist.fval_vol(inner_it)     = fval(end);
    hist.n_iters                = inner_it;

    % ---- instrumentation on the accepted iterate ----
    dn = xnew(2:end);
    % Bound activity uses a tolerance RELATIVE to each variable's own box, so
    % that MMA's asymptotic approach to a vertex is registered as "at bound".
    bw   = max(drho_ub - drho_lb, eps);
    tolb = 1e-6 * bw;
    hist.n_at_lb(inner_it)       = sum(dn <= drho_lb + tolb);
    hist.n_at_ub(inner_it)       = sum(dn >= drho_ub - tolb);
    hist.frac_at_bound(inner_it) = (hist.n_at_lb(inner_it) + ...
                                    hist.n_at_ub(inner_it)) / nEl;
    hist.frac_near_bound(inner_it) = sum(dn <= drho_lb + 0.01*bw | ...
                                         dn >= drho_ub - 0.01*bw) / nEl;
    hist.drho_inf(inner_it)      = max(abs(dn));

    xold2 = xold1;
    xold1 = xval;
    xval  = xnew;

    if drho_change < inner_tol * sqrt(nEl)
        hist.converged = true;
        hist.termination_reason = 'convergence';
        break
    end
end

if ~hist.converged
    hist.hit_max_iter = hist.n_iters >= inner_max_iter;
    if hist.hit_max_iter
        hist.termination_reason = 'max_iterations';
    else
        hist.termination_reason = 'stopped';
    end
end

fn = fieldnames(hist);
for fi = 1:numel(fn)
    v = hist.(fn{fi});
    if (isnumeric(v) || iscell(v)) && size(v,1) == inner_max_iter
        hist.(fn{fi}) = v(1:hist.n_iters, :);
    end
end

drho       = xval(2:end);
beta_final = xval(1) * lambda_ref;
end
