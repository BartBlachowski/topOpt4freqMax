function [drho, beta_final, hist, state] = inner_loop_mma_persistent_phase2( ...
    rho, lambda_bar, fsk, lambda_J, dlam_J, volfrac, rho_min, ...
    inner_max_iter, inner_tol, move_lim, outer_move, state)
%INNER_LOOP_MMA_PERSISTENT_PHASE2  Phase-2 persistent MMA state test.
%
% This helper keeps the Du-Olhoff inner subproblem and Svanberg MMA update
% rules unchanged, but preserves the MMA asymptote state across outer
% iterations.  To make xold1/xold2 physically comparable across outer
% iterations, the MMA variables are [beta_hat; rho_candidate].  The
% subproblem constraints are still written in terms of
% Delta_rho = rho_candidate - rho and the candidate bounds are exactly the
% original Delta_rho bounds shifted by the current rho.

rho = rho(:);
nEl = numel(rho);
N = size(fsk, 2);
has_J = isfinite(lambda_J) && ~isempty(dlam_J);
if has_J
    dlam_J = dlam_J(:);
end

fsk2D = reshape(fsk, nEl, N*N);
n_var = nEl + 1;
m = N + 1 + has_J;
lambda_ref = lambda_bar;

if nargin < 11 || isempty(outer_move)
    outer_move = 0.2;
end

beta_max_hat = 1e6;
drho_lb = max(rho_min - rho, -outer_move * ones(nEl, 1));
drho_ub = min(1 - rho, outer_move * ones(nEl, 1));
rho_lb = rho + drho_lb;
rho_ub = rho + drho_ub;

xmin = [0; rho_lb];
xmax = [beta_max_hat; rho_ub];
xval = [(1 - 1e-6); rho];

if isempty(state) || ~isfield(state, 'initialized') || ~state.initialized
    state = struct();
    state.initialized = true;
    state.iter = 0;
    state.low = xmin;
    state.upp = xmax;
    state.xold1 = xval;
    state.xold2 = xval;
else
    state.low = localResizeAndProject(state.low, xmin, xmax, xval);
    state.upp = localResizeAndProject(state.upp, xmin, xmax, xval);
    state.xold1 = localResizeAndProject(state.xold1, xmin, xmax, xval);
    state.xold2 = localResizeAndProject(state.xold2, xmin, xmax, xval);
end

a0 = 1;
a = zeros(m, 1);
c = 1e3 * ones(m, 1);
d = ones(m, 1);

hist.drho_change = nan(inner_max_iter, 1);
hist.beta = nan(inner_max_iter, 1);
hist.fval_cluster = nan(inner_max_iter, 1);
hist.fval_vol = nan(inner_max_iter, 1);
hist.n_iters = 0;
hist.converged = false;
hist.hit_max_iter = false;
hist.termination_reason = 'not_started';
hist.asym_width_min = NaN;
hist.asym_width_mean = NaN;
hist.asym_width_max = NaN;
hist.asym_expand_count = NaN;
hist.asym_contract_count = NaN;
hist.asym_same_count = NaN;

for inner_it = 1:inner_max_iter
    beta_hat = xval(1);
    rho_candidate = xval(2:end);
    Delta_rho = rho_candidate - rho;

    F_vec = fsk2D' * Delta_rho;
    F_mat = reshape(F_vec, N, N);
    [Q, Mu_D] = eig(F_mat);
    mu_raw = real(diag(Mu_D));
    [mu, si] = sort(mu_raw, 'ascend');
    Q = real(Q(:, si));

    f0 = -beta_hat;
    df0 = zeros(n_var, 1);
    df0(1) = -1;

    fval = zeros(m, 1);
    dfdx = zeros(m, n_var);

    for i = 1:N
        fval(i) = beta_hat - 1 - mu(i) / lambda_ref;
        dfdx(i, 1) = 1;
        q_i = Q(:, i);
        dmu_i = fsk2D * kron(q_i, q_i);
        dfdx(i, 2:end) = -dmu_i' / lambda_ref;
    end

    row = N + 1;
    if has_J
        fval(row) = beta_hat - lambda_J / lambda_ref ...
            - (dlam_J' * Delta_rho) / lambda_ref;
        dfdx(row, 1) = 1;
        dfdx(row, 2:end) = -dlam_J' / lambda_ref;
        row = row + 1;
    end

    fval(row) = mean(rho_candidate) - volfrac;
    dfdx(row, 1) = 0;
    dfdx(row, 2:end) = 1 / nEl;

    mma_iter = state.iter + 1;
    [xnew, ~, ~, ~, ~, ~, ~, ~, ~, low, upp, asyinfo] = ...
        mmasub_persist_experiment(m, n_var, mma_iter, xval, xmin, xmax, ...
            state.xold1, state.xold2, f0, df0, fval, dfdx, ...
            state.low, state.upp, a0, a, c, d);

    if isfinite(move_lim) && move_lim > 0
        xnew(2:end) = min(max(xnew(2:end), xval(2:end) - move_lim), ...
                          xval(2:end) + move_lim);
    end
    xnew = max(xmin, min(xmax, xnew));

    drho_change = norm(xnew(2:end) - xval(2:end));
    hist.drho_change(inner_it) = drho_change;
    hist.beta(inner_it) = beta_hat * lambda_ref;
    hist.fval_cluster(inner_it) = max(fval(1:N));
    hist.fval_vol(inner_it) = fval(end);
    hist.n_iters = inner_it;

    state.iter = mma_iter;
    state.xold2 = state.xold1;
    state.xold1 = xval;
    state.low = low;
    state.upp = upp;
    xval = xnew;

    rho_width = asyinfo.width(2:end);
    rho_expand = asyinfo.expand(2:end);
    rho_contract = asyinfo.contract(2:end);
    hist.asym_width_min = min(rho_width);
    hist.asym_width_mean = mean(rho_width);
    hist.asym_width_max = max(rho_width);
    hist.asym_expand_count = nnz(rho_expand);
    hist.asym_contract_count = nnz(rho_contract);
    hist.asym_same_count = nEl - hist.asym_expand_count - hist.asym_contract_count;

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
    if isnumeric(v) && numel(v) == inner_max_iter
        hist.(fn{fi}) = v(1:hist.n_iters);
    end
end

drho = xval(2:end) - rho;
beta_final = xval(1) * lambda_ref;
end

function x = localResizeAndProject(x, xmin, xmax, fallback)
if numel(x) ~= numel(fallback)
    x = fallback;
else
    x = x(:);
end
x = max(xmin, min(xmax, x));
end
