function [drho, beta_final, hist, state] = inner_loop_mma_persist_step_experiment( ...
    rho, lambda_bar, fsk, lambda_J, dlam_J, volfrac, rho_min, state, asyinit)
% INNER_LOOP_MMA_PERSIST_STEP_EXPERIMENT  One MMA step with persistent asymptotes.
%
% Experiment-only variant for testing whether standard MMA asymptote
% adaptation across outer iterations explains the Olhoff benchmark gap.
% The mathematical subproblem remains the increment formulation, but MMA uses
% x = [beta_hat; rho_candidate] so xold1/xold2 describe the actual evolving
% density field. Constraints use Delta_rho = rho_candidate - rho.

rho = rho(:);
nEl = numel(rho);
N = size(fsk, 2);
has_J = isfinite(lambda_J) && ~isempty(dlam_J);
if has_J, dlam_J = dlam_J(:); end

fsk2D = reshape(fsk, nEl, N*N);
n_var = nEl + 1;
m = N + 1 + has_J;
lambda_ref = lambda_bar;

beta_max_hat = 1e6;  % inactive finite bound required by mmasub.
xmin = [0; rho_min * ones(nEl, 1)];
xmax = [beta_max_hat; ones(nEl, 1)];
xval = [(1 - 1e-6); rho];

if nargin < 9 || isempty(asyinit), asyinit = 0.02; end
if isempty(state) || ~isfield(state, 'iter')
    state.iter = 0;
    state.low = xmin;
    state.upp = xmax;
    state.xold1 = xval;
    state.xold2 = xval;
end

Delta_rho = xval(2:end) - rho;
F_vec = fsk2D' * Delta_rho;
F_mat = reshape(F_vec, N, N);
[Q, Mu_D] = eig(F_mat);
mu_raw = real(diag(Mu_D));
[mu, si] = sort(mu_raw, 'ascend');
Q = real(Q(:, si));

f0 = -xval(1);
df0 = zeros(n_var, 1);
df0(1) = -1;

fval = zeros(m, 1);
dfdx = zeros(m, n_var);

for i = 1:N
    fval(i) = xval(1) - 1 - mu(i) / lambda_ref;
    dfdx(i, 1) = 1;
    q_i = Q(:, i);
    dmu_i = fsk2D * kron(q_i, q_i);
    dfdx(i, 2:end) = -dmu_i' / lambda_ref;
end

row = N + 1;
if has_J
    fval(row) = xval(1) - lambda_J / lambda_ref ...
        - (dlam_J' * Delta_rho) / lambda_ref;
    dfdx(row, 1) = 1;
    dfdx(row, 2:end) = -dlam_J' / lambda_ref;
    row = row + 1;
end

fval(row) = mean(xval(2:end)) - volfrac;
dfdx(row, 1) = 0;
dfdx(row, 2:end) = 1 / nEl;

a0 = 1;
a = zeros(m, 1);
c = 1e3 * ones(m, 1);
d = ones(m, 1);

iter = state.iter + 1;
[xnew, ~, ~, ~, ~, ~, ~, ~, ~, low, upp, asyinfo] = ...
    mmasub_persist_experiment(m, n_var, iter, xval, xmin, xmax, ...
        state.xold1, state.xold2, f0, df0, fval, dfdx, ...
        state.low, state.upp, a0, a, c, d, asyinit);

xnew = max(xmin, min(xmax, xnew));

drho = xnew(2:end) - rho;
beta_final = xnew(1) * lambda_ref;

rho_width = asyinfo.width(2:end);
rho_expand = asyinfo.expand(2:end);
rho_contract = asyinfo.contract(2:end);

hist.n_iters = 1;
hist.beta = beta_final;
hist.fval_cluster = max(fval(1:N));
hist.fval_vol = fval(end);
hist.drho_change = norm(drho);
hist.asym_width_min = min(rho_width);
hist.asym_width_mean = mean(rho_width);
hist.asym_width_max = max(rho_width);
hist.asym_expand_count = nnz(rho_expand);
hist.asym_contract_count = nnz(rho_contract);
hist.asym_same_count = nEl - hist.asym_expand_count - hist.asym_contract_count;

state.iter = iter;
state.low = low;
state.upp = upp;
state.xold2 = state.xold1;
state.xold1 = xval;
end
