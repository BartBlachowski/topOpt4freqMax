function result = topReciprocalSQP(nelx, nely, volfrac, penal, rmin, opts)
%TOPRECIPROCALSQP Experimental reciprocal-variable topology optimizer.
%
%   This is a practical sequential reciprocal approximation for the same
%   minimum-compliance MBB benchmark used by topIQP.  It is intentionally
%   separate from topIQP.m: topIQP remains the Labanda reference path, while
%   this file tests the reciprocal-variable idea used in Long et al. (2019).
%
%   The design variable is x = rho^(-penal).  Each iteration builds a local
%   diagonal convex model in x, enforces move limits and the exact volume
%   by a scalar multiplier search, then accepts the step by compliance
%   backtracking.  No dense QP or quadprog call is used.
%
%   Usage:
%     result = topReciprocalSQP(nelx, nely, volfrac, penal, rmin)
%     result = topReciprocalSQP(nelx, nely, volfrac, penal, rmin, opts)
%
%   Options:
%     maxIter        maximum iterations                 default 200
%     move           density move limit per iteration   default 0.10
%     rhoMin         minimum design density             default 1e-3
%     tolChange      infinity-norm density stop         default 1e-3
%     tolRelObj      relative compliance stop           default 1e-5
%     minIter        minimum iterations before rel stop  default 20
%     curvatureScale reciprocal proximal curvature      default 1.0
%     displayEvery   print every k iterations           default 1

if nargin < 6 || isempty(opts)
    opts = struct();
end

max_iter = getOption(opts, 'maxIter', 200);
move = getOption(opts, 'move', 0.10);
rho_min = getOption(opts, 'rhoMin', 1e-3);
tol_change = getOption(opts, 'tolChange', 1e-3);
tol_rel_obj = getOption(opts, 'tolRelObj', 1e-5);
min_iter = getOption(opts, 'minIter', 20);
curvature_scale = getOption(opts, 'curvatureScale', 1.0);
display_every = getOption(opts, 'displayEvery', 1);

E1 = 1e2;
Ev = 1e-1;
nu = 0.3;

nEl = nelx * nely;
nNodes = (nelx + 1) * (nely + 1);
nDof = 2 * nNodes;
KE = keElastic(1.0, 1.0, nu);

% Element connectivity: LL -> LR -> UR -> UL.
elxV = floor((0:nEl-1)' / nely);
elyV = mod((0:nEl-1)', nely);
n1 = (nely+1) * elxV + elyV + 1;
n2 = (nely+1) * (elxV + 1) + elyV + 1;
n3 = n2 + 1;
n4 = n1 + 1;
edofMat = int32([2*n1-1, 2*n1, 2*n2-1, 2*n2, 2*n3-1, 2*n3, 2*n4-1, 2*n4]);
iK = reshape(kron(edofMat, ones(8,1,'int32'))', 64*nEl, 1);
jK = reshape(kron(edofMat, ones(1,8,'int32'))', 64*nEl, 1);

% MBB boundary conditions, matching topIQP.m.
leftNodes = 1:(nely+1);
botRightNode = (nely+1) * nelx + 1;
fixedDofs = unique([2*leftNodes-1, 2*botRightNode]');
freeDofs = setdiff((1:nDof)', fixedDofs);
topLeftNode = nely + 1;
F = zeros(nDof, 1);
F(2*topLeftNode) = -1.0;

Hnorm = buildDensityFilter(nelx, nely, rmin);

rho = volfrac * ones(nEl, 1);
rhoPhys = full(Hnorm * rho);

compHistory = zeros(max_iter, 1);
changeHistory = zeros(max_iter, 1);
alphaHistory = zeros(max_iter, 1);
lambdaHistory = zeros(max_iter, 1);

fprintf('TopReciprocalSQP - minimum compliance\n');
fprintf('  mesh %d x %d,  volfrac %.3f,  penal %.1f,  rmin %.2f\n', ...
        nelx, nely, volfrac, penal, rmin);
fprintf('  %5s  %14s  %8s  %6s  %9s  %9s\n', ...
        'Iter', 'Compliance', 'Vol', 'alpha', 'change', 'relDrop');

prev_comp = inf;
last_iter = 0;
for iter = 1:max_iter
    [f_val, df_drho, rhoPhys] = evaluateCompliance( ...
        rho, Hnorm, KE, E1, Ev, penal, iK, jK, edofMat, freeDofs, F, nDof);

    x = rho .^ (-penal);
    drho_dx = -(1 / penal) * rho ./ x;
    df_dx = df_drho .* drho_dx;

    % Positive diagonal model curvature.  This is a proximal term for the
    % reciprocal model; the scalar multiplier search handles the volume.
    curv = curvature_scale * abs(df_dx) ./ max(abs(x), 1);
    curv = max(curv, 1e-14);

    rho_low = max(rho_min, rho - move);
    rho_high = min(1.0, rho + move);
    x_low = rho_high .^ (-penal);
    x_high = rho_low .^ (-penal);

    [x_trial, lambda] = solveReciprocalStep( ...
        x, df_dx, drho_dx / nEl, curv, x_low, x_high, volfrac, penal);
    rho_trial = min(1.0, max(rho_min, x_trial .^ (-1 / penal)));

    alpha = 1.0;
    accepted = false;
    f_new = f_val;
    rho_new = rho;
    rhoPhys_new = rhoPhys;
    for ls = 1:12
        rho_candidate = (1 - alpha) * rho + alpha * rho_trial;
        rho_candidate = enforceVolume(rho_candidate, volfrac, rho_min);
        [f_candidate, ~, rhoPhys_candidate] = evaluateCompliance( ...
            rho_candidate, Hnorm, KE, E1, Ev, penal, iK, jK, ...
            edofMat, freeDofs, F, nDof);
        if f_candidate <= f_val || alpha < 1e-3
            accepted = true;
            f_new = f_candidate;
            rho_new = rho_candidate;
            rhoPhys_new = rhoPhys_candidate;
            break;
        end
        alpha = 0.5 * alpha;
    end

    if ~accepted
        warning('TopReciprocalSQP: no acceptable step at iteration %d.', iter);
        break;
    end

    change = norm(rho_new - rho, Inf);
    rel_drop = (prev_comp - f_new) / max(abs(prev_comp), 1);
    if ~isfinite(rel_drop)
        rel_drop = inf;
    end

    rho = rho_new;
    rhoPhys = rhoPhys_new;
    compHistory(iter) = f_new;
    changeHistory(iter) = change;
    alphaHistory(iter) = alpha;
    lambdaHistory(iter) = lambda;
    last_iter = iter;

    if mod(iter, display_every) == 0 || iter == 1
        fprintf('  %5d  %14.6g  %8.4f  %6.4f  %9.3e  %9.3e\n', ...
                iter, f_new, mean(rho), alpha, change, rel_drop);
    end

    if change < tol_change
        fprintf('  Converged by density change at iteration %d.\n', iter);
        break;
    end
    if iter >= min_iter && abs(rel_drop) < tol_rel_obj
        fprintf('  Converged by relative objective change at iteration %d.\n', iter);
        break;
    end

    prev_comp = f_new;
end

if last_iter == 0
    last_iter = 1;
    compHistory(1) = f_val;
    changeHistory(1) = 0;
    alphaHistory(1) = 0;
    lambdaHistory(1) = 0;
end

result.xPhys = reshape(rhoPhys, nely, nelx);
result.xTilde = reshape(rho, nely, nelx);
result.compliance = compHistory(last_iter);
result.nIter = last_iter;
result.compHistory = compHistory(1:last_iter);
result.changeHistory = changeHistory(1:last_iter);
result.alphaHistory = alphaHistory(1:last_iter);
result.lambdaHistory = lambdaHistory(1:last_iter);
result.method = 'reciprocal-scalar-qp';
end

function [f_val, df_drho, rhoPhys] = evaluateCompliance( ...
    rho, Hnorm, KE, E1, Ev, penal, iK, jK, edofMat, freeDofs, F, nDof)

rhoPhys = full(Hnorm * rho);
nEl = numel(rho);
Ee = Ev + (E1 - Ev) * rhoPhys(:) .^ penal;
sK = reshape(KE(:) * Ee', 64*nEl, 1);
K = sparse(double(iK), double(jK), sK, nDof, nDof);
K = (K + K') / 2;
u = zeros(nDof, 1);
u(freeDofs) = K(freeDofs, freeDofs) \ F(freeDofs);
ue = u(edofMat);
ce = sum((ue * KE) .* ue, 2);
f_val = Ee' * ce;
df_dphys = -penal * (E1 - Ev) * rhoPhys(:).^(penal-1) .* ce;
df_drho = Hnorm' * df_dphys;
end

function [x_new, lambda] = solveReciprocalStep( ...
    x, grad, vol_grad, curv, x_low, x_high, volfrac, penal)

    lambda = 0.0;
    x_new = projectX(x - grad ./ curv, x_low, x_high);
    if mean(x_new .^ (-1 / penal)) <= volfrac
        return;
    end

    lo = 0.0;
    hi = 1.0;
    for k = 1:80
        x_hi = projectX(x - (grad + hi * vol_grad) ./ curv, x_low, x_high);
        if mean(x_hi .^ (-1 / penal)) <= volfrac
            break;
        end
        hi = 2.0 * hi;
    end

    for k = 1:80
        lambda = 0.5 * (lo + hi);
        x_mid = projectX(x - (grad + lambda * vol_grad) ./ curv, x_low, x_high);
        if mean(x_mid .^ (-1 / penal)) > volfrac
            lo = lambda;
        else
            hi = lambda;
        end
    end

    lambda = hi;
    x_new = projectX(x - (grad + lambda * vol_grad) ./ curv, x_low, x_high);
end

function x = projectX(x, x_low, x_high)
    x = min(x_high, max(x_low, x));
end

function rho = enforceVolume(rho, volfrac, rho_min)
    rho = min(1.0, max(rho_min, rho));
    current = mean(rho);
    if current <= volfrac
        return;
    end
    lo = rho_min;
    hi = max(rho);
    for k = 1:80
        eta = 0.5 * (lo + hi);
        projected = min(rho, eta);
        if mean(projected) > volfrac
            hi = eta;
        else
            lo = eta;
        end
    end
    rho = min(rho, lo);
end

function Hnorm = buildDensityFilter(nelx, nely, rmin)
    nEl = nelx * nely;
    if rmin <= 0
        Hnorm = speye(nEl);
        return;
    end

    filterRadius = ceil(rmin) - 1;
    maxEntries = nEl * (2 * filterRadius + 1)^2;
    iH = zeros(maxEntries, 1);
    jH = zeros(maxEntries, 1);
    sH = zeros(maxEntries, 1);
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
    H = sparse(iH(1:kH), jH(1:kH), sH(1:kH), nEl, nEl);
    Hs = full(sum(H, 2));
    Hnorm = spdiags(1 ./ Hs, 0, nEl, nEl) * H;
end

function value = getOption(opts, name, defaultValue)
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = defaultValue;
    end
end

function KE = keElastic(hx, hy, nu)
    D = (1/(1-nu^2)) * [1 nu 0; nu 1 0; 0 0 (1-nu)/2];
    iJ = diag([2/hx, 2/hy]);
    dJ = 0.25 * hx * hy;
    gp = 1/sqrt(3);
    KE = zeros(8,8);
    for xi = [-gp, gp]
        for eta = [-gp, gp]
            dNdxi = 0.25 * [-(1-eta), (1-eta), (1+eta), -(1+eta)];
            dNdeta = 0.25 * [-(1-xi), -(1+xi), (1+xi), (1-xi)];
            dNxy = iJ * [dNdxi; dNdeta];
            dNdx = dNxy(1,:);
            dNdy = dNxy(2,:);
            B = zeros(3,8);
            B(1,1:2:end) = dNdx;
            B(2,2:2:end) = dNdy;
            B(3,1:2:end) = dNdy;
            B(3,2:2:end) = dNdx;
            KE = KE + (B' * D * B) * dJ;
        end
    end
end
