function [omega, Phi, info] = structural_modes(prob, x, nOut, opts)
%STRUCTURAL_MODES  Common evaluator of the paper: the lowest nOut STRUCTURAL
%   eigenfrequencies of a design (Sections 2.4 and 4.1).
%
%   [omega, Phi, info] = structural_modes(prob, x)          nOut = 2
%   [omega, Phi, info] = structural_modes(prob, x, nOut, opts)
%
%   Model: SIMP stiffness E = Emin + x^3 (E0 - Emin) with Emin = 1e-6 E0 and linear
%   mass rho = 1e-6 + x (1 - 1e-6) (opts.Emin, opts.rhoMin override).  x in the common
%   numbering (fe_mesh).  A computed mode is kept when its eigen-residual is <= 1e-6,
%   less than half of its kinetic and of its strain energy lies in elements with
%   x <= opts.tLow (0.1), and its kinetic-energy-weighted density exceeds opts.pThr
%   (0.5; skipped on a uniform design).  Modes failing these tests are localized in
%   the void and are not frequencies of the structure.
%
%   info.lambdaRaw / info.keep / info.Vraw: all computed eigenpairs (free dofs) and the
%   selection mask, info.Mf: mass matrix on the free dofs,
%   info.idx: raw indices of the returned modes, Phi: modes on all dofs.
if nargin < 3 || isempty(nOut), nOut = 2; end
if nargin < 4, opts = struct(); end
Emin = getopt(opts, 'Emin', 1e-6*prob.E);  rhoMin = getopt(opts, 'rhoMin', 1e-6*prob.rho);
tLow = getopt(opts, 'tLow', 0.1);  sCut = getopt(opts, 'sCut', 0.5);  pThr = getopt(opts, 'pThr', 0.5);
mdl = getopt(opts, 'mdl', []);  if isempty(mdl), mdl = fe_mesh(prob); end
x = min(1, max(0, x(:)));
[K, M, Ee, re] = assemble_KM(mdl, x, prob.E, Emin, prob.rho, rhoMin);
f = mdl.free;  Kf = K(f,f);  Mf = M(f,f);
limit = max(1, size(Kf,1) - 1);
eo = struct('tol', 1e-10, 'maxit', 20000, 'v0', sin((1:size(Kf,1))'*0.7071067811865476) + 0.5);
k = min(max(getopt(opts, 'kStart', 6), nOut), limit);
low = x <= tLow;  uniform = max(x) - min(x) <= 1e-12;
U = zeros(mdl.ndof, 1);
while true
    [V, D] = eigs(Kf, Mf, k, 'smallestabs', eo);
    [lam, ix] = sort(real(diag(D)), 'ascend');  V = V(:, ix);
    keep = false(numel(lam), 1);
    for j = 1:numel(lam)
        u = V(:, j);
        if ~(isfinite(lam(j)) && lam(j) > 0), continue; end
        res = norm(Kf*u - lam(j)*(Mf*u)) / (norm(Kf*u) + abs(lam(j))*norm(Mf*u) + eps);
        if ~(res <= 1e-6), continue; end
        U(f) = u;  ue = U(mdl.edofMat);
        ke = max(re .* sum((ue*mdl.ME) .* ue, 2), 0);  se = max(Ee .* sum((ue*mdl.KE) .* ue, 2), 0);
        if ~(sum(ke) > 0 && sum(se) > 0), continue; end
        ke = ke/sum(ke);  se = se/sum(se);
        keep(j) = sum(ke(low)) < sCut && sum(se(low)) < sCut && (uniform || sum(ke .* x) > pThr);
    end
    if nnz(keep) >= nOut || k >= limit, break; end
    k = min(2*k, limit);
end
sel = find(keep);  nn = min(nOut, numel(sel));
omega = nan(nOut, 1);  omega(1:nn) = sqrt(lam(sel(1:nn)));
Phi = zeros(mdl.ndof, nOut);  Phi(f, 1:nn) = V(:, sel(1:nn));
info = struct('lambdaRaw', lam, 'keep', keep, 'Vraw', V, 'Mf', Mf, 'free', f, 'idx', sel(1:nn), ...
    'nRejectedBelow', sum(~keep(1:max([sel(1:nn); 0]))));
end

function v = getopt(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
