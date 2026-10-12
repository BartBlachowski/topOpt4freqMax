function [omega, Phi, info] = lowest_structural_modes(Kf, Mf, free, ndof, edofMat, KE, ME, Ee, rhoEl, xEl, nOut, kStart)
%LOWEST_STRUCTURAL_MODES  The nOut lowest STRUCTURAL eigenmodes of K phi = lambda M phi.
%
%   [omega, Phi, info] = LOWEST_STRUCTURAL_MODES(Kf, Mf, free, ndof, edofMat,
%       KE, ME, Ee, rhoEl, xEl, nOut, kStart)
%
%   Arguments as STRUCTURAL_MODE_MASK, which decides what is structural.
%   Modes are requested in a batch of kStart (default max(6, 2*nOut)) that
%   doubles until nOut structural ones are found, so a design without void
%   modes costs exactly one solve of kStart modes.
%
%   Returns omega (nOut x 1, rad/s, NaN where fewer were found), Phi (free DOFs
%   x nOut), info.nComputed (modes in the last batch) and info.nVoid (modes
%   rejected below the highest one returned).
%
%   See also STRUCTURAL_MODE_MASK.

if nargin < 11 || isempty(nOut), nOut = 1; end
if nargin < 12 || isempty(kStart), kStart = max(6, 2*nOut); end
limit = max(1, size(Kf, 1) - 1);
opts = struct('tol', 1e-10, 'maxit', 20000, 'v0', localStartVector(size(Kf, 1)));

k = min(max(kStart, nOut), limit);
while true
    [V, D] = eigs(Kf, Mf, k, 'smallestabs', opts);
    [lam, ix] = sort(real(diag(D)), 'ascend');
    V = V(:, ix);
    keep = structural_mode_mask(V, lam, Kf, Mf, free, ndof, edofMat, KE, ME, Ee, rhoEl, xEl);
    if nnz(keep) >= nOut || k >= limit, break; end
    k = min(2*k, limit);
end

sel = find(keep);
n = min(nOut, numel(sel));
omega = nan(nOut, 1);
Phi = zeros(size(Kf, 1), nOut);
omega(1:n) = sqrt(lam(sel(1:n)));
Phi(:, 1:n) = V(:, sel(1:n));
top = 0; if n > 0, top = sel(n); end
info = struct('nComputed', k, 'nVoid', sum(~keep(1:top)));
end

function v = localStartVector(n)
% Deterministic, so the same design always gives the same modes.
v = sin((1:n)' * 0.7071067811865476) + 0.5;
end
