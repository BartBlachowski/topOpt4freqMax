function [omega, Phi, lambda, info] = eigSolve(K, M, J, solver, v0, opts)
%EIGSOLVE  Lowest J eigenpairs of K*phi = lambda*M*phi on the reduced system.
%
%   [omega,Phi,lambda,info] = EIGSOLVE(K,M,J,solver,v0)
%
%   solver = 'dense'  full LAPACK eig(K,M).  Deterministic ordering, the
%                     reference per CLAUDE.md sec.6.  Cost O(n^3).
%          = 'eigs'   ARPACK shift-invert.  MANDATORY fixed start vector v0 --
%                     ARPACK's default start is random, which makes mode
%                     ordering non-deterministic near the degeneracies this
%                     study is about.  If v0 is not supplied a fixed
%                     deterministic vector is generated here.
%
%   opts (optional) for 'eigs': fields tol (default 1e-12), maxit (5000) and
%   pFactor (4, Krylov size p = max(20, pFactor*J)).  The defaults are the
%   constants this function always used, so omitting opts is bitwise identical.
%
%   Modes are returned M-orthonormalised: Phi'*M*Phi = I, so that the
%   generalized gradients f_sk of eq. (19) can be formed directly.

if nargin < 4 || isempty(solver), solver = 'dense'; end
if nargin < 6 || isempty(opts), opts = struct(); end
eigTol   = local_opt(opts, 'tol',     1e-12);
eigMaxit = local_opt(opts, 'maxit',   5000);
eigPfac  = local_opt(opts, 'pFactor', 4);
n = size(K,1);
t0 = tic;

switch lower(solver)
    case 'dense'
        [V, Dm] = eig(full(K), full(M), 'chol');
        d = diag(Dm);
        [d, idx] = sort(real(d), 'ascend');
        V = V(:, idx);
        lambda = d(1:J);
        Phi    = V(:, 1:J);
        info.solver = 'dense';

    case 'eigs'
        if nargin < 5 || isempty(v0)
            % deterministic, reproducible, and not orthogonal to the low modes
            v0 = sin((1:n)'*0.7071067811865476) + 0.5;
        end
        eopts = struct('v0', v0, 'tol', eigTol, 'maxit', eigMaxit, ...
                       'p', min(n, max(20, eigPfac*J)));
        [V, Dm, flag] = eigs(K, M, J, 'smallestabs', eopts);
        if flag ~= 0
            error('eigSolve:noconv','eigs did not converge (flag=%d)',flag);
        end
        d = diag(Dm);
        [d, idx] = sort(real(d), 'ascend');
        V = V(:, idx);
        lambda = d;
        Phi    = V;
        info.solver = 'eigs';

    otherwise
        error('eigSolve:solver','unknown solver %s',solver);
end

% ---- M-orthonormalise (eig 'chol' already does, eigs does; enforce anyway) --
for j = 1:size(Phi,2)
    s = sqrt(Phi(:,j)'*M*Phi(:,j));
    Phi(:,j) = Phi(:,j)/s;
end

lambda = lambda(:);
omega  = sqrt(max(lambda,0));
info.time = toc(t0);
info.J = J;
end

function v = local_opt(s, name, dflt)
if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = dflt; end
end
