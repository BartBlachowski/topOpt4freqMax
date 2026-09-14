function [omega, Phi, lambda, info] = eigSolve(K, M, J, solver, v0)
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
%   Modes are returned M-orthonormalised: Phi'*M*Phi = I, so that the
%   generalized gradients f_sk of eq. (19) can be formed directly.

if nargin < 4 || isempty(solver), solver = 'dense'; end
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
        opts = struct('v0', v0, 'tol', 1e-12, 'maxit', 5000, ...
                      'p', min(n, max(20, 4*J)));
        [V, Dm, flag] = eigs(K, M, J, 'smallestabs', opts);
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
