function omega = e1_structural_omegas(xHist, nelx, nely, nOut)
%E1_STRUCTURAL_OMEGAS  First nOut structural eigenfrequencies under evaluator E1.
%
%   omega = E1_STRUCTURAL_OMEGAS(xHist, nelx, nely, nOut) evaluates every
%   column of xHist (n_e x n_iter physical densities, element order
%   e = ey + ex*nely + 1) with the frozen common evaluator E1 of
%   examples/Performance/benchmark_profile/study_evaluate_design.m and returns
%   an n_iter x nOut matrix [rad/s] of the nOut lowest STRUCTURAL modes.
%
%   E1 and its classifier are reproduced here verbatim: SIMP p = 3 with
%   E_min = 1e-6 E_0, linear mass with rho_min = 1e-6, pins at the mid-height
%   nodes of both ends; a mode is structural when its eigenpair residual is
%   <= 1e-6, its kinetic- and strain-energy shares in elements with
%   x_e <= 0.1 are both below 0.5, and its kinetic-energy-weighted density
%   participation exceeds 0.5.  One deviation: on a uniform design (the
%   starting design x_e = V_f) every mode has a density participation of
%   exactly 0.5, the last test is undefined there and is skipped; on every
%   other iterate of the Fig. 9 runs the selection is identical to E1's.
%   study_evaluate_design reports only the FIRST
%   structural mode; this helper keeps requesting modes (doubling, as that
%   evaluator does) until nOut structural modes are found.  Missing modes are
%   NaN.  For the simply supported beam benchmark only.

if nargin < 4, nOut = 3; end
[KE,ME] = q4_matrices(8/nelx, 1/nely, 0.3, 1.0);
[iK,jK,edof] = assembly_indices(nelx, nely);
ndof = 2*(nelx+1)*(nely+1);
jMid = round(nely/2); nL = jMid; nR = nelx*(nely+1) + jMid;
fixed = [2*nL+1; 2*nL+2; 2*nR+1; 2*nR+2];
free = setdiff((1:ndof)', fixed);
limit = numel(free) - 1;

nIter = size(xHist, 2);
omega = nan(nIter, nOut);
for it = 1:nIter
    z = max(0, min(1, double(xHist(:,it))));
    Ee = 1e7*(1e-6 + (1-1e-6)*z.^3);
    rr = 1e-6 + (1-1e-6)*z;
    K = sparse(iK, jK, reshape(KE(:)*Ee', [], 1), ndof, ndof); K = (K+K')/2;
    M = sparse(iK, jK, reshape(ME(:)*rr', [], 1), ndof, ndof); M = (M+M')/2;
    Kf = K(free,free); Mf = M(free,free);
    k = min(max(6, 2*nOut), limit);
    while true
        [om, valid] = structural_batch(Kf, Mf, free, ndof, edof, KE, ME, Ee, rr, z, k);
        sel = om(valid);
        if numel(sel) >= nOut || k >= limit, break; end
        k = min(2*k, limit);
    end
    n = min(nOut, numel(sel));
    omega(it,1:n) = sel(1:n);
end
end

function [omega, valid] = structural_batch(Kf, Mf, free, ndof, edof, KE, ME, Ee, rr, zeff, k)
opts = struct('disp',0,'maxit',200000,'tol',1e-10,'v0',deterministic_v0(size(Kf,1)));
try
    [V,D] = eigs(Kf, Mf, k, 'smallestabs', opts);
catch
    [V,D] = eigs(Kf, Mf, k, 'sm', opts);
end
lam = real(diag(D)); [lam,ix] = sort(lam, 'ascend'); V = V(:,ix);
omega = sqrt(max(lam,0)); U = zeros(ndof,k); U(free,:) = V;
low = zeff <= 0.1; valid = false(k,1);
uniform = max(zeff) - min(zeff) <= 1e-12;
for j = 1:k
    u = V(:,j); denom = norm(Kf*u) + abs(lam(j))*norm(Mf*u) + eps;
    residual = norm(Kf*u - lam(j)*(Mf*u))/denom;
    if ~(isfinite(lam(j)) && lam(j) > 0 && isfinite(residual) && residual <= 1e-6), continue; end
    ue = reshape(U(edof,j), size(edof));
    ke = max(rr.*sum((ue*ME).*ue,2), 0); se = max(Ee.*sum((ue*KE).*ue,2), 0);
    if ~(sum(ke) > 0 && sum(se) > 0), continue; end
    ken = ke/sum(ke); sen = se/sum(se);
    valid(j) = sum(ken(low)) < 0.5 && sum(sen(low)) < 0.5 && (uniform || sum(ken.*zeff) > 0.5);
end
end

function v = deterministic_v0(n)
s = RandStream('twister','Seed',42); v = randn(s,n,1); v = v/norm(v);
end

function [iK,jK,edof] = assembly_indices(nelx, nely)
nEl = nelx*nely; edof = zeros(nEl,8);
for ex = 0:nelx-1
    for ey = 0:nely-1
        e = ey + ex*nely + 1; n1 = (nely+1)*ex + ey; n2 = (nely+1)*(ex+1) + ey;
        edof(e,:) = [2*n1+1 2*n1+2 2*n2+1 2*n2+2 2*(n2+1)+1 2*(n2+1)+2 2*(n1+1)+1 2*(n1+1)+2];
    end
end
iK = reshape(kron(edof,ones(1,8))',[],1); jK = reshape(kron(edof,ones(8,1))',[],1);
end

function [KE,ME] = q4_matrices(hx, hy, nu, t)
D = (1/(1-nu^2))*[1 nu 0; nu 1 0; 0 0 0.5*(1-nu)]; invJ = [2/hx 0; 0 2/hy];
detJ = 0.25*hx*hy; gp = 1/sqrt(3); KE = zeros(8);
for xi = [-gp gp]
    for eta = [-gp gp]
        a = 0.25*[-(1-eta) (1-eta) (1+eta) -(1+eta)];
        b = 0.25*[-(1-xi) -(1+xi) (1+xi) (1-xi)]; d = invJ*[a; b]; B = zeros(3,8);
        B(1,1:2:end) = d(1,:); B(2,2:2:end) = d(2,:); B(3,1:2:end) = d(2,:); B(3,2:2:end) = d(1,:);
        KE = KE + B'*D*B*detJ;
    end
end
KE = t*KE; Ms = (hx*hy/36)*[4 2 1 2; 2 4 2 1; 1 2 4 2; 2 1 2 4]; ME = t*kron(Ms, eye(2));
end
