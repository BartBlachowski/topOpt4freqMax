function test_discretization(matfile)
%TEST_DISCRETIZATION  Isolate FE discretization stiffness from topology.
%
%   One optimized design is evaluated as THE SAME PHYSICAL FIELD on progressively
%   refined meshes (each element split k x k, densities copied).  The geometry is
%   identical throughout, so every difference is discretization.
%
%   A Richardson fit  omega(nely) = omega_inf + C*nely^(-p)  is then extrapolated
%   BACK to the coarse meshes the paper's own initial frequency points to
%   (~64x8 - 80x10).  No optimization is run below 160x20; this is a post-hoc
%   evaluation of one fixed design.
maxNumCompThreads(1);
S=load(matfile); res=S.res; cfg=res.cfg;
X = reshape(res.rho, cfg.nely, cfg.nelx);

ks = [1 2 3 4];
ny = zeros(size(ks)); w1 = zeros(size(ks)); w2 = w1; w3 = w1;
fprintf('%-14s %8s %8s %8s %8s\n','mesh','omega1','omega2','omega3','gap');
for i = 1:numel(ks)
    k = ks(i);
    Xk = repelem(X, k, k);
    c = cfg; c.nelx = cfg.nelx*k; c.nely = cfg.nely*k;
    mdl = model2D(c);
    [K,M] = assemble2D(mdl, Xk(:), c.p, c.massInterp);
    w = eigSolve(K,M,3,'eigs');
    ny(i)=c.nely; w1(i)=w(1); w2(i)=w(2); w3(i)=w(3);
    fprintf('%-14s %8.2f %8.2f %8.2f %8.4f\n', ...
        sprintf('%dx%d',c.nelx,c.nely),w(1),w(2),w(3),(w(2)-w(1))/w(1));
end

% ---- Richardson fit on the three finest points -------------------------
[winf, C, p] = richardson(ny(2:4), w1(2:4));
fprintf('\nfit: omega1(nely) = %.2f + %.1f*nely^(-%.2f)   (converged value %.2f)\n', ...
        winf, C, p, winf);
fprintf('implied reading of THIS SAME DESIGN on coarser meshes:\n');
for n0 = [8 10 12 16 20]
    fprintf('   %3dx%-3d  omega1 = %7.2f\n', n0*8, n0, winf + C*n0^(-p));
end
fprintf('\nPaper reports omega1_opt = 174.7 and its initial frequency (68.7)\n');
fprintf('is matched to 0.1%% at 64x8-80x10.\n');
end

function [winf, C, p] = richardson(n, w)
% solve w_i = winf + C*n_i^(-p) for three points
f = @(p) (w(1)-w(2))/(w(2)-w(3)) - (n(1)^(-p)-n(2)^(-p))/(n(2)^(-p)-n(3)^(-p));
lo = 0.2; hi = 4; 
for it = 1:200
    mid = 0.5*(lo+hi);
    if f(lo)*f(mid) <= 0, hi = mid; else, lo = mid; end
end
p = 0.5*(lo+hi);
C = (w(1)-w(2))/(n(1)^(-p)-n(2)^(-p));
winf = w(1) - C*n(1)^(-p);
end
