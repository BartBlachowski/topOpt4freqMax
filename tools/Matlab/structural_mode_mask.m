function keep = structural_mode_mask(V, lam, Kf, Mf, free, ndof, edofMat, KE, ME, Ee, rhoEl, xEl)
%STRUCTURAL_MODE_MASK  Which computed eigenmodes are modes of the STRUCTURE.
%
%   keep = STRUCTURAL_MODE_MASK(V, lam, Kf, Mf, free, ndof, edofMat, KE, ME,
%       Ee, rhoEl, xEl)
%
%   V, lam    eigenvectors (free DOFs x k) and eigenvalues (k x 1) of
%             Kf phi = lambda Mf phi
%   Kf, Mf    stiffness and mass restricted to the free DOFs
%   free      free DOF indices into the ndof global DOFs
%   edofMat   nEl x 8 element DOF map (1-based, global numbering)
%   KE, ME    8 x 8 unit element stiffness and mass matrices
%   Ee, rhoEl element stiffness and mass factors, so that K = sum Ee*KE and
%             M = sum rhoEl*ME (the caller's own material interpolation)
%   xEl       element physical densities in [0, 1]
%
%   On a design with near-void elements, a SIMP stiffness E_min + x^p E_0 next
%   to a linear mass leaves those elements with a tiny local frequency, so the
%   lowest computed modes can be vibrations of the void region alone.  They are
%   not frequencies of the structure.  A mode is kept with the classifier of
%   the frozen common evaluator E1
%   (examples/Performance/benchmark_profile/study_evaluate_design.m): its
%   eigenpair residual is <= 1e-6, its kinetic- and strain-energy shares in
%   elements with x <= 0.1 are both below 0.5, and its kinetic-energy-weighted
%   density participation exceeds 0.5 (skipped on a uniform design, where it is
%   exactly 0.5 for every mode).
%
%   See also LOWEST_STRUCTURAL_MODES.

xEl = max(0, min(1, double(xEl(:))));
Ee = double(Ee(:));  rhoEl = double(rhoEl(:));
lam = real(lam(:));
low = xEl <= 0.1;
uniform = max(xEl) - min(xEl) <= 1e-12;
keep = false(numel(lam), 1);
U = zeros(ndof, 1);
for j = 1:numel(lam)
    if ~(isfinite(lam(j)) && lam(j) > 0), continue; end
    u = V(:, j);
    res = norm(Kf*u - lam(j)*(Mf*u)) / (norm(Kf*u) + abs(lam(j))*norm(Mf*u) + eps);
    if ~(isfinite(res) && res <= 1e-6), continue; end
    U(free) = u;
    ue = U(edofMat);
    ke = max(rhoEl .* sum((ue*ME) .* ue, 2), 0);
    se = max(Ee .* sum((ue*KE) .* ue, 2), 0);
    if ~(sum(ke) > 0 && sum(se) > 0), continue; end
    ken = ke / sum(ke);  sen = se / sum(se);
    keep(j) = sum(ken(low)) < 0.5 && sum(sen(low)) < 0.5 && ...
              (uniform || sum(ken .* xEl) > 0.5);
end
end
