function T = check_mode_selection(quick)
%CHECK_MODE_SELECTION  Sections 4.1-4.2: sensitivity of the structural-mode selection.
%   For the final designs of the proposed method (three examples, alpha = 1) and of
%   Du-Olhoff (simply supported beam): omega_1, omega_2 for the 36 combinations of the
%   thresholds (density 0.05/0.1/0.2/0.3, energy share 0.3/0.5/0.7, participation
%   0.4/0.5/0.6), for Emin = 1e-9 E0, and for removal of the elements with x < 0.1
%   (no classifier).  About 15 min (quick: coarse meshes, 1 min).
if nargin < 1, quick = false; end
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
if quick, probs = {problem_ss_beam(160, 20), problem_clamped_beam(160, 20), problem_building(40, 120)};
else,     probs = {problem_ss_beam(), problem_clamped_beam(), problem_building()}; end
designs = {};
for k = 1:3, designs(end+1, :) = {probs{k}, 'proposed', proposed_freq(probs{k}).x}; end %#ok<AGROW>
designs(end+1, :) = {probs{1}, 'du_olhoff', du_olhoff(probs{1}).x};
rows = {};
for k = 1:size(designs, 1)
    [prob, meth, x] = designs{k, :};
    mdl = fe_mesh(prob);
    [om0, ~, info] = structural_modes(prob, x, 2, struct('mdl', mdl));
    g = [];
    for t = [0.05 0.1 0.2 0.3], for s = [0.3 0.5 0.7], for pt = [0.4 0.5 0.6]
        g(end+1, :) = structural_modes(prob, x, 2, struct('mdl', mdl, 'tLow', t, 'sCut', s, 'pThr', pt))'; %#ok<AGROW>
    end, end, end
    o9 = structural_modes(prob, x, 2, struct('mdl', mdl, 'Emin', 1e-9*prob.E, 'rhoMin', 1e-9));
    xk = x;  xk(x < 0.1) = 0;               % void removed: stiffness and mass zero there
    ok = removeVoid(prob, mdl, xk);
    fprintf(['%-13s %-9s omega1/2 = %8.2f / %8.2f (%d rejected below omega2) | 36 thresholds: omega1 %.2f-%.2f, ' ...
        'omega2 %.2f-%.2f | Emin 1e-9: %.2f/%.2f | void removed: %.2f/%.2f\n'], prob.name, meth, om0, ...
        info.nRejectedBelow, min(g(:,1)), max(g(:,1)), min(g(:,2)), max(g(:,2)), o9, ok);
    rows(end+1, :) = {prob.name, meth, om0(1), om0(2), min(g(:,1)), max(g(:,1)), min(g(:,2)), max(g(:,2)), ...
        o9(1), o9(2), ok(1), ok(2)}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'case', 'method', 'omega1', 'omega2', 'grid_omega1_min', 'grid_omega1_max', ...
    'grid_omega2_min', 'grid_omega2_max', 'emin1e9_omega1', 'emin1e9_omega2', 'void_removed_omega1', 'void_removed_omega2'});
end

function om = removeVoid(prob, mdl, x)
on = x > 0;
Ee = double(on).*x.^3*prob.E;  re = double(on).*x*prob.rho;
K = sparse(mdl.iK, mdl.jK, reshape(mdl.KE(:)*Ee', [], 1), mdl.ndof, mdl.ndof);  K = (K + K')/2;
M = sparse(mdl.iK, mdl.jK, reshape(mdl.ME(:)*re', [], 1), mdl.ndof, mdl.ndof);  M = (M + M')/2;
used = false(mdl.ndof, 1);  used(unique(mdl.edofMat(on, :))) = true;
f = mdl.free(used(mdl.free));
k = 12;                                     % loose fragments give near-zero mechanisms:
while true                                  % compute until two modes above 1 rad/s are found
    lam = sort(real(eigs(K(f,f), M(f,f), k, -1)));
    if nnz(lam > 1) >= 2 || k >= 384, break; end
    k = 2*k;
end
lam = lam(lam > 1);
om = [NaN NaN];  om(1:min(2, numel(lam))) = sqrt(lam(1:min(2, numel(lam))));
end
