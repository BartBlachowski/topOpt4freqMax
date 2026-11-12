function T = sec4_3_reference_mode_study()
%SEC4_3_REFERENCE_MODE_STUDY  Section 4.3: building (alpha = 1) with three reference
%   shapes, about 1 min: the solid-domain mode (the method), the fundamental mode of a
%   nonuniform feasible design (density decreasing linearly with height at the same
%   volume) and the solid mode with a smooth 10% perturbation (mass norm, supports kept).
%   Output: results/reference_mode/reference_mode.csv.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('reference_mode');
prob = problem_building();
base = proposed_freq(prob, struct('maxIter', 1));               % only for the mesh and phi0
mdl = base.mdl;  f = mdl.free;  v = base.phi0(:,1);  om0 = base.omega0(1);
[~, Ms] = assemble_KM(mdl, ones(mdl.nEl,1), prob.E, prob.Emin, prob.rho, prob.rhoMin);
% graded design: active density 1.6 at the base to 0.4 at the top (relative), same volume
x = prob.volfrac*ones(mdl.nEl,1);  x(mdl.pas) = 1;
x(mdl.act) = (prob.volfrac*mdl.nEl - numel(mdl.pas))/numel(mdl.act);
yc = repmat(((1:mdl.nely)' - 0.5)/mdl.nely, mdl.nelx, 1);  prof = 1.6 - 1.2*yc;
x(mdl.act) = x(mdl.act).*prof(mdl.act)/mean(prof(mdl.act));  x = min(1, max(1e-3, x));
[K, M] = assemble_KM(mdl, x, prob.E, prob.Emin, prob.rho, prob.rhoMin);
[V, D] = eigs(K(f,f), M(f,f), 1, 'smallestabs', struct('tol', 1e-12, 'v0', sin((1:numel(f))'*0.7071067811865476) + 0.5));
vg = zeros(mdl.ndof,1);  vg(f) = V/sqrt(V'*Ms(f,f)*V);
% perturbed solid mode
nn = mdl.ndof/2;  ids = (1:nn)';
nx = floor((ids-1)/(mdl.nely+1))/mdl.nelx;  ny = mod(ids-1, mdl.nely+1)/mdl.nely;
pert = zeros(mdl.ndof,1);  pert(1:2:end) = sin(3*pi*ny).*cos(pi*nx);  pert(2:2:end) = 0.5*sin(2*pi*ny).*sin(2*pi*nx);
pert(setdiff(1:mdl.ndof, f)) = 0;
vp = v + 0.10*pert/sqrt(pert(f)'*Ms(f,f)*pert(f));  vp = vp/sqrt(vp(f)'*Ms(f,f)*vp(f));
refs = {'solid', v, om0;  'graded', vg, sqrt(D);  'perturbed', vp, om0};
rows = {};
for k = 1:3
    P = proposed_freq(prob, struct('reference', struct('omega0', refs{k,3}, 'phi0', refs{k,2})));
    om = structural_modes(prob, P.x);
    fprintf('%-9s reference: MAC with solid mode %.3f, %d iterations, omega1/2 = %.2f / %.2f rad/s\n', refs{k,1}, ...
        mac_mass(refs{k,2}(f), v(f), Ms(f,f)), P.iterations, om);
    rows(end+1, :) = {refs{k,1}, mac_mass(refs{k,2}(f), v(f), Ms(f,f)), P.iterations, om(1), om(2)}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'reference', 'MAC_with_solid', 'iterations', 'omega1', 'omega2'});
writetable(T, fullfile(out, 'reference_mode.csv'));
end
