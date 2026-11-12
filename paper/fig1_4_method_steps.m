function fig1_4_method_steps()
%FIG1_4_METHOD_STEPS  Section 2.2, Figures 1-4: the steps of the proposed method on the
%   simply supported beam (400x50): (1) first mode shape of the solid design domain,
%   (2) its inertial forces f = omega^2 M phi on the domain, (3) the inertial forces on
%   the final topology, (4) the final topology.  Under 1 min.  Output: results/fig1_4/.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('fig1_4');
prob = problem_ss_beam();
P = proposed_freq(prob);
mdl = P.mdl;  phi = P.phi0(:,1);  s = P.omega0(1)^2;
[X, Y] = meshgrid((0:mdl.nelx)*mdl.hx, (0:mdl.nely)*mdl.hy);     % nodes, column-major from bottom
ux = reshape(phi(1:2:end), mdl.nely+1, mdl.nelx+1);  uy = reshape(phi(2:2:end), mdl.nely+1, mdl.nelx+1);
sc = 0.3/max(abs(phi));
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1000 200]);
mesh(X + sc*ux, Y + sc*uy, zeros(size(X)), 'EdgeColor', [0.4 0.4 0.4]);  view(2);  axis equal off;
exportgraphics(fig, fullfile(out, 'fig1_mode_shape.png'), 'Resolution', 200);  close(fig);
plotForces(mdl, prob, ones(mdl.nEl,1), phi, s, fullfile(out, 'fig2_inertial_forces_domain.png'), X, Y);
plotForces(mdl, prob, P.x, phi, s, fullfile(out, 'fig3_inertial_forces_topology.png'), X, Y);
save_design_png(prob, P.x, fullfile(out, 'fig4_final_topology.png'));
fprintf('omega_1^(0) = %.2f rad/s, %d iterations, final structural omega1 = %.2f rad/s\n', ...
    P.omega0(1), P.iterations, structural_modes(prob, P.x, 1));
end

function plotForces(mdl, prob, x, phi, s, file, X, Y)
[~, M] = assemble_KM(mdl, x, prob.E, prob.Emin, prob.rho, prob.rhoMin);
f = s*(M*phi);
fx = reshape(f(1:2:end), mdl.nely+1, mdl.nelx+1);  fy = reshape(f(2:2:end), mdl.nely+1, mdl.nelx+1);
k = 1:8:mdl.nelx+1;  j = 1:5:mdl.nely+1;                       % every 8th / 5th node
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1000 220]);
imagesc([0 prob.L], [0 prob.H], 1 - reshape(x, prob.nely, prob.nelx));  colormap(gray);  caxis([0 1]);
set(gca, 'YDir', 'normal');  hold on;
quiver(X(j,k), Y(j,k), fx(j,k), fy(j,k), 1.5, 'r');  axis equal off;
exportgraphics(fig, file, 'Resolution', 200);  close(fig);
end
