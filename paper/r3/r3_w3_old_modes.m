function T = r3_w3_old_modes()
%R3_W3_OLD_MODES  Reviewer 3, W3: the low omega1 of the SUBMITTED clamped-beam results
%   (275.28 rad/s at alpha = 1, 347.70 rad/s at alpha = 0.75; old code a2261f7).
%   Loads the regenerated old designs (results/r3/w3_old_designs.mat from
%   r3_w3_regenerate_old) and evaluates them
%   (1) with the CURRENT common evaluator and structural-mode selection
%       (structural_modes: SIMP p = 3, E_min = 1e-6 E0, linear mass, rho_min = 1e-6),
%   (2) on the model with all elements x < 0.1 REMOVED (no floor material there; dofs
%       of removed regions deleted).
%   For the lowest raw modes of (1): selection verdict, kinetic / strain energy share
%   in x <= 0.1 and in 0.1 < x < 0.3, kinetic-energy-weighted density, mass-MAC with
%   phi1 of the uniform design, kinetic-energy share per region, mode-shape PNG.
%   Output: results/r3/w3_old_modes.csv, w3_removed_model.csv, w3_alpha*_mode*.png,
%   w3_old_designs.png.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
maxNumCompThreads(1);
out = results_dir('r3');
S = load(fullfile(out, 'w3_old_designs.mat'));  old = S.old;
prob = problem_clamped_beam(400, 50);  mdl = fe_mesh(prob);  f = mdl.free;
nely = prob.nely;  nelx = prob.nelx;
% centroids of the elements (common numbering)
[cy, cx] = ndgrid(((1:nely) - 0.5)*mdl.hy, ((1:nelx) - 0.5)*mdl.hx);  cx = cx(:);  cy = cy(:);
% phi1 of the uniform design (x = V_f): same shapes as the solid domain
xu = prob.volfrac*ones(mdl.nEl,1);
[Ku, Mu] = assemble_KM(mdl, xu, prob.E, 1e-6*prob.E, prob.rho, 1e-6*prob.rho);
[Vu, Du] = eigs(Ku(f,f), Mu(f,f), 2, 'smallestabs');  [~, iu] = sort(diag(Du));  phiU = zeros(mdl.ndof,1);  phiU(f) = Vu(:, iu(1));
fprintf('uniform design: omega1 = %.2f rad/s\n', sqrt(min(diag(Du))));
rows = {};  rrows = {};  nRaw = 12;
for a = 1:numel(old)
    xo = double(old(a).x(:));  al = old(a).alpha;
    % orientation: old code numbering -> common numbering, chosen by reproducing omega_1
    cand = {xo, reshape(flipud(reshape(xo, nely, nelx)), [], 1), reshape(reshape(xo, nelx, nely)', [], 1)};
    w1 = zeros(1,3);
    for c = 1:3
        [Kc, Mc] = assemble_KM(mdl, min(1, max(0, cand{c})), prob.E, 1e-6*prob.E, prob.rho, 1e-6*prob.rho);
        w1(c) = sqrt(eigs(Kc(f,f), Mc(f,f), 1, 'smallestabs'));
    end
    [dmin, c] = min(abs(w1 - old(a).omega(1)));  x = min(1, max(0, cand{c}));
    fprintf('alpha = %.2f: old code omega1 %.4f, here %.4f (orientation %d)\n', al, old(a).omega(1), w1(c), c);
    assert(dmin < 1e-3*old(a).omega(1));
    old(a).xc = x;
    % (1) current selection
    [om, ~, info] = structural_modes(prob, x, 2, struct('kStart', nRaw));
    Mf = info.Mf;  lam = info.lambdaRaw;  V = info.Vraw;
    [K, M, Ee, re] = assemble_KM(mdl, x, prob.E, 1e-6*prob.E, prob.rho, 1e-6*prob.rho);
    low = x <= 0.1;  mid = x > 0.1 & x < 0.3;
    for j = 1:min(nRaw, numel(lam))
        U = zeros(mdl.ndof,1);  U(f) = V(:,j);  ue = U(mdl.edofMat);
        ke = max(re .* sum((ue*mdl.ME).*ue, 2), 0);  se = max(Ee .* sum((ue*mdl.KE).*ue, 2), 0);
        ke = ke/sum(ke);  se = se/sum(se);
        mac = mac_mass(U(f), phiU(f), Mf);
        % kinetic energy per region: left / right quarter, middle half; top flange / web / bottom flange
        regL = cx < 2;  regR = cx > 6;  regC = ~regL & ~regR;
        top = cy > 0.8;  bot = cy < 0.2;  web = ~top & ~bot;
        rows(end+1,:) = {al, j, sqrt(lam(j)), info.keep(j), sum(ke(low)), sum(se(low)), sum(ke(mid)), sum(se(mid)), ...
            sum(ke.*x), mac, sum(ke(regL)), sum(ke(regC)), sum(ke(regR)), sum(ke(top)), sum(ke(web)), sum(ke(bot)), ...
            sum(ke(x >= 0.3))}; %#ok<AGROW>
        if j <= 4, plotMode(prob, mdl, x, U, ke, sqrt(lam(j)), al, j, info.keep(j), mac, fullfile(out, sprintf('w3_alpha%03d_mode%d.png', round(100*al), j))); end
    end
    fprintf('alpha = %.2f: current evaluator structural omega = %.2f, %.2f rad/s; raw = %s\n', al, om, mat2str(sqrt(lam(1:6))', 6));
    % (2) elements x < 0.1 removed
    keepE = x >= 0.1;
    Ee2 = prob.E*x.^3;  re2 = prob.rho*x;  Ee2(~keepE) = 0;  re2(~keepE) = 0;
    K2 = sparse(mdl.iK, mdl.jK, reshape(mdl.KE(:)*Ee2', [], 1), mdl.ndof, mdl.ndof);  K2 = (K2+K2')/2;
    M2 = sparse(mdl.iK, mdl.jK, reshape(mdl.ME(:)*re2', [], 1), mdl.ndof, mdl.ndof);  M2 = (M2+M2')/2;
    act = false(mdl.ndof,1);  act(unique(mdl.edofMat(keepE,:))) = true;  f2 = f(act(f));
    [V2, D2] = eigs(K2(f2,f2), M2(f2,f2), 8, 'smallestabs');  [l2, i2] = sort(real(diag(D2)));  V2 = V2(:, i2);
    for j = 1:8
        U = zeros(mdl.ndof,1);  U(f2) = V2(:,j);
        rrows(end+1,:) = {al, j, sqrt(l2(j)), mac_mass(U(f2), phiU(f2), M2(f2,f2)), nnz(keepE)}; %#ok<AGROW>
    end
    fprintf('alpha = %.2f: x<0.1 removed (%d elements kept): omega = %s rad/s\n', al, nnz(keepE), mat2str(sqrt(l2(1:4))', 6));
    [~, ~, ~] = deal([]);
end
T = cell2table(rows, 'VariableNames', {'alpha','raw_mode','omega','kept_by_selection','KE_share_x_le_0p1', ...
    'SE_share_x_le_0p1','KE_share_0p1_0p3','SE_share_0p1_0p3','KE_weighted_density','MAC_phi1_uniform', ...
    'KE_left_quarter','KE_middle_half','KE_right_quarter','KE_top_0p2','KE_web','KE_bottom_0p2','KE_x_ge_0p3'});
writetable(T, fullfile(out, 'w3_old_modes.csv'));
R2 = cell2table(rrows, 'VariableNames', {'alpha','mode','omega','MAC_phi1_uniform','n_elements_kept'});
writetable(R2, fullfile(out, 'w3_removed_model.csv'));
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1400 400]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for a = 1:numel(old)
    ax = nexttile(tl);
    imagesc(ax, [0 prob.L], [0 prob.H], 1 - reshape(old(a).xc, nely, nelx));
    set(ax, 'YDir', 'normal');  colormap(ax, gray);  clim(ax, [0 1]);  axis(ax, 'image', 'off');
    title(ax, sprintf('submitted version, clamped beam 400x50, \\alpha = %.2f (old code a2261f7)', old(a).alpha), 'FontWeight', 'normal');
end
exportgraphics(fig, fullfile(out, 'w3_old_designs.png'), 'Resolution', 200);  close(fig);
disp(T);  disp(R2);
end

function plotMode(prob, mdl, x, U, ke, omega, al, j, keep, mac, file)
% left: density with the kinetic-energy density of the mode (red); right: deformed solid part
nelx = prob.nelx;  nely = prob.nely;
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1500 330]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl);
g = 1 - reshape(x, nely, nelx);  k = reshape(ke/max(ke), nely, nelx).^0.5;
rgb = cat(3, max(g, k), g.*(1-k), g.*(1-k));
image(ax, [0 prob.L], [0 prob.H], rgb);  set(ax, 'YDir', 'normal');  axis(ax, 'image', 'off');
title(ax, sprintf('\\alpha = %.2f, raw mode %d, \\omega = %.2f rad/s: kinetic energy (red) on the design', al, j, omega), 'FontWeight', 'normal');
ax = nexttile(tl);  hold(ax, 'on');
nodes = (0:(nelx+1)*(nely+1)-1)';  nx = floor(nodes/(nely+1))*mdl.hx;  ny = mod(nodes, nely+1)*mdl.hy;
ux = U(1:2:end);  uy = U(2:2:end);  s = 0.08*prob.L/max(abs([ux; uy]));
sel = find(x >= 0.3);
conn = (mdl.edofMat(sel, [2 4 6 8]))/2;
patch(ax, 'Faces', conn, 'Vertices', [nx + s*ux, ny + s*uy], 'FaceVertexCData', x(sel), 'FaceColor', 'flat', 'EdgeColor', 'none');
colormap(ax, flipud(gray));  clim(ax, [0 1]);  axis(ax, 'image', 'off');
if keep, v = 'kept'; else, v = 'REJECTED'; end
title(ax, sprintf('deformed elements x \\geq 0.3; selection: %s, MAC_M with \\phi_1 of uniform design %.3f', v, mac), 'FontWeight', 'normal');
exportgraphics(fig, file, 'Resolution', 150);  close(fig);
end
