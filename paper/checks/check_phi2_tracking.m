function check_phi2_tracking()
%CHECK_PHI2_TRACKING  Sections 2.4, 4.2 and 4.3: why the tracked phi_2^(0) mode has a low
%   MAC in some designs of Tables 6 and 8.  About 2 min.
%
%   For each design of the alpha sweep (clamped beam 400x50, building 80x240) it prints
%   how phi_2^(0) is distributed over the structural modes of the final design: the four
%   largest mass-weighted MAC values (Eq. (9)) and the share of the mass norm of phi_2^(0)
%   captured by the 100 lowest modes.  The MAC values of one shape with all modes sum to
%   one, so a capture of at least 0.99 means that no mode above the 100th can be a better
%   match than the one reported.
%
%   It also writes one picture per design, results/phi2_tracking/*.png: phi_2^(0) (blue)
%   and the best-matching structural mode (red; also the second-best one when its MAC exceeds 0.3) drawn on the solid elements (x > 0.5) of
%   the final design, both normalized to unit mass norm and with the same sign.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
out = results_dir('phi2_tracking');
alphas = [1 0.75 0.5 0.25 0];
for prob = {problem_clamped_beam(), problem_building()}
    p = prob{1};  mdl = fe_mesh(p);
    nn = (mdl.nelx+1)*(mdl.nely+1);  id = (1:nn)';
    X = floor((id-1)/(mdl.nely+1))*mdl.hx;  Y = mod(id-1, mdl.nely+1)*mdl.hy;
    E = mdl.edofMat(:, 2:2:8)/2;                 % element -> node numbers
    for a = alphas
        P = proposed_freq(p, struct('alpha', a));
        [~, ~, info] = structural_modes(p, P.x, 1, struct('mdl', mdl, 'kStart', 100));
        sel = find(info.keep);  f = info.free;  Mf = info.Mf;  V = info.Vraw;
        v = P.phi0(f, 2);
        mac = zeros(numel(sel), 1);
        for k = 1:numel(sel), mac(k) = mac_mass(v, V(:, sel(k)), Mf); end
        [ms, ord] = sort(mac, 'descend');
        Vn = V ./ sqrt(sum(V .* (Mf*V), 1));     % mass-normalized raw modes
        share = (Vn' * (Mf*v)).^2 / (v'*Mf*v);
        fprintf('%-12s alpha=%.2f | MAC: mode %2d %.3f, mode %2d %.3f, mode %2d %.3f, mode %2d %.3f | 100 modes capture %.3f (structural %.3f)\n', ...
            p.name, a, ord(1), ms(1), ord(2), ms(2), ord(3), ms(3), ord(4), ms(4), sum(share), sum(share(sel)));
        % pictures: best match, and the second-best match when it is a large share too
        for r = 1:2
            if r == 2 && ms(2) < 0.3, break; end
            drawOverlay(p, mdl, X, Y, E, P.x, v, V(:, sel(ord(r))), Mf, f, ...
                sqrt(info.lambdaRaw(sel(ord(r)))), ord(r), ms(r), r, a, ord(3-r), ms(3-r), out);
        end
    end
end
end

function drawOverlay(p, mdl, X, Y, E, x, v, psi, Mf, f, om, kMode, macK, rank, a, kOther, macOther, out)
v = v/sqrt(v'*Mf*v);  psi = psi/sqrt(psi'*Mf*psi);
if v'*Mf*psi < 0, psi = -psi; end
U1 = zeros(mdl.ndof, 1);  U1(f) = v;  U2 = zeros(mdl.ndof, 1);  U2(f) = psi;
solid = x > 0.5;  sn = unique(E(solid, :));
sc = 0.25*min(p.L, p.H) / max(hypot(U1(2*sn-1), U1(2*sn)));
fig = figure('Visible', 'off', 'Color', 'w');
if p.H > p.L, set(fig, 'Position', [0 0 700 1400]); else, set(fig, 'Position', [0 0 1800 520]); end
ax = axes(fig);  hold(ax, 'on');
h0 = patch(ax, 'Faces', E(solid, :), 'Vertices', [X Y], 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'none');
h1 = patch(ax, 'Faces', E(solid, :), 'Vertices', [X + sc*U1(1:2:end), Y + sc*U1(2:2:end)], ...
    'FaceColor', [0 0.35 0.9], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
h2 = patch(ax, 'Faces', E(solid, :), 'Vertices', [X + sc*U2(1:2:end), Y + sc*U2(2:2:end)], ...
    'FaceColor', [0.9 0.1 0.1], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
axis(ax, 'equal', 'off');
names = {'best match', 'second-best match'};  other = {'second-best match', 'best match'};
legend(ax, [h1 h2 h0], {'\phi_2^{(0)} (reference shape, solid domain)', ...
    sprintf('%s: structural mode %d, \\omega = %.1f rad/s', names{rank}, kMode, om), ...
    'undeformed design (x > 0.5)'}, 'Location', 'southoutside', 'Box', 'off');
title(ax, sprintf('%s, \\alpha = %.2f:  MAC_M = %.3f   (%s: mode %d, MAC %.3f)', ...
    strrep(p.name, '_', ' '), a, macK, other{rank}, kOther, macOther), 'FontWeight', 'normal');
suffix = {'', '_second_best'};
exportgraphics(fig, fullfile(out, sprintf('%s_alpha_%.2f_phi2_overlay%s.png', p.name, a, suffix{rank})), 'Resolution', 150);
close(fig);
end
