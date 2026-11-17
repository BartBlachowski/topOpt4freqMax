function T = r3_w5_interpolation()
%R3_W5_INTERPOLATION  Reviewer 3, W5: sensitivity of the proposed method to the material
%   interpolation of the OPTIMIZATION model.  Simply supported beam 400x50, V_f = 0.5,
%   native settings except (i) p = 2, 3, 4 with E_min = 1e-9 E0 (native), and (ii) at
%   p = 3, E_min = 1e-9, 1e-6, 1e-4 E0 (rho_min stays native, 1e-9).  Frequencies:
%   common evaluator (SIMP p = 3, E_min = 1e-6 E0, linear mass, structural modes).
%   Output: results/r3/w5_interpolation.csv, w5_designs.png, w5_designs.mat.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
addpath(fileparts(mfilename('fullpath')));
maxNumCompThreads(1);
out = results_dir('r3');
prob = problem_ss_beam(400, 50);
cases = {2, 1e-9; 3, 1e-9; 4, 1e-9; 3, 1e-6; 3, 1e-4};
rows = {};  X = cell(size(cases,1),1);
r3_proposed_freq_penal(prob);                                   % warm-up (not reported)
for k = 1:size(cases,1)
    p = cases{k,1};  em = cases{k,2};
    t0 = tic;
    P = r3_proposed_freq_penal(prob, struct('penal', p, 'Emin', em*prob.E));
    o = structural_modes(prob, P.x, 2);
    tw = toc(t0);
    X{k} = P.x;
    rows(end+1, :) = {p, em, P.iterations, P.stop, P.time.total, tw, o(1), o(2), mean(4*P.x.*(1-P.x))}; %#ok<AGROW>
    fprintf('p = %d  Emin = %.0e E0 | %3d it (%s) %.2f s (incl. modal %.2f s) | omega %.2f %.2f | grey %.3f\n', ...
        p, em, P.iterations, P.stop, P.time.total, tw, o, rows{end,9});
end
T = cell2table(rows, 'VariableNames', {'p','Emin_rel','iterations','stop','solver_time_s','total_time_s','omega1','omega2','greyness'});
writetable(T, fullfile(out, 'w5_interpolation.csv'));
save(fullfile(out, 'w5_designs.mat'), 'X', 'cases');
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1500 560]);
tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:size(cases,1)
    ax = nexttile(tl);
    imagesc(ax, [0 prob.L], [0 prob.H], 1 - reshape(X{k}, prob.nely, prob.nelx));
    set(ax, 'YDir', 'normal');  colormap(ax, gray);  clim(ax, [0 1]);  axis(ax, 'image', 'off');
    title(ax, sprintf('p = %d, E_{min} = 10^{%d} E_0:  \\omega_1 = %.1f, \\omega_2 = %.1f rad/s', ...
        cases{k,1}, round(log10(cases{k,2})), T.omega1(k), T.omega2(k)), 'FontWeight', 'normal');
end
exportgraphics(fig, fullfile(out, 'w5_designs.png'), 'Resolution', 200);  close(fig);
end
