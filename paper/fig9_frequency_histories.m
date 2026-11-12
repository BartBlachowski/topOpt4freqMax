function fig9_frequency_histories()
%FIG9_FREQUENCY_HISTORIES  Section 4.1, Figure 9: three lowest eigenfrequencies over the
%   iterations at 240x30 for Du-Olhoff (its own material model, as used by its bound
%   formulation) and for the proposed method (every iterate analysed afterwards with the
%   common evaluator, structural modes only).  About 3 min.  Output: results/fig9/.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('fig9');
prob = problem_ss_beam(240, 30);
D = du_olhoff(prob);
P = proposed_freq(prob, struct('keepDesigns', true));
nP = numel(P.designs) - 1;                                 % analysed designs
wP = zeros(nP, 3);  mdl = fe_mesh(prob);
for k = 1:nP, wP(k, :) = structural_modes(prob, P.designs{k}, 3, struct('mdl', mdl))'; end
fprintf('Du-Olhoff: %d outer iterations, final omega1..3 = %.2f %.2f %.2f rad/s\n', D.nOuter, D.omega(1:3));
fprintf('proposed : %d iterations, final structural omega1..3 = %.2f %.2f %.2f rad/s\n', P.iterations, wP(end, :));
fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 700 700]);
subplot(2,1,1);  plot(D.hist.omega(1:3, :)', 'LineWidth', 1.5);  grid on;
title('(a) Du-Olhoff');  xlabel('outer iteration');  ylabel('rad/s');  legend('\omega_1', '\omega_2', '\omega_3');
subplot(2,1,2);  plot(wP, 'LineWidth', 1.5);  grid on;
title('(b) proposed');  xlabel('iteration');  ylabel('rad/s');  legend('\omega_1', '\omega_2', '\omega_3');
exportgraphics(fig, fullfile(out, 'fig9_frequency_histories.png'), 'Resolution', 200);  close(fig);
end
