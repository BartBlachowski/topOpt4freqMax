function T = table2_fig8_benchmark(quick)
%TABLE2_FIG8_BENCHMARK  Section 4.1: nine-mesh benchmark on the simply supported beam,
%   Table 2 (iterations and wall times) and Figure 8 (time versus mesh size).
%
%   T = table2_fig8_benchmark()       meshes 160x20 ... 800x100; Du-Olhoff dominates the
%                                     run time (about 2.5 h single-threaded on a laptop)
%   T = table2_fig8_benchmark(true)   quick: 160x20 and 240x30, ~5 min
%
%   All methods with their native settings (Table 3) and the stagnation_x stop
%   (W = 10, 1e-2), one run per entry, single thread.  Times depend on the machine; the
%   paper's values were measured on Apple silicon with MATLAB R2025b.  Frequencies:
%   lowest structural omega_1 under the common evaluator.  Output: results/table2/.
if nargin < 1, quick = false; end
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('table2');
maxNumCompThreads(1);
meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
if quick, meshes = meshes(1:2, :); end
rows = {};
for k = 1:size(meshes, 1)
    prob = problem_ss_beam(meshes(k,1), meshes(k,2));
    P = proposed_freq(prob);  Y = yuksel_freq(prob);  D = du_olhoff(prob);
    oP = structural_modes(prob, P.x, 1);  oY = structural_modes(prob, Y.x, 1);  oD = structural_modes(prob, D.x, 1);
    fprintf(['%4dx%-3d | Du-Olhoff %3d outer %5d inner %8.1f s (MMA %5.1f%%) | Yuksel %3d+%-3d %6.2f s | ' ...
        'proposed 1+%3d %6.2f s | omega1 %.1f / %.1f / %.1f\n'], meshes(k,:), D.nOuter, D.nInner, D.time.total, ...
        100*D.time.inner/D.time.total, Y.iterations, Y.time.total, P.iterations, P.time.total, oD, oY, oP);
    rows(end+1, :) = {sprintf('%dx%d', meshes(k,:)), prod(meshes(k,:)), D.nOuter, D.nInner, D.time.outerExclInner, ...
        D.time.inner, D.time.total, Y.iterations(1), Y.iterations(2), Y.time.stage1, Y.time.stage2, Y.time.total, ...
        P.iterations, P.time.reference, P.time.loop, P.time.total, oD, oY, oP}; %#ok<AGROW>
    if meshes(k,1) == 400
        save_design_png(prob, D.x, fullfile(out, 'fig5_du_olhoff_400x50.png'));
        save_design_png(prob, Y.x, fullfile(out, 'fig6_yuksel_400x50.png'));
        save_design_png(prob, P.x, fullfile(out, 'fig7_proposed_400x50.png'));
    end
end
T = cell2table(rows, 'VariableNames', {'mesh', 'Ne', 'do_outer', 'do_inner', 'do_time_outer_s', 'do_time_inner_s', ...
    'do_total_s', 'yu_stage1', 'yu_stage2', 'yu_time1_s', 'yu_time2_s', 'yu_total_s', 'prop_iter', ...
    'prop_time_ref_s', 'prop_time_loop_s', 'prop_total_s', 'do_omega1', 'yu_omega1', 'prop_omega1'});
writetable(T, fullfile(out, 'table2.csv'));
% Figure 8: total time versus number of elements (log-log) with power-law fits
fig = figure('Visible', 'off', 'Color', 'w');
loglog(T.Ne, T.do_total_s, 'o-', T.Ne, T.yu_total_s, 's-', T.Ne, T.prop_total_s, '^-');
grid on;  xlabel('number of elements N_e');  ylabel('total time [s]');
legend('Du-Olhoff', 'Yuksel-Yilmaz', 'proposed', 'Location', 'northwest');
for c = {'do_total_s', 'yu_total_s', 'prop_total_s'}
    q = polyfit(log(T.Ne), log(T.(c{1})), 1);
    fprintf('Figure 8 fit %-13s time ~ Ne^%.2f\n', c{1}, q(1));
end
exportgraphics(fig, fullfile(out, 'fig8_time_vs_mesh.png'), 'Resolution', 200);  close(fig);
end
