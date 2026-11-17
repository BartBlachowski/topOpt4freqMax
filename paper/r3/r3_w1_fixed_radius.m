function T = r3_w1_fixed_radius()
%R3_W1_FIXED_RADIUS  Reviewer 3, W1(a): proposed and Yuksel-Yilmaz methods with a FIXED
%   PHYSICAL sensitivity-filter radius R = 0.06 m (the Du-Olhoff value) on the nine
%   meshes 160x20 ... 800x100 of Table 2; all other settings native (stagnation_x,
%   W = 10, eps = 1e-2, single thread).  Frequencies: common evaluator (two lowest
%   structural modes).  Output: results/r3/w1a_fixed_radius.csv, w1a_designs.mat.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
maxNumCompThreads(1);
out = results_dir('r3');
R = 0.06;
meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
rows = {};  designs = struct('mesh', {}, 'method', {}, 'x', {});
for k = 1:size(meshes, 1)
    prob = problem_ss_beam(meshes(k,1), meshes(k,2));
    P = proposed_freq(prob, struct('filterRadius', R));
    Y = yuksel_freq(prob, struct('filterRadius', R));
    oP = structural_modes(prob, P.x, 2);  oY = structural_modes(prob, Y.x, 2);
    rows(end+1, :) = {sprintf('%dx%d', meshes(k,:)), prod(meshes(k,:)), 'proposed', R, R/prob.L*prob.nelx, ...
        sprintf('%d', P.iterations), P.stop, P.time.total, oP(1), oP(2)}; %#ok<AGROW>
    rows(end+1, :) = {sprintf('%dx%d', meshes(k,:)), prod(meshes(k,:)), 'yuksel', R, R/prob.L*prob.nelx, ...
        sprintf('%d+%d', Y.iterations), Y.stop, Y.time.total, oY(1), oY(2)}; %#ok<AGROW>
    designs(end+1) = struct('mesh', meshes(k,:), 'method', 'proposed', 'x', P.x); %#ok<AGROW>
    designs(end+1) = struct('mesh', meshes(k,:), 'method', 'yuksel', 'x', Y.x); %#ok<AGROW>
    fprintf('%4dx%-3d R=%.3f m (%.2f el) | proposed %s %.2f s  omega %.2f %.2f | Yuksel %s %.2f s  omega %.2f %.2f\n', ...
        meshes(k,:), R, R/prob.L*prob.nelx, rows{end-1,6}, P.time.total, oP, rows{end,6}, Y.time.total, oY);
    save(fullfile(out, 'w1a_designs.mat'), 'designs');
end
T = cell2table(rows, 'VariableNames', {'mesh','Ne','method','R_m','R_elements','iterations','stop','solver_time_s','omega1','omega2'});
writetable(T, fullfile(out, 'w1a_fixed_radius.csv'));
end
