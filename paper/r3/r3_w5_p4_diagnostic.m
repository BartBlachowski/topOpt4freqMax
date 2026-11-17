function r3_w5_p4_diagnostic()
%R3_W5_P4_DIAGNOSTIC  Why the p = 4 run of r3_w5_interpolation hits the 2000-iteration cap:
%   history of the relative design change d and greyness, and omega_1 (common evaluator)
%   of the design after 50, 100, 200, 500, 1000, 2000 iterations.  Output:
%   results/r3/w5_p4_diagnostic.csv, w5_p4_history.png.
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
addpath(fileparts(mfilename('fullpath')));  maxNumCompThreads(1);  out = results_dir('r3');
prob = problem_ss_beam(400, 50);  rows = {};
for n = [50 100 200 500 1000 2000]
    P = r3_proposed_freq_penal(prob, struct('penal', 4, 'maxIter', n));
    o = structural_modes(prob, P.x, 2);
    rows(end+1,:) = {n, P.c, o(1), o(2), mean(4*P.x.*(1-P.x))}; %#ok<AGROW>
    fprintf('%5d it: c = %.6e  omega %.2f %.2f  grey %.4f\n', n, P.c, o, rows{end,5});
end
writetable(cell2table(rows, 'VariableNames', {'iterations','compliance','omega1','omega2','greyness'}), fullfile(out, 'w5_p4_diagnostic.csv'));
h = P.hist;  fig = figure('Visible', 'off', 'Color', 'w');
subplot(3,1,1); semilogy(h.d); ylabel('d = ||\Delta x||/||x||'); grid on; yline(1e-2, 'r--');
subplot(3,1,2); plot(h.grey); ylabel('greyness'); grid on;
subplot(3,1,3); plot(h.c); ylabel('compliance c'); xlabel('iteration'); grid on;
exportgraphics(fig, fullfile(out, 'w5_p4_history.png'), 'Resolution', 150);  close(fig);
fprintf('last 10 d: %s\n', mat2str(h.d(end-9:end)', 3));
end
