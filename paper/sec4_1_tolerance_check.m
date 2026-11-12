function T = sec4_1_tolerance_check()
%SEC4_1_TOLERANCE_CHECK  Section 4.1: the 400x50 benchmark with stagnation_x tolerances
%   1e-2 (as in Table 2) and 1e-3, with the cost breakdown of each method.
%   About 20 min (Du-Olhoff).  Output: results/tolerance/tolerance.csv.
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('tolerance');
maxNumCompThreads(1);
prob = problem_ss_beam(400, 50);
rows = {};
for tol = [1e-2 1e-3]
    P = proposed_freq(prob, struct('stopTol', tol));
    Y = yuksel_freq(prob, struct('stopTol', tol));
    D = du_olhoff(prob, struct('stopTol', tol));
    o = [structural_modes(prob, P.x, 1), structural_modes(prob, Y.x, 1), structural_modes(prob, D.x, 1)];
    fprintf(['tol %.0e: proposed %d it %.1f s (reference eigenanalysis %.2f s) | Yuksel %d+%d it %.1f s | ' ...
        'Du-Olhoff %d outer %d inner %.1f s (eig %.1f%%, MMA %.1f%%) | omega1 %.1f / %.1f / %.1f\n'], tol, ...
        P.iterations, P.time.total, P.time.reference, Y.iterations, Y.time.total, D.nOuter, D.nInner, ...
        D.time.total, 100*D.time.eig/D.time.total, 100*D.time.inner/D.time.total, o);
    rows(end+1, :) = {tol, P.iterations, P.time.total, sum(Y.iterations), Y.time.total, D.nOuter, D.nInner, ...
        D.time.total, D.time.eig, D.time.inner, o(1), o(2), o(3)}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'tol', 'prop_iter', 'prop_s', 'yu_iter', 'yu_s', 'do_outer', 'do_inner', ...
    'do_s', 'do_eig_s', 'do_mma_s', 'prop_omega1', 'yu_omega1', 'do_omega1'});
writetable(T, fullfile(out, 'tolerance.csv'));
end
