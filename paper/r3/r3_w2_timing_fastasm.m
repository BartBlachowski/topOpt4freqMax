function T = r3_w2_timing_fastasm(nRep, meshIdx)
%R3_W2_TIMING_FASTASM  As r3_w2_timing, proposed method only, with r3_proposed_freq_fastasm
%   (lower-triangle assembly; identical matrices).  Reviewer 3, W2: repeated wall-time measurement of the proposed and the
%   Yuksel-Yilmaz methods (simply supported beam, V_f = 0.5, native settings, nine
%   meshes 160x20 ... 800x100, stagnation_x W = 10, eps = 1e-2, single thread).
%   Per mesh and method: one discarded warm-up, then nRep (default 5) timed runs.
%   Timed quantity ("Total" of Table 2): the complete solve INCLUDING the final modal
%   analysis of the design (common evaluator, two structural modes), measured by tic/toc
%   around both calls.  The solver-internal total (without the final modal analysis) is
%   also recorded.  Output: results/r3/w2_timing_runs.csv, w2_timing.csv.
if nargin < 1 || isempty(nRep), nRep = 5; end
run(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'setup_paths.m'));
addpath(fileparts(mfilename('fullpath')));
maxNumCompThreads(1);
out = results_dir('r3');
meshes = [160 20; 240 30; 320 40; 400 50; 480 60; 560 70; 640 80; 720 90; 800 100];
if nargin < 2, meshIdx = []; end
if ~isempty(meshIdx), meshes = meshes(meshIdx, :); end
% Table 2 single-run totals [s] (manuscript, tab:performance): proposed, Yuksel-Yilmaz
paperP = [1.13 1.52 2.50 3.75 5.69 7.50 10.92 20.15 28.22];
paperY = [1.35 2.08 3.31 5.52 8.06 11.86 17.77 31.18 39.36];
allMeshes = [160 240 320 400 480 560 640 720 800];
runs = {};  summ = {};
for k = 1:size(meshes, 1)
    prob = problem_ss_beam(meshes(k,1), meshes(k,2));
    for m = 1
        tt = nan(nRep,1);  ts = nan(nRep,1);  it = '';  om = nan(1,2);
        for r = 0:nRep
            t0 = tic;
            if m == 1, R = r3_proposed_freq_fastasm(prob); else, R = yuksel_freq(prob); end
            o = structural_modes(prob, R.x, 2);
            tw = toc(t0);
            if r == 0, continue; end                      % discarded warm-up
            tt(r) = tw;  ts(r) = R.time.total;
            if m == 1, it = sprintf('%d', R.iterations); else, it = sprintf('%d+%d', R.iterations); end
            om = o(:)';
            runs(end+1, :) = {sprintf('%dx%d', meshes(k,:)), methodName(m), r, tw, R.time.total, it, om(1), om(2)}; %#ok<AGROW>
            fprintf('%4dx%-3d %-8s run %d  total %8.3f s (solver %8.3f)  it %s  omega %.2f %.2f\n', ...
                meshes(k,:), methodName(m), r, tw, R.time.total, it, om);
        end
        j = find(allMeshes == meshes(k,1));
        if m == 1, pv = paperP(j); else, pv = paperY(j); end
        summ(end+1, :) = {sprintf('%dx%d', meshes(k,:)), prod(meshes(k,:)), methodName(m), median(tt), min(tt), max(tt), ...
            median(ts), it, om(1), om(2), pv, 100*(median(tt)/pv - 1)}; %#ok<AGROW>
    end
end
writetable(cell2table(runs, 'VariableNames', {'mesh','method','run','total_s','solver_total_s','iterations','omega1','omega2'}), ...
    fullfile(out, sprintf('w2_timing_fastasm_runs%s.csv', suffix(meshIdx))));
T = cell2table(summ, 'VariableNames', {'mesh','Ne','method','median_s','min_s','max_s','median_solver_s', ...
    'iterations','omega1','omega2','table2_total_s','diff_pct'});
writetable(T, fullfile(out, sprintf('w2_timing_fastasm%s.csv', suffix(meshIdx))));
end
function s = methodName(m), if m == 1, s = 'proposed_fastasm'; else, s = 'yuksel'; end, end
function s = suffix(idx), if ~isempty(idx), s = sprintf('_subset%s', sprintf('_%d', idx)); else, s = ''; end, end
