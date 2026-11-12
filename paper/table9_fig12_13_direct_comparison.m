function T = table9_fig12_13_direct_comparison(quick)
%TABLE9_FIG12_13_DIRECT_COMPARISON  Section 4.4: proposed method vs direct eigenfrequency
%   optimization (Du-Olhoff), Table 9 and Figures 12-13.
%
%   T = table9_fig12_13_direct_comparison()       full run, about 45 min (single thread)
%   T = table9_fig12_13_direct_comparison(true)   quick check on coarse meshes, ~2 min
%
%   Cases: simply supported beam 400x50 at V_f = 0.3/0.5/0.7, clamped beam 160x20 and
%   400x50 (alpha = 1), building 80x240 (alpha = 1; Du-Olhoff with passive frame,
%   filter radius three elements).  Frequencies: lowest two STRUCTURAL modes under the
%   common evaluator (structural_modes).  Output: results/table9/table9.csv and the
%   design images of Figures 12 (clamped beam) and 13 (building).
if nargin < 1, quick = false; end
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('table9');
if quick
    cases = {problem_ss_beam(160, 20, 0.5), problem_clamped_beam(80, 10), problem_building(40, 120)};
else
    cases = {problem_ss_beam(400, 50, 0.3), problem_ss_beam(400, 50, 0.5), problem_ss_beam(400, 50, 0.7), ...
             problem_clamped_beam(160, 20), problem_clamped_beam(400, 50), problem_building(80, 240)};
end
paperGap = [-5.8 -4.1 -6.7 -17.4 -10.8 -1.7];                 % Table 9, %
rows = {};
for k = 1:numel(cases)
    prob = cases{k};
    P = proposed_freq(prob);
    R = 0.06;  if strcmp(prob.name, 'building'), R = 3*prob.L/prob.nelx; end
    D = du_olhoff(prob, struct('filterRadius', R));
    oP = structural_modes(prob, P.x);  oD = structural_modes(prob, D.x);
    gap = 100*(oP(1)/oD(1) - 1);
    tag = sprintf('%s_%dx%d_vf%02.0f', prob.name, prob.nelx, prob.nely, 100*prob.volfrac);
    fprintf('%-26s proposed %3d it %6.1f s  %7.2f/%7.2f | Du-Olhoff %3d outer %6.1f s  %7.2f/%7.2f | gap %+5.1f%%\n', ...
        tag, P.iterations, P.time.total, oP, D.nOuter, D.time.total, oD, gap);
    if ~quick, compare_with_paper(['gap ' tag], gap, paperGap(k), '%'); end
    rows(end+1, :) = {tag, P.iterations, P.time.total, oP(1), oP(2), D.nOuter, D.time.total, oD(1), oD(2), gap}; %#ok<AGROW>
    save_design_png(prob, P.x, fullfile(out, [tag '_proposed.png']));
    save_design_png(prob, D.x, fullfile(out, [tag '_du_olhoff.png']));
end
T = cell2table(rows, 'VariableNames', {'case', 'prop_iter', 'prop_time_s', 'prop_omega1', 'prop_omega2', ...
    'do_outer', 'do_time_s', 'do_omega1', 'do_omega2', 'gap_pct'});
writetable(T, fullfile(out, 'table9.csv'));
disp(T);
end
