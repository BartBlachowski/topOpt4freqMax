function T = table1_gradient_ablation(quick)
%TABLE1_GRADIENT_ABLATION  Section 3.2, Table 1: effect of the omitted load sensitivity,
%   of the optimizer and of the volume constraint (alpha = 1).
%
%   T = table1_gradient_ablation()       nominal meshes, about 5 min
%   T = table1_gradient_ablation(true)   coarse meshes, < 1 min
%
%   Five variants per example: frozen-load sensitivity g_e with OC (the proposed
%   method), g_e with MMA and the volume as inequality / equality, and the full
%   derivative g_e + g_e^F (Eq. (13)) with MMA and inequality / equality.  The equality
%   is enforced by an exact projection after each MMA step (uniform shift, bisection).
%   Output: results/table1/table1.csv and the designs.
if nargin < 1, quick = false; end
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'setup_paths.m'));
out = results_dir('table1');
if quick, probs = {problem_ss_beam(160, 20), problem_clamped_beam(160, 20), problem_building(40, 120)};
else,     probs = {problem_ss_beam(), problem_clamped_beam(), problem_building()}; end
variants = {'frozen', 'OC', 'equality';  'frozen', 'MMA', 'inequality';  'frozen', 'MMA', 'equality'; ...
            'full',   'MMA', 'inequality';  'full',  'MMA', 'equality'};
paperOmega1 = [160.0 159.7 160.0 100.2 105.3; 368.2 368.4 364.1 NaN 129.5; 87.1 83.8 83.4 63.0 65.0];
rows = {};
for ip = 1:numel(probs)
    prob = probs{ip};
    for iv = 1:size(variants, 1)
        o = struct('sensitivity', variants{iv,1}, 'optimizer', variants{iv,2}, 'volume', variants{iv,3});
        P = proposed_freq(prob, o);
        om = structural_modes(prob, P.x);
        fprintf('%-13s %-6s %-3s %-10s  %4d it  vol %.3f  c %.3e  omega1/2 = %8.1f / %8.1f\n', prob.name, ...
            variants{iv,:}, P.iterations, mean(P.x), P.c, om);
        if ~quick && ~isnan(paperOmega1(ip, iv))
            compare_with_paper(sprintf('%s %s/%s/%s omega1', prob.name, variants{iv,:}), om(1), paperOmega1(ip, iv), 'rad/s');
        end
        rows(end+1, :) = {prob.name, variants{iv,1}, variants{iv,2}, variants{iv,3}, P.iterations, mean(P.x), P.c, om(1), om(2)}; %#ok<AGROW>
        save_design_png(prob, P.x, fullfile(out, sprintf('%s_%s_%s_%s.png', prob.name, variants{iv,:})));
    end
end
T = cell2table(rows, 'VariableNames', {'case', 'sensitivity', 'optimizer', 'volume', 'iterations', ...
    'volume_fraction', 'compliance', 'omega1', 'omega2'});
writetable(T, fullfile(out, 'table1.csv'));
end
