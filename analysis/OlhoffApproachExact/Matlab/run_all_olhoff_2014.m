function T = run_all_olhoff_2014(which_cases, overrides)
% RUN_ALL_OLHOFF_2014  Run the paper's 2D examples and print the comparison table.
%
%   T = run_all_olhoff_2014()
%   T = run_all_olhoff_2014({'ss_n1','cs_n1','cc_n1'})
%   T = run_all_olhoff_2014([], struct('nelx',240,'nely',30))
%
%   Covers Olhoff & Du (2014):
%     section 3.1  max omega_1, SS / CS / CC          (Figs. 2, 3, 4, 5)
%     section 3.2  max omega_2, SS / CS / CC          (Fig. 6)
%     section 3.3  max (omega_3 - omega_2), CC + m_c  (Fig. 7)
%
%   The 3D plate examples of sections 3.4-3.6 are out of scope by design.
%
%   Prints one row per case with the paper value, the computed value, the
%   multiplicity, the connectivity, the stop reason and the declared-rule
%   verdict, and writes the table to experiments/paper_examples/summary.csv.
%   The optimized topologies of all cases are stacked into a single figure,
%   experiments/paper_examples/topologies_all.png.

if nargin < 1 || isempty(which_cases)
    which_cases = {'ss_n1','cs_n1','cc_n1','ss_n2','cs_n2','cc_n2','cc_gap23'};
end
if nargin < 2, overrides = struct(); end

here = fileparts(mfilename('fullpath'));
addpath(here);
addpath(fullfile(here, '..', '..', '..', 'tools', 'Matlab'));
outroot = fullfile(here, '..', 'experiments', 'paper_examples');
if ~exist(outroot, 'dir'), mkdir(outroot); end

rows = {};
res  = cell(1, numel(which_cases));
for k = 1:numel(which_cases)
    r = run_olhoff_case(which_cases{k}, overrides);
    res{k} = r;
    rows{end+1} = struct( ...
        'case',        string(r.name), ...
        'figure',      string(r.target.figure), ...
        'paper',       r.target_value, ...
        'computed',    r.value, ...
        'err_pct',     r.err_pct, ...
        'initial',     r.value0, ...
        'N_final',     r.final_N, ...
        'N_paper',     r.target.multiplicity, ...
        'N_min_tail',  r.N_tail_min, ...
        'components',  r.components, ...
        'volume',      r.volume, ...
        'iters',       r.outer_iters, ...
        'stop_reason', string(r.stop_reason), ...
        'verdict',     string(r.verdict), ...
        'seconds',     r.elapsed_s); %#ok<AGROW>
end

T = struct2table(cell2mat(rows));
writetable(T, fullfile(outroot, 'summary.csv'));

panel = plot_all_topologies(res, fullfile(outroot, 'topologies_all.png'));

fprintf('\n\n%s\n', repmat('=', 1, 112));
fprintf('  OLHOFF & DU (2014) 2D EXAMPLES -- SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 112));
fprintf('%-10s %-10s %9s %10s %9s %6s %6s %6s %-18s %s\n', ...
    'case','figure','paper','computed','err %','N','Npap','comp','stop_reason','verdict');
fprintf('%s\n', repmat('-', 1, 112));
for k = 1:height(T)
    fprintf('%-10s %-10s %9.1f %10.3f %+9.2f %6g %6d %6d %-18s %s\n', ...
        T.case(k), T.figure(k), T.paper(k), T.computed(k), T.err_pct(k), ...
        T.N_final(k), T.N_paper(k), T.components(k), T.stop_reason(k), T.verdict(k));
end
fprintf('%s\n', repmat('=', 1, 112));
fprintf(' verdicts: %d PASS, %d PARTIAL, %d FAIL   (rule: PLAN_Olhoff2014_exact.md section 7.1)\n', ...
    sum(T.verdict == "PASS"), sum(T.verdict == "PARTIAL"), sum(T.verdict == "FAIL"));
fprintf(' summary written to %s\n', fullfile(outroot, 'summary.csv'));
fprintf(' topologies written to %s\n\n', panel);
end
