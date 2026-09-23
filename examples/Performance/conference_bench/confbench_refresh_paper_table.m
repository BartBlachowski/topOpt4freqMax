function files = confbench_refresh_paper_table(campaignDir)
%CONFBENCH_REFRESH_PAPER_TABLE Rewrite only conference_performance_table.tex.
% Uses the recorded runs in benchmark_results.json; never runs a
% solver and leaves every CSV, JSON and note untouched.  Schema-1 Proposed rows
% are re-accounted exactly as CONFBENCH_REFRESH_TABLES does.
% Example (with conference_bench and analysis/Olhoff on the MATLAB path):
%   confbench_refresh_paper_table('examples/Performance/conference_benchmark/nine_mesh_comparison_pedersen_b21483b')
source = jsondecode(fileread(fullfile(campaignDir, 'benchmark_results.json')));
R = source.runs;
if iscell(R); R = vertcat(R{:}); end
for i = 1:numel(R)
    T = R(i).times;
    if strcmp(R(i).method_key, 'proposed') && ...
            strcmp(T.time1_name, 'stage1_eigenanalysis_and_preparation_s')
        R(i).times = confbench_proposed_times(T.stage1_time_s, ...
            T.stage1_reference_eigen_time_s, T.time2, ...
            T.total_wall_time_s, T.solver_self_report_wall_s);
    end
end
% The column notes follow the current export, as a fresh compose would write
% them; only the recorded dagger tolerance is carried from the campaign.
cav = confbench_caveats();
if isfield(source, 'caveats') && isfield(source.caveats, 'omega1_native_flag_tol')
    cav.omega1_native_flag_tol = source.caveats.omega1_native_flag_tol;
end
files = confbench_export(struct('outputDir', campaignDir), R, struct(), struct(), ...
    struct('latex_only', true, 'caveats', cav));
end
