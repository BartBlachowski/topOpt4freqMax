function files = confbench_refresh_complexity_plots(campaignDir)
%CONFBENCH_REFRESH_COMPLEXITY_PLOTS Plot recorded stage times without a solve.
% Add conference_bench and analysis/Olhoff to the MATLAB path before calling.
source = jsondecode(fileread(fullfile(campaignDir, 'benchmark_results.json')));
records = source.runs;
if iscell(records); records = vertcat(records{:}); end
for i = 1:numel(records)
    T = records(i).times;
    if strcmp(records(i).method_key, 'proposed') && ...
            strcmp(T.time1_name, 'stage1_eigenanalysis_and_preparation_s')
        records(i).times = confbench_proposed_times(T.stage1_time_s, ...
            T.stage1_reference_eigen_time_s, T.time2, ...
            T.total_wall_time_s, T.solver_self_report_wall_s);
    end
end
cfg = struct('outputDir', campaignDir, 'fitScaling', source.scaling.fitted, ...
    'performanceCampaign', source.performance_campaign, ...
    'scientificEvidence', source.scientific_evidence);
scaling = confbench_scaling_fit(cfg, records);
files = confbench_complexity_plots(cfg, records, scaling);
end
