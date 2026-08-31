function report=generate_scaling_outputs(rows,analysisDir,isProduction)
%GENERATE_SCALING_OUTPUTS Actual rows only; certified primary-P support.
% Frozen eligibility: primary P and a fit-eligible status. Both the per-series
% "available" support and the explicitly intersected "common" support are
% emitted, and the exact mesh list behind every fit is written alongside it so
% the support of any published curve is auditable from the artifact.
T=struct2table(rmfield(rows,{'source_hashes'}));
ix=T.P==100&ismember(string(T.status),["PASS","PASS_WITH_LATER_SOLVER_TERMINATION"]);
T=T(ix,:);
metrics={'k_enter','k_cert','native_total_time_to_enter','native_total_time_to_cert','mean_native_iteration_time', ...
    'native_iterations','yuksel_stage1_iterations','yuksel_stage2_iterations','yuksel_total_iterations', ...
    'olhoff_outer_updates','olhoff_lp_calls','olhoff_lp_backend_iterations', ...
    'olhoff_mma_total_inner_iterations','olhoff_mma_mean_inner_iterations','olhoff_mma_p95_inner_iterations'};available={};
for i=1:numel(metrics),if ismember(metrics{i},T.Properties.VariableNames)&&any(isfinite(T.(metrics{i}))),available{end+1}=metrics{i};end,end %#ok<AGROW>
if isempty(available),report=struct('generated',false,'reason','no finite certified scaling metrics');return;end
S=T;S.method=string(T.method)+"-"+string(T.method_variant);
[fits,support]=iefinal.fit_scaling_table(S,available);
writetable(fits,fullfile(analysisDir,'scaling_fits.csv'));
writetable(support,fullfile(analysisDir,'scaling_common_support.csv'));
isCommon=fits.support=="common";
report=struct('generated',true,'production',isProduction,'C_reported',true,'p_reported',true, ...
    'fit_count',height(fits),'common_support_enforced',true, ...
    'common_support_fit_count',sum(isCommon&fits.fitted), ...
    'common_support_infeasible_metrics',sum(~support.common_fit_feasible), ...
    'support_disclosure','analysis/scaling_common_support.csv');
end
