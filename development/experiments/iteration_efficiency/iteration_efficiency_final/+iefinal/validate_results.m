function report=validate_results(rows)
%VALIDATE_RESULTS Strict method-neutral schema and N/A semantics checks.
p=iefinal.paths();schema=jsondecode(fileread(p.schema));required=cellstr(string(schema.required));errors={};
for i=1:numel(rows)
    missing=setdiff(required,fieldnames(rows(i)));
    if ~isempty(missing),errors{end+1}=sprintf('row %d missing %s',i,strjoin(missing,','));end %#ok<AGROW>
    if ~strcmp(rows(i).trajectory_dtype,'double'),errors{end+1}=sprintf('row %d non-double trajectory',i);end %#ok<AGROW>
    if ~strcmp(rows(i).evaluator_id,'candidate_c_adaptive_structural_mode_v1'),errors{end+1}=sprintf('row %d stale evaluator',i);end %#ok<AGROW>
    if strcmp(rows(i).method_variant,'lp')&&isfinite(rows(i).olhoff_mma_total_inner_iterations),errors{end+1}='LP row contains MMA accounting';end %#ok<AGROW>
    if strcmp(rows(i).method_variant,'mma')&&isfinite(rows(i).olhoff_lp_calls)&&rows(i).olhoff_lp_calls~=0,errors{end+1}='MMA row contains LP calls';end %#ok<AGROW>
    if strcmp(rows(i).status,'RUN_ERROR')
        % A failed cell must carry its exception identity and must not present
        % any scientific quantity as if it had been measured.
        if isempty(rows(i).error_identifier),errors{end+1}=sprintf('row %d RUN_ERROR without error_identifier',i);end %#ok<AGROW>
        fabricated={'k_enter','k_cert','b_ref','B_meas','E1','E2','E3','Q', ...
            'topology_pass','volume_pass','hard_gate_pass','native_iterations', ...
            'native_total_time','native_total_time_to_enter','native_total_time_to_cert', ...
            'mean_native_iteration_time'};
        for f=1:numel(fabricated)
            if ~isnan(rows(i).(fabricated{f}))
                errors{end+1}=sprintf('row %d RUN_ERROR has non-N/A %s',i,fabricated{f}); %#ok<AGROW>
            end
        end
    end
end
report=struct('pass',isempty(errors),'row_count',numel(rows),'errors',{errors});
if ~report.pass,error('iefinal:ResultSchema','Result schema failed: %s',strjoin(errors,'; '));end
end
