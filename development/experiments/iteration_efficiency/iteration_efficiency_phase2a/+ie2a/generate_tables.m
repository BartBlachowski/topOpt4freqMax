function artifacts = generate_tables(rows, outputDir, writeFiles)
%GENERATE_TABLES Produce frozen-schema endpoint and status tables.
arguments
    rows table
    outputDir (1,:) char
    writeFiles (1,1) logical = false
end
required={'method','nelx','nely','q','status','b_ref','B_meas','tail_truncated','k_enter','k_cert','Q_E1','Q_E2','Q_E3'};
assert(all(ismember(required,rows.Properties.VariableNames)),'ie2a:TableSchema','Result rows lack required frozen columns.');
artifacts=struct('endpoints',rows(:,required),'all_status_rows_preserved',true,'files',strings(0,1));
if writeFiles
    ie2a.assert_output_isolated(outputDir,'production'); if ~isfolder(outputDir),mkdir(outputDir);end
    f=fullfile(outputDir,'iteration_efficiency_endpoints.csv');writetable(artifacts.endpoints,f);artifacts.files=string(f);
end
end
