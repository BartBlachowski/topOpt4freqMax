function write_results(rows,out)
%WRITE_RESULTS Machine-readable authoritative rows and paper-facing CSV.
iefinal.validate_results(rows);jsonPath=fullfile(out,'tables','results.json');
fid=fopen(jsonPath,'w');assert(fid>0);c=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(rows,PrettyPrint=true));clear c
flat=rmfield(rows,{'source_hashes'});T=struct2table(flat);
writetable(T,fullfile(out,'tables','results.csv'));
quality=T(:,{'method','method_variant','mesh','q','P','status','k_enter','k_cert','E1','E2','E3','Q','topology_pass','volume_pass','hard_gate_pass'});
writetable(quality,fullfile(out,'tables','absolute_quality_and_acceptance.csv'));
end
