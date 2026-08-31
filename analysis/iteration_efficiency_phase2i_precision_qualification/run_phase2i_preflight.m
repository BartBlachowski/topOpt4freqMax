% Phase 2I WP22: verify fail-closed precision blocker after a negative result.
repo=fileparts(fileparts(fileparts(mfilename('fullpath'))));outDir=fileparts(mfilename('fullpath'));
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'));
r=ie2a.production_preflight(SelectedOlhoffVariant='lp',ThrowOnFailure=false);
fid=fopen(fullfile(outDir,'raw','preflight_after.json'),'w');fprintf(fid,'%s\n',jsonencode(r,PrettyPrint=true));fclose(fid);
fprintf('preflight_pass=%d precision=%d cross_method=%d reference_length=%d\n',r.pass, ...
 r.checks.candidate_c_precision,r.checks.candidate_c_cross_method,r.checks.candidate_c_reference_length);
