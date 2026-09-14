function evidence = run_phase2h_preflight_tests
%RUN_PHASE2H_PREFLIGHT_TESTS Stale and malformed qualification negative controls.
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'));
c=ie2a.load_contract();p=ie2a.paths();
base=struct('schema_version','candidate_c_precision_qualification_v1','pass',true, ...
    'scope','precision','candidate','C','classifier_version',c.quality.classifier_version, ...
    'evaluator_sha256',c.quality.source_sha256,'contract_sha256',ie2a.sha256_file(p.contract), ...
    'olhoff_variant','lp','input_provenance_sha256',struct('fixture',repmat('a',1,64)), ...
    'results',struct('decisions_identical',true));
cases={ ...
    'old_evaluator',setfield(base,'evaluator_sha256',repmat('0',1,64)); ... %#ok<SFLD>
    'old_contract',setfield(base,'contract_sha256',repmat('1',1,64)); ... %#ok<SFLD>
    'wrong_candidate',setfield(base,'candidate','A'); ... %#ok<SFLD>
    'wrong_classifier',setfield(base,'classifier_version','voidKE_only_v0'); ... %#ok<SFLD>
    'wrong_scope',setfield(base,'scope','old_precision'); ... %#ok<SFLD>
    'wrong_variant',setfield(base,'olhoff_variant','mma'); ... %#ok<SFLD>
    'pass_false',setfield(base,'pass',false)}; %#ok<SFLD>
name=strings(size(cases,1),1);rejected=false(size(cases,1),1);reason=strings(size(cases,1),1);
for i=1:size(cases,1)
    path=[tempname '.json'];writeJson(path,cases{i,2});
    r=ie2a.validate_qualification(path,'precision',c,SelectedOlhoffVariant='lp');
    name(i)=cases{i,1};rejected(i)=~r.pass;reason(i)=strjoin(string(r.errors),'; ');
    delete(path);
end
pre=ie2a.production_preflight(ThrowOnFailure=false,SelectedOlhoffVariant='lp');
missingBlocked=~pre.pass&&~pre.checks.candidate_c_precision&& ...
    ~pre.checks.candidate_c_cross_method&&~pre.checks.candidate_c_reference_length;
evidence=struct('schema_version','phase2h_preflight_negative_controls_v1', ...
    'pass',all(rejected)&&missingBlocked,'cases',table2struct(table(name,rejected,reason)), ...
    'missing_qualifications_block_preflight',missingBlocked,'preflight_checks',pre.checks);
writeJson(fullfile(here,'preflight_negative_controls.json'),evidence);
assert(evidence.pass,'phase2h:PreflightNegativeControls','Preflight negative controls failed.');
end
function writeJson(path,value)
fid=fopen(path,'w');cleanup=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true));
end
