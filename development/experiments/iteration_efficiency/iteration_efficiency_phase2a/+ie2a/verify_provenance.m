function report = verify_provenance()
%VERIFY_PROVENANCE Re-hash all protected methodology, audit, profile, and numerical inputs.
p=ie2a.paths();m=jsondecode(fileread(fullfile(p.phase2a,'implementation_provenance.json')));
groups={'audit_records','protected_numerical_sources','profile_sources'};rows={};
for g=1:numel(groups)
    items=m.(groups{g});
    for i=1:numel(items)
        path=fullfile(p.repo,items(i).path);actual=ie2a.sha256_file(path);rows(end+1,:)={groups{g},items(i).path,items(i).sha256,actual,strcmp(actual,items(i).sha256)}; %#ok<AGROW>
    end
end
T=cell2table(rows,'VariableNames',{'group','path','expected_sha256','actual_sha256','unchanged'});
c=ie2a.load_contract();for i=1:numel(c.normative_documents)
    d=c.normative_documents(i);actual=ie2a.sha256_file(fullfile(p.repo,d.path));
    T(end+1,:)={'normative_documents',char(d.path),char(d.sha256),char(actual),strcmp(actual,d.sha256)}; %#ok<AGROW>
end
report=struct('pass',all(T.unchanged),'files',T,'contract_sha256',ie2a.sha256_file(p.contract), ...
    'contract_unchanged',strcmp(ie2a.sha256_file(p.contract),ie2a.frozen_contract_sha256()));
assert(report.pass&&report.contract_unchanged,'ie2a:ProvenanceMismatch','Protected hashes changed.');
end
