function report = run_negative_controls()
%RUN_NEGATIVE_CONTROLS Prove scientific deviations fail closed.
c=ie2a.load_contract();
cases={}; labels={};
b=c;b.reference.P=99;cases{end+1}=b;labels{end+1}='P changed';
b=c;b.production_meshes(end,:)=[];cases{end+1}=b;labels{end+1}='mesh removed';
b=c;b.quality.evaluators=b.quality.evaluators(1);cases{end+1}=b;labels{end+1}='E1 only';
b=c;b.topology.aggregate_detached_area_role='hard_gate';cases{end+1}=b;labels{end+1}='aggregate veto';
b=c;b.measurement_budget.formula='B_meas = B0';cases{end+1}=b;labels{end+1}='B_meas reverts to B0';
b=c;b.reference.B_ref=3199;cases{end+1}=b;labels{end+1}='B_ref changed';
b=c;b.quality.levels=[.98 .99];cases{end+1}=b;labels{end+1}='quality levels changed';
b=c;if iscell(b.methods),b.methods{1}.profile_id='changed_profile';else,b.methods(1).profile_id='changed_profile';end;cases{end+1}=b;labels{end+1}='method profile changed';
rejected=false(numel(cases),1); error_id=strings(numel(cases),1);
for i=1:numel(cases)
    try, ie2a.validate_contract(cases{i},VerifyFiles=false);
    catch ME, rejected(i)=true; error_id(i)=string(ME.identifier); end
end
report=table(string(labels(:)),rejected,error_id,'VariableNames',{'control','rejected','error_id'});
assert(all(rejected),'ie2a:NegativeControlFailure','At least one frozen-contract mutation was not rejected.');
end
