function results = run_phase2h_tests
%RUN_PHASE2H_TESTS No-production MATLAB tests for Candidate C.
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(repo,'analysis','three_method_parametric_study'));
tests={@()testFirstBatch(repo),@()testParking(repo),@()testK252(repo), ...
    @()testK594(repo),@()testEscalation(repo),@()testMaximum(repo), ...
    @()testLate(repo),@()testDSeparation(repo),@testFailureSemantics, ...
    @testReferenceAndStatus,@testOlhoffSelectorAndAccounting};
name=strings(numel(tests),1);passed=false(numel(tests),1);message=strings(numel(tests),1);
for i=1:numel(tests)
    name(i)=string(func2str(tests{i}));
    try,tests{i}();passed(i)=true;message(i)="PASS";
    catch ME,message(i)=string(getReport(ME,'extended','hyperlinks','off'));end
end
results=table(name,passed,message);writetable(results,fullfile(here,'matlab_test_results.csv'));
evidence=struct('schema_version','phase2h_matlab_tests_v1','pass',all(passed), ...
    'test_count',height(results),'passed',sum(passed),'production_results',false,'optimizer_run',false);
fid=fopen(fullfile(here,'matlab_test_evidence.json'),'w');cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(evidence,PrettyPrint=true));clear cleanup
disp(results(:,{'name','passed'}));assert(all(passed),'phase2h:TestFailure','Phase-2H MATLAB tests failed.');
end

function testFirstBatch(~)
e=study_evaluate_design(ones(16,1),8,2,.5,ComputeBinaryDiagnostic=false);
assert(strcmp(e.status,'PASS'));assert(all([e.selected_ordinal_raw_E1,e.selected_ordinal_raw_E2,e.selected_ordinal_raw_E3]==1));
assert(all([e.modal_raw_E1.modes_requested_final,e.modal_raw_E2.modes_requested_final,e.modal_raw_E3.modes_requested_final]==3));
end
function testParking(repo)
x=state(repo,'160x20',80);e=study_evaluate_design(x,160,20,.5,ComputeBinaryDiagnostic=false);
assert(e.selected_ordinal_raw_E2==1);assert(rel(e.selected_omega_raw_E2,129.42514405462543)<1e-8);
end
function testK252(repo)
x=state(repo,'160x20',252);e=study_evaluate_design(x,160,20,.5,ComputeBinaryDiagnostic=false);
assert(e.selected_ordinal_raw_E2==4&&e.selected_ordinal_raw_E3==5);
assert(e.modal_raw_E2.modes_requested_final==6&&e.modal_raw_E2.escalation_count==1);
assert(rel(e.selected_omega_raw_E2,166.3670755250195)<1e-8);
assert(all(e.modal_raw_E2.selected_condition_margins>0));
end
function testK594(repo)
x=state(repo,'240x30',594);e=study_evaluate_design(x,240,30,.5,ComputeBinaryDiagnostic=false);
for id={'E2','E3'}
    m=e.(['modal_raw_' id{1}]);j=find(m.voidKE<.5,1,'first');
    assert(j<m.selected_ordinal&&m.voidKE(j)<.5&&m.voidSE(j)>.5&&m.densityParticipation(j)<.5);
    assert(~m.valid_structural(j)&&m.selected_ordinal==4);
end
end
function testEscalation(repo)
x=state(repo,'480x60',194);e=study_evaluate_design(x,480,60,.5,ComputeBinaryDiagnostic=false);
m=e.modal_raw_E3;assert(m.selected_ordinal==13&&m.modes_requested_final==24&&m.escalation_count==3);
assert(isequal(m.batch_schedule,[3 6 12 24]));assert(rel(m.selected_omega,167.77870624792712)<2e-6);
end
function testMaximum(repo)
x=state(repo,'720x90',411);e=study_evaluate_design(x,720,90,.5,ComputeBinaryDiagnostic=false);
m=e.modal_raw_E3;assert(m.selected_ordinal==18&&m.modes_requested_final==24&&m.escalation_count==3);
assert(rel(m.selected_omega,172.08702087402344)<2e-6);
end
function testLate(repo)
x=state(repo,'160x20',1600);e=study_evaluate_design(x,160,20,.5,ComputeBinaryDiagnostic=false);
assert(e.selected_ordinal_raw_E2==1&&rel(e.selected_omega_raw_E2,167.04934725228287)<1e-8);
end
function testDSeparation(repo)
x=state(repo,'400x50',833);e=ie2a.evaluate_common(x,400,50,.5,IncludeBinaryDiagnostic=true);
assert(strcmp(e.status,'PASS')&&rel(e.Q(2),170.93147031839717)<1e-8);
% MATLAB eigs and the independent Phase-2G SciPy solve agree within 0.2 ppm.
assert(rel(e.Q_binary_endpoint_diagnostic(2),4.672712513392596)<2e-7);
assert(~isequal(e.Q,e.Q_binary_endpoint_diagnostic));
assert(strcmp(e.Q_source,'ACTUAL_GRAY_LOWEST_UNANIMOUS_VALID_STRUCTURAL_MODE'));
end
function testFailureSemantics
x=ones(16,1);
e=ie2a.evaluate_common(x,8,2,.5,InjectEigensolverFailure=true);
assert(strcmp(e.status,'STRUCTURAL_MODE_NOT_FOUND')&&all(isnan(e.Q)));
e=ie2a.evaluate_common(x,8,2,.5,TechnicalMaxModes=3,InjectInvalidEigenpairs=true);
assert(strcmp(e.status,'STRUCTURAL_MODE_NOT_FOUND')&&all(isnan(e.Q)));
e=ie2a.evaluate_common(x,8,2,.5,TechnicalMaxModes=3,InjectNonfiniteDiagnostics=true);
assert(strcmp(e.status,'STRUCTURAL_MODE_NOT_FOUND')&&all(isnan(e.Q)));
e=study_evaluate_design(ones(2,1),1,2,.5,ComputeBinaryDiagnostic=false,InjectInvalidEigenpairs=true);
assert(strcmp(e.status,'STRUCTURAL_MODE_NOT_FOUND'));
end
function testReferenceAndStatus
Q=ones(700,3);H=false(700,1);valid=false(700,1);
r=ie2a.reference_phase(Q,H,EvaluatorValid=valid);assert(strcmp(r.status,'STRUCTURAL_MODE_NOT_FOUND'));
f=struct('reference_status','PASS','endpoint_found',false,'structural_mode_not_found',true, ...
    'solver_terminated',false,'solver_termination_after_cert',false, ...
    'topology_persistence_possible',true,'quality_persistence_possible',true,'pointwise_acceptance_seen',false);
assert(strcmp(ie2a.classify_status(f),'STRUCTURAL_MODE_NOT_FOUND'));
f.reference_status='REFERENCE_SOLVER_TERMINATION';
assert(strcmp(ie2a.classify_status(f),'STRUCTURAL_MODE_NOT_FOUND'));
end
function testOlhoffSelectorAndAccounting
assert(height(ie2a.olhoff_variant_plan('lp'))==1&&height(ie2a.olhoff_variant_plan('mma'))==1);
p=ie2a.olhoff_variant_plan('both');assert(height(p)==2&&isequal(p.route_id,["lp";"mma"]));
a=ie2a.account_iterations('Olhoff',struct('variant','lp','lp_calls',12),12);assert(a.lp_calls==12&&a.outer_iterations==12);
n=struct('variant','mma','inner_iterations',[80;100;300],'inner_converged',[true;true;false], ...
    'inner_cap_hit',[false;false;true]);a=ie2a.account_iterations('Olhoff',n,3);
assert(a.total_mma_inner_iterations==480&&a.inner_cap_hits==1&&abs(a.converged_inner_fraction-2/3)<eps);
end
function x=state(repo,mesh,k)
S=load(fullfile(repo,'examples','Performance','final_campaign','raw','olhoff',['s1_' mesh '.mat']),'res');
x=double(S.res.rho_snapshots(:,k+1));
end
function y=rel(a,b),y=abs(a-b)/abs(b);end
