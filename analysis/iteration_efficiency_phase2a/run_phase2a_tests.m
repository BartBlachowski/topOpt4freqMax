function results = run_phase2a_tests(opts)
%RUN_PHASE2A_TESTS No-production regression suite for the frozen harness.
arguments
    opts.RunTinySolverSmoke (1,1) logical = false
end
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(here,fullfile(repo,'tools','Matlab'),fullfile(repo,'analysis','three_method_parametric_study'), ...
    fullfile(repo,'examples','Performance'),fullfile(repo,'analysis','iteration_efficiency_study_design'));
tests={@testContract,@testBudget,@testReference,@testProjection,@testTopology,@testPersistence, ...
    @testStatus,@testAccounting,@testTimingScaling,@testEvaluatorRegression,@testRendererReuse,@testObserverUnit};
names={};passed=[];messages={};
for i=1:numel(tests)
    names{i,1}=func2str(tests{i}); %#ok<AGROW>
    try,tests{i}();passed(i,1)=true;messages{i,1}='PASS'; %#ok<AGROW>
    catch ME,passed(i,1)=false;messages{i,1}=getReport(ME,'extended','hyperlinks','off');end %#ok<AGROW>
end
if opts.RunTinySolverSmoke
    names{end+1,1}='testTinyObserverIdentity';
    try,testTinyObserverIdentity(repo,here);passed(end+1,1)=true;messages{end+1,1}='PASS';
    catch ME,passed(end+1,1)=false;messages{end+1,1}=getReport(ME,'extended','hyperlinks','off');end
end
results=table(string(names),passed,string(messages),'VariableNames',{'test','passed','message'});
disp(results(:,{'test','passed'}));
assert(all(results.passed),'ie2a:TestFailure','Phase-2A suite has failures.');
fprintf('PHASE2A_TESTS_PASS=%d\n',height(results));
end

function testContract
c=ie2a.load_contract();ie2a.validate_contract(c);r=ie2a.run_negative_controls();assert(all(r.rejected));
end
function testBudget
for B0=[900 2000 3200]
    last=-Inf;
    for b=600:100:3200
        a=ie2a.measurement_budget(B0,b,100,3200);z=ie2a.measurement_budget(B0,b,100,3200);
        assert(isequal(a,z)&&a.B_meas>=B0&&a.B_meas<=3200&&a.B_meas>=last);last=a.B_meas;
        if b+99<=3200,assert(a.B_meas>=b+99&&~a.certification_tail_truncated);end
    end
end
a=ie2a.measurement_budget(900,3200,100,3200);assert(a.B_meas==3200&&a.tail_truncation==99&&a.certification_tail_truncated);
assert(ie2a.measurement_budget(900,700,100,3200).B_meas==ie2a.measurement_budget(900,700,100,3200).B_meas);
end
function testReference
Q=ones(900,3);Q(701:end,:)=2;h=true(900,1);r=ie2a.reference_phase(Q,h);assert(r.b_ref==600&&all(r.Q_ref==1));
h(1:40)=false;r2=ie2a.reference_phase(Q,h);assert(r2.b_ref>=600);
r3=ie2a.reference_phase(Q(1:500,:),true(500,1),SolverTerminated=true);assert(strcmp(r3.status,'REFERENCE_SOLVER_TERMINATION')&&isnan(r3.b_ref));
r4=ie2a.reference_phase(Q(1:500,:),true(500,1));assert(strcmp(r4.status,'REFERENCE_NOT_ESTABLISHED'));
end
function testProjection
x=ones(10,1);b=ie2a.exact_count_binary(x,.5);assert(isequal(find(b), (1:5)'));
end
function testTopology
nx=160;ny=20;base=false(ny,nx);base(1:9,1:159)=true;base(10,:)=true;base(11,1:9)=true;assert(nnz(base)==1600);
t=ie2a.topology_metrics(double(base(:)),nx,ny);assert(t.hard_gate_pass&&t.aggregate_detached_elements==0);
x=base;x(1,157:159)=false;x(20,80:82)=true;t=ie2a.topology_metrics(double(x(:)),nx,ny);assert(t.topology_pass&&t.max_detached_elements==3);
x=base;x(1,156:159)=false;x(20,80:83)=true;t=ie2a.topology_metrics(double(x(:)),nx,ny);assert(~t.topology_pass&&t.max_detached_elements==4);
x=base;remove=[154:159,148:153];x(1,remove)=false;for j=1:4,x(18:20,20*j)=true;end
t=ie2a.topology_metrics(double(x(:)),nx,ny);assert(t.topology_pass&&t.aggregate_detached_elements==12&&t.aggregate_detached_area>.01);
x=base;x(10,160)=false;x(20,100)=true;t=ie2a.topology_metrics(double(x(:)),nx,ny);assert(~t.required_connected);
end
function testPersistence
A=false(250,3);A(3:102,1)=true;A(50:148,2)=true;A(151:250,3)=true;r=ie2a.scan_persistence(A,100);
assert(r.k_enter(1)==3&&r.k_cert(1)==102&&isnan(r.k_enter(2))&&r.k_enter(3)==151&&r.k_cert(3)==250);
r50=ie2a.scan_persistence(A,50);r200=ie2a.scan_persistence(A,200);assert(r50.k_enter(1)==3&&isnan(r200.k_enter(1)));
end
function testStatus
f=struct('reference_status','PASS','endpoint_found',true,'structural_mode_not_found',false,'solver_terminated',false,'solver_termination_after_cert',false,'topology_persistence_possible',true,'quality_persistence_possible',true,'pointwise_acceptance_seen',true);assert(strcmp(ie2a.classify_status(f),'PASS'));
f.endpoint_found=false;f.topology_persistence_possible=false;assert(strcmp(ie2a.classify_status(f),'INVALID_TOPOLOGY'));
f.topology_persistence_possible=true;f.quality_persistence_possible=false;assert(strcmp(ie2a.classify_status(f),'QUALITY_NOT_REACHED'));
f.quality_persistence_possible=true;assert(strcmp(ie2a.classify_status(f),'PERSISTENT_NONACCEPTANCE'));
f.solver_terminated=true;assert(strcmp(ie2a.classify_status(f),'SOLVER_TERMINATION'));
f.solver_terminated=false;f.structural_mode_not_found=true;assert(strcmp(ie2a.classify_status(f),'STRUCTURAL_MODE_NOT_FOUND'));
end
function testAccounting
a=ie2a.account_iterations('Proposed',struct(),12);assert(a.chronological_update==12);
a=ie2a.account_iterations('Yuksel',struct('stage1_updates',27),12);assert(a.chronological_update==39&&a.eligible_iteration==12);
a=ie2a.account_iterations('Olhoff',struct('variant','lp','lp_calls',12),12);assert(a.lp_calls==12&&isnan(a.genuine_solver_iterations));
n=struct('variant','mma','inner_iterations',[80;100;300],'inner_converged',[true;true;false],'inner_cap_hit',[false;false;true]);
a=ie2a.account_iterations('Olhoff',n,3);assert(a.total_mma_inner_iterations==480&&a.inner_cap_hits==1);
try,ie2a.account_iterations('Olhoff',struct('nInner_as_solver_iterations',true),1);error('not rejected');catch ME,assert(strcmp(ME.identifier,'ie2a:OlhoffAccounting'));end
end
function testTimingScaling
p=ie2a.timing_replay_plan(["Proposed";"Yuksel";"Olhoff"],[10;20;30]);assert(height(p)==12&&sum(p.discarded_warmup)==3&&all(p.threads==1));
f=ie2a.fit_power_law([100;200;400;800],[10;20;40;80],["a";"b";"c";"d"]);assert(abs(f.p-1)<1e-12&&f.n_valid==4&&~f.weakly_identified);
end
function testEvaluatorRegression
p=ie2a.paths();S=load(fullfile(p.repo,'examples','Performance','final_campaign','raw','olhoff','s1_160x20.mat'),'res');
o=ie2a.evaluate_common(double(S.res.rho_snapshots(:,253)),160,20,.5,IncludeBinaryDiagnostic=true);
assert(strcmp(o.status,'PASS')&&isequal(o.selected_ordinal,[1 4 5]));
assert(max(abs(o.Q-[165.869052911 166.367075524 166.367366818])./[165.869052911 166.367075524 166.367366818])<1e-8);
assert(~isequal(o.Q,o.Q_binary_endpoint_diagnostic)&&strcmp(o.Q_source,'ACTUAL_GRAY_LOWEST_UNANIMOUS_VALID_STRUCTURAL_MODE'));
end
function testRendererReuse
p=ie2a.paths();src=fileread(fullfile(p.repo,'analysis','iteration_efficiency_study_design','render_iteration_efficiency_topology_grid.m'));
assert(contains(src,'renderTopologyDensity')&&~contains(src,'imagesc('));
end
function testObserverUnit
p=ie2a.paths();out=fullfile(p.validation,'observer_unit.mat');if ~isfolder(p.validation),mkdir(p.validation);end
H=topopt_history_init(2,struct());rec=struct('iter',1,'xPhys',[.2;.8],'volfrac',.5);
H0=topopt_history_record(H,rec);clean=ie2a.install_observer(out,2,2);H1=topopt_history_record(H,rec);clear clean
assert(isequaln(H0,H1));m=matfile(out);assert(m.n_observed==1&&isequal(m.xPhys(:,1),rec.xPhys));
end
function testTinyObserverIdentity(repo,here)
addpath(fullfile(repo,'analysis','three_method_parametric_study'));
methods={"proposed","yuksel"};
for i=1:2
    prm=struct('max_iters',3,'record_history',true,'extend_beyond_native_stop',true);
    if methods{i}=="yuksel",prm.stage1_max_iters=3;end
    cfg=study_base_config(char(methods{i}),40,5,prm);[x0,w0,~,n0]=run_topopt_from_json(cfg);
    out=fullfile(here,'validation_outputs',['observer_' char(methods{i}) '.mat']);clean=ie2a.install_observer(out,200,10);
    [x1,w1,~,n1]=run_topopt_from_json(cfg);clear clean
    assert(isequaln(x0,x1)&&isequaln(w0,w1)&&n0==n1);
end
end
