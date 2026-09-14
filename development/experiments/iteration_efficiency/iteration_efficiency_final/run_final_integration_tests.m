function results=run_final_integration_tests
%RUN_FINAL_INTEGRATION_TESTS No-production final-harness regression suite.
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(here,fullfile(repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(repo,'analysis','three_method_parametric_study'), ...
    fullfile(repo,'analysis','iteration_efficiency_study_design'),fullfile(repo,'tools','Matlab'));
tests={@testManifestAndPreflight,@testCandidateAnchors,@testTopologyReferencePersistence, ...
    @testAccounting,@testResultSchema,@testTimingFirewall,@testSelectorEvidence, ...
    @testDoubleStorageEvidence,@testScalingAndRendering,@testStaleAndProductionLocks,@testOutputIsolation, ...
    @testB1YukselTimingWorkIdentity,@testB2CommonSupport,@testB3FailureContainment};
names=strings(numel(tests),1);passed=false(numel(tests),1);messages=strings(numel(tests),1);
for i=1:numel(tests)
    names(i)=string(func2str(tests{i}));
    try,tests{i}();passed(i)=true;messages(i)="PASS";
    catch ME,messages(i)=string(getReport(ME,'extended','hyperlinks','off'));end
end
results=table(names,passed,messages);if ~isfolder(fullfile(here,'validation')),mkdir(fullfile(here,'validation'));end
writetable(results,fullfile(here,'validation','matlab_test_results.csv'));
evidence=struct('schema_version','iteration_efficiency_final_tests_v1','pass',all(passed), ...
    'test_count',height(results),'passed',sum(passed),'production_campaign_run',false, ...
    'matlab_version',version,'results',table2struct(results));localWrite(fullfile(here,'validation','matlab_test_results.json'),evidence);
disp(results(:,{'names','passed'}));assert(all(passed),'iefinal:TestsFailed','Final integration tests failed.');
end

function testManifestAndPreflight
for v={'lp','mma','both'},c=iefinal.config('smoke',v{1});r=iefinal.preflight(c);assert(r.pass);end
c=iefinal.config('production','lp');assert(isequal(c.meshes,[160 20;240 30;320 40;400 50;480 60;560 70;640 80;720 90;800 100]));
assert(c.B_ref==3200&&c.P_primary==100&&isequal(c.P_sensitivity,[50 200])&&isequal(c.quality_levels,[.98 .99 .995]));
end
function testCandidateAnchors
e=ie2a.evaluate_common(ones(3200,1),160,20,.5);assert(strcmp(e.status,'PASS')&&all(e.selected_ordinal==1));
p=iefinal.paths();S=load(fullfile(p.repo,'examples','Performance','final_campaign','raw','olhoff','s1_480x60.mat'),'res');
e=ie2a.evaluate_common(double(S.res.rho_snapshots(:,195)),480,60,.5);assert(strcmp(e.status,'PASS'));
assert(e.selected_ordinal(3)==13&&e.modal{3}.escalation_count==3&&isequal(e.modal{3}.batch_schedule,[3 6 12 24]));
end
function testTopologyReferencePersistence
nx=160;ny=20;x=false(ny,nx);x(1:9,1:159)=true;x(10,:)=true;x(11,1:9)=true;
t=ie2a.topology_metrics(double(x(:)),nx,ny);assert(t.hard_gate_pass);
Q=ones(900,3);h=true(900,1);r=ie2a.reference_phase(Q,h);assert(r.b_ref==600);
b=ie2a.measurement_budget(900,3200,100,3200);assert(b.B_meas==3200&&b.certification_tail_truncated);
A=false(250,1);A(3:102)=true;s=ie2a.scan_persistence(A,100);assert(s.k_enter==3&&s.k_cert==102);
end
function testAccounting
a=ie2a.account_iterations('Yuksel',struct('stage1_updates',7),3);assert(a.stage1_updates==7&&a.chronological_update==10);
a=ie2a.account_iterations('Olhoff',struct('variant','lp','lp_calls',3,'genuine_solver_iterations',12),3);assert(a.lp_calls==3&&a.genuine_solver_iterations==12);
n=struct('variant','mma','inner_iterations',[70;46;63],'inner_converged',true(3,1),'inner_cap_hit',false(3,1));
a=ie2a.account_iterations('Olhoff',n,3);assert(a.total_mma_inner_iterations==179&&a.inner_cap_hits==0&&a.converged_inner_fraction==1);
end
function testResultSchema
p=iefinal.paths();E=load(fullfile(p.repo,'analysis','iteration_efficiency_phase2i_precision_qualification','raw','reference_evaluation.mat'),'Qd','hardD','validD');
ref=ie2a.reference_phase(E.Qd,E.hardD&E.validD,EvaluatorValid=E.validD);cfg=iefinal.config('smoke','lp');budget=ie2a.measurement_budget(900,ref.b_ref,100,3200);
a=struct('Q',E.Qd,'H0',E.hardD&E.validD,'H_topology',logical(E.hardD),'H_volume',true(3200,1),'evaluator_valid',logical(E.validD));
tr=struct('method','Proposed','method_variant','proposed','nelx',96,'nely',12,'x_post',zeros(1152,3200), ...
    'solver_terminated',false,'trajectory_dtype','double','native',struct('total_updates',3200));
rows=iefinal.build_rows(tr,a,ref,budget,cfg,struct('test','schema'));r=iefinal.validate_results(rows);assert(r.pass&&numel(rows)==9);
bad=ref;bad.status='REFERENCE_NOT_ESTABLISHED';bad.b_ref=NaN;
failed=iefinal.build_rows(tr,struct(),bad,struct('B0',900),cfg,struct('test','schema'));
r=iefinal.validate_results(failed);assert(r.pass&&all(strcmp({failed.status},'REFERENCE_NOT_ESTABLISHED'))&&all(isnan([failed.k_enter])));
end
function testTimingFirewall
z=struct('method','Proposed','method_variant','proposed','mesh','160x20','nelx',160,'nely',20,'P',100, ...
    'k_enter',1,'k_cert',1,'native_iterations',1,'native_total_time',NaN,'native_total_time_to_enter',NaN, ...
    'native_total_time_to_cert',NaN,'mean_native_iteration_time',NaN);
cells=struct('id','proposed_160x20','trajectory',struct('method','Proposed','method_variant','proposed', ...
    'nelx',160,'nely',20,'native',struct('total_updates',1)));
out=tempname('/tmp');mkdir(out);cfg=iefinal.config('smoke','lp');cfg.timing_repetitions=1;
[z,r]=iefinal.run_timing_firewall(z,out,cfg,cells);
assert(r.pass&&r.fixed_horizon&&~r.trajectory_capture&&isfinite(z.native_total_time));
assert(r.work_identity_verified&&r.work_identity_checks>0);
% Firewall exclusions must remain intact (B-1 correction must not widen timing).
src=fileread(which('iefinal.run_timing_firewall'));
assert(~contains(src,'evaluate_common')&&contains(src,'captureTrajectory=false'));
assert(~contains(src,'topology_metrics')&&~contains(src,'scan_persistence'));
assert(~contains(src,'render_')&&~contains(src,'exportgraphics'));
assert(contains(src,'record_history'',false'));
% Timing evidence is mandatory: an unverifiable replay must be refused.
try,iefinal.run_timing_firewall(z,out,cfg,struct([]));error('missing evidence accepted');
catch ME,assert(strcmp(ME.identifier,'iefinal:TimingEvidenceMissing'));end
end

function testB1YukselTimingWorkIdentity
% The Stage-2 timing horizon must not clamp the Stage-1 budget.
base=struct('record_history',false,'extend_beyond_native_stop',true, ...
    'move',.1,'stage1_tol',.01,'stage2_tol',.01,'rmin_element',2.5);
p=base;p.max_iters=3200;p.stage1_max_iters=2000;
[~,~,~,~,~,nsRef,telRef]=run_topopt_from_json(study_base_config('yuksel',160,20,p));
s1=nsRef.stage1;assert(isfinite(s1)&&s1>1);
assert(telRef.yuksel.stage1_max_iters==2000,'reference Stage-1 budget must be unclamped');
H=s1-1; % horizon strictly below the native Stage-1 length
% Default (flag absent) preserves the historical clamp: the documented defect.
p=base;p.max_iters=H;p.stage1_max_iters=2000;
[~,~,~,~,~,nsOld,telOld]=run_topopt_from_json(study_base_config('yuksel',160,20,p));
assert(telOld.yuksel.stage1_max_iters==H&&nsOld.stage1==H,'clamp default must be unchanged');
% Timing path declares the Stage-1 budget independent: identity restored.
p=base;p.max_iters=H;p.stage1_max_iters=2000;p.stage1_budget_independent=true;
[~,~,~,nitNew,~,nsNew,telNew]=run_topopt_from_json(study_base_config('yuksel',160,20,p));
assert(telNew.yuksel.stage1_max_iters==2000);
assert(nsNew.stage1==s1,'Stage-1 work must equal the native Stage-1 work');
assert(nsNew.stage2==H,'Stage-2 work must equal the horizon');
assert(nitNew==nsNew.stage1+nsNew.stage2,'total must equal Stage 1 + Stage 2');
end

function testB2CommonSupport
Ne=[3200;7200;12800;20000];
mk=@(m,ne,y)struct('method',m,'element_count',ne,'k_enter',y);
build=@(spec)localBuildScalingTable(spec,mk);
% 1. all methods have all meshes -> common support is all four
T=build({"A",Ne;"B",Ne});[f,s]=iefinal.fit_scaling_table(T,{'k_enter'});
assert(s.n_support==4&&strcmp(s.common_meshes,'3200,7200,12800,20000'));
assert(all(f.n_valid(f.support=="common")==4)&&all(f.fitted(f.support=="common")));
% 2. one method lacks one mesh -> intersection drops it
T=build({"A",Ne;"B",Ne([1 2 3])});[f,s]=iefinal.fit_scaling_table(T,{'k_enter'});
assert(s.n_support==3&&strcmp(s.common_meshes,'3200,7200,12800'));
assert(all(f.n_valid(f.support=="common")==3));
assert(f.n_valid(f.support=="available"&f.method=="A")==4); % available stays honest
% 3. different methods lack different meshes
T=build({"A",Ne([1 2 3]);"B",Ne([2 3 4])});[f,s]=iefinal.fit_scaling_table(T,{'k_enter'});
assert(s.n_support==2&&strcmp(s.common_meshes,'7200,12800'));
assert(~s.common_fit_feasible&&all(~f.fitted(f.support=="common")));
assert(all(isnan(f.C(f.support=="common")))&&all(isnan(f.p(f.support=="common"))));
% 4. only one common point remains -> fail closed
T=build({"A",Ne([1 2]);"B",Ne([2 3])});[~,s]=iefinal.fit_scaling_table(T,{'k_enter'});
assert(s.n_support==1&&~s.common_fit_feasible);
% 5. zero common points remain -> fail closed
T=build({"A",Ne([1 2]);"B",Ne([3 4])});[f,s]=iefinal.fit_scaling_table(T,{'k_enter'});
assert(s.n_support==0&&~s.common_fit_feasible&&all(~f.fitted(f.support=="common")));
% 6. a RUN_ERROR cell must never enter the common support
cfg=iefinal.config('smoke','lp');
meshes=[160 20;240 30;320 40;400 50];
rows=struct([]);n=0;
for mi=1:4
    for spec={{'Proposed','proposed'},{'Olhoff','lp'}}
        s=spec{1};
        % Olhoff fails at the third mesh (320x40) only.
        isFail=strcmp(s{1},'Olhoff')&&mi==3;
        n=n+1;z=iefinal.empty_result_row();
        z.schema_version='iteration_efficiency_result_v1';z.method=s{1};z.method_variant=s{2};
        z.nelx=meshes(mi,1);z.nely=meshes(mi,2);z.element_count=z.nelx*z.nely;
        z.mesh=sprintf('%dx%d',z.nelx,z.nely);z.q=.99;z.P=100;z.B0=3200;z.B_ref=3200;
        z.trajectory_dtype='double';z.evaluator_id='candidate_c_adaptive_structural_mode_v1';
        z.contract_hash=cfg.manifest.scientific_contract.sha256;
        if isFail
            z.status='RUN_ERROR';z.censoring_reason='RUN_ERROR';
            z.error_identifier='iefinal:OptimizerFailure';z.error_message='LP failed';
        else
            z.status='PASS';z.k_enter=100+10*mi;z.k_cert=z.k_enter+99;z.native_iterations=z.k_enter;
        end
        if n==1,rows=z;else,rows(n)=z;end
    end
end
iefinal.validate_results(rows);
td=tempname('/tmp');mkdir(td);
sc=iefinal.generate_scaling_outputs(rows,td,false);
assert(sc.generated&&sc.common_support_enforced);
sup=readtable(fullfile(td,'scaling_common_support.csv'),'TextType','string');
ke=sup(sup.metric=="k_enter",:);
% 320x40 = 12800 elements failed for Olhoff, so it must be absent from S_common
assert(ke.n_support==3,'common support must drop the RUN_ERROR mesh');
assert(strcmp(ke.common_meshes,'3200,7200,20000'),char(ke.common_meshes));
ft=readtable(fullfile(td,'scaling_fits.csv'),'TextType','string');
ck=ft(ft.support=="common"&ft.metric=="k_enter",:);
assert(all(ck.n_valid==3)&&all(~contains(ck.included_meshes,"12800")));
% available support for Proposed still legitimately has all four
ak=ft(ft.support=="available"&ft.metric=="k_enter"&ft.method=="Proposed-proposed",:);
assert(ak.n_valid==4);
end

function T=localBuildScalingTable(spec,mk)
rows=struct([]);n=0;
for i=1:size(spec,1)
    ne=spec{i,2};
    for j=1:numel(ne),n=n+1;r=mk(spec{i,1},ne(j),100*(ne(j)/3200)^0.5);if n==1,rows=r;else,rows(n)=r;end,end
end
T=struct2table(rows);
end

function testB3FailureContainment
% Classification boundary: scientific execution failures are cell-local,
% integrity and programming failures stay campaign-fatal.
scientific={'iefinal:OptimizerFailure','iefinal:MissingTrajectory', ...
    'iefinal:NonfiniteTrajectory','iefinal:MissingReferenceTrajectory', ...
    'MATLAB:eigs:ARPACKroutineError','MATLAB:nomem','MATLAB:singularMatrix'};
for i=1:numel(scientific)
    v=iefinal.classify_cell_failure(MException(scientific{i},'x'));
    assert(v.cell_local&&strcmp(v.status,'RUN_ERROR'),scientific{i});
end
integrity={'iefinal:PreflightFailed','iefinal:ResultSchema','iefinal:OutputCollision', ...
    'iefinal:FingerprintMismatch','iefinal:TrajectoryPrecision','iefinal:TrajectoryIdentity', ...
    'iefinal:StateIndex','iefinal:TimingWorkMismatch','ie2a:OutputIsolation', ...
    'MATLAB:badsubscript','MATLAB:undefinedFunction','MATLAB:nonExistentField'};
for i=1:numel(integrity)
    v=iefinal.classify_cell_failure(MException(integrity{i},'x'));
    assert(~v.cell_local&&strcmp(v.status,'INTEGRITY_FAILURE'),integrity{i});
end
% RUN_ERROR rows: schema-valid, identity preserved, every quantity N/A.
cfg=iefinal.config('smoke','lp');
v=iefinal.classify_cell_failure(MException('iefinal:OptimizerFailure','LP failed at outer 627'));
rows=iefinal.build_error_rows('Olhoff','lp','Olhoff-LP',400,50,3200,cfg,struct('t','x'),v);
assert(numel(rows)==9);
r=iefinal.validate_results(rows);assert(r.pass);
assert(all(strcmp({rows.status},'RUN_ERROR'))&&all(strcmp({rows.censoring_reason},'RUN_ERROR')));
assert(all(strcmp({rows.error_identifier},'iefinal:OptimizerFailure')));
assert(~isempty(rows(1).error_message));
assert(all(isnan([rows.k_enter]))&&all(isnan([rows.k_cert]))&&all(isnan([rows.E1]))&&all(isnan([rows.Q])));
assert(all(isnan([rows.native_iterations]))&&all(isnan([rows.native_total_time])));
assert(all(isnan([rows.b_ref]))&&all(isnan([rows.B_meas])));
assert(all([rows.nelx]==400)&&all([rows.nely]==50)&&all(strcmp({rows.mesh},'400x50')));
assert(all(strcmp({rows.method},'Olhoff'))&&all(strcmp({rows.method_variant},'lp')));
assert(all(strcmp({rows.contract_hash},cfg.manifest.scientific_contract.sha256)));
% A fabricated scientific value in a RUN_ERROR row must be rejected.
bad=rows(1);bad.k_enter=42;
try,iefinal.validate_results(bad);error('fabricated RUN_ERROR value accepted');
catch ME,assert(strcmp(ME.identifier,'iefinal:ResultSchema'));end
bad=rows(1);bad.error_identifier='';
try,iefinal.validate_results(bad);error('RUN_ERROR without identifier accepted');
catch ME,assert(strcmp(ME.identifier,'iefinal:ResultSchema'));end
% Topology must be suppressed for a failed cell and not fabricated.
td=tempname('/tmp');mkdir(td);
art=iefinal.render_topologies(struct([]),rows,td); %#ok<NASGU>
assert(isfile(fullfile(td,'accepted_q0p99_400x50.png')));
% RUN_ERROR must be excluded from scaling entirely.
sc=iefinal.generate_scaling_outputs(rows,tempdir,false);
assert(~sc.generated&&contains(sc.reason,'no finite'));
end
function testSelectorEvidence
for v={'lp','mma','both'}
    d=localLatestRun(v{1});s=jsondecode(fileread(fullfile(d,'analysis','run_summary.json')));assert(strcmp(s.status,'SMOKE_INTEGRATION_PASS'));
    if strcmp(v{1},'lp'),assert(s.selector.lp_executed&&~s.selector.mma_executed);
    elseif strcmp(v{1},'mma'),assert(~s.selector.lp_executed&&s.selector.mma_executed);
    else,assert(s.selector.lp_executed&&s.selector.mma_executed&&numel(s.records)==4);end
end
end
function testDoubleStorageEvidence
p=iefinal.paths();files=dir(fullfile(p.runs,'smoke','lp','*','validation','double_storage_identity.json'));assert(~isempty(files));
[~,ix]=max([files.datenum]);d=jsondecode(fileread(fullfile(files(ix).folder,files(ix).name)));
assert(d.pass&&strcmp(d.trajectory_dtype,'double')&&d.stored_post_update_equals_authoritative_checkpoint&&d.exact_count_binary_identity&&d.hard_gate_identity);
end
function testScalingAndRendering
d=localLatestRun('both');assert(isfile(fullfile(d,'analysis','SMOKE_SYNTHETIC_SCALING_FITS_NOT_SCIENTIFIC.csv')));
T=readtable(fullfile(d,'analysis','SMOKE_SYNTHETIC_SCALING_FITS_NOT_SCIENTIFIC.csv'));
assert(all(isfinite(T.C(logical(T.fitted)))&isfinite(T.p(logical(T.fitted)))));
assert(any(strcmp(string(T.support),"common"))&&any(strcmp(string(T.support),"available")));
assert(isfile(fullfile(d,'analysis','SMOKE_SYNTHETIC_SCALING_SUPPORT_NOT_SCIENTIFIC.csv')));
assert(isfile(fullfile(d,'topologies','accepted_q0p99_160x20.png'))&&isfile(fullfile(d,'topologies','accepted_q0p99_160x20.pdf')));
end
function testStaleAndProductionLocks
c=iefinal.config('smoke','lp');c.manifest.common_evaluator.sha256=repmat('0',1,64);
try,iefinal.preflight(c);error('stale accepted');catch ME,assert(strcmp(ME.identifier,'iefinal:PreflightFailed'));end
c=iefinal.config('production','lp');try,iefinal.preflight(c);error('production accepted');catch ME,assert(strcmp(ME.identifier,'iefinal:PreflightFailed'));end
end
function testOutputIsolation
a=localLatestRun('lp');b=localLatestRun('mma');c=localLatestRun('both');assert(~strcmp(a,b)&&~strcmp(a,c)&&~strcmp(b,c));
assert(contains(a,[filesep 'runs' filesep 'smoke' filesep 'lp' filesep]));
end
function d=localLatestRun(v)
p=iefinal.paths();x=dir(fullfile(p.runs,'smoke',v,'*','analysis','run_summary.json'));assert(~isempty(x));[~,i]=max([x.datenum]);d=fileparts(x(i).folder);
end
function localWrite(path,v)
if ~isfolder(fileparts(path)),mkdir(fileparts(path));end
fid=fopen(path,'w');assert(fid>0);c=onCleanup(@()fclose(fid));fprintf(fid,'%s\n',jsonencode(v,PrettyPrint=true));clear c
end
