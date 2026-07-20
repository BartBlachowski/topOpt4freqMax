function results = test_a4_phase2(opts)
%TEST_A4_PHASE2  Executable validators for Phase-2 specification §11.
% opts.run_tiny_nonperturbation (default true) runs a 40x5 diagnostics on/off
% optimization comparison. opts.run_window_recovery (default false) executes
% the production 400x50 frozen trajectory through iteration 30 and validates
% audit recovery at iterations 25/30. Full N=inf and finite-N replay validators
% consume the completed Phase-2 artifacts and therefore remain production-run
% gates in a4_eigenpair_refresh.m.
if nargin<1, opts=struct(); end
runTiny=localOpt(opts,'run_tiny_nonperturbation',true);
runRecovery=localOpt(opts,'run_window_recovery',false);
thisDir=fileparts(mfilename('fullpath')); repoRoot=fileparts(fileparts(thisDir));
rv1=fullfile(repoRoot,'examples','Revision_v1');
addpath(thisDir); addpath(rv1); addpath(fullfile(repoRoot,'tools','Matlab'));
addpath(fullfile(repoRoot,'analysis','ourApproach','Matlab'));

fprintf('\n=== test_a4_phase2 ===\n'); nPass=0; nFail=0;
c=a4_phase2_constants();
[nPass,nFail]=ck('I-1 exact ladder/ceiling/thresholds/grid', ...
    isequal(c.window_ladder,[20 40 80 160 320]) && c.M_max==320 && ...
    c.tau_mac==.8 && c.tau_stab==.99 && c.tau_kin==.5 && ...
    c.tau_strain==.5 && c.x_low==.1 && numel(c.diagnostic_grid)==25,nPass,nFail);

% Deterministic synthetic search fixtures.
[ctx,x,phi0,Kf,Mf,free,ndof]=localOneElementFixture();
deep=@(m)localMockModes(m,ndof,25,'deep');
rDeep=a4_adaptive_mode_search(Kf,Mf,free,ndof,x,ctx,phi0,phi0,struct('solve_modes',deep));
meta=struct('arm_N','fixture','iteration',25,'event_kind','diagnostic','event_id',1);
[eDeep,cDeep]=a4_build_event_telemetry(rDeep,meta);
[eDeep.event_classes,~]=a4_classify_event(eDeep,cDeep);
[nPass,nFail]=ck('V-P2-7 E-1 above index 20, never E-2/B3', ...
    strcmp(rDeep.search_outcome,'SELECTED') && rDeep.selected_index==25 && ...
    isequal(rDeep.window_rungs_solved,[20 40 80]) && ...
    any(strcmp(eDeep.event_classes,'E-1')) && ~any(contains(eDeep.event_classes,'E-2')),nPass,nFail);

clean=@(m)localMockModes(m,ndof,1,'clean');
r0=a4_adaptive_mode_search(Kf,Mf,free,ndof,x,ctx,phi0,phi0,struct('solve_modes',clean));
[e0,c0]=a4_build_event_telemetry(r0,meta); [cl0,~]=a4_classify_event(e0,c0);
[nPass,nFail]=ck('V-P2-7 E-0 clean confirmed selection',isequal(cl0,{'E-0'}),nPass,nFail);

none=@(m)localMockModes(m,ndof,0,'none_continuous');
r2b=a4_adaptive_mode_search(Kf,Mf,free,ndof,x,ctx,phi0,phi0,struct('solve_modes',none));
[e2b,c2b]=a4_build_event_telemetry(r2b,meta); [cl2b,~]=a4_classify_event(e2b,c2b);
[nPass,nFail]=ck('V-P2-7 E-2b physical modes without continuity',isequal(cl2b,{'E-2b'}),nPass,nFail);

[ctxD,xD,phiD,Kd,Md,fd,nd]=localDisconnectedFixture();
noPhysical=@(m)localMockModes(m,nd,1,'clean');
r2a=a4_adaptive_mode_search(Kd,Md,fd,nd,xD,ctxD,phiD,phiD,struct('solve_modes',noPhysical));
[e2a,c2a]=a4_build_event_telemetry(r2a,meta); [cl2a,det2a]=a4_classify_event(e2a,c2a);
[nPass,nFail]=ck('V-P2-7 E-2a no connected candidate',any(strcmp(cl2a,'E-2a')),nPass,nFail);
[nPass,nFail]=ck('V-P2-7 E-4 requires measured topology and modal condition', ...
    any(strcmp(cl2a,'E-4')) && e2a.n_solid_components>=2 && ...
    det2a.best_mac_support_kinetic_fraction<c.tau_kin,nPass,nFail);

e3=e0; c3=c0; e3.reference_changed=true; c3(e3.selected_index).admissible=false;
[cl3,~]=a4_classify_event(e3,c3);
[nPass,nFail]=ck('V-P2-7 E-3 escalates to E-5',all(ismember({'E-3','E-5'},cl3)),nPass,nFail);
e5=e0; e5.search_outcome='SOLVER_FAILURE'; e5.failure_message='fixture';
[cl5,~]=a4_classify_event(e5,c0);
[nPass,nFail]=ck('V-P2-7 E-5 solver fixture',isequal(cl5,{'E-5'}),nPass,nFail);

tie=@(m)localMockModes(m,ndof,1,'tie');
rTie=a4_adaptive_mode_search(Kf,Mf,free,ndof,x,ctx,phi0,phi0,struct('solve_modes',tie));
[nPass,nFail]=ck('I-2 deterministic lower-index tie break', ...
    rTie.selected_index==1 && rTie.tie_flag,nPass,nFail);
rRepeat=a4_adaptive_mode_search(Kf,Mf,free,ndof,x,ctx,phi0,phi0,struct('solve_modes',deep));
[nPass,nFail]=ck('V-P2-4 screening symmetry (arm-independent inputs)', ...
    isequal(rRepeat.window_rungs_solved,rDeep.window_rungs_solved) && ...
    rRepeat.selected_index==rDeep.selected_index,nPass,nFail);
[nPass,nFail]=ck('V-P2-5 ladder determinism', ...
    isequaln(rRepeat.candidates,rDeep.candidates),nPass,nFail);

mandatory={'arm_N','iteration','event_kind','event_id','window_m_final','mode_index', ...
    'omega','mac_prev','mac_phi0','mac_solid','support_kinetic_fraction', ...
    'low_density_strain_fraction','low_density_kinetic_fraction','support_connectivity', ...
    'cond_kinetic_pass','cond_supports_pass','cond_strain_pass','cond_mac_pass', ...
    'rejection_reason','admissible','selected','tie_flag','eigensolver_status'};
[nPass,nFail]=ck('§6.1 every mandatory candidate field present', ...
    isequal(fieldnames(cDeep)',mandatory),nPass,nFail);

% V-P2-8 corrected base hash and negative test.
base=fullfile(rv1,'a4_ss_400x50_base.json'); h=a4_hash_file(base);
tmp=[tempname '.json']; fid=fopen(tmp,'w'); fprintf(fid,'{"different":true}\n'); fclose(fid);
h2=a4_hash_file(tmp); delete(tmp);
[nPass,nFail]=ck('V-P2-8 base hash and negative hash fixture', ...
    strcmp(h,'fnv1a32_c141e407') && ~strcmp(h,h2),nPass,nFail);

tinyProof=struct('executed',false,'pass',false);
if runTiny
    tinyProof=localTinyNonperturbation(rv1);
    [nPass,nFail]=ck('V-P2-1 tiny diagnostics on/off bit identity',tinyProof.pass,nPass,nFail);
end

recovery=struct('executed',false,'pass',false);
if runRecovery
    recovery=localWindowRecovery(rv1);
    [nPass,nFail]=ck('V-P2-3 production iteration-25/30 window recovery',recovery.pass,nPass,nFail);
end

phase1Pass=true;
try, evalc('test_a4_phase1();'); catch, phase1Pass=false; end
[nPass,nFail]=ck('V-P2-9 Phase-1 regressions remain 10/10',phase1Pass,nPass,nFail);

results=struct('passed',nPass,'failed',nFail,'tiny_nonperturbation',tinyProof, ...
    'window_recovery',recovery,'production_pending', ...
    {{'V-P2-2 full N=inf gate','V-P2-6 full finite-N replay','R-1..R-5/D-3..D-5 artifact evidence'}});
fprintf('  passed: %d   failed: %d\n',nPass,nFail);
if nFail>0,error('test_a4_phase2:Failed','%d Phase-2 validator(s) failed.',nFail);end
end

function [ctx,x,phi,K,M,free,ndof]=localOneElementFixture()
ndof=8; free=(1:ndof)'; K=eye(ndof); M=eye(ndof); x=1; phi=zeros(ndof,1); phi(1)=1;
ctx=struct('nelx',1,'nely',1,'edofMat',1:8,'KE',eye(8),'ME',eye(8), ...
    'M',M,'free',free,'Emax',1,'Emin',1e-9,'rho0',1,'rho_min',1e-9, ...
    'penal',3,'massInterp',struct('mode','linear','pmass',1));
end
function [ctx,x,phi,K,M,free,ndof]=localDisconnectedFixture()
ndof=24; free=(1:ndof)'; K=eye(ndof); M=eye(ndof); x=[1;0;1]; phi=zeros(ndof,1); phi(1)=1;
ctx=struct('nelx',3,'nely',1,'edofMat',reshape(1:24,8,3)', ...
    'KE',eye(8),'ME',eye(8),'M',M,'free',free,'Emax',1,'Emin',1e-9, ...
    'rho0',1,'rho_min',1e-9,'penal',3,'massInterp',struct('mode','linear','pmass',1));
end
function [omegas,Phi,residuals]=localMockModes(m,ndof,index,kind)
e1=zeros(ndof,1);e1(1)=1;e2=zeros(ndof,1);e2(min(2,ndof))=1;
Phi=repmat(e2,1,m); omegas=(1:m)'; residuals=zeros(m,1);
switch kind
    case 'deep', if m>=40,Phi(:,index)=e1;end
    case 'clean', Phi(:,index)=e1;
    case 'tie', Phi(:,1)=e1;Phi(:,2)=e1;
    case 'none_continuous'
end
end
function proof=localTinyNonperturbation(rv1)
cfg=jsondecode(fileread(fullfile(rv1,'a4_ss_400x50_base.json')));
cfg.domain.mesh.nelx=40;cfg.domain.mesh.nely=5;cfg.optimization.max_iters=4;
cfg.optimization.convergence_tol=1e-16;cfg.optimization.volume_fraction=.9;
cfg.domain.load_cases(1).loads(1).update_after=0;
cfg.optimization.a4_endpoint_export=true;cfg.optimization.a4_phase2_enabled=true;
cfg.optimization.a4_checkpoint_path=[tempname '.mat'];
cfg.optimization.a4_diagnostics_enabled=true;
[x1,~,~,n1,~,i1]=run_topopt_from_json(cfg);
cfg.optimization.a4_checkpoint_path=[tempname '.mat'];cfg.optimization.a4_diagnostics_enabled=false;
[x2,~,~,n2,~,i2]=run_topopt_from_json(cfg);
proof=struct('executed',true,'pass',n1==n2 && isequal(x1,x2) && ...
    isequaln(i1.a4_phase2.iteration_histories,i2.a4_phase2.iteration_histories), ...
    'iterations',[n1 n2],'topology_bit_identical',isequal(x1,x2), ...
    'trajectory_bit_identical',isequaln(i1.a4_phase2.iteration_histories,i2.a4_phase2.iteration_histories));
end
function proof=localWindowRecovery(rv1)
cfg=jsondecode(fileread(fullfile(rv1,'a4_ss_400x50_base.json')));
cfg.optimization.max_iters=30;cfg.optimization.convergence_tol=1e-16;
cfg.domain.load_cases(1).loads(1).update_after=0;cfg.optimization.a4_endpoint_export=true;
cfg.optimization.a4_phase2_enabled=true;cfg.optimization.a4_diagnostics_enabled=true;
cfg.optimization.a4_checkpoint_path=[tempname '.mat'];
[~,~,~,~,~,info]=run_topopt_from_json(cfg); ev=info.a4_phase2.screening_events;
e25=ev([ev.iteration]==25);e30=ev([ev.iteration]==30);
proof=struct('executed',true,'pass',false,'iteration25',e25,'iteration30',e30);
if ~isempty(e25)&&~isempty(e30)
    proof.pass=e25.selected_index==49 && e30.selected_index==37 && ...
        abs(e25.selected_mac_prev-.978)<.01 && abs(e30.selected_mac_prev-.966)<.01;
end
end
function v=localOpt(s,n,d),v=d;if isstruct(s)&&isfield(s,n),v=s.(n);end,end
function [p,f]=ck(name,cond,p,f),if cond,fprintf('  [PASS] %s\n',name);p=p+1;else,fprintf(2,'  [FAIL] %s\n',name);f=f+1;end,end
