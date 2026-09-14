function tr=run_trajectory(method,variant,nelx,nely,horizon,rawDir,opts)
%RUN_TRAJECTORY Lossless discovery/reference trajectory with explicit states.
arguments
    method (1,:) char {mustBeMember(method,{'Proposed','Yuksel','Olhoff'})}
    variant (1,:) char
    nelx (1,1) double {mustBeInteger,mustBePositive}
    nely (1,1) double {mustBeInteger,mustBePositive}
    horizon (1,1) double {mustBeInteger,mustBePositive}
    rawDir (1,:) char
    opts.Stage1MaxIterations (1,1) double {mustBeInteger,mustBePositive} = 2000
end
p=iefinal.paths();if ~isfolder(rawDir),mkdir(rawDir);end
maxNumCompThreads(1);
addpath(fullfile(p.repo,'tools','Matlab'),fullfile(p.repo,'analysis','three_method_parametric_study'), ...
    fullfile(p.repo,'analysis','iteration_efficiency_phase2a'), ...
    fullfile(p.repo,'analysis','olhoff_stabilization_audit'), ...
    fullfile(p.repo,'Matlab','reproduction2007','runner'));

switch method
    case {'Proposed','Yuksel'}
        tr=localOc(method,nelx,nely,horizon,rawDir,opts.Stage1MaxIterations);
    case 'Olhoff'
        tr=localOlhoff(variant,nelx,nely,horizon);
end
assert(isa(tr.x_initial,'double')&&isa(tr.x_post,'double'),'iefinal:TrajectoryPrecision', ...
    'Authoritative trajectories must be MATLAB double.');
assert(size(tr.x_post,1)==nelx*nely&&numel(tr.x_initial)==nelx*nely,'iefinal:TrajectoryShape','Density shape mismatch.');
assert(isequal(tr.state_index,(0:size(tr.x_post,2)).'),'iefinal:StateIndex','State indexing must be initial 0 then post-updates 1..N.');
tr.method=method;tr.method_variant=variant;tr.nelx=nelx;tr.nely=nely;
tr.horizon_requested=horizon;tr.trajectory_dtype=class(tr.x_post);
tr.fingerprint=ie2a.trajectory_fingerprint(tr.x_post);
end

function tr=localOc(method,nx,ny,H,rawDir,stage1Max)
key=lower(method);obsFile=fullfile(rawDir,[key '_trajectory.mat']);
prm=struct('record_history',true,'extend_beyond_native_stop',true,'max_iters',H);
if strcmp(method,'Proposed')
    prm.move=.2;prm.tol=.01;prm.rmin_element=2;
else
    prm.move=.1;prm.stage1_tol=.01;prm.stage2_tol=.01;prm.rmin_element=2.5;prm.stage1_max_iters=stage1Max;
end
cfg=study_base_config(key,nx,ny,prm);
cleanup=ie2a.install_observer(obsFile,nx*ny,H+2100);
try
    [~,~,~,nIter,~,nStage,tel]=run_topopt_from_json(cfg);
catch ME
    clear cleanup;rethrow(ME)
end
clear cleanup
m=matfile(obsFile);n=m.n_observed;X=double(m.xPhys(:,1:n));stage=double(m.stage(1:n,1));
if strcmp(method,'Yuksel'),eligible=stage==2;else,eligible=true(n,1);end
idx=find(eligible);assert(~isempty(idx),'iefinal:MissingTrajectory','No eligible %s post-update states.',method);
first=idx(1);nativeStage=round(stage(first));hasPrev=logical(m.has_first_xPhysPrev(nativeStage,1));
if hasPrev,x0=double(m.first_xPhysPrev(:,nativeStage));elseif first>1,x0=X(:,first-1);else,x0=.5*ones(nx*ny,1);end
Xp=X(:,idx);assert(all(isfinite(Xp),'all'),'iefinal:NonfiniteTrajectory','Nonfinite density state.');
if strcmp(method,'Yuksel')
    s1=localField(nStage,'stage1',NaN);s2=numel(idx);native=struct('stage1_updates',s1,'stage2_updates',s2,'total_updates',s1+s2);
else
    native=struct('total_updates',numel(idx));
end
tr=struct('x_initial',double(x0(:)),'x_post',Xp,'state_index',(0:size(Xp,2)).', ...
    'method_gate',true(size(Xp,2),1),'solver_terminated',false,'native',native, ...
    'native_total_reported',nIter,'telemetry',tel,'discovery_timing_not_publishable',true, ...
    'stage1_budget',stage1Max);
end

function tr=localOlhoff(variant,nx,ny,H)
guard=repro2007_paths(); %#ok<NASGU>
switch variant
    case 'lp'
        [cfg,~]=repro2007_config('fig3a_best');cfg.nelx=nx;cfg.nely=ny;cfg.maxOuter=H;
        cfg.verbose=false;cfg.threads=1;cfg.captureTrajectory=true;cfg.authoritativeTrajectory=true;
        policy=struct('id','S1','move_sequence',[.005 .0025],'gap_threshold',.01,'persistence',100);
        r=olhoffOptStabilized(cfg,policy);
        assert(~strcmp(r.status,'SOLVER_FAILURE'),'iefinal:OptimizerFailure','Olhoff LP failed at outer %g.',r.failure_iteration);
        flags=double(r.hist.lpFlag(:));backend=double(r.hist.lpBackendIterations(:));
        native=struct('outer_updates',r.nOuter,'lp_calls',numel(flags), ...
            'failed_lp_calls',sum(flags~=1),'lp_backend_iterations',localFiniteSumOrNaN(backend), ...
            'lp_backend_iterations_observed',any(isfinite(backend)));
        gate=r.hist.policyStage(:)==2&r.hist.N(:)==2&r.hist.gap12(:)<=.01;
        routeRole='principal';nativeStatus=r.status;
    case 'mma'
        [cfg,~]=repro2007_config('fig3a_best');cfg.nelx=nx;cfg.nely=ny;cfg.maxOuter=H;
        cfg.verbose=false;cfg.threads=1;cfg.rminEl=3;cfg.rminPhys=.06;cfg.move=.01;
        cfg.tolMult=.05;cfg.innerSolver='mma';cfg.offDiag=true;cfg.filterMode='diag';
        cfg.maxInner=300;cfg.tolInner=.01;cfg.minInner=5;cfg.captureTrajectory=true;
        cfg.captureInnerHistories=true;cfg.extendBeyondNativeStop=true;
        r=olhoffOpt(cfg);
        assert(strcmpi(r.cfg.innerSolver,'mma')&&r.cfg.offDiag,'iefinal:WrongOlhoffRoute','Nested MMA route provenance is invalid.');
        assert(abs(r.cfg.rminEl-(.06/(r.cfg.b/r.cfg.nely)))<1e-12,'iefinal:FilterRadius','Effective MMA filter radius is stale.');
        inner=double(r.hist.nInner(:));conv=logical(r.hist.innerConv(:));cap=inner>=r.cfg.maxInner&~conv;
        native=struct('outer_updates',r.nOuter,'inner_iterations',inner,'inner_converged',conv, ...
            'inner_cap_hit',cap,'lp_calls',0,'failed_lp_calls',0,'lp_backend_iterations',NaN);
        gap=(r.hist.omega(2,:)-r.hist.omega(1,:))./r.hist.omega(1,:);
        gate=r.hist.N(:)==2&gap(:)<=.01;routeRole='secondary_paper_native_uncontrolled_vs_lp';nativeStatus=r.status;
    otherwise
        error('iefinal:OlhoffVariant','Unknown Olhoff variant %s.',variant);
end
assert(isa(r.rho_snapshots,'double'),'iefinal:TrajectoryPrecision','Olhoff authoritative snapshots are not double.');
assert(isequal(r.rho,r.rho_snapshots(:,end)),'iefinal:TrajectoryIdentity','Stored terminal post-update state differs from optimizer state.');
tr=struct('x_initial',r.rho_snapshots(:,1),'x_post',r.rho_snapshots(:,2:end), ...
    'state_index',(0:r.nOuter).','method_gate',logical(gate),'solver_terminated',false, ...
    'native',native,'native_status',nativeStatus,'route_role',routeRole,'post_call_cfg',r.cfg, ...
    'telemetry',r.hist,'discovery_timing_not_publishable',true);
end

function x=localFiniteSumOrNaN(v)
if any(isfinite(v)),x=sum(v(isfinite(v)));else,x=NaN;end
end
function x=localField(s,n,d)
if isstruct(s)&&isfield(s,n),x=double(s.(n));else,x=d;end
end
