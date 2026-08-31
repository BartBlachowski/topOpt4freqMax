function tr = run_method_trajectory(method, nelx, nely, horizon, rawDir, opts)
%RUN_METHOD_TRAJECTORY Frozen-profile discovery run with return-field capture.
arguments
    method (1,:) char {mustBeMember(method,{'Proposed','Yuksel','Olhoff'})}
    nelx (1,1) double {mustBeInteger,mustBePositive}
    nely (1,1) double {mustBeInteger,mustBePositive}
    horizon (1,1) double {mustBeInteger,mustBePositive}
    rawDir (1,:) char
    opts.OlhoffVariant (1,:) char {mustBeMember(opts.OlhoffVariant,{'lp','mma'})} = 'lp'
end
p=ie2a.paths();ie2a.assert_output_isolated(rawDir,'production');if ~isfolder(rawDir),mkdir(rawDir);end
maxNumCompThreads(1);
addpath(fullfile(p.repo,'tools','Matlab'),fullfile(p.repo,'analysis','three_method_parametric_study'), ...
    fullfile(p.repo,'analysis','olhoff_stabilization_audit'),fullfile(p.repo,'Matlab','reproduction2007','runner'));
switch method
    case {'Proposed','Yuksel'}
        key=lower(method); obsFile=fullfile(rawDir,[key '_trajectory.mat']);
        prm=struct('record_history',true,'extend_beyond_native_stop',true,'max_iters',horizon);
        if strcmp(method,'Proposed')
            prm.move=.2;prm.tol=.01;prm.rmin_element=2;
        else
            prm.move=.1;prm.stage1_tol=.01;prm.stage2_tol=.01;prm.rmin_element=2.5;prm.stage1_max_iters=2000;
        end
        cfg=study_base_config(key,nelx,nely,prm);cleanup=ie2a.install_observer(obsFile,nelx*nely,horizon+2000);
        try
            [~,~,~,nIter,~,nStage,tel]=run_topopt_from_json(cfg);runError=[];
        catch ME
            runError=ME;nIter=0;nStage=struct();tel=struct();
        end
        clear cleanup;m=matfile(obsFile);n=m.n_observed;X=m.xPhys(:,1:n);stage=m.stage(1:n,1);
        if strcmp(method,'Yuksel'),eligible=stage==2;else,eligible=true(n,1);end
        tr=struct('method',method,'xPhys',X(:,eligible),'method_gate',true(sum(eligible),1), ...
            'solver_terminated',~isempty(runError),'error',localError(runError),'native_total',nIter, ...
            'stage_counts',nStage,'telemetry',tel,'discovery_timing_not_publishable',true);
    case 'Olhoff'
        if strcmp(opts.OlhoffVariant,'mma')
            error('ie2a:SecondaryRouteQualificationRequired', ...
                ['Nested MMA is the secondary paper-literal route, but its production trajectory ' ...
                 'runner is qualification-gated. Complete the route qualification before dispatch.']);
        end
        out=run_stabilization_case('S1',nelx,nely,rawDir,horizon);S=load(out,'res');r=S.res;
        X=double(r.rho_snapshots(:,2:end));gate=r.hist.policyStage(:)==2&r.hist.N(:)==2&r.hist.gap12(:)<=.01;
        tr=struct('method',method,'xPhys',X,'method_gate',gate,'solver_terminated',strcmp(r.status,'SOLVER_FAILURE'), ...
            'error',struct('identifier','','message',''),'native_total',r.nOuter,'stage_counts',struct(), ...
            'telemetry',r.hist,'discovery_timing_not_publishable',true, ...
            'new_snapshot_precision','source-native single converted to double; requires pre-production qualification', ...
            'olhoff_variant','lp','olhoff_route_role','principal');
end
tr.nelx=nelx;tr.nely=nely;tr.horizon=horizon;tr.fingerprint=ie2a.trajectory_fingerprint(tr.xPhys);
end
function e=localError(ME)
if isempty(ME),e=struct('identifier','','message','');else,e=struct('identifier',ME.identifier,'message',ME.message);end
end
