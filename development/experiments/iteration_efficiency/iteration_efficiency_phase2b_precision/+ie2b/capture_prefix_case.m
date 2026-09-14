function artifact=capture_prefix_case(nelx,nely,baselineHorizon,pairIterations,label)
%CAPTURE_PREFIX_CASE Obtain genuine pairs from unmodified deterministic cap runs.
arguments
    nelx (1,1) double {mustBeInteger,mustBePositive}
    nely (1,1) double {mustBeInteger,mustBePositive}
    baselineHorizon (1,1) double {mustBeInteger,mustBePositive}
    pairIterations (1,:) double {mustBeInteger,mustBeNonnegative}
    label (1,:) char
end
p=ie2b.paths();if ~isfolder(p.runs),mkdir(p.runs);end
addpath(p.phase2a,fullfile(p.repo,'analysis','olhoff_stabilization_audit'), ...
    fullfile(p.repo,'Matlab','reproduction2007','runner'));
guard=repro2007_paths(); %#ok<NASGU>
[cfg,~]=repro2007_config('fig3a_best');cfg.nelx=nelx;cfg.nely=nely;cfg.maxOuter=baselineHorizon;cfg.threads=1;cfg.verbose=false;
policy=struct('id','S1','move_sequence',[.005 .0025],'gap_threshold',.01,'persistence',100);
pairIterations=unique(double(pairIterations(:).'),'stable');assert(all(pairIterations<=baselineHorizon));
fprintf('Phase2B prefix baseline %s %dx%d horizon=%d\n',label,nelx,nely,baselineHorizon);
baseline=olhoffOptStabilized(cfg,policy);assert(baseline.nOuter==baselineHorizon,'ie2b:BaselineTermination','Baseline did not reach its cap.');
n=numel(pairIterations);x_double=nan(nelx*nely,n);x_single=nan(nelx*nely,n,'single');
prefix_single_identical=false(n,1);prefix_history_identical=false(n,1);cast_identity=false(n,1);status=strings(n,1);policy_stage=nan(n,1);
for i=1:n
    k=pairIterations(i);
    if k==0
        xd=double(cfg.rho0)*ones(nelx*nely,1);xs=baseline.rho_snapshots(:,1);
        prefix_single_identical(i)=isequal(xs,baseline.rho_snapshots(:,1));prefix_history_identical(i)=true;status(i)="INITIAL";policy_stage(i)=1;
    else
        c=cfg;c.maxOuter=k;r=olhoffOptStabilized(c,policy);
        assert(r.nOuter==k,'ie2b:PrefixTermination','Prefix run k=%d did not reach its cap.',k);
        xd=r.rho;xs=r.rho_snapshots(:,end);status(i)=string(r.status);policy_stage(i)=r.final_policy_stage;
        prefix_single_identical(i)=isequal(xs,baseline.rho_snapshots(:,k+1));
        expectedTriggers=baseline.trigger_iterations(baseline.trigger_iterations<=k);
        prefix_history_identical(i)=localHistoryPrefix(r.hist,baseline.hist,k)&&isequal(r.trigger_iterations(:),expectedTriggers(:));
    end
    x_double(:,i)=double(xd);x_single(:,i)=xs;cast_identity(i)=isequal(single(xd),xs);
    assert(prefix_single_identical(i)&&prefix_history_identical(i)&&cast_identity(i), ...
        'ie2b:PrefixMismatch','Prefix/cast identity failed at k=%d [single=%d history=%d cast=%d].', ...
        k,prefix_single_identical(i),prefix_history_identical(i),cast_identity(i));
    fprintf('  paired k=%d (%d/%d)\n',k,i,n);
end
cfgRecord=cfg;policyRecord=policy;baselineScientific=localStripTiming(baseline); %#ok<NASGU>
pairFile=fullfile(p.runs,[label '_paired_states.mat']);
save(pairFile,'x_double','x_single','pairIterations','prefix_single_identical','prefix_history_identical', ...
    'cast_identity','status','policy_stage','cfgRecord','policyRecord','baselineScientific','-v7.3');
logTable=table(pairIterations(:),status,policy_stage,prefix_single_identical,prefix_history_identical,cast_identity, ...
    'VariableNames',{'iteration','status','policy_stage','prefix_single_identical','prefix_history_identical','cast_identity'});
logFile=fullfile(p.runs,[label '_prefix_log.csv']);writetable(logTable,logFile);
artifact=struct('label',label,'nelx',nelx,'nely',nely,'baseline_horizon',baselineHorizon, ...
    'pair_iterations',pairIterations,'pair_file',pairFile,'log_file',logFile,'n_pairs',n, ...
    'trigger_iterations',baseline.trigger_iterations,'baseline_status',baseline.status);
end
function ok=localHistoryPrefix(a,b,k)
fields=setdiff(fieldnames(a),{'tEig','tGrad','tInner'});ok=true;
for i=1:numel(fields)
    av=a.(fields{i});bv=b.(fields{i});
    if size(av,2)==k && size(bv,2)>=k
        bv=bv(:,1:k);
    else
        bv=bv(1:numel(av));bv=reshape(bv,size(av));
    end
    ok=ok&&isequaln(av,bv);if ~ok,return;end
end
assert(numel(a.N)==k);
end
function r=localStripTiming(r)
r=rmfield(r,{'wallclock','mdl'});r.hist=rmfield(r.hist,{'tEig','tGrad','tInner'});
end
