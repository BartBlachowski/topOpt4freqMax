function res=olhoffOptStabilizedDiagnostic(cfg,policy)
%OLHOFFOPTSTABILIZEDDIAGNOSTIC External diagnostic mirror of frozen S1 loop.
% Numerical operations and ordering match olhoffOptStabilized.m.  Only the
% failed LP attempt receives additional retained output after LINPROG returns.

if nargin<2||isempty(policy),policy=struct();end
policy=local_policy(policy,cfg.move);
maxNumCompThreads(cfg.threads);t0=tic;
mdl=model2D(cfg);NE=mdl.nele;rho=cfg.rho0*ones(NE,1);
if isfield(cfg,'rminPhys')&&~isempty(cfg.rminPhys)&&cfg.rminPhys>0
    cfg.rminEl=cfg.rminPhys/(cfg.b/cfg.nely);
end
flt=prepFilter(cfg.nelx,cfg.nely,cfg.rminEl);
n=cfg.n;Nmax=cfg.Nmax;Jcalc=n+Nmax;cumInner=0;
hist=struct('omega',[],'N',[],'beta',[],'nInner',[],'dxOuter',[],'vol',[], ...
    'tEig',[],'tGrad',[],'tInner',[],'degen',[],'multJ',[],'innerConv',[], ...
    'cumInner',[],'moveLimit',[],'policyStage',[],'trigger',[],'gap12',[], ...
    'dRms',[],'moveBoundFraction',[],'stronglyMovingFraction',[],'lpFlag',[], ...
    'finiteOk',[],'volumeResidual',[]);
snapshots=NaN(NE,cfg.maxOuter+1,'single');snapshots(:,1)=single(rho);
stage=1;counter=0;status='RUNNING';failureIteration=NaN;log={};failedAttempt=struct();

for outer=1:cfg.maxOuter
    te=tic;[K,M]=assemble2D(mdl,rho,cfg.p,cfg.massInterp);
    [w,Phi,lam]=eigSolve(K,M,Jcalc,cfg.solver);tEig=toc(te);
    N=1;
    while n+N<=Jcalc-1&&abs(w(n+N)-w(n))/w(n)<cfg.tolMult,N=N+1;end
    J=n+N;multJ=(J+1<=Jcalc)&&abs(w(J+1)-w(J))/w(J)<cfg.tolMult;
    gap12=(w(2)-w(1))/w(1);
    condition=(N==2)&&(gap12<=policy.gap_threshold);
    if condition,counter=counter+1;else,counter=0;end
    trigger=false;
    if counter>=policy.persistence&&stage<numel(policy.move_sequence)
        stage=stage+1;counter=0;trigger=true;
        log{end+1}=sprintf('iter %d: stabilization stage %d, move %.8g',outer,stage,policy.move_sequence(stage)); %#ok<AGROW>
    end
    currentMove=policy.move_sequence(stage);

    tg=tic;idx=n:(n+N-1);lamTild=mean(lam(idx));
    F=genGrad(mdl,rho,cfg.p,cfg.massInterp,Phi,lamTild,idx);
    FJ=genGrad(mdl,rho,cfg.p,cfg.massInterp,Phi,lam(J),J);fJJ=FJ(:,1,1);
    switch lower(cfg.filterMode)
        case 'diag'
            for j=1:N,F(:,j,j)=applyFilter(flt,rho,F(:,j,j));end
        case 'all'
            for s=1:N
                for k=s:N,v=applyFilter(flt,rho,F(:,s,k));F(:,s,k)=v;F(:,k,s)=v;end
            end
        case 'none'
        otherwise,error('olhoffOptStabilizedDiagnostic:filterMode','unknown filterMode %s',cfg.filterMode)
    end
    fJJ=applyFilter(flt,rho,fJJ);tGrad=toc(tg);

    ti=tic;ctx=struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
        'rho',rho,'rhomin',cfg.rhomin,'volfrac',cfg.volfrac,'move',currentMove, ...
        'maxInner',cfg.maxInner,'tolInner',cfg.tolInner,'minInner',cfg.minInner, ...
        'offDiag',cfg.offDiag);
    [drho,st,lpdiag]=innerLoopLPDiagnostic(ctx);tInner=toc(ti);
    if ~st.conv||st.lpFlag~=1||any(~isfinite(drho))
        status='SOLVER_FAILURE';failureIteration=outer;
        prev=struct('iteration',outer-1,'max_dx',last_or_nan(hist.dxOuter), ...
            'rms_dx',last_or_nan(hist.dRms),'omega1',last_or_nan_row(hist.omega,1), ...
            'omega2',last_or_nan_row(hist.omega,2),'omega3',last_or_nan_row(hist.omega,3), ...
            'N',last_or_nan(hist.N),'gap12',last_or_nan(hist.gap12), ...
            'move',last_or_nan(hist.moveLimit),'volume',mean(rho));
        failedAttempt=struct('attempted_outer_iteration',outer,'exitflag',st.lpFlag, ...
            'lp',lpdiag,'omega',w,'lambda',lam,'mode_shapes',Phi,'N',N,'J',J, ...
            'gap12',gap12,'lamref',lam(idx(1)),'multiplicity_next',multJ, ...
            'policy_stage',stage,'trigger_counter',counter,'trigger_this_iteration',trigger, ...
            'condition_N2_gap',condition,'current_move',currentMove,'volume',mean(rho), ...
            'density_min',min(rho),'density_max',max(rho),'density_mean',mean(rho), ...
            'density_grayness',mean(4*rho.*(1-rho)),'density_finite',all(isfinite(rho)), ...
            'spectrum_finite',all(isfinite(w)),'preceding_valid_update',prev, ...
            'ctx',ctx,'eig_time_s',tEig,'gradient_time_s',tGrad,'lp_time_s',tInner);
        log{end+1}=sprintf('iter %d: LP failure flag=%d conv=%d',outer,st.lpFlag,st.conv); %#ok<AGROW>
        break
    end
    rho=min(1,max(cfg.rhomin,rho+drho));snapshots(:,outer+1)=single(rho);
    dx=max(abs(drho));cumInner=cumInner+st.nInner;
    hist.omega(:,outer)=w;hist.N(outer)=N;hist.beta(outer)=st.beta;
    hist.nInner(outer)=st.nInner;hist.dxOuter(outer)=dx;hist.vol(outer)=mean(rho);
    hist.tEig(outer)=tEig;hist.tGrad(outer)=tGrad;hist.tInner(outer)=tInner;
    hist.degen(outer)=st.degenHits;hist.multJ(outer)=multJ;hist.innerConv(outer)=st.conv;
    hist.cumInner(outer)=cumInner;hist.moveLimit(outer)=currentMove;
    hist.policyStage(outer)=stage;hist.trigger(outer)=trigger;hist.gap12(outer)=gap12;
    hist.dRms(outer)=sqrt(mean(drho.^2));
    hist.moveBoundFraction(outer)=mean(abs(abs(drho)-currentMove)<1e-12);
    hist.stronglyMovingFraction(outer)=mean(abs(drho)>0.98*currentMove);
    hist.lpFlag(outer)=st.lpFlag;hist.finiteOk(outer)=all(isfinite(rho))&&all(isfinite(w))&&isfinite(st.beta);
    hist.volumeResidual(outer)=mean(rho)-cfg.volfrac;
end

nDone=numel(hist.N);snapshots=snapshots(:,1:nDone+1);
if strcmp(status,'RUNNING'),status='CAP_HIT';end
if ~strcmp(status,'SOLVER_FAILURE')
    [K,M]=assemble2D(mdl,rho,cfg.p,cfg.massInterp);[w,Phi,lam]=eigSolve(K,M,Jcalc,cfg.solver);
    modeTable=classifyModes(mdl,M,Phi,w);
else
    w=NaN(Jcalc,1);lam=w;modeTable=table();
end
res=struct('cfg',cfg,'policy',policy,'rho',rho,'omega',w,'lambda',lam, ...
    'hist',hist,'modeTable',modeTable,'log',{log},'nOuter',nDone,'wallclock',toc(t0), ...
    'mdl',mdl,'rho_snapshots',snapshots,'status',status,'failure_iteration',failureIteration, ...
    'final_policy_stage',stage,'trigger_iterations',find(hist.trigger), ...
    'failed_attempt',failedAttempt,'diagnostic_only',true);
end

function p=local_policy(p,baselineMove)
defaults=struct('id','S0','move_sequence',baselineMove,'gap_threshold',0.01,'persistence',100);
fn=fieldnames(defaults);for i=1:numel(fn),if ~isfield(p,fn{i}),p.(fn{i})=defaults.(fn{i});end,end
p.move_sequence=double(p.move_sequence(:)');
assert(~isempty(p.move_sequence)&&abs(p.move_sequence(1)-baselineMove)<1e-15);
assert(all(diff(p.move_sequence)<0)&&all(p.move_sequence<=baselineMove)||numel(p.move_sequence)==1);
assert(p.persistence==100&&p.gap_threshold==0.01,'Policy differs from preregistration.');
end

function v=last_or_nan(x)
if isempty(x),v=NaN;else,v=x(end);end
end

function v=last_or_nan_row(x,r)
if isempty(x)||size(x,1)<r,v=NaN;else,v=x(r,end);end
end
