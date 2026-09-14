function res = olhoffOptTelemetry(cfg, opts)
%OLHOFFOPTTELEMETRY Audit-only mirror of frozen OLHOFFOPT with observation.
%
% This file is intentionally outside Matlab/reproduction2007.  Numerical
% decisions are copied verbatim from the frozen outer loop; additions only
% observe already-computed values, retain density states, and optionally apply
% a separately specified online stopping detector after the frozen update.
% Identity against the authoritative frozen trajectory is mandatory.

if nargin < 2, opts = struct(); end
opts = localOptions(opts, cfg);

maxNumCompThreads(cfg.threads);
t0 = tic;

mdl = model2D(cfg);
NE  = mdl.nele;
rho = cfg.rho0*ones(NE,1);

if isfield(cfg,'rminPhys') && ~isempty(cfg.rminPhys) && cfg.rminPhys > 0
    dyEl = cfg.b/cfg.nely;
    cfg.rminEl = cfg.rminPhys/dyEl;
end
flt = prepFilter(cfg.nelx, cfg.nely, cfg.rminEl);

n     = cfg.n;
Nmax  = cfg.Nmax;
Jcalc = n + Nmax;

hist = struct('omega',[],'N',[],'beta',[],'nInner',[],'dxOuter',[], ...
              'vol',[],'tEig',[],'tGrad',[],'tInner',[],'degen',[],'multJ',[], ...
              'innerConv',[],'cumInner',[]);
log = {};
cumInner = 0;
telemetry = localTelemetryInit(NE, Jcalc, cfg.maxOuter, opts, rho);
PhiPrev = [];
wPrev = [];
NPrev = NaN;
rhoPrev = rho;
rhoTwoBack = [];

if cfg.verbose
    fprintf('%4s %9s %9s %9s %4s %9s %6s %6s %8s %9s %7s\n', ...
            'it','omega1','omega2','omega3','N','sqrt(beta)', ...
            'inner','cumIn','maxdrho','vol','conv');
end

for outer = 1:cfg.maxOuter
    te = tic;
    [K,M] = assemble2D(mdl, rho, cfg.p, cfg.massInterp);
    lastwarn('');
    [w, Phi, lam, eigInfo] = eigSolve(K, M, Jcalc, cfg.solver);
    [eigWarn,eigWarnId] = lastwarn;
    tEig = toc(te);

    N = 1;
    while n+N <= Jcalc-1 && abs(w(n+N)-w(n))/w(n) < cfg.tolMult
        N = N + 1;
    end
    if N >= Nmax
        log{end+1} = sprintf('iter %d: detected N=%d >= Nmax=%d, J may be truncated',outer,N,Nmax); %#ok<AGROW>
    end
    J = n + N;
    multJ = (J+1 <= Jcalc) && abs(w(J+1)-w(J))/w(J) < cfg.tolMult;
    if multJ
        log{end+1} = sprintf('iter %d: omega_J (J=%d) is itself multiple -- (25b) undefined',outer,J); %#ok<AGROW>
    end

    tg = tic;
    idx     = n:(n+N-1);
    lamTild = mean(lam(idx));
    F       = genGrad(mdl, rho, cfg.p, cfg.massInterp, Phi, lamTild, idx);
    FJ      = genGrad(mdl, rho, cfg.p, cfg.massInterp, Phi, lam(J), J);
    fJJ     = FJ(:,1,1);

    switch lower(cfg.filterMode)
        case 'diag'
            for j = 1:N
                F(:,j,j) = applyFilter(flt, rho, F(:,j,j));
            end
        case 'all'
            for s = 1:N
                for k = s:N
                    v = applyFilter(flt, rho, F(:,s,k));
                    F(:,s,k) = v;  F(:,k,s) = v;
                end
            end
        case 'none'
        otherwise
            error('olhoffOptTelemetry:filterMode','unknown filterMode %s',cfg.filterMode);
    end
    fJJ = applyFilter(flt, rho, fJJ);
    tGrad = toc(tg);

    ti = tic;
    ctx = struct('F',F,'fJJ',fJJ,'lam',lam(idx),'lamJ',lam(J), ...
                 'rho',rho,'rhomin',cfg.rhomin,'volfrac',cfg.volfrac, ...
                 'move',cfg.move,'maxInner',cfg.maxInner, ...
                 'tolInner',cfg.tolInner,'minInner',cfg.minInner, ...
                 'offDiag',cfg.offDiag);
    if strcmpi(cfg.innerSolver,'lp')
        [drho, st] = innerLoopLP(ctx);
        if ~st.conv
            log{end+1} = sprintf('iter %d: LP inner solve failed (flag=%d)',outer,st.lpFlag); %#ok<AGROW>
        end
    else
        [drho, st] = innerLoop(ctx);
    end
    tInner = toc(ti);

    rho = min(1, max(cfg.rhomin, rho + drho));
    dxOuter = max(abs(drho));

    hist.omega(:,outer)  = w(1:min(Jcalc,numel(w)));
    hist.N(outer)        = N;
    hist.beta(outer)     = st.beta;
    hist.nInner(outer)   = st.nInner;
    cumInner             = cumInner + st.nInner;
    hist.cumInner(outer) = cumInner;
    hist.innerConv(outer)= st.conv;
    hist.dxOuter(outer)  = dxOuter;
    hist.vol(outer)      = mean(rho);
    hist.tEig(outer)     = tEig;
    hist.tGrad(outer)    = tGrad;
    hist.tInner(outer)   = tInner;
    hist.degen(outer)    = st.degenHits;
    hist.multJ(outer)    = multJ;

    telemetry = localObserve(telemetry, outer, rho, drho, w, lam, Phi, ...
        PhiPrev, wPrev, rhoTwoBack, M, N, NPrev, st, ctx, eigInfo, ...
        eigWarn, eigWarnId, cfg, opts);
    PhiPrev = Phi;
    wPrev = w;
    NPrev = N;
    rhoTwoBack = rhoPrev;
    rhoPrev = rho;

    if cfg.verbose
        fprintf('%4d %9.2f %9.2f %9.2f %4d %9.2f %6d %6d %8.4f %9.3f %7s\n', ...
                outer, w(1), w(2), w(min(3,end)), N, sqrt(max(st.beta,0)), ...
                st.nInner, cumInner, dxOuter, mean(rho), localYesNo(st.conv));
    end

    nativeBreak = dxOuter < cfg.tolOuter;
    if opts.suppress_native_stop
        nativeBreak = false;
    end
    detectorBreak = false;
    if opts.detector_enabled && st.conv && telemetry.finite_ok(outer)
        detectorBreak = nativeConvergenceDetector(hist, telemetry, outer, cfg, opts.detector);
    end
    telemetry.detector_condition(outer) = detectorBreak;

    if nativeBreak
        log{end+1} = sprintf('converged at outer iteration %d (max|drho| = %.3e)',outer,dxOuter); %#ok<AGROW>
        telemetry.break_reason = 'frozen_outer_increment_below_tolerance';
        break
    elseif detectorBreak && opts.detector_active_stop
        log{end+1} = sprintf('audit detector stopped at outer iteration %d',outer); %#ok<AGROW>
        telemetry.break_reason = 'audit_native_detector';
        break
    end
end

[K,M] = assemble2D(mdl, rho, cfg.p, cfg.massInterp);
[w, Phi, lam] = eigSolve(K, M, Jcalc, cfg.solver);
T = classifyModes(mdl, M, Phi, w);

nDone = numel(hist.N);
telemetry = localTelemetryTrim(telemetry, nDone);
if isempty(telemetry.break_reason)
    if nDone >= cfg.maxOuter
        telemetry.break_reason = 'max_outer_iterations';
    else
        telemetry.break_reason = 'terminated_early';
    end
end

res = struct('cfg',cfg,'rho',rho,'omega',w,'lambda',lam,'hist',hist, ...
             'modeTable',T,'log',{log},'nOuter',nDone, ...
             'wallclock',toc(t0),'mdl',mdl,'telemetry',telemetry);
end

function opts = localOptions(opts, cfg)
defaults = struct( ...
    'density_thresholds',[1e-4 5e-4 1e-3 2.5e-3 4.9e-3], ...
    'bound_threshold',0.01, ...
    'store_density_every',1, ...
    'detector_enabled',false, ...
    'detector_active_stop',false, ...
    'suppress_native_stop',false, ...
    'detector',struct(), ...
    'run_label','');
fn=fieldnames(defaults);
for i=1:numel(fn)
    if ~isfield(opts,fn{i}), opts.(fn{i})=defaults.(fn{i}); end
end
if opts.store_density_every < 1 || mod(opts.store_density_every,1)~=0
    error('olhoffOptTelemetry:SnapshotStride','store_density_every must be a positive integer.');
end
if opts.detector_active_stop && ~opts.detector_enabled
    error('olhoffOptTelemetry:DetectorConfig','active stop requires detector_enabled.');
end
opts.profile = struct('nelx',cfg.nelx,'nely',cfg.nely,'rminEl',cfg.rminEl, ...
    'move',cfg.move,'rhomin',cfg.rhomin,'tolMult',cfg.tolMult, ...
    'filterMode',cfg.filterMode,'innerSolver',cfg.innerSolver, ...
    'maxOuter',cfg.maxOuter,'support',cfg.support,'axial',cfg.axial);
end

function t = localTelemetryInit(NE,J,maxOuter,opts,rho0)
nThr=numel(opts.density_thresholds);
nSnap=floor(maxOuter/opts.store_density_every)+2;
t=struct();
t.schema_version='1.0';
t.run_label=opts.run_label;
t.density_thresholds=opts.density_thresholds(:).';
t.d_rms=NaN(1,maxOuter); t.d_mean=NaN(1,maxOuter);
t.rho_phase_rms=NaN(1,maxOuter); t.topology_phase_turnover=NaN(1,maxOuter);
t.moving_fraction=NaN(nThr,maxOuter);
t.move_bound_count=NaN(1,maxOuter); t.move_bound_fraction=NaN(1,maxOuter);
t.near_lower_count=NaN(1,maxOuter); t.near_lower_fraction=NaN(1,maxOuter);
t.near_upper_count=NaN(1,maxOuter); t.near_upper_fraction=NaN(1,maxOuter);
t.gaps_rel=NaN(J-1,maxOuter); t.gap_change_abs=NaN(J-1,maxOuter);
t.mode_prev_index=NaN(J,maxOuter); t.mode_best_mac=NaN(J,maxOuter);
t.mode_order_changed=false(1,maxOuter); t.N_changed=false(1,maxOuter);
t.omega_jump_rel=NaN(J,maxOuter);
t.lp_flag=NaN(1,maxOuter); t.lp_max_ineq_violation=NaN(1,maxOuter);
t.lp_max_eq_violation=NaN(1,maxOuter);
t.eig_ok=false(1,maxOuter); t.eig_warning=false(1,maxOuter);
t.eig_warning_id=cell(1,maxOuter); t.eig_warning_message=cell(1,maxOuter);
t.finite_ok=false(1,maxOuter); t.detector_condition=false(1,maxOuter);
t.rho_snapshot_iter=NaN(1,nSnap);
t.rho_snapshots=NaN(NE,nSnap,'single');
t.rho_snapshot_iter(1)=0; t.rho_snapshots(:,1)=single(rho0);
t.n_snapshots=1; t.snapshot_stride=opts.store_density_every;
t.break_reason='';
end

function t = localObserve(t,k,rho,drho,w,lam,Phi,PhiPrev,wPrev,rhoTwoBack,M,N,NPrev,st,ctx, ...
        eigInfo,eigWarn,eigWarnId,cfg,opts)
NE=numel(rho); ad=abs(drho);
t.d_rms(k)=sqrt(mean(drho.^2));
t.d_mean(k)=mean(ad);
if ~isempty(rhoTwoBack)
    t.rho_phase_rms(k)=sqrt(mean((rho-rhoTwoBack).^2));
    t.topology_phase_turnover(k)=mean((rho>=0.5)~=(rhoTwoBack>=0.5));
end
for j=1:numel(t.density_thresholds)
    t.moving_fraction(j,k)=mean(ad>t.density_thresholds(j));
end
t.move_bound_count(k)=nnz(abs(ad-cfg.move)<1e-12);
t.move_bound_fraction(k)=t.move_bound_count(k)/NE;
t.near_lower_count(k)=nnz(rho<=cfg.rhomin+opts.bound_threshold);
t.near_lower_fraction(k)=t.near_lower_count(k)/NE;
t.near_upper_count(k)=nnz(rho>=1-opts.bound_threshold);
t.near_upper_fraction(k)=t.near_upper_count(k)/NE;
t.gaps_rel(:,k)=(w(2:end)-w(1:end-1))./max(abs(w(1:end-1)),eps);
if k>1
    t.gap_change_abs(:,k)=abs(t.gaps_rel(:,k)-t.gaps_rel(:,k-1));
    t.N_changed(k)=N~=NPrev;
end
if ~isempty(PhiPrev)
    C=abs(PhiPrev'*M*Phi).^2;
    oldNorm=real(diag(PhiPrev'*M*PhiPrev));
    newNorm=real(diag(Phi'*M*Phi));
    C=C./max(oldNorm*newNorm.',eps);
    [best,idx]=max(C,[],1);
    t.mode_prev_index(:,k)=idx(:);
    t.mode_best_mac(:,k)=best(:);
    t.mode_order_changed(k)=any(idx(:)~=(1:numel(idx))');
end
if ~isempty(wPrev)
    t.omega_jump_rel(:,k)=abs(w-wPrev)./max(abs(wPrev),eps);
end
t.lp_flag(k)=st.lpFlag;
[t.lp_max_ineq_violation(k),t.lp_max_eq_violation(k)] = localLpResidual(ctx,drho,st);
t.eig_ok(k)=isfield(eigInfo,'solver') && all(isfinite(w)) && all(isfinite(lam));
t.eig_warning(k)=~isempty(eigWarnId) || ~isempty(eigWarn);
t.eig_warning_id{k}=eigWarnId; t.eig_warning_message{k}=eigWarn;
t.finite_ok(k)=all(isfinite(rho))&&all(isfinite(drho))&&all(isfinite(w))&& ...
    all(isfinite(lam))&&isfinite(st.beta)&&isfinite(t.lp_max_ineq_violation(k));
if mod(k,opts.store_density_every)==0
    t.n_snapshots=t.n_snapshots+1;
    t.rho_snapshot_iter(t.n_snapshots)=k;
    t.rho_snapshots(:,t.n_snapshots)=single(rho);
end
end

function [ineq,eq] = localLpResidual(ctx,drho,st)
if ~isfield(st,'lpFlag') || st.lpFlag~=1
    ineq=Inf; eq=Inf; return
end
N=numel(ctx.lam); pred=NaN(N+2,1);
for j=1:N
    pred(j)=st.beta-(ctx.lam(j)+ctx.F(:,j,j)'*drho);
end
pred(N+1)=st.beta-(ctx.lamJ+ctx.fJJ'*drho);
pred(N+2)=sum(ctx.rho+drho)-ctx.volfrac*numel(ctx.rho);
ineq=max([0; pred(1:N+1)/max(abs(ctx.lam(1)),eps); pred(N+2)/numel(ctx.rho)]);
eqVals=[];
for s=1:N
    for j=s+1:N
        eqVals(end+1)=ctx.F(:,s,j)'*drho/max(abs(ctx.lam(1)),eps); %#ok<AGROW>
    end
end
eq=max([0 abs(eqVals)]);
end

function t = localTelemetryTrim(t,n)
vectorFields={'d_rms','d_mean','rho_phase_rms','topology_phase_turnover', ...
    'move_bound_count','move_bound_fraction', ...
    'near_lower_count','near_lower_fraction','near_upper_count','near_upper_fraction', ...
    'mode_order_changed','N_changed','lp_flag','lp_max_ineq_violation', ...
    'lp_max_eq_violation','eig_ok','eig_warning','finite_ok','detector_condition'};
for i=1:numel(vectorFields), t.(vectorFields{i})=t.(vectorFields{i})(1:n); end
matrixFields={'moving_fraction','gaps_rel','gap_change_abs','mode_prev_index', ...
    'mode_best_mac','omega_jump_rel'};
for i=1:numel(matrixFields), t.(matrixFields{i})=t.(matrixFields{i})(:,1:n); end
t.eig_warning_id=t.eig_warning_id(1:n);
t.eig_warning_message=t.eig_warning_message(1:n);
t.rho_snapshot_iter=t.rho_snapshot_iter(1:t.n_snapshots);
t.rho_snapshots=t.rho_snapshots(:,1:t.n_snapshots);
end

function s=localYesNo(tf)
if tf, s='yes'; else, s='NO'; end
end
