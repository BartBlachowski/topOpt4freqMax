function ref = reference_phase(Q, H0, cfg)
%REFERENCE_PHASE Causal frozen reference-quality construction.
arguments
    Q (:,3) double
    H0 (:,1) logical
    cfg.P (1,1) double {mustBeInteger,mustBePositive} = 100
    cfg.LRef (1,1) double {mustBeInteger,mustBePositive} = 500
    cfg.EpsilonRef (1,1) double {mustBeNonnegative} = 0.001
    cfg.BRef (1,1) double {mustBeInteger,mustBePositive} = 3200
    cfg.SolverTerminated (1,1) logical = false
    cfg.EvaluatorValid (:,1) logical = false(0,1)
end
assert(size(Q,1)==numel(H0), 'ie2a:TrajectoryLength', 'Q and H0 lengths differ.');
if isempty(cfg.EvaluatorValid),cfg.EvaluatorValid=true(size(H0));end
assert(numel(cfg.EvaluatorValid)==numel(H0),'ie2a:TrajectoryLength','EvaluatorValid and H0 lengths differ.');
n=min(size(Q,1),cfg.BRef); Q=Q(1:n,:); H0=H0(1:n);
evaluatorValid=cfg.EvaluatorValid(1:n);
F=nan(n,3); best=nan(1,3); validWindow=false(n,1);
for b=1:n
    if b>=cfg.P && all(H0(b-cfg.P+1:b)) && all(all(isfinite(Q(b-cfg.P+1:b,:))))
        floorQ=min(Q(b-cfg.P+1:b,:),[],1); validWindow(b)=true;
        if any(isnan(best)), best=floorQ; else, best=max(best,floorQ); end
    end
    F(b,:)=best;
end
gain=nan(n,3); candidate=false(n,1); bRef=NaN;
for b=cfg.P:cfg.P:n
    if b-cfg.LRef>=1 && all(isfinite(F(b,:))) && all(isfinite(F(b-cfg.LRef,:))) && all(F(b,:)>0)
        gain(b,:)=(F(b,:)-F(b-cfg.LRef,:))./F(b,:);
        candidate(b)=all(gain(b,:)<=cfg.EpsilonRef);
        if isnan(bRef) && candidate(b), bRef=b; end
    end
end
ref=struct('F',F,'gain',gain,'valid_window_endpoint',validWindow, ...
    'freeze_candidate',candidate,'b_ref',bRef,'Q_ref',[NaN NaN NaN], ...
    'B_ref',cfg.BRef,'P',cfg.P,'L_ref',cfg.LRef,'epsilon_ref',cfg.EpsilonRef, ...
    'status','REFERENCE_NOT_ESTABLISHED');
if isfinite(bRef)
    assert(bRef>=cfg.P+cfg.LRef, 'ie2a:ReferenceTooEarly', 'Reference endpoint violates frozen minimum.');
    ref.Q_ref=F(bRef,:); ref.status='PASS';
elseif any(~evaluatorValid)
    ref.status='STRUCTURAL_MODE_NOT_FOUND';
elseif cfg.SolverTerminated
    ref.status='REFERENCE_SOLVER_TERMINATION';
end
end
