repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'));
[p,guard]=ie2br.setup_paths(); %#ok<ASGLU>
nelx=96; nely=12; H=1600;
[cfg,policy]=ie2br.olhoff_cfg(nelx,nely,H);
t=tic; base=olhoffOptStabilized(cfg,policy); fprintf('baseline %dx%d H=%d status=%s nOuter=%d wall=%.1fs\n',nelx,nely,H,base.status,base.nOuter,toc(t));
X=base.rho_snapshots;  ns=size(X,2);
fprintf('snapshot dims %dx%d  class=%s\n',size(X,1),ns,class(X));
Q=nan(ns-1,3); H0=false(ns-1,1);
t=tic;
for k=1:ns-1                    % state after k accepted updates = column k+1
    x=double(X(:,k+1));
    ev=study_evaluate_design(x,nelx,nely,0.5);
    Q(k,:)=[ev.omega_raw_E1(1) ev.omega_raw_E2(1) ev.omega_raw_E3(1)];
    tm=ie2a.topology_metrics(x,nelx,nely);
    H0(k)=tm.hard_gate_pass;
    if mod(k,200)==0, fprintf('  eval %d/%d  (%.1fs)\n',k,ns-1,toc(t)); end
end
fprintf('eval total %.1fs   H0 true: %d/%d  first true=%d  longest run=%d\n', toc(t), nnz(H0),numel(H0),find(H0,1), localLongest(H0));
ref=ie2a.reference_phase(Q,H0);
fprintf('REFERENCE status=%s  b_ref=%s  Q_ref=%s\n',ref.status,mat2str(ref.b_ref),mat2str(ref.Q_ref,8));
save(fullfile(p.runs,'probe_96x12_H1600.mat'),'Q','H0','ref','-v7.3');
function L=localLongest(v)
L=0;c=0; for i=1:numel(v), if v(i), c=c+1; L=max(L,c); else, c=0; end, end
end
