repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'));
[p,guard]=ie2br.setup_paths(); %#ok<ASGLU>
nelx=96; nely=12; H=3200;
[cfg,policy]=ie2br.olhoff_cfg(nelx,nely,H);
t=tic; base=olhoffOptStabilized(cfg,policy);
fprintf('baseline %dx%d H=%d status=%s nOuter=%d wall=%.1fs\n',nelx,nely,H,base.status,base.nOuter,toc(t));
X=base.rho_snapshots; ns=size(X,2); s01=single(0.1); s001=single(1e-3);
n=ns-1; Qlo=nan(n,3); Qhi=nan(n,3); H0=false(n,1); atrisk=zeros(n,1); atrisk3=zeros(n,1); cutTie=false(n,1);
lastLE = 0.1;   % double 0.1 satisfies x<=0.1, forcing the x^6 branch
t=tic;
for k=1:n
    xs=X(:,k+1); x=double(xs);
    ev=ie2a.evaluate_common(x,nelx,nely,0.5); Qlo(k,:)=ev.Q_raw;
    tm=ie2a.topology_metrics(x,nelx,nely); H0(k)=tm.hard_gate_pass;
    hit = (xs==s01); atrisk(k)=nnz(hit); atrisk3(k)=nnz(xs==s001);
    if atrisk(k)>0
        xh=x; xh(hit)=lastLE;
        evh=ie2a.evaluate_common(xh,nelx,nely,0.5); Qhi(k,:)=evh.Q_raw;
    else
        Qhi(k,:)=Qlo(k,:);
    end
    ne=numel(x); nsl=round(0.5*ne); [~,o]=sortrows([-x,(1:ne).'],[1 2]);
    cutTie(k) = xs(o(nsl))==xs(o(nsl+1));
    if mod(k,400)==0, fprintf('  eval %d/%d (%.1fs)\n',k,n,toc(t)); end
end
fprintf('eval %.1fs\n',toc(t));
refLo=ie2a.reference_phase(Qlo,H0);
fprintf('SINGLE reference: status=%s b_ref=%s\n',refLo.status,mat2str(refLo.b_ref));
fprintf('at-risk(0.1) states: %d/%d  max elems=%d  at-risk(1e-3) states=%d  cutoff-tie states=%d\n', ...
   nnz(atrisk>0),n,max(atrisk),nnz(atrisk3>0),nnz(cutTie));
rel=(Qhi-Qlo)./Qlo;
fprintf('branch-interval rel width: E1 max=%.3e  E2 max=%.3e  E3 max=%.3e\n',max(rel(:,1)),max(rel(:,2)),max(rel(:,3)));
save(fullfile(p.runs,'probe_96x12_H3200.mat'),'Qlo','Qhi','H0','refLo','atrisk','atrisk3','cutTie','-v7.3');
