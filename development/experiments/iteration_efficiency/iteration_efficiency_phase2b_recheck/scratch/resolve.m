repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'analysis','iteration_efficiency_phase2b_recheck'));
[p,guard]=ie2br.setup_paths(); %#ok<ASGLU>
D=load(fullfile(p.runs,'decide_96x12.mat'));
S=load(fullfile(p.runs,'probe_96x12_H3200.mat'));
levels=[.98 .99 .995];
dis=find(any(D.passLo~=D.passHi,2));
fprintf('bracket-disagreement states: %d  (min k=%d max k=%d)\n',numel(dis),min(dis),max(dis));
sample=unique([dis(:); (100:100:800).'; 2100; 2200]);
sample=sample(sample<=3200);
fprintf('prefix reruns to perform: %d  total iterations=%d\n',numel(sample),sum(sample));
nelx=96;nely=12;
[cfg,policy]=ie2br.olhoff_cfg(nelx,nely,3200);
% baseline for prefix determinism cross-check
base=olhoffOptStabilized(cfg,policy);
nS=numel(sample);
Qd=nan(nS,3); Qs=nan(nS,3); Qh=nan(nS,3); prefixBit=false(nS,1); castOk=false(nS,1);
nAt=zeros(nS,1); nBelow=zeros(nS,1); binDiff=zeros(nS,1); hgD=false(nS,1); hgS=false(nS,1);
s01=single(0.1);
t=tic;
for i=1:nS
    k=sample(i); c=cfg; c.maxOuter=k;
    r=olhoffOptStabilized(c,policy);
    assert(r.nOuter==k,'prefix did not reach cap');
    xd=r.rho; xs=r.rho_snapshots(:,end);
    castOk(i)=isequal(single(xd),xs);
    prefixBit(i)=isequal(xs,base.rho_snapshots(:,k+1));
    xdd=double(xd); xsd=double(xs);
    ed=ie2a.evaluate_common(xdd,nelx,nely,0.5); es=ie2a.evaluate_common(xsd,nelx,nely,0.5);
    Qd(i,:)=ed.Q_raw; Qs(i,:)=es.Q_raw; Qh(i,:)=S.Qhi(k,:);
    hit=(xs==s01); nAt(i)=nnz(hit); nBelow(i)=nnz(hit & xdd<=0.1);
    td=ie2a.topology_metrics(xdd,nelx,nely); ts=ie2a.topology_metrics(xsd,nelx,nely);
    hgD(i)=td.hard_gate_pass; hgS(i)=ts.hard_gate_pass;
    binDiff(i)=nnz(td.binary~=ts.binary);
    if mod(i,10)==0, fprintf('  %d/%d (%.1fs)\n',i,nS,toc(t)); end
end
fprintf('prefix reruns done %.1fs\n',toc(t));
fprintf('cast identity all: %d | prefix bit-identical all: %d\n',all(castOk),all(prefixBit));
fprintf('at-risk elements: total=%d  of which double<=0.1 (flip): %d  (%.1f%%)\n',sum(nAt),sum(nBelow),100*sum(nBelow)/max(1,sum(nAt)));
relSD=abs(Qd-Qs)./abs(Qd);
fprintf('genuine paired |Q_double-Q_single|/Q : E1 max=%.3e  E2 max=%.3e  E3 max=%.3e\n',max(relSD(:,1)),max(relSD(:,2)),max(relSD(:,3)));
atHi=max(abs(Qd-Qh)./abs(Qd),[],1);
fprintf('distance of true double from UPPER bracket: E1=%.3e E2=%.3e E3=%.3e\n',atHi(1),atHi(2),atHi(3));
% per-state acceptance under the common (single-derived) Q_ref
qref=D.refLo.Q_ref;
robD=min(Qd./qref,[],2); robS=min(Qs./qref,[],2);
fprintf('\nper-state spectral+gate acceptance under common Q_ref (b_ref=2100):\n');
for j=1:3
  pD=hgD&robD>=levels(j); pS=hgS&robS>=levels(j);
  d=find(pD~=pS);
  fprintf('  q=%.3f : %d of %d sampled states FLIP',levels(j),numel(d),nS);
  if ~isempty(d), fprintf('  -> k = %s',mat2str(sample(d).')); end
  fprintf('\n');
end
T=table(sample,nAt,nBelow,castOk,prefixBit,binDiff,hgD,hgS,Qd(:,1),Qs(:,1),Qd(:,2),Qs(:,2),Qd(:,3),Qs(:,3),robD,robS, ...
 'VariableNames',{'k','n_atrisk','n_branch_flip','cast_identity','prefix_bit_identical','binary_diff_elems', ...
 'hard_gate_double','hard_gate_single','E1_double','E1_single','E2_double','E2_single','E3_double','E3_single','robust_double','robust_single'});
writetable(T,fullfile(p.phase2br,'PAIRED_RESOLUTION.csv'));
save(fullfile(p.runs,'resolve_96x12.mat'),'sample','Qd','Qs','Qh','nAt','nBelow','hgD','hgS','binDiff','prefixBit','castOk','-v7.3');
