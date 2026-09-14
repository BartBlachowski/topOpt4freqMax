function fi_gcmma_run(name)
[S,P,L,R,E]=fi_setup();N=P.NE;n=P.nvar;m=P.m;
assert(isfile(fullfile(E,'gcmma_validation.json')));
up=fullfile(fileparts(mfilename('fullpath')),'upstream','GCMMA-MMA-code-1.5');addpath(up,'-end');
safe=~strcmp(name,'G0_UNSAFE');epsi=1e-7;if strcmp(name,'G2_GCMMA_ACCURATE'),epsi=1e-12;end
checkeps=1e-7;
x=[zeros(N,1);1];xold1=x;xold2=x;xmin=P.xmin;xmax=P.xmax;low=xmin;upp=xmax;
raa0=.01;raa=.01*ones(m,1);re0=1e-6;re=1e-6*ones(m,1);
H=[];CK=[];Trials=[];calls=0;evals=0;grads=0;wall=0;start=tic;consecutive=0;stable=0;prevA=[];stop='budget';firstProduction=NaN;
for it=1:500
 [f,J]=P.evalProd(x);grads=grads+1;evals=evals+1;f0=-x(end);df0=P.f;
 t=tic;[low,upp,raa0,raa]=asymp(it,n,x,xold1,xold2,xmin,xmax,low,upp,raa0,raa,re0,re,df0,J);wall=wall+toc(t);
 accepted=false;
 for j=0:16
  if calls>=500 || wall>=1800,break,end
  t=tic;
  [xm,ym,zm,mu,xi,et,~,~,~,f0a,fa]=gcmmasub(m,n,it,epsi,x,xmin,xmax,low,upp,raa0,raa,f0,df0,f,J,1,zeros(m,1),1000*ones(m,1),zeros(m,1));
  wall=wall+toc(t);calls=calls+1;
  cn=P.evalProd(xm);evals=evals+1;conserv=concheck(m,checkeps,f0a,-xm(end),fa,cn);
  tr=struct('iter',it,'correction',j,'calls',calls,'conservative',logical(conserv),'violation',max([-xm(end)-f0a;cn-fa]),'raa0',raa0,'raa',raa.','bs',xm(end),'solverWall',wall);
  if isempty(Trials),Trials=tr;else,Trials(end+1)=tr;end
  if ~safe || conserv,accepted=true;break,end
  if j==16,stop='correction_limit';break,end
  [raa0,raa]=raaupdate(xm,x,xmin,xmax,low,upp,-xm(end),cn,f0a,fa,raa0,raa,re0,re,checkeps);
 end
 if ~accepted,break,end
 [M,K,A]=fi_metric(P,R,xm,mu,xi,et);
 if isequal(A,prevA),stable=stable+1;else,stable=0;end;prevA=A;
 if M.fidelity,consecutive=consecutive+1;else,consecutive=0;end
 dx=max(abs(xm(1:N)-x(1:N)));rel=dx/max(max(abs(xm(1:N))),1e-12);
 if isnan(firstProduction) && it>=L.ctx.minInner && rel<L.ctx.tolInner,firstProduction=it;end
 M.iter=it;M.calls=calls;M.nonlinearEvaluations=evals;M.gradientEvaluations=grads;M.auditEvaluations=it;
 M.solverWall=wall;M.wall=toc(start);M.relStep=rel;M.dx=dx;M.activeStableCount=stable;M.corrections=j;M.conservativeViolation=tr.violation;
 al=(x-low)./(xmax-xmin);au=(upp-x)./(xmax-xmin);
 M.asyMin=min([al(1:N);au(1:N)]);M.asyMedian=median([al(1:N);au(1:N)]);M.asyMax=max([al(1:N);au(1:N)]);
 M.asyAtMin=mean(abs([al(1:N);au(1:N)]-.01)<1e-10);M.asyAtMax=mean(abs([al(1:N);au(1:N)]-10)<1e-10);
 M.ymax=max(ym);M.z=zm;
 if isempty(H),H=M;else,H(end+1)=M;end
 xold2=xold1;xold1=x;x=xm;
 if ismember(it,[1 5 10 19 20 50 100 200 500]) || calls>=500
  ck=struct('iter',it,'x',x,'lam',mu,'xsi',xi,'eta',et,'low',low,'upp',upp,'metric',M,'kkt',rmfield(K,'masks'));
  if isempty(CK),CK=ck;else,CK(end+1)=ck;end
  fprintf('%s %d calls=%d gain=%.6f d2=%.5f KKT=%.3e corrections=%d solve=%.1fs\n',name,it,calls,M.gainRecovery,M.d2,M.kkt,j,wall);
  fi_json(fullfile(E,[name '_progress.json']),M);
 end
 if it>=100 && consecutive>=5,stop='fidelity';break,end
 if calls>=500 || wall>=1800,break,end
end
if CK(end).iter~=M.iter,CK(end+1)=struct('iter',M.iter,'x',x,'lam',mu,'xsi',xi,'eta',et,'low',low,'upp',upp,'metric',M,'kkt',rmfield(K,'masks'));end
out=struct('name',name,'epsimin',epsi,'concheckTolerance',checkeps,'safe',safe,'iterations',numel(H),'calls',calls,'firstProductionStop',firstProduction,'stop',stop,'solverWall',wall,'wall',toc(start),'fidelity',M.fidelity,'correctionCalls',calls-numel(H));
save(fullfile(E,[name '.mat']),'CK','H','out','Trials','-v7.3');
fi_json(fullfile(E,[name '.json']),struct('out',out,'history',H,'trials',Trials));
end
