function fi_mma_run(name)
[S,P,L,R,E]=fi_setup(); assert(isfile(fullfile(E,'oracle_identity.json')));
N=P.NE; n=P.nvar; m=P.m;
cfg=struct('asyinit',.5,'asymax',.2,'epsimin',1e-7);
unit=strcmp(name,'S5_UNIT_BOX');
if strcmp(name,'S2_ASYINIT_001'),cfg.asyinit=.01;end
if any(strcmp(name,{'S3_CANONICAL_CLAMP','S34_CLAMP_ACCURACY'})),cfg.asymax=10;end
if any(strcmp(name,{'S4_SUBSOLV_ACCURACY','S34_CLAMP_ACCURACY'})),cfg.epsimin=1e-12;end
offset=zeros(n,1); scale=ones(n,1);
if unit,offset(1:N)=P.xmin(1:N);scale(1:N)=P.xmax(1:N)-P.xmin(1:N);end
xmin=(P.xmin-offset)./scale; xmax=(P.xmax-offset)./scale;
x=([zeros(N,1);1]-offset)./scale; xold1=x;xold2=x;low=xmin;upp=xmax;
budget=500; if strcmp(name,'S1_PERSISTENT'),budget=50;end
CK=struct([]); H=struct([]); consecutive=0;stable=0;prevA=[];
saveAt=[1 5 10 19 20 50 100 200 500];
solverWall=0;start=tic; firstProduction=NaN;
if strcmp(name,'B0_CURRENT_REPEATED_MMA')
 t=tic;[dp,st]=innerLoop(L.ctx); prodtime=toc(t);
 assert(isequal(dp,S.drho386) && isequal(st.beta,S.hist.beta(end)) && st.nInner==19);
 fi_json(fullfile(E,'production_reproduction.json'),struct('bitwise_drho',true,'bitwise_beta',true,'nInner',st.nInner,'wall_s',prodtime,'st',st));
end
for it=1:budget
 physical=offset+scale.*x;
 if ~unit,physical=x;end
 [fval,dfdx]=P.evalProd(physical); dfdx=dfdx.*scale.';
 bs=x(end); f0val=-bs;df0dx=zeros(n,1);df0dx(n)=-1;
 t=tic;
 if any(strcmp(name,{'B0_CURRENT_REPEATED_MMA','S1_PERSISTENT','S5_UNIT_BOX'}))
  [xm,ym,zm,mu,xi,et,~,~,~,low,upp]=mmasub(m,n,it,x,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,1,zeros(m,1),1000*ones(m,1),zeros(m,1));
 else
  [xm,ym,zm,mu,xi,et,~,~,~,low,upp]=fi_mmasub(m,n,it,x,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,1,zeros(m,1),1000*ones(m,1),zeros(m,1),cfg);
 end
 solverWall=solverWall+toc(t);
 xp=offset+scale.*xm; if ~unit,xp=xm;end
 dx=max(abs(xp(1:N)-physical(1:N))); relStep=dx/max(max(abs(xp(1:N))),1e-12);
 [M,K,A]=fi_metric(P,R,xp,mu,xi./scale,et./scale);
 if isequal(A,prevA),stable=stable+1;else,stable=0;end;prevA=A;
 if M.fidelity,consecutive=consecutive+1;else,consecutive=0;end
 if isnan(firstProduction) && it>=L.ctx.minInner && relStep<L.ctx.tolInner,firstProduction=it;end
 M.iter=it;M.calls=it;M.nonlinearEvaluations=it;M.gradientEvaluations=it;
 M.auditEvaluations=it;M.solverWall=solverWall;M.wall=toc(start);M.relStep=relStep;M.dx=dx;
 M.activeStableCount=stable;M.ymax=max(ym);M.z=zm;
 al=(x-low)./(xmax-xmin); au=(upp-x)./(xmax-xmin);
 M.asyMin=min([al(1:N);au(1:N)]);M.asyMedian=median([al(1:N);au(1:N)]);M.asyMax=max([al(1:N);au(1:N)]);
 M.asyAtMin=mean(abs([al(1:N);au(1:N)]-.01)<1e-10);M.asyAtMax=mean(abs([al(1:N);au(1:N)]-cfg.asymax)<1e-10);
 M.asyBetaLow=al(end);M.asyBetaHigh=au(end);
 % Reconstruct approximation values, observational only, after the solve.
 [f0app,fapp]=fi_mma_app(x,xm,xmin,xmax,low,upp,f0val,df0dx,fval,dfdx);
 cn=P.evalProd(xp);M.conservativeViolation=max([(-xp(end))-f0app;cn-fapp]);
 if isempty(H),H=M;else,H(it)=M;end
 xold2=xold1;xold1=x;x=xm;
 if ismember(it,saveAt) || it==budget
  ck=struct('iter',it,'x',xp,'lam',mu,'xsi',xi./scale,'eta',et./scale,'low',offset+scale.*low,'upp',offset+scale.*upp,'metric',M,'kkt',rmfield(K,'masks'));
  if isempty(CK),CK=ck;else,CK(end+1)=ck;end
  fprintf('%s %d gain=%.6f d2=%.5f KKT=%.3e step=%.3e solve=%.1fs\n',name,it,M.gainRecovery,M.d2,M.kkt,relStep,solverWall);
  fi_json(fullfile(E,[name '_progress.json']),M);
 end
 if strcmp(name,'B0_CURRENT_REPEATED_MMA') && it==19,assert(isequal(xp,L.xP19));end
 if strcmp(name,'B0_CURRENT_REPEATED_MMA') && it==500,assert(isequal(xp,L.xM500));end
 if strcmp(name,'S1_PERSISTENT') && it==19
  save(fullfile(E,'S1_state_boundary.mat'),'x','xold1','xold2','low','upp','it');
  clear x xold1 xold2 low upp
  Z=load(fullfile(E,'S1_state_boundary.mat'));x=Z.x;xold1=Z.xold1;xold2=Z.xold2;low=Z.low;upp=Z.upp;
 end
 if solverWall>=1800 || (it>=100 && consecutive>=5),break,end
end
if CK(end).iter~=it,CK(end+1)=struct('iter',it,'x',xp,'lam',mu,'xsi',xi./scale,'eta',et./scale,'low',offset+scale.*low,'upp',offset+scale.*upp,'metric',M,'kkt',rmfield(K,'masks'));end
out=struct('name',name,'cfg',cfg,'iterations',it,'firstProductionStop',firstProduction,'solverWall',solverWall,'wall',toc(start),'fidelity',M.fidelity,'label','FROZEN ONLY; no density applied');
if strcmp(name,'S1_PERSISTENT')
 B=load(fullfile(E,'B0_CURRENT_REPEATED_MMA.mat'),'CK');
 out.bitwise50=isequal(xp,B.CK([B.CK.iter]==50).x);assert(out.bitwise50);
end
save(fullfile(E,[name '.mat']),'CK','H','out','-v7.3');
fi_json(fullfile(E,[name '.json']),struct('out',out,'history',H));
end
