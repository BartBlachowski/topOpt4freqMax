function probe_components()
repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'Matlab','reproduction2007','runner'));
g=repro2007_paths(); %#ok<NASGU>
cfg=repro2007_config('fig3a_best');
cfg.nelx=160;cfg.nely=20;cfg.a=8;cfg.b=1;cfg.massInterp='4b';cfg.rminEl=1.3;
cfg.support='mid';cfg.axial='both';cfg.bc='a';
mdl=model2D(cfg);NE=mdl.nele;rho=.5*ones(NE,1);
t=tic;[K,M]=assemble2D(mdl,rho,3,'4b');ta=toc(t);
t=tic;[w,Phi,lam]=eigSolve(K,M,5,'eigs');te=toc(t);
t=tic;[w2,~,~]=eigSolve(K,M,5,'dense');td=toc(t);
flt=prepFilter(160,20,1.3);
t=tic;F=genGrad(mdl,rho,3,'4b',Phi,lam(1),1);tg=toc(t);
fprintf('assemble=%.3fs eigs=%.3fs dense=%.3fs genGrad=%.3fs\n',ta,te,td,tg);
fprintf('w(eigs)=%.6f w(dense)=%.6f\n',w(1),w2(1));
% time mmasub inner iteration
ctx=struct('F',F,'fJJ',F(:,1,1),'lam',lam(1),'lamJ',lam(5),'rho',rho, ...
  'rhomin',1e-3,'volfrac',.5,'currentVolume',sum(rho), ...
  'volumeWeights',flt.H'*(ones(NE,1)./flt.Hs),'move',.005,'maxInner',20, ...
  'tolInner',1e-2,'minInner',20,'offDiag',true);
t=tic;
nvar=NE+1;m=3;xmin=[-0.005*ones(NE,1);0];xmax=[0.005*ones(NE,1);5];
x=[zeros(NE,1);1];xold1=x;xold2=x;low=xmin;upp=xmax;
for it=1:20
  fval=rand(m,1);dfdx=rand(m,nvar);
  [xmma,~,~,~,~,~,~,~,~,low,upp]=mmasub(m,nvar,it,x,xmin,xmax,xold1,xold2, ...
     -1,[zeros(NE,1);-1],fval,dfdx,low,upp,1,zeros(m,1),1000*ones(m,1),zeros(m,1));
  xold2=xold1;xold1=x;x=xmma;
end
fprintf('20 mmasub calls = %.3fs (%.4fs each)\n',toc(t),toc(t)/20);
end
