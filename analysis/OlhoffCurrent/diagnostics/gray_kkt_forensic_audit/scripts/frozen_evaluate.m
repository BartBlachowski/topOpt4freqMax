function frozen_evaluate()
% OFFLINE FROZEN-STATE ANALYSIS. NOT OPTIMIZATION.
% Allowed kernel calls only. No inner solver, MMA, update, restart or continuation.
here=fileparts(mfilename('fullpath')); study=fileparts(here);
root=fileparts(fileparts(study));addpath(root);guard=olhoffcurrent_paths(); %#ok<NASGU>
maxNumCompThreads(1);
for nx=[400,480,800]
 if isfile(fullfile(study,'evaluations',sprintf('fd_%d.mat',nx))),continue;end
 ny=nx/8;
 if nx==400
  path=fullfile(root,'evidence','two_branch_controller_validation','C400x50_trajectory.mat');k=466;
 else
  path=fullfile(root,'evidence','three_rung_canary_preflight',sprintf('C%dx%d_three_rung_trajectory.mat',nx,ny));
  if nx==480,k=386;else,k=468;end
 end
 mf=matfile(path);cfg=mf.cfg;rho=mf.RHO(:,k);frozen=rho;
 flat=olh.config.toLegacy(cfg);mdl=model2D(flat);p=cfg.material.stiffness.p;mass=cfg.material.mass;
 flt=prepFilter(nx,ny,cfg.filter.radiusPhysical/(1/ny));
 [K,M]=assemble2D(mdl,rho,p,mass);[omega,Phi,lam]=eigSolve(K,M,5,cfg.eigen.solver);
 [N,~]=olh.multi.detect(cfg,omega,1,5,[]);assert(N==2);idx=1:N;J=N+1;
 Fraw=genGrad(mdl,rho,p,mass,Phi,lam(1),idx);
 GK=genGrad(mdl,rho,p,mass,Phi,0,idx);
 for j=1:N
  Gj=genGrad(mdl,rho,p,mass,Phi,lam(j),j);Fraw(:,j,j)=Gj(:,1,1);
 end
 GM=Fraw-GK;Ffiltered=Fraw;
 for s=1:N,for t=s:N
  ff=applyFilter(flt,rho,Fraw(:,s,t));Ffiltered(:,s,t)=ff;Ffiltered(:,t,s)=ff;
 end,end
 FJ=genGrad(mdl,rho,p,mass,Phi,lam(J),J);fJraw=FJ(:,1,1);fJfiltered=applyFilter(flt,rho,fJraw);
 dOff=lam(idx)-lam(1);[~,draw,Vraw]=deltaLambda(Fraw,zeros(size(rho)),dOff);
 [~,dfiltered,Vfiltered]=deltaLambda(Ffiltered,zeros(size(rho)),dOff);
 massOrth=norm(Phi'*M*Phi-eye(5),'fro');eigResidual=zeros(5,1);
 for j=1:5,eigResidual(j)=norm(K*Phi(:,j)-lam(j)*M*Phi(:,j))/(norm(K*Phi(:,j))+norm(lam(j)*M*Phi(:,j)));end
 gK=GK(:,1,1);gM=GM(:,1,1);gRaw=draw(:,1);gFiltered=dfiltered(:,1);
 out=fullfile(study,'evaluations',sprintf('spectral_%d.mat',nx));
 save(out,'rho','omega','lam','Fraw','Ffiltered','GK','GM','fJraw','fJfiltered','gK','gM','gRaw','gFiltered','massOrth','eigResidual','Vraw','Vfiltered','dOff','-v7');
 fprintf('FROZEN %d omega1 %.12g gap12 %.12g\n',nx,omega(1),(omega(2)-omega(1))/omega(1));
 sample=load(fullfile(study,'evaluations',sprintf('sample_%d.mat',nx)));ids=double(sample.ids);deltas=double(sample.deltas);fd=[];
 for e=ids(:)'
  for h=deltas(:)'
   central=(rho(e)-h>=cfg.design.minimum)&&(rho(e)+h<=1)&&~((rho(e)-h<.1)&&(rho(e)+h>.1));
   if central
    rp=rho;rm=rho;rp(e)=rho(e)+h;rm(e)=rho(e)-h;
    lp=eval_lambda(mdl,rp,p,mass,cfg);lm=eval_lambda(mdl,rm,p,mass,cfg);numeric=(lp-lm)/(2*h);scheme=0;
   else
    sg=1;if rho(e)>.1,sg=-1;end
    if rho(e)-h<cfg.design.minimum,sg=1;end
    if rho(e)+h>1,sg=-1;end
    % choose direction away from the mass-law junction when needed
    if rho(e)>.1 && rho(e)-h<.1,sg=1;end
    if rho(e)<.1 && rho(e)+h>.1,sg=-1;end
    rp=rho;rpp=rho;rp(e)=rho(e)+sg*h;rpp(e)=rho(e)+2*sg*h;
    assert(all(rpp>=cfg.design.minimum)&all(rpp<=1));
    lp=eval_lambda(mdl,rp,p,mass,cfg);lpp=eval_lambda(mdl,rpp,p,mass,cfg);numeric=sg*(-3*lam(1)+4*lp-lpp)/(2*h);scheme=sg;
   end
   u=zeros(size(rho));u(e)=h;
   a=deltaLambda(Fraw,u,dOff);b=deltaLambda(Fraw,-u,dOff);subraw=(a(1)-b(1))/(2*h);
   a=deltaLambda(Ffiltered,u,dOff);b=deltaLambda(Ffiltered,-u,dOff);subfiltered=(a(1)-b(1))/(2*h);
   fd(end+1,:)=[e,rho(e),h,scheme,gRaw(e),numeric,gFiltered(e),subraw,subfiltered]; %#ok<AGROW>
   fprintf('FD %d e=%d h=%g raw=%g numeric=%g\n',nx,e,h,gRaw(e),numeric);
  end
 end
 assert(isequal(rho,frozen),'frozen state changed');
 save(fullfile(study,'evaluations',sprintf('fd_%d.mat',nx)),'fd','ids','deltas','-v7');
 fprintf('COMPLETED FROZEN %d NO OPTIMIZATION\n',nx);
end
end
function l=eval_lambda(mdl,rhoTrial,p,mass,cfg)
[K,M]=assemble2D(mdl,rhoTrial,p,mass);[~,~,lam]=eigSolve(K,M,5,cfg.eigen.solver);l=lam(1);
end
