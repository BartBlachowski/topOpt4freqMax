function outFile=evaluate_stabilized_e1(matFile,profileId,nelx,nely)
%EVALUATE_STABILIZED_E1 Exact R3 raw-E1 spectrum at every saved candidate state.
maxNumCompThreads(1); here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
addpath(fullfile(repo,'analysis','three_method_parametric_study'));
S=load(matFile,'res');res=S.res;X=double(res.rho_snapshots);nState=size(X,2);
[KE,ME]=q4_matrices(8/nelx,1/nely,0.3,1);[iK,jK]=assembly_indices(nelx,nely);
ndof=2*(nelx+1)*(nely+1);jMid=round(nely/2);nL=jMid;nR=nelx*(nely+1)+jMid;
fixed=[2*nL+1;2*nL+2;2*nR+1;2*nR+2];free=setdiff((1:ndof)',fixed);
opts=struct('disp',0,'maxit',1000,'tol',1e-8,'v0',deterministic_v0(numel(free)));
omega=NaN(nState,3);evalTime=NaN(nState,1);fallback=false(nState,1);
for q=1:nState
 z=max(0,min(1,X(:,q)));t=tic;Ee=1e7*(1e-6+(1-1e-6)*z.^3);rr=1e-6+(1-1e-6)*z;
 K=sparse(iK,jK,reshape(KE(:)*Ee',[],1),ndof,ndof);K=(K+K')/2;
 M=sparse(iK,jK,reshape(ME(:)*rr',[],1),ndof,ndof);M=(M+M')/2;
 try,[~,D]=eigs(K(free,free),M(free,free),3,'smallestabs',opts);
 catch,[~,D]=eigs(K(free,free),M(free,free),3,'sm',opts);fallback(q)=true;end
 lam=sort(real(diag(D)),'ascend');omega(q,:)=sqrt(max(lam,0));evalTime(q)=toc(t);
 if mod(q-1,200)==0,fprintf('%s %dx%d E1 k=%d omega1=%.9f\n',profileId,nelx,nely,q-1,omega(q,1));end
end
iteration=(0:nState-1)';mesh=repmat(string(sprintf('%dx%d',nelx,nely)),nState,1);profile=repmat(string(upper(profileId)),nState,1);
baseCheckpoint=readtable(fullfile(repo,'analysis','olhoff_fixed_budget_audit','checkpoint_metrics.csv'),'TextType','string');
br=baseCheckpoint(baseCheckpoint.mesh==mesh(1)&baseCheckpoint.iteration==1600,:);refOmega=br.common_raw_E1_omega1;
ratio=omega(:,1)/refOmega;loss=(refOmega-omega(:,1))/refOmega;
T=table(profile,mesh,iteration,omega(:,1),omega(:,2),omega(:,3),ratio,loss,evalTime,fallback, ...
 'VariableNames',{'profile','mesh','iteration','common_raw_E1_omega1','common_raw_E1_omega2','common_raw_E1_omega3', ...
 'ratio_to_baseline_k1600','loss_to_baseline_k1600','evaluation_time_s','fallback_sm'});
outFile=fullfile(here,'raw',sprintf('e1_%s_%dx%d.csv',lower(profileId),nelx,nely));writetable(T,outFile);
check=[50 100 150 200 250 300 400 600 800 1200 min(1600,nState-1)];check=unique(check(check<=nState-1));maxAbs=0;
for k=check
 ev=study_evaluate_design(X(:,k+1),nelx,nely,0.5);maxAbs=max(maxAbs,max(abs(omega(k+1,:)'-ev.omega_raw_E1)));
end
assert(maxAbs<1e-8);fprintf('Wrote %s max-checkpoint-difference %.3g\n',outFile,maxAbs);
end

function v=deterministic_v0(n),s=RandStream('twister','Seed',42);v=randn(s,n,1);v=v/norm(v);end
function [iK,jK]=assembly_indices(nelx,nely)
nEl=nelx*nely;edof=zeros(nEl,8);for ex=0:nelx-1,for ey=0:nely-1,e=ey+ex*nely+1;n1=(nely+1)*ex+ey;n2=(nely+1)*(ex+1)+ey;edof(e,:)=[2*n1+1 2*n1+2 2*n2+1 2*n2+2 2*(n2+1)+1 2*(n2+1)+2 2*(n1+1)+1 2*(n1+1)+2];end,end
iK=reshape(kron(edof,ones(1,8))',[],1);jK=reshape(kron(edof,ones(8,1))',[],1);end
function [KE,ME]=q4_matrices(hx,hy,nu,t)
D=(1/(1-nu^2))*[1 nu 0;nu 1 0;0 0 .5*(1-nu)];invJ=[2/hx 0;0 2/hy];detJ=.25*hx*hy;gp=1/sqrt(3);KE=zeros(8);
for xi=[-gp gp],for eta=[-gp gp],a=.25*[-(1-eta) (1-eta) (1+eta) -(1+eta)];b=.25*[-(1-xi) -(1+xi) (1+xi) (1-xi)];d=invJ*[a;b];B=zeros(3,8);B(1,1:2:end)=d(1,:);B(2,2:2:end)=d(2,:);B(3,1:2:end)=d(2,:);B(3,2:2:end)=d(1,:);KE=KE+B'*D*B*detJ;end,end
KE=t*KE;Ms=(hx*hy/36)*[4 2 1 2;2 4 2 1;1 2 4 2;2 1 2 4];ME=t*kron(Ms,eye(2));end
