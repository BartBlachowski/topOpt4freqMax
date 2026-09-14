function outFile = run_e1_trajectory(nelx,nely)
%RUN_E1_TRAJECTORY Evaluate the frozen R3 common-E1 raw model at every x_k.
% The FE/eigensolver definitions below are a direct extraction of
% study_evaluate_design.m. Verification against that authoritative function
% is mandatory at every preregistered checkpoint before the CSV is accepted.

maxNumCompThreads(1);
here=fileparts(mfilename('fullpath')); repo=fileparts(fileparts(here));
addpath(fullfile(repo,'analysis','three_method_parametric_study'));
src=fullfile(repo,'analysis','olhoff_native_convergence','results', ...
    sprintf('development_%dx%d.mat',nelx,nely));
S=load(src,'res'); X=double(S.res.telemetry.rho_snapshots);
assert(size(X,2)==1601 && S.res.telemetry.snapshot_stride==1);

[KE,ME]=q4_matrices(8/nelx,1/nely,0.3,1.0);
[iK,jK]=assembly_indices(nelx,nely);
ndof=2*(nelx+1)*(nely+1); jMid=round(nely/2);
nL=jMid; nR=nelx*(nely+1)+jMid;
fixed=[2*nL+1;2*nL+2;2*nR+1;2*nR+2]; free=setdiff((1:ndof)',fixed);
opts=struct('disp',0,'maxit',1000,'tol',1e-8,'v0',deterministic_v0(numel(free)));

iteration=(0:1600)'; omega1=NaN(1601,1); omega2=omega1; omega3=omega1;
evaluation_time_s=omega1; fallback_sm=false(1601,1);
for q=1:1601
    z=max(0,min(1,X(:,q))); t=tic;
    Ee=1e7*(1e-6+(1-1e-6)*z.^3); rr=1e-6+(1-1e-6)*z;
    K=sparse(iK,jK,reshape(KE(:)*Ee',[],1),ndof,ndof); K=(K+K')/2;
    M=sparse(iK,jK,reshape(ME(:)*rr',[],1),ndof,ndof); M=(M+M')/2;
    try
        [~,D]=eigs(K(free,free),M(free,free),3,'smallestabs',opts);
    catch
        [~,D]=eigs(K(free,free),M(free,free),3,'sm',opts); fallback_sm(q)=true;
    end
    lam=sort(real(diag(D)),'ascend'); w=sqrt(max(lam,0));
    omega1(q)=w(1); omega2(q)=w(2); omega3(q)=w(3);
    evaluation_time_s(q)=toc(t);
    if mod(iteration(q),100)==0
        fprintf('%dx%d k=%d E1raw=%.9f elapsed=%.3fs\n', ...
            nelx,nely,iteration(q),omega1(q),evaluation_time_s(q));
    end
end

T=table(repmat(string(sprintf('%dx%d',nelx,nely)),1601,1),iteration, ...
    omega1,omega2,omega3,evaluation_time_s,fallback_sm, ...
    'VariableNames',{'mesh','iteration','common_raw_E1_omega1', ...
    'common_raw_E1_omega2','common_raw_E1_omega3','evaluation_time_s','fallback_sm'});

pre=jsondecode(fileread(fullfile(here,'study_preregistration.json')));
maxAbs=0;
for k=double(pre.predeclared_checkpoints(:))'
    ev=study_evaluate_design(X(:,k+1),nelx,nely,0.5);
    maxAbs=max(maxAbs,max(abs([omega1(k+1);omega2(k+1);omega3(k+1)]-ev.omega_raw_E1(:))));
end
assert(maxAbs<1e-8,'Extracted E1 evaluator failed exact-definition verification: %.3g',maxAbs);
outFile=fullfile(here,'raw',sprintf('e1_raw_trajectory_%dx%d.csv',nelx,nely));
writetable(T,outFile);
fprintf('Wrote %s; verification max abs %.3g; total eval %.3fs\n', ...
    outFile,maxAbs,sum(evaluation_time_s));
end

function v=deterministic_v0(n)
s=RandStream('twister','Seed',42); v=randn(s,n,1); v=v/norm(v);
end

function [iK,jK]=assembly_indices(nelx,nely)
nEl=nelx*nely; edof=zeros(nEl,8);
for ex=0:nelx-1
    for ey=0:nely-1
        e=ey+ex*nely+1; n1=(nely+1)*ex+ey; n2=(nely+1)*(ex+1)+ey;
        edof(e,:)=[2*n1+1 2*n1+2 2*n2+1 2*n2+2 2*(n2+1)+1 2*(n2+1)+2 2*(n1+1)+1 2*(n1+1)+2];
    end
end
iK=reshape(kron(edof,ones(1,8))',[],1); jK=reshape(kron(edof,ones(8,1))',[],1);
end

function [KE,ME]=q4_matrices(hx,hy,nu,t)
D=(1/(1-nu^2))*[1 nu 0;nu 1 0;0 0 0.5*(1-nu)];
invJ=[2/hx 0;0 2/hy]; detJ=0.25*hx*hy; gp=1/sqrt(3); KE=zeros(8);
for xi=[-gp gp]
    for eta=[-gp gp]
        a=0.25*[-(1-eta) (1-eta) (1+eta) -(1+eta)];
        b=0.25*[-(1-xi) -(1+xi) (1+xi) (1-xi)];
        d=invJ*[a;b]; B=zeros(3,8);
        B(1,1:2:end)=d(1,:); B(2,2:2:end)=d(2,:);
        B(3,1:2:end)=d(2,:); B(3,2:2:end)=d(1,:);
        KE=KE+B'*D*B*detJ;
    end
end
KE=t*KE; Ms=(hx*hy/36)*[4 2 1 2;2 4 2 1;1 2 4 2;2 1 2 4]; ME=t*kron(Ms,eye(2));
end
