function modes=evaluate_proposed_native_modes(x,cfg,nModes)
%EVALUATE_PROPOSED_NATIVE_MODES Offline native-model modes and localization.
% This duplicates the final eigensolve model in topopt_freq.m without changing
% or feeding information back into the optimization.
if nargin<3,nModes=3;end
x=max(0,min(1,double(x(:))));nx=cfg.domain.mesh.nelx;ny=cfg.domain.mesh.nely;
L=cfg.domain.size.length;H=cfg.domain.size.height;t=cfg.domain.thickness;
E0=cfg.material.E;Emin=E0*cfg.void_material.E_min_ratio;nu=cfg.material.nu;
rho0=cfg.material.rho;rhoMin=cfg.void_material.rho_min;p=cfg.optimization.penalization;
[KE,ME]=q4_matrices(L/nx,H/ny,nu,t);[edof,iK,jK]=assembly_indices(nx,ny);
ndof=2*(nx+1)*(ny+1);Ee=Emin+x.^p*(E0-Emin);rr=rhoMin+x*(rho0-rhoMin);
K=sparse(iK,jK,reshape(KE(:)*Ee',[],1),ndof,ndof);K=(K+K')/2;
M=sparse(iK,jK,reshape(ME(:)*rr',[],1),ndof,ndof);M=(M+M')/2;
fixed=supportsToFixedDofs(cfg.bc.supports,nx,ny,L,H);free=setdiff((1:ndof)',fixed(:));
opts=struct('v0',deterministic_v0(numel(free)));
[V,D]=eigs(K(free,free),M(free,free),nModes,'smallestabs',opts);
[lam,ord]=sort(real(diag(D)),'ascend');V=V(:,ord);omega=sqrt(max(lam,0));
phi=zeros(ndof,nModes);localization=struct([]);
for k=1:nModes
    v=V(:,k);mn=real(v'*(M(free,free)*v));v=v/sqrt(mn);phi(free,k)=v;
    pe=reshape(phi(edof(:),k),size(edof));disp2=mean(pe.^2,2);
    kinetic=zeros(numel(x),1);strain=kinetic;
    for e=1:numel(x)
        q=phi(edof(e,:),k);q=q(:);
        kinetic(e)=rr(e)*real(q'*ME*q);strain(e)=Ee(e)*real(q'*KE*q);
    end
    low=x<=0.1;gray=x>0.1&x<0.9;solid=x>=0.9;
    entry=struct('mode',k,'omega',omega(k), ...
        'displacement_fraction_low',fraction(disp2,low), ...
        'displacement_fraction_gray',fraction(disp2,gray), ...
        'displacement_fraction_solid',fraction(disp2,solid), ...
        'kinetic_fraction_low',fraction(kinetic,low), ...
        'kinetic_fraction_gray',fraction(kinetic,gray), ...
        'kinetic_fraction_solid',fraction(kinetic,solid), ...
        'strain_fraction_low',fraction(strain,low), ...
        'strain_fraction_gray',fraction(strain,gray), ...
        'strain_fraction_solid',fraction(strain,solid), ...
        'density_weighted_by_displacement',sum(x.*disp2)/max(sum(disp2),eps), ...
        'peak_to_median_element_displacement',max(disp2)/max(median(disp2),eps), ...
        'participation_elements_90pct',participation_count(disp2,0.9));
    if k==1,localization=entry;else,localization(k)=entry;end
end
modes=struct('omega',omega,'lambda',lam,'phi',phi,'free',free,'fixed',fixed(:), ...
    'K',K,'M',M,'edof',edof,'localization',localization, ...
    'model','Proposed native: Emin/E0=1e-9, stiffness x^3, linear mass with rho_min=1e-9');
end

function y=fraction(v,mask)
y=sum(v(mask))/max(sum(v),eps);
end

function n=participation_count(v,target)
v=sort(v,'descend');n=find(cumsum(v)>=target*sum(v),1,'first');if isempty(n),n=0;end
end

function v=deterministic_v0(n)
s=RandStream('twister','Seed',42);v=randn(s,n,1);v=v/norm(v);
end

function [edof,iK,jK]=assembly_indices(nx,ny)
edof=zeros(nx*ny,8);
for ex=0:nx-1
    for ey=0:ny-1
        e=ey+ex*ny+1;n1=(ny+1)*ex+ey;n2=(ny+1)*(ex+1)+ey;
        edof(e,:)=[2*n1+1 2*n1+2 2*n2+1 2*n2+2 2*(n2+1)+1 2*(n2+1)+2 2*(n1+1)+1 2*(n1+1)+2];
    end
end
iK=reshape(kron(edof,ones(1,8))',[],1);jK=reshape(kron(edof,ones(8,1))',[],1);
end

function [KE,ME]=q4_matrices(hx,hy,nu,t)
D=(1/(1-nu^2))*[1 nu 0;nu 1 0;0 0 0.5*(1-nu)];invJ=[2/hx 0;0 2/hy];
detJ=0.25*hx*hy;gp=1/sqrt(3);KE=zeros(8);
for xi=[-gp gp]
    for eta=[-gp gp]
        a=0.25*[-(1-eta) (1-eta) (1+eta) -(1+eta)];
        b=0.25*[-(1-xi) -(1+xi) (1+xi) (1-xi)];d=invJ*[a;b];B=zeros(3,8);
        B(1,1:2:end)=d(1,:);B(2,2:2:end)=d(2,:);B(3,1:2:end)=d(2,:);B(3,2:2:end)=d(1,:);
        KE=KE+B'*D*B*detJ;
    end
end
KE=t*KE;Ms=(hx*hy/36)*[4 2 1 2;2 4 2 1;1 2 4 2;2 1 2 4];ME=t*kron(Ms,eye(2));
end
