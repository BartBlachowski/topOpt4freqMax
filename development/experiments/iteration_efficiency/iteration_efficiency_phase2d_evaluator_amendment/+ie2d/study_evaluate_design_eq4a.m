function ev = study_evaluate_design_eq4a(x, nelx, nely, volfrac)
%STUDY_EVALUATE_DESIGN_EQ4A  Phase-2D amended common evaluator (OFFLINE_AMENDMENT_VALIDATION).
%   Byte-identical to the frozen study_evaluate_design.m except for the E2 and E3
%   low-density mass branch, which uses Du & Olhoff (2007) Eq. (4a) with c0 = 1e5
%   in place of the discontinuous Eq. (4).  Source: references/Du2007_Topological.pdf
%   section 2.2 -- "the coefficient c0 = 10^5 enforces the C0 continuity at the value
%   rho_e = 0.1".  Equivalent existing implementations: massScale.m case 4a and
%   mass_interp.m mode du2007_c0 (both left untouched; they carry NATIVE semantics).
%   Stiffness laws, floors, mesh, supports, eigensolver and tie-break are UNCHANGED.

if nargin < 4, volfrac = 0.5; end
x = max(0, min(1, double(x(:))));
assert(numel(x) == nelx*nely, 'Density vector size does not match mesh.');

ev = struct();
ev.volume = mean(x);
ev.volume_residual = mean(x) - volfrac;
ev.minimum_density = min(x);
ev.maximum_density = max(x);
ev.mean_density = mean(x);
ev.std_density = std(x);
ev.grayness = mean(4*x.*(1-x));
ev.gray_fraction_01_09 = mean(x > 0.1 & x < 0.9);
ev.connectivity_raw_05 = connectivity(x, nely, nelx, 0.5);

% Exact-count volume-preserving binary representation with stable index tie-break.
nSolid = round(volfrac*numel(x));
[~, order] = sortrows([-x, (1:numel(x))'], [1 2]);
xb = zeros(size(x));
xb(order(1:nSolid)) = 1;
ev.binary_solid_count = nSolid;
ev.binary_volume = mean(xb);
ev.connectivity_binary = connectivity(xb, nely, nelx, 0.5);

models = {'E1','E2','E3'};
for i = 1:numel(models)
    id = models{i};
    ev.(['omega_raw_' id]) = solve_modes(x, nelx, nely, id);
    ev.(['omega_binary_' id]) = solve_modes(xb, nelx, nely, id);
end
end

function omega = solve_modes(z, nelx, nely, model)
[KE, ME] = q4_matrices(8/nelx, 1/nely, 0.3, 1.0);
[iK,jK] = assembly_indices(nelx,nely);
switch model
    case 'E1'
        Ee = 1e7 * (1e-6 + (1-1e-6)*z.^3);
        rr = 1e-6 + (1-1e-6)*z;
    case 'E2'
        Ee = 1e7 * (1e-9 + (1-1e-9)*z.^3);
        g = z;
        low = z <= 0.1;
        g(low) = 1e5 * z(low).^6;     % Du & Olhoff Eq. (4a), c0 = 1e5 (was Eq. (4): z.^6)
        rr = 1e-9 + (1-1e-9)*g;
    case 'E3'
        z3 = max(z,1e-3);
        Ee = 1e7*z3.^3;
        g = z3;
        low = z3 <= 0.1;
        g(low) = 1e5 * z3(low).^6;    % Du & Olhoff Eq. (4a), c0 = 1e5 (was Eq. (4): z3.^6)
        rr = g;
    otherwise
        error('Unknown evaluator model %s.',model);
end
ndof=2*(nelx+1)*(nely+1);
K=sparse(iK,jK,reshape(KE(:)*Ee',[],1),ndof,ndof); K=(K+K')/2;
M=sparse(iK,jK,reshape(ME(:)*rr',[],1),ndof,ndof); M=(M+M')/2;
jMid=round(nely/2);
nL=jMid; nR=nelx*(nely+1)+jMid;
fixed=[2*nL+1;2*nL+2;2*nR+1;2*nR+2];
free=setdiff((1:ndof)',fixed);
opts=struct('disp',0,'maxit',1000,'tol',1e-8,'v0',deterministic_v0(numel(free)));
try
    [~,D]=eigs(K(free,free),M(free,free),3,'smallestabs',opts);
catch
    [~,D]=eigs(K(free,free),M(free,free),3,'sm',opts);
end
lam=sort(real(diag(D)),'ascend');
omega=sqrt(max(lam(1:min(3,numel(lam))),0));
omega(end+1:3,1)=NaN;
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
iK=reshape(kron(edof,ones(1,8))',[],1);
jK=reshape(kron(edof,ones(8,1))',[],1);
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

function c=connectivity(rho,nely,nelx,threshold)
B=reshape(rho,nely,nelx)>=threshold; visited=false(size(B)); labels=zeros(size(B)); sizes=[]; component=0;
for r=1:nely
    for col=1:nelx
        if ~B(r,col)||visited(r,col), continue; end
        component=component+1; qr=zeros(nnz(B),1); qc=qr; head=1; tail=1; qr(1)=r; qc(1)=col; visited(r,col)=true; count=0;
        while head<=tail
            rr=qr(head); cc=qc(head); head=head+1; count=count+1; labels(rr,cc)=component;
            nbr=[rr-1 cc;rr+1 cc;rr cc-1;rr cc+1];
            for k=1:4
                r2=nbr(k,1); c2=nbr(k,2);
                if r2>=1&&r2<=nely&&c2>=1&&c2<=nelx&&B(r2,c2)&&~visited(r2,c2)
                    tail=tail+1; qr(tail)=r2; qc(tail)=c2; visited(r2,c2)=true;
                end
            end
        end
        sizes(component)=count; %#ok<AGROW>
    end
end
left=unique(labels(:,1)); left(left==0)=[]; right=unique(labels(:,end)); right(right==0)=[];
c=struct('n_components',component,'left_right_connected',~isempty(intersect(left,right)), ...
    'largest_component_fraction',max([0 sizes])/max(1,nnz(B)),'solid_fraction',mean(B(:)));
end
