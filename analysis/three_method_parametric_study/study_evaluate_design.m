function ev = study_evaluate_design(x, nelx, nely, volfrac, opts)
%STUDY_EVALUATE_DESIGN Candidate-C common post-hoc evaluator.
% The scientific output is the lowest algebraically ordered mode satisfying
% all three structural conditions on the ACTUAL gray field. Exact-count binary
% spectra are retained only as endpoint/manufacturability diagnostics.
arguments
    x {mustBeNumeric,mustBeReal,mustBeFinite}
    nelx (1,1) double {mustBeInteger,mustBePositive}
    nely (1,1) double {mustBeInteger,mustBePositive}
    volfrac (1,1) double = 0.5
    opts.ComputeBinaryDiagnostic (1,1) logical = true
    opts.TechnicalMaxModes (1,1) double {mustBePositive} = Inf
    opts.InjectEigensolverFailure (1,1) logical = false
    opts.InjectInvalidEigenpairs (1,1) logical = false
    opts.InjectNonfiniteDiagnostics (1,1) logical = false
end
x = max(0, min(1, double(x(:))));
assert(numel(x) == nelx*nely, 'Density vector size does not match mesh.');

ev = struct();
ev.evaluator_candidate = 'C';
ev.density_representation = 'ACTUAL_GRAY';
ev.modal_classifier_version = 'candidate_c_unanimous_v1';
ev.volume = mean(x);
ev.volume_residual = mean(x) - volfrac;
ev.minimum_density = min(x);
ev.maximum_density = max(x);
ev.mean_density = mean(x);
ev.std_density = std(x);
ev.grayness = mean(4*x.*(1-x));
ev.gray_fraction_01_09 = mean(x > 0.1 & x < 0.9);
ev.connectivity_raw_05 = connectivity(x, nely, nelx, 0.5);

% Existing exact-count topology representation. It does not define Q.
nSolid = round(volfrac*numel(x));
[~, order] = sortrows([-x, (1:numel(x))'], [1 2]);
xb = zeros(size(x)); xb(order(1:nSolid)) = 1;
ev.binary_solid_count = nSolid;
ev.binary_volume = mean(xb);
ev.connectivity_binary = connectivity(xb, nely, nelx, 0.5);
ev.binary_role = 'ENDPOINT_MANUFACTURABILITY_TOPOLOGY_DIAGNOSTIC_EXCLUDED_FROM_Q';

models = {'E1','E2','E3'}; allPass = true;
for i = 1:numel(models)
    id = models{i};
    modal = solve_structural_modes(x, nelx, nely, id, opts.TechnicalMaxModes, ...
        opts.InjectEigensolverFailure,opts.InjectInvalidEigenpairs,opts.InjectNonfiniteDiagnostics);
    ev.(['modal_raw_' id]) = modal;
    ev.(['omega_raw_' id]) = modal.omega;
    ev.(['selected_ordinal_raw_' id]) = modal.selected_ordinal;
    ev.(['selected_lambda_raw_' id]) = modal.selected_lambda;
    ev.(['selected_omega_raw_' id]) = modal.selected_omega;
    ev.(['status_raw_' id]) = modal.status;
    allPass = allPass && strcmp(modal.status,'PASS');
    if opts.ComputeBinaryDiagnostic
        bd = solve_binary_diagnostic(xb, nelx, nely, id, opts.InjectEigensolverFailure);
        ev.(['binary_diagnostic_' id]) = bd;
        ev.(['omega_binary_' id]) = bd.omega;
    else
        ev.(['binary_diagnostic_' id]) = struct('status','NOT_REQUESTED', ...
            'role','ENDPOINT_DIAGNOSTIC_EXCLUDED_FROM_Q','omega',nan(0,1));
        ev.(['omega_binary_' id]) = nan(0,1);
    end
end
if allPass, ev.status = 'PASS'; else, ev.status = 'STRUCTURAL_MODE_NOT_FOUND'; end
end

function out = solve_structural_modes(z, nelx, nely, model, technicalMaxModes, injectFailure,injectInvalid,injectNonfinite)
[Kf,Mf,md,Ee,rr,zeff] = pencil(z,nelx,nely,model);
nFree = size(Kf,1); dimensionLimit = max(0,nFree-1);
technicalLimit = min(dimensionLimit,floor(technicalMaxModes));
out = empty_modal(model); out.matrix_free_dofs = nFree; out.technical_mode_limit = technicalLimit;
if injectFailure
    out.failure_reason = 'INJECTED_EIGENSOLVER_FAILURE'; out.solver_status = 'EIGENSOLVER_FAILURE'; return
end
if technicalLimit < 1
    out.failure_reason = 'MATRIX_DIMENSION_EXHAUSTED'; out.solver_status = 'DIMENSION_EXHAUSTED'; return
end

requested = min(3,technicalLimit); escalation = 0; schedule = [];
while true
    schedule(end+1) = requested; %#ok<AGROW>
    [batch,ok,message] = solve_batch(Kf,Mf,md,Ee,rr,zeff,requested);
    if ~ok
        out.failure_reason = message; out.solver_status = 'EIGENSOLVER_FAILURE';
        out.modes_requested_final = requested; out.escalation_count = escalation; out.batch_schedule = schedule; return
    end
    if injectInvalid
        batch.eigenpair_valid(:)=false;batch.valid_structural(:)=false;
    end
    if injectNonfinite
        batch.voidKE(:)=NaN;batch.diagnostic_finite(:)=false;batch.valid_structural(:)=false;
    end
    out = batch; out.model = model; out.modes_requested_final = requested;
    out.escalation_count = escalation; out.batch_schedule = schedule;
    valid = find(batch.valid_structural,1,'first');
    if ~isempty(valid)
        out.status = 'PASS'; out.solver_status = 'PASS'; out.failure_reason = '';
        out.selected_ordinal = valid; out.selected_lambda = batch.lambda(valid);
        out.selected_omega = batch.omega(valid); out.selected_voidKE = batch.voidKE(valid);
        out.selected_voidSE = batch.voidSE(valid);
        out.selected_densityParticipation = batch.densityParticipation(valid);
        out.selected_IPR = batch.IPR(valid);
        out.selected_condition_margins = batch.condition_margins(valid,:);
        out.selected_minimum_margin = batch.minimum_margin(valid); return
    end
    if requested >= technicalLimit
        out.status = 'STRUCTURAL_MODE_NOT_FOUND'; out.solver_status = 'NO_VALID_MODE';
        if technicalLimit < dimensionLimit, out.failure_reason = 'TECHNICAL_RESOURCE_LIMIT';
        else, out.failure_reason = 'MATRIX_DIMENSION_EXHAUSTED'; end
        return
    end
    requested = min(2*requested,technicalLimit); escalation = escalation + 1;
end
end

function out = solve_binary_diagnostic(z,nelx,nely,model,injectFailure)
[Kf,Mf,md,Ee,rr,zeff] = pencil(z,nelx,nely,model);
k = min(6,max(0,size(Kf,1)-1));
out = struct('status','STRUCTURAL_MODE_NOT_FOUND','role', ...
    'ENDPOINT_MANUFACTURABILITY_TOPOLOGY_DIAGNOSTIC_EXCLUDED_FROM_Q','omega',nan(0,1));
if injectFailure || k < 1, return; end
[batch,ok,message] = solve_batch(Kf,Mf,md,Ee,rr,zeff,k);
if ok
    out = batch; out.status = 'PASS';
    out.role = 'ENDPOINT_MANUFACTURABILITY_TOPOLOGY_DIAGNOSTIC_EXCLUDED_FROM_Q';
else, out.failure_reason = message; end
end

function [out,ok,message] = solve_batch(Kf,Mf,md,Ee,rr,zeff,k)
out = empty_modal(''); ok = false; message = '';
opts=struct('disp',0,'maxit',200000,'tol',1e-10,'v0',deterministic_v0(size(Kf,1)));
try
    try, [V,D]=eigs(Kf,Mf,k,'smallestabs',opts);
    catch, [V,D]=eigs(Kf,Mf,k,'sm',opts); end
catch ME
    message = ['EIGENSOLVER_FAILURE:' ME.identifier]; return
end
lam=real(diag(D)); [lam,ix]=sort(lam,'ascend'); V=V(:,ix);
omega=sqrt(max(lam,0)); ndof=md.ndof; U=zeros(ndof,k); U(md.free,:)=V;
voidKE=nan(k,1);voidSE=nan(k,1);dwp=nan(k,1);ipr=nan(k,1);
keTotal=nan(k,1);seTotal=nan(k,1);residual=nan(k,1);
diagnosticFinite=false(k,1);eigenpairValid=false(k,1);low=zeff<=0.1;
for j=1:k
    u=V(:,j); denom=norm(Kf*u)+abs(lam(j))*norm(Mf*u)+eps;
    residual(j)=norm(Kf*u-lam(j)*(Mf*u))/denom;
    eigenpairValid(j)=isfinite(lam(j))&&lam(j)>0&&isfinite(residual(j))&&residual(j)<=1e-6;
    ue=reshape(U(md.edof,j),size(md.edof));
    ke=rr.*sum((ue*md.ME).*ue,2); se=Ee.*sum((ue*md.KE).*ue,2);
    ke=max(ke,0);se=max(se,0);keTotal(j)=sum(ke);seTotal(j)=sum(se);
    if isfinite(keTotal(j))&&keTotal(j)>0&&isfinite(seTotal(j))&&seTotal(j)>0
        ken=ke/keTotal(j);sen=se/seTotal(j);voidKE(j)=sum(ken(low));voidSE(j)=sum(sen(low));
        dwp(j)=sum(ken.*zeff);ipr(j)=sum(ken.^2);
        diagnosticFinite(j)=all(isfinite([voidKE(j),voidSE(j),dwp(j),ipr(j)]));
    end
end
margins=[0.5-voidKE,0.5-voidSE,dwp-0.5];
valid=eigenpairValid&diagnosticFinite&all(margins>0,2);
out.lambda=lam;out.omega=omega;out.eigenpair_residual=residual;out.eigenpair_valid=eigenpairValid;
out.diagnostic_finite=diagnosticFinite;out.voidKE=voidKE;out.voidSE=voidSE;
out.densityParticipation=dwp;out.IPR=ipr;out.kinetic_energy_total=keTotal;
out.strain_energy_total=seTotal;out.condition_margins=margins;
out.minimum_margin=min(margins,[],2);out.valid_structural=valid;ok=true;
end

function out=empty_modal(model)
out=struct('model',model,'classifier','UNANIMOUS_ALL_THREE','rho_void_threshold',0.1, ...
    'voidKE_threshold',0.5,'voidSE_threshold',0.5,'densityParticipation_threshold',0.5, ...
    'IPR_role','NONBINDING_QA','status','STRUCTURAL_MODE_NOT_FOUND','solver_status','NOT_RUN', ...
    'failure_reason','','modes_requested_final',0,'escalation_count',0,'batch_schedule',nan(0,1), ...
    'lambda',nan(0,1),'omega',nan(0,1),'eigenpair_residual',nan(0,1), ...
    'eigenpair_valid',false(0,1),'diagnostic_finite',false(0,1),'voidKE',nan(0,1), ...
    'voidSE',nan(0,1),'densityParticipation',nan(0,1),'IPR',nan(0,1), ...
    'kinetic_energy_total',nan(0,1),'strain_energy_total',nan(0,1), ...
    'condition_margins',nan(0,3),'minimum_margin',nan(0,1),'valid_structural',false(0,1), ...
    'selected_ordinal',NaN,'selected_lambda',NaN,'selected_omega',NaN, ...
    'selected_voidKE',NaN,'selected_voidSE',NaN,'selected_densityParticipation',NaN, ...
    'selected_IPR',NaN,'selected_condition_margins',nan(1,3), ...
    'selected_minimum_margin',NaN,'matrix_free_dofs',NaN,'technical_mode_limit',NaN);
end

function [Kf,Mf,md,Ee,rr,zeff]=pencil(z,nelx,nely,model)
[KE,ME]=q4_matrices(8/nelx,1/nely,0.3,1.0);[iK,jK,edof]=assembly_indices(nelx,nely);
switch model
    case 'E1'
        zeff=z;Ee=1e7*(1e-6+(1-1e-6)*z.^3);rr=1e-6+(1-1e-6)*z;
    case 'E2'
        zeff=z;Ee=1e7*(1e-9+(1-1e-9)*z.^3);g=z;low=z<=0.1;
        g(low)=1e5*z(low).^6;rr=1e-9+(1-1e-9)*g;
    case 'E3'
        zeff=max(z,1e-3);Ee=1e7*zeff.^3;g=zeff;low=zeff<=0.1;
        g(low)=1e5*zeff(low).^6;rr=g;
    otherwise, error('ie2a:UnknownEvaluator','Unknown evaluator model %s.',model);
end
ndof=2*(nelx+1)*(nely+1);K=sparse(iK,jK,reshape(KE(:)*Ee',[],1),ndof,ndof);K=(K+K')/2;
M=sparse(iK,jK,reshape(ME(:)*rr',[],1),ndof,ndof);M=(M+M')/2;
jMid=round(nely/2);nL=jMid;nR=nelx*(nely+1)+jMid;
fixed=[2*nL+1;2*nL+2;2*nR+1;2*nR+2];free=setdiff((1:ndof)',fixed);
Kf=K(free,free);Mf=M(free,free);md=struct('KE',KE,'ME',ME,'edof',edof,'ndof',ndof,'free',free);
end

function v=deterministic_v0(n)
s=RandStream('twister','Seed',42);v=randn(s,n,1);v=v/norm(v);
end

function [iK,jK,edof]=assembly_indices(nelx,nely)
nEl=nelx*nely;edof=zeros(nEl,8);
for ex=0:nelx-1
    for ey=0:nely-1
        e=ey+ex*nely+1;n1=(nely+1)*ex+ey;n2=(nely+1)*(ex+1)+ey;
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

function c=connectivity(rho,nely,nelx,threshold)
B=reshape(rho,nely,nelx)>=threshold;visited=false(size(B));labels=zeros(size(B));sizes=[];component=0;
for r=1:nely
    for col=1:nelx
        if ~B(r,col)||visited(r,col),continue;end
        component=component+1;qr=zeros(nnz(B),1);qc=qr;head=1;tail=1;qr(1)=r;qc(1)=col;visited(r,col)=true;count=0;
        while head<=tail
            rr=qr(head);cc=qc(head);head=head+1;count=count+1;labels(rr,cc)=component;
            nbr=[rr-1 cc;rr+1 cc;rr cc-1;rr cc+1];
            for k=1:4
                r2=nbr(k,1);c2=nbr(k,2);
                if r2>=1&&r2<=nely&&c2>=1&&c2<=nelx&&B(r2,c2)&&~visited(r2,c2)
                    tail=tail+1;qr(tail)=r2;qc(tail)=c2;visited(r2,c2)=true;
                end
            end
        end
        sizes(component)=count; %#ok<AGROW>
    end
end
left=unique(labels(:,1));left(left==0)=[];right=unique(labels(:,end));right(right==0)=[];
c=struct('n_components',component,'left_right_connected',~isempty(intersect(left,right)), ...
    'largest_component_fraction',max([0 sizes])/max(1,nnz(B)),'solid_fraction',mean(B(:)));
end
