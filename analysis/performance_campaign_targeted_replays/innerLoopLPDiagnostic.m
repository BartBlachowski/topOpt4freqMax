function [drho,st,diag] = innerLoopLPDiagnostic(ctx)
%INNERLOOPLPDIAGNOSTIC Numerical mirror of frozen innerLoopLP plus retention.
% The LP objective, matrices, bounds, options and update decision are identical.
% The fourth LINPROG output and post-solve residual summaries are diagnostic only.

NE=numel(ctx.rho);N=numel(ctx.lam);lamref=ctx.lam(1);
Vtot=ctx.volfrac*NE;nvar=NE+1;
lo=max(ctx.rhomin-ctx.rho,-ctx.move);
hi=min(1-ctx.rho,ctx.move);
lb=[lo;0];ub=[hi;5];
f=zeros(nvar,1);f(end)=-1;
nIneq=N+2;A=zeros(nIneq,nvar);b=zeros(nIneq,1);
for j=1:N
    A(j,1:NE)=-ctx.F(:,j,j).'/lamref;
    A(j,nvar)=1;
    b(j)=ctx.lam(j)/lamref;
end
A(N+1,1:NE)=-ctx.fJJ.'/lamref;
A(N+1,nvar)=1;
b(N+1)=ctx.lamJ/lamref;
A(N+2,1:NE)=1;
b(N+2)=Vtot-sum(ctx.rho);
npair=N*(N-1)/2;Aeq=zeros(npair,nvar);beq=zeros(npair,1);r=0;
for s=1:N
    for k=s+1:N
        r=r+1;Aeq(r,1:NE)=ctx.F(:,s,k).'/lamref;
    end
end

opts=optimoptions('linprog','Display','none','Algorithm','dual-simplex-highs');
[x,fval,flag,output]=linprog(f,A,b,Aeq,beq,lb,ub,opts);

st=struct('nInner',1,'degenHits',0,'conv',flag==1,'dxHist',[],'relHist',[], ...
    'lpFlag',flag);
diag=struct('exitflag',flag,'output',output,'objective_value',fval, ...
    'returned_x',x,'lamref',lamref,'algorithm','dual-simplex-highs', ...
    'options',opts,'matrix_dimensions',[size(A,1),size(A,2),size(Aeq,1),size(Aeq,2)], ...
    'nnz_A',nnz(A),'nnz_Aeq',nnz(Aeq));
if flag~=1 || isempty(x)
    diag=complete_diagnostics(diag,A,b,Aeq,beq,lb,ub,f,x);
    drho=zeros(NE,1);st.beta=ctx.lam(1);return
end
drho=x(1:NE);st.beta=x(end)*lamref;
end

function d=complete_diagnostics(d,A,b,Aeq,beq,lb,ub,f,x)
d.A=A;d.b=b;d.Aeq=Aeq;d.beq=beq;d.lb=lb;d.ub=ub;d.f=f;
d.finite_matrices=all(isfinite(A(:)))&&all(isfinite(b))&& ...
    all(isfinite(Aeq(:)))&&all(isfinite(beq))&&all(isfinite(lb))&&all(isfinite(ub));
d.row_norm_A=sqrt(sum(A.^2,2));
d.row_norm_Aeq=sqrt(sum(Aeq.^2,2));
d.row_norm_ratio_A=max(d.row_norm_A)/max(min(d.row_norm_A),eps);
if isempty(Aeq)
    d.row_norm_ratio_Aeq=NaN;d.constraint_row_rank=rank(A);d.normalized_gram_rcond=rcond(normalize_rows(A)*normalize_rows(A)');
else
    d.row_norm_ratio_Aeq=max(d.row_norm_Aeq)/max(min(d.row_norm_Aeq),eps);
    C=[normalize_rows(A);normalize_rows(Aeq)];
    d.constraint_row_rank=rank(C,1e-10);
    d.normalized_gram_rcond=rcond(C*C');
end
d.bound_width_min=min(ub-lb);d.bound_width_max=max(ub-lb);
d.bound_width_ratio=d.bound_width_max/max(d.bound_width_min,eps);
d.b_abs_min=min(abs(b));d.b_abs_max=max(abs(b));
d.has_returned_point=~isempty(x);d.returned_point_finite=~isempty(x)&&all(isfinite(x));
if d.returned_point_finite
    ineq=A*x-b;eq=Aeq*x-beq;low=lb-x;high=x-ub;
    d.max_inequality_violation=max([0;ineq]);
    d.max_equality_residual=max([0;abs(eq)]);
    d.max_lower_bound_violation=max([0;low]);
    d.max_upper_bound_violation=max([0;high]);
    tol=1e-9*max(1,max(abs([lb;ub])));
    d.active_lower_count=sum(abs(x-lb)<=tol);d.active_upper_count=sum(abs(x-ub)<=tol);
    d.active_bound_fraction=(d.active_lower_count+d.active_upper_count)/numel(x);
    d.returned_objective=f'*x;
else
    d.max_inequality_violation=NaN;d.max_equality_residual=NaN;
    d.max_lower_bound_violation=NaN;d.max_upper_bound_violation=NaN;
    d.active_lower_count=NaN;d.active_upper_count=NaN;d.active_bound_fraction=NaN;
    d.returned_objective=NaN;
end
end

function C=normalize_rows(C)
if isempty(C),return;end
n=sqrt(sum(C.^2,2));C=C./max(n,eps);
end
