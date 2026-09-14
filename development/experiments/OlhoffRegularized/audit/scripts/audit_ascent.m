function out=audit_ascent(G,volW,lo,hi,volSlack,radius)
%AUDIT_ASCENT  Maximum feasible first-order ascent rate of lambda_min.
%
%   G      NE x N x N  design-space generalized gradients f_sk (already mapped
%          through the filter chain rule).  The directional derivative of the
%          N-fold eigenvalue cluster along d is the spectrum of
%          A(d)_sk = f_sk' d ; the ascent rate of the MINIMUM of the cluster is
%          lambda_min(A(d)) = min_{|v|=1} v' A(d) v .
%
%   Solves      max_d  min_{|v|=1}  sum_sk v_s v_k (f_sk' d)
%               s.t.   volW' d <= volSlack
%                      max(lo,-radius) <= d <= min(hi,radius)
%
%   by Kelley cutting planes on v with an LP master.  For N = 1 the first LP
%   is already exact.  This is deliberately independent of deltaLambda / MMA /
%   the production convergence test: it only uses linprog and eig on an N x N.
if nargin<6||isempty(radius),radius=1;end
[NE,N,~]=size(G);
l=max(lo(:),-radius);u=min(hi(:),radius);
l=min(l,0);u=max(u,0);                 % d = 0 must stay feasible
nvar=NE+1;
f=zeros(nvar,1);f(end)=-1;             % maximise t
lb=[l;-inf];ub=[u;inf];
Aineq=[volW(:).' 0];bineq=volSlack;
V=eye(N);                              % initial cuts: the coordinate directions
opts=optimoptions('linprog','Display','none','Algorithm','dual-simplex-highs');
d=zeros(NE,1);tval=0;flag=1;hist=zeros(0,3);
for it=1:60
    nc=size(V,2);
    Ac=zeros(nc,nvar);
    for c=1:nc
        v=V(:,c);g=zeros(NE,1);
        for s=1:N,for k=1:N
            w=v(s)*v(k);if w~=0,g=g+w*G(:,s,k);end
        end,end
        Ac(c,1:NE)=-g.';Ac(c,nvar)=1;   % t - g'd <= 0
    end
    [z,fv,flag]=linprog(f,[Aineq;Ac],[bineq;zeros(nc,1)],[],[],lb,ub,opts);
    if flag~=1||isempty(z),break,end
    d=z(1:NE);tval=-fv;
    A=zeros(N);
    for s=1:N,for k=s:N,a=G(:,s,k).'*d;A(s,k)=a;A(k,s)=a;end,end
    A=(A+A')/2;
    [Vv,Dd]=eig(A);[lmin,ix]=min(real(diag(Dd)));v=Vv(:,ix);v=v/norm(v);
    hist(end+1,:)=[it tval lmin]; %#ok<AGROW>
    if lmin>=tval-1e-10*max(1,abs(tval)),break,end
    V=[V v]; %#ok<AGROW>
end
% the certified ascent rate is the TRUE lambda_min along the returned d
A=zeros(N);
for s=1:N,for k=s:N,a=G(:,s,k).'*d;A(s,k)=a;A(k,s)=a;end,end
A=(A+A')/2;
out=struct('rate',min(real(eig((A+A')/2))),'masterBound',tval,'d',d, ...
  'flag',flag,'iters',size(hist,1),'hist',hist,'radius',radius, ...
  'volUse',volW(:).'*d,'dInf',max(abs(d)),'dRms',sqrt(mean(d.^2)));
end
