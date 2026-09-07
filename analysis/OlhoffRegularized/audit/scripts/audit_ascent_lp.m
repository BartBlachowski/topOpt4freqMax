function out=audit_ascent_lp(G,volW,lo,hi,volSlack,radius)
%AUDIT_ASCENT_LP  The Eq.(22) Krog&Olhoff RESTRICTED ascent rate:
%   max_d min_j f_jj' d   s.t.  f_sk' d = 0 (s<k), volume, box.
%   Provided only as a control against AUDIT_ASCENT, whose feasible set is
%   the true one.  rate_lp <= rate_true always.
[NE,N,~]=size(G);
l=min(max(lo(:),-radius),0);u=max(min(hi(:),radius),0);
nvar=NE+1;f=zeros(nvar,1);f(end)=-1;
A=zeros(N+1,nvar);b=zeros(N+1,1);
for j=1:N,A(j,1:NE)=-G(:,j,j).';A(j,nvar)=1;end
A(N+1,1:NE)=volW(:).';b(N+1)=volSlack;
np=N*(N-1)/2;Aeq=zeros(np,nvar);beq=zeros(np,1);r=0;
for s=1:N,for k=s+1:N,r=r+1;Aeq(r,1:NE)=G(:,s,k).';end,end
o=optimoptions('linprog','Display','none','Algorithm','dual-simplex-highs');
[z,fv,flag]=linprog(f,A,b,Aeq,beq,[l;-inf],[u;inf],o);
if flag~=1||isempty(z),out=struct('rate',NaN,'flag',flag,'d',zeros(NE,1),'offDiagResid',NaN);return,end
d=z(1:NE);rt=-fv;
od=0;for s=1:N,for k=s+1:N,od=max(od,abs(G(:,s,k).'*d));end,end
out=struct('rate',rt,'flag',flag,'d',d,'offDiagResid',od,'radius',radius);
end
