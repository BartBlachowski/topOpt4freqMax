function mdl=audit_model(bcType,nelx,nely,volfrac,tipMassFraction)
%AUDIT_MODEL  Independent rebuild of the OlhoffRegularized FE model.
%   Written from the documented problem definition, NOT by calling the
%   production localModel (which is a private nested function).  It is
%   cross-checked against the production omega in audit_stationarity.
if nargin<4||isempty(volfrac),volfrac=0.5;end
if nargin<5||isempty(tipMassFraction),tipMassFraction=0;end
E=1e7;nu=0.3;rhom=1;t=1;
switch lower(bcType)
    case {'simply','ss'},        a=8;b=1;
    case {'fixedpinned','cs'},   a=8;b=1;
    case {'cantilever','cf'},    a=15;b=10;
    otherwise,error('audit_model:bc','unknown %s',bcType);
end
dx=a/nelx;dy=b/nely;
nele=nelx*nely;nnode=(nelx+1)*(nely+1);ndof=2*nnode;
nodenrs=reshape(1:nnode,1+nely,1+nelx);
edofVec=reshape(2*nodenrs(1:end-1,1:end-1)+1,nele,1);
edofMat=repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nele,1);
iK=reshape(kron(edofMat,ones(8,1))',64*nele,1);
jK=reshape(kron(edofMat,ones(1,8))',64*nele,1);
[K0,M0]=elemMats2D(dx,dy,E,nu,rhom,t,'Q4','consistent');
rowMid=round(nely/2)+1;leftCol=1;rightCol=nelx+1;
dofsOf=@(nd,c)2*nd(:)-2+c;
clampFace=@(col)reshape([dofsOf(nodenrs(:,col),1);dofsOf(nodenrs(:,col),2)],[],1);
switch lower(bcType)
    case {'simply','ss'}
        L=nodenrs(rowMid,leftCol);R=nodenrs(rowMid,rightCol);
        fixed=[dofsOf(L,1);dofsOf(L,2);dofsOf(R,1);dofsOf(R,2)];
    case {'fixedpinned','cs'}
        R=nodenrs(rowMid,rightCol);
        fixed=[clampFace(leftCol);dofsOf(R,1);dofsOf(R,2)];
    case {'cantilever','cf'}
        fixed=clampFace(leftCol);
end
fixed=unique(fixed(:));free=setdiff((1:ndof)',fixed);
tipDofs=zeros(0,1);tipVal=0;
if tipMassFraction>0
    tn=nodenrs(rowMid,rightCol);tipDofs=[dofsOf(tn,1);dofsOf(tn,2)];
    tipVal=tipMassFraction*volfrac*a*b*t*rhom;
end
mdl=struct('nelx',nelx,'nely',nely,'dx',dx,'dy',dy,'nele',nele,'nnode',nnode, ...
  'ndof',ndof,'nodenrs',nodenrs,'edofMat',edofMat,'iK',iK,'jK',jK,'K0',K0,'M0',M0, ...
  'free',free,'fixed',fixed,'a',a,'b',b,'t',t,'E',E,'nu',nu,'rhom',rhom, ...
  'tipMassDofs',tipDofs,'tipMassValue',tipVal,'volfrac',volfrac);
end
