function S=audit_stationarity(tag)
%AUDIT_STATIONARITY  WP2 independent first-order stationarity certificate.
%   Rebuilds the model, spectrum and multiple-eigenvalue first-order model from
%   scratch, solves a feasible directional-ascent problem with its own solver,
%   and verifies the maximising direction by PHYSICAL fixed-t eigensolves.
%   It never reads the production convergence boolean.
repo='/Users/piotrek/Programming/topOpt4freqMax';
addpath(fullfile(repo,'Matlab','reproduction2007','runner'));
g=repro2007_paths(); %#ok<NASGU>
addpath(fullfile(repo,'analysis','OlhoffRegularized','audit','scripts'));
d=fullfile(repo,'analysis','OlhoffRegularized','audit','results',tag);
L=load(fullfile(d,'run.mat'));
m=L.meta;penal=3;massInterp='4b';rhomin=1e-3;volfrac=0.5;Jc=5;
tipf=0;if strcmpi(m.bcType,'cantilever'),tipf=0.20;end

mdl=audit_model(m.bcType,m.nelx,m.nely,volfrac,tipf);
flt=prepFilter(m.nelx,m.nely,m.rmin);
x=L.design(:);NE=mdl.nele;
rho=(flt.H*x)./flt.Hs;
volW=flt.H'*(ones(NE,1)./flt.Hs);

S=struct('tag',tag,'meta',m,'status',L.status);
S.rhoMatchesProduction=max(abs(rho-L.rho(:)));
S.volume=mean(rho);S.volResidual=sum(rho)-volfrac*NE;
S.grayness=4*mean(rho.*(1-rho));

[K,M]=audit_assemble(mdl,rho,penal,massInterp);
[w,Phi,lam]=eigSolve(K,M,Jc,'eigs');
% independent eigensolve cross-check with a different start vector / basis size
nfree=size(K,1);
v0b=cos((1:nfree)'*0.31830988618)+0.25;
optsb=struct('v0',v0b,'tol',1e-13,'maxit',20000,'p',min(nfree,max(40,8*Jc)));
[~,Db]=eigs(K,M,Jc,'smallestabs',optsb);
lamB=sort(real(diag(Db)),'ascend');
S.eigCrossCheck=max(abs(lamB(:)-lam(:))./abs(lam(:)));
S.omega=w(:).';S.omegaProduction=L.omega(:).';
S.omegaMatchesProduction=max(abs(w(1:3)-L.omega(1:3)));
S.gaps=(w(2:Jc)-w(1))./w(1);
S.Nfor=[localMult(w,0.001) localMult(w,0.01) localMult(w,0.05)];
S.NfromProduction=L.hist.N(end);
if isfield(L.hist,'certificateSlope')
    S.productionCertSlope=L.hist.certificateSlope(end);
    S.productionCertN=L.hist.certificateN(end);
    S.productionStatMeasure=L.hist.stationarityMeasure(end);
else
    S.productionCertSlope=NaN;S.productionCertN=NaN;S.productionStatMeasure=NaN;
end
S.productionPredSlope=L.hist.predictedSlope(end);
if isfield(L,'regcfg')&&isfield(L.regcfg,'convergenceStationarityTol')
    S.convergenceTol=L.regcfg.convergenceStationarityTol;
else
    S.convergenceTol=L.regcfg.stationarityTol;
end
S.certRadiusProduction=NaN;
if isfield(L.regcfg,'certRadius'),S.certRadiusProduction=L.regcfg.certRadius;end

lamFinalTrust=L.hist.trustNext(end);
radii=[1 5e-3 max(lamFinalTrust,1e-9)];
lo=rhomin-x;hi=1-x;volSlack=max(volfrac*NE-sum(rho),0);

Ns=unique([1 2 3 S.Nfor(3)]);Ns=Ns(Ns>=1&Ns<=4);
S.cert=struct([]);c=0;
for N=Ns
    idx=1:N;
    F=genGrad(mdl,rho,penal,massInterp,Phi,mean(lam(idx)),idx);
    G=zeros(NE,N,N);
    for s=1:N,for k=s:N
        v=flt.H'*(F(:,s,k)./flt.Hs);G(:,s,k)=v;G(:,k,s)=v;
    end,end
    for r=radii
        a=audit_ascent(G,volW,lo,hi,volSlack,r);
        b=audit_ascent_lp(G,volW,lo,hi,volSlack,r);
        c=c+1;
        S.cert(c).N=N;S.cert(c).radius=r;S.cert(c).rate=a.rate;
        S.cert(c).masterBound=a.masterBound;S.cert(c).cuts=a.iters;
        S.cert(c).normRate=a.rate/(max(abs(lam(1)),1)*r);   % comparable with predSlope
        S.cert(c).relGain=a.rate/max(abs(lam(1)),1);        % best relative dlambda for ONE step of size r
        S.cert(c).dInf=a.dInf;S.cert(c).volUse=a.volUse;S.cert(c).flag=a.flag;
        S.cert(c).rateLpRestricted=b.rate;
        S.cert(c).normRateLp=b.rate/(max(abs(lam(1)),1)*r);
        A0=zeros(N);for s=1:N,for k=s:N,q=G(:,s,k).'*a.d;A0(s,k)=q;A0(k,s)=q;end,end
        S.cert(c).Ad=A0;
        S.cert(c).offDiagOfOptimalDir=max(max(abs(A0-diag(diag(A0)))));
        if r==1,S.dir{N}=a.d;S.rateFull(N)=a.rate;end
    end
end

% ---- PHYSICAL fixed-t verification along each full-box ascent direction -----
ts=[1e-1 3e-2 1e-2 3e-3 1e-3 1e-4 1e-5];
S.phys=struct([]);q=0;
for N=Ns
    dd=S.dir{N};pr=S.rateFull(N);
    for t=ts
        xt=min(1,max(rhomin,x+t*dd));
        rt=(flt.H*xt)./flt.Hs;
        [Kt,Mt]=audit_assemble(mdl,rt,penal,massInterp);
        [wt,~,lt]=eigSolve(Kt,Mt,Jc,'eigs');
        q=q+1;
        S.phys(q).N=N;S.phys(q).t=t;
        S.phys(q).predDlam=t*pr;
        S.phys(q).actDlam=lt(1)-lam(1);
        S.phys(q).actDomega=wt(1)-w(1);
        S.phys(q).ratio=(lt(1)-lam(1))/max(t*pr,realmin);
        S.phys(q).volume=mean(rt);
        S.phys(q).volFeasible=(sum(rt)-volfrac*NE)<=1e-9*volfrac*NE;
        S.phys(q).dlamAll=(lt(1:min(4,Jc))-lam(1:min(4,Jc))).';
        S.phys(q).Nafter=localMult(wt,0.05);
    end
end
% self-consistent cluster on the audit side, by the same fixed point the
% production certificate now uses: the smallest N whose predicted gain at the
% reference radius does not reach the first EXCLUDED eigenvalue.
S.selfConsistentN=NaN;
for N=Ns
    k=find([S.cert.N]==N & abs([S.cert.radius]-5e-3)<1e-12,1);
    if isempty(k),continue,end
    if N+1>numel(lam),S.selfConsistentN=N;break,end
    if (lam(N+1)-lam(1))>=S.cert(k).rate,S.selfConsistentN=N;break,end
end
if isnan(S.selfConsistentN),S.selfConsistentN=max(Ns);end
k=find([S.cert.N]==S.selfConsistentN & abs([S.cert.radius]-5e-3)<1e-12,1);
S.selfConsistentAscent=S.cert(k).relGain;
S.selfConsistentSlope=S.cert(k).normRate;
save(fullfile(d,'stationarity.mat'),'S','-v7.3');
localPrint(S);
end

function N=localMult(w,tol)
N=1;while N<numel(w)&&abs(w(N+1)-w(1))/w(1)<tol,N=N+1;end
end

function localPrint(S)
fprintf('\n===== WP2 INDEPENDENT STATIONARITY AUDIT : %s =====\n',S.tag);
fprintf('production status      : %s\n',S.status);
fprintf('rho rebuild max|diff|  : %.3e   (independent model vs production rho)\n',S.rhoMatchesProduction);
fprintf('omega rebuild max|diff|: %.3e\n',S.omegaMatchesProduction);
fprintf('eigs cross-check relerr: %.3e\n',S.eigCrossCheck);
fprintf('omega(1:5)             : %s\n',mat2str(S.omega,10));
fprintf('gaps (w_j-w_1)/w_1     : %s\n',mat2str(S.gaps(:).',4));
fprintf('N at tol .001/.01/.05  : %d / %d / %d   (production reported N=%d)\n',S.Nfor,S.NfromProduction);
fprintf('mean(rho)=%.10f  volResidual=%.3e  grayness=%.4f\n',S.volume,S.volResidual,S.grayness);
fprintf(['production certificate: slope=%.6e (N=%g, radius=%g), route slope=%.6e, ' ...
    'statMeasure=%.6e, convergence tol=%.3e\n'],S.productionCertSlope,S.productionCertN, ...
    S.certRadiusProduction,S.productionPredSlope,S.productionStatMeasure,S.convergenceTol);
fprintf('\n-- maximum feasible first-order ascent rate of lambda_min --\n');
fprintf('%3s %10s %16s %14s %12s %14s %12s %5s\n','N','radius','rate dlam/dt', ...
  'rate/(lam*r)','relgain@r','LPrestr rate','LP/(lam*r)','cuts');
for i=1:numel(S.cert)
    c=S.cert(i);
    fprintf('%3d %10.3e %16.8e %14.6e %12.4e %14.6e %12.4e %5d\n', ...
      c.N,c.radius,c.rate,c.normRate,c.relGain,c.rateLpRestricted,c.normRateLp,c.cuts);
end
fprintf('  (rate/(lam*r) is exactly the quantity the production optimizer reports as predSlope)\n');
for i=1:numel(S.cert)
    c=S.cert(i);
    if c.radius==1
        fprintf('  N=%d optimal-direction A(d) = %s ; max|offdiag| = %.4e\n', ...
            c.N,mat2str(c.Ad,6),c.offDiagOfOptimalDir);
    end
end
fprintf(['\nself-consistent cluster (fixed point: gain <= gap to first excluded mode): ' ...
    'N = %d, slope = %.6e, max feasible ascent at r=5e-3 = %.4e\n'], ...
    S.selfConsistentN,S.selfConsistentSlope,S.selfConsistentAscent);
fprintf('\n-- physical fixed-t verification (ordered spectrum) --\n');
fprintf('%3s %9s %14s %14s %12s %9s %12s %3s\n','N','t','pred dlam','act dlam1','act domega1','act/pred','mean(rho)','N''');
for i=1:numel(S.phys)
    p=S.phys(i);
    fprintf('%3d %9.1e %14.6e %14.6e %12.5e %9.4f %12.9f %3d\n', ...
      p.N,p.t,p.predDlam,p.actDlam,p.actDomega,p.ratio,p.volume,p.Nafter);
end
fprintf('===== END %s =====\n\n',S.tag);
end
