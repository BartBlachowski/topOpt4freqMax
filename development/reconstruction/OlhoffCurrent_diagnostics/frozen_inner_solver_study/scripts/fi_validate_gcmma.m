function fi_validate_gcmma()
here=fileparts(mfilename('fullpath'));E=fullfile(fileparts(here),'evaluations');
up=fullfile(here,'upstream','GCMMA-MMA-code-1.5');addpath(up,'-begin');maxNumCompThreads(1);
gctoyinit;maxoutit=9;gctoymain;
o=struct('official_toy_x',xval.','official_toy_f0',f0val,'official_toy_kkt',kktnorm);
assert(max(abs(xval-[2.0175;1.7800;1.2375]))<=5e-5);
assert(abs(f0val-8.7702)<=5e-5);
maxoutit=100;kkttol=1e-6;gctoymain;
o.official_toy_converged_kkt=kktnorm;assert(kktnorm<1e-6);
% Analytic convex quadratic: optimum [.5;.5], objective .5, multiplier 1.
n=2;m=1;x=[1;1];xold1=x;xold2=x;xmin=zeros(n,1);xmax=2*ones(n,1);low=xmin;upp=xmax;
raa0=1e-2;raa=1e-2;re=1e-6;epsi=1e-9;corrections=0;
for it=1:60
 f0=x.'*x;df0=2*x;f=1-sum(x);J=-ones(1,2);
 [low,upp,raa0,raa]=asymp(it,n,x,xold1,xold2,xmin,xmax,low,upp,raa0,raa,re,re,df0,J);
 for j=0:16
  [xm,ym,zm,la,xi,et,~,~,~,f0a,fa]=gcmmasub(m,n,it,epsi,x,xmin,xmax,low,upp,raa0,raa,f0,df0,f,J,1,0,1000,0);
  f0n=xm.'*xm;fn=1-sum(xm);c=concheck(m,epsi,f0a,f0n,fa,fn);
  if c,break,end
  assert(j<16);
  [raa0,raa]=raaupdate(xm,x,xmin,xmax,low,upp,f0n,fn,f0a,fa,raa0,raa,re,re,epsi);corrections=corrections+1;
 end
 xold2=xold1;xold1=x;x=xm;
end
o.quadratic=struct('x',x.','f0',x.'*x,'lambda',la,'constraint',1-sum(x),'stationarity',norm(2*x-la*ones(2,1)-xi+et,inf),'corrections',corrections);
assert(norm(x-[.5;.5],inf)<1e-6 && abs(x.'*x-.5)<1e-6 && o.quadratic.stationarity<1e-6);
o.verdict='GCMMA_IMPLEMENTATION_VALIDATED';o.source='official smoptit.se GCMMA-MMA-code-1.5, unmodified';
fi_json(fullfile(E,'gcmma_validation.json'),o);disp(o);
end
