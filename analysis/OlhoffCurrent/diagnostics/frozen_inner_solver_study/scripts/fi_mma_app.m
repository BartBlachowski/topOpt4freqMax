function [f0app,fapp]=fi_mma_app(x,xm,xmin,xmax,low,upp,f0,df0,fval,dfdx)
n=numel(x);m=numel(fval);w=max(xmax-xmin,1e-5*ones(n,1));
u=upp-x;l=x-low;
p0=max(df0,0);q0=max(-df0,0);reg=.001*(p0+q0)+1e-5./w;
p0=(p0+reg).*u.^2;q0=(q0+reg).*l.^2;
P=max(dfdx,0);Q=max(-dfdx,0);reg=.001*(P+Q)+1e-5*ones(m,1)*(1./w).';
P=(P+reg).*u.'.^2;Q=(Q+reg).*l.'.^2;
du=1./(upp-xm)-1./u;dl=1./(xm-low)-1./l;
f0app=f0+p0.'*du+q0.'*dl;fapp=fval+P*du+Q*dl;
end
