function test_stabilized_mirror()
%TEST_STABILIZED_MIRROR Prove S0 reproduces the frozen trajectory prefix.
here=fileparts(mfilename('fullpath'));repo=fileparts(fileparts(here));
tmp=fullfile(here,'raw'); out=run_stabilization_case('S0',240,30,tmp,20);
S=load(out,'res');B=load(fullfile(repo,'analysis','olhoff_native_convergence','results','development_240x30.mat'),'res');
r=S.res;b=B.res;n=20;
assert(isequaln(r.hist.omega,b.hist.omega(:,1:n)));
assert(isequaln(r.hist.N,b.hist.N(1:n)));
assert(isequaln(r.hist.beta,b.hist.beta(1:n)));
assert(isequaln(r.hist.nInner,b.hist.nInner(1:n)));
assert(isequaln(r.hist.dxOuter,b.hist.dxOuter(1:n)));
assert(isequaln(r.hist.vol,b.hist.vol(1:n)));
assert(isequaln(r.rho_snapshots,double_to_single(b.telemetry.rho_snapshots(:,1:n+1))));
fprintf('STABILIZED_MIRROR_IDENTITY_OK n=%d\n',n);
end
function x=double_to_single(x),x=single(x);end
