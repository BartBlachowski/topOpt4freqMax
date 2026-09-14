function diag_inner()
maxNumCompThreads(1);
cfg = defaultCfg();
mdl = model2D(cfg); NE = mdl.nele; rho = 0.5*ones(NE,1);
flt = prepFilter(cfg.nelx,cfg.nely,cfg.rminEl);
[K,M] = assemble2D(mdl,rho,cfg.p,cfg.massInterp);
[~,Phi,lam] = eigSolve(K,M,5,'eigs');
J = 2;
F  = genGrad(mdl,rho,cfg.p,cfg.massInterp,Phi,lam(1),1);
FJ = genGrad(mdl,rho,cfg.p,cfg.massInterp,Phi,lam(J),J);
F(:,1,1) = applyFilter(flt,rho,F(:,1,1));
fJJ = applyFilter(flt,rho,FJ(:,1,1));
ctx = struct('F',F,'fJJ',fJJ,'lam',lam(1),'lamJ',lam(J),'rho',rho, ...
   'rhomin',cfg.rhomin,'volfrac',cfg.volfrac,'move',cfg.move, ...
   'maxInner',80,'tolInner',1e-4,'offDiag',true);
[drho,st] = innerLoop(ctx);
fprintf('nInner=%d  conv=%d  beta^.5=%.3f\n',st.nInner,st.conv,sqrt(st.beta));
fprintf('dxHist (max |x_new - x_old| per MMA sub-iterate):\n');
h = st.dxHist;
for i=1:numel(h)
    fprintf('%3d: %10.3e', i, h(i));
    if mod(i,6)==0, fprintf('\n'); end
end
fprintf('\nfinal drho: min=%.4f max=%.4f  #at+move=%d #at-move=%d  sum=%.3e\n', ...
    min(drho),max(drho),sum(drho>cfg.move-1e-9),sum(drho<-cfg.move+1e-9),sum(drho));
end
