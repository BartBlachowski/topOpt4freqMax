function sweep_lp_rmin240()
%SWEEP_LP_RMIN240  Filter radius at the conclusive mesh, 240x30.
rs = [1.1 1.3 1.5 1.8 2.2];
fprintf('%-7s %8s %8s %8s %8s %7s %7s %6s\n','rminEl','omega1','omega2','omega3','gap','Mnd','bimod%','secs');
for r = rs
    cfg = defaultCfg();
    cfg.nelx=240; cfg.nely=30; cfg.rminEl=r; cfg.rminPhys=[];
    cfg.move=0.005; cfg.tolMult=0.05; cfg.maxOuter=1600;
    cfg.verbose=false; cfg.innerSolver='lp'; cfg.offDiag=false;
    res = olhoffOpt(cfg);
    h=res.hist; n=res.nOuter; last=max(1,n-49):n;
    gap=(res.omega(2)-res.omega(1))/res.omega(1);
    Mnd=sum(4*res.rho.*(1-res.rho))/numel(res.rho);
    fprintf('%-7.2f %8.2f %8.2f %8.2f %8.4f %7.3f %7.0f %6.1f\n', ...
        r,res.omega(1),res.omega(2),res.omega(3),gap,Mnd,100*mean(h.N(last)>=2),res.wallclock);
    save(sprintf('results/lp240_rmin%g.mat',r),'res','-v7.3');
end
fprintf('\nTarget: omega1 = omega2 = 174.7 (bimodal), omega3 = 284.9\n');
end
