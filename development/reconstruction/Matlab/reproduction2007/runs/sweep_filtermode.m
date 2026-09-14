function sweep_filtermode()
%SWEEP_FILTERMODE  CLAUDE.md sec.5: the filter takes ONE sensitivity vector but
%   the multiple case has N(N+1)/2 vectors f_sk.  Filtering only the diagonal
%   f_jj versus all of them are different algorithms; which reproduces Fig. 4a
%   is a result.  Distinguishable only once N >= 2, which now happens.
fprintf('%-8s %-10s %8s %8s %8s %8s %7s %7s\n', ...
        'route','filterMode','omega1','omega2','omega3','gap','Mnd','bimod%');
for route = {'lp','mma'}
  for fm = {'diag','all','none'}
    cfg = defaultCfg();
    cfg.rminEl=1.2; cfg.move=0.005; cfg.tolMult=0.05; cfg.maxOuter=1600;
    cfg.verbose=false; cfg.filterMode=fm{1};
    if strcmp(route{1},'lp')
        cfg.innerSolver='lp'; cfg.offDiag=false;
    else
        cfg.innerSolver='mma'; cfg.offDiag=true; cfg.maxOuter=400; cfg.move=0.01;
    end
    res = olhoffOpt(cfg);
    h=res.hist; n=res.nOuter; last=max(1,n-49):n;
    fprintf('%-8s %-10s %8.2f %8.2f %8.2f %8.4f %7.3f %7.0f\n', route{1}, fm{1}, ...
        res.omega(1),res.omega(2),res.omega(3), ...
        (res.omega(2)-res.omega(1))/res.omega(1), ...
        sum(4*res.rho.*(1-res.rho))/numel(res.rho), 100*mean(h.N(last)>=2));
    save(sprintf('results/fm_%s_%s.mat',route{1},fm{1}),'res','-v7.3');
  end
end
end
