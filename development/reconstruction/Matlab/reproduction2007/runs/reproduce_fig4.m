function reproduce_fig4()
%REPRODUCE_FIG4  The paper's Fig. 4a converges in ~80 outer iterations with the
%   omega_1/omega_2 coalescence at ~20.  The move limit -- unstated -- sets that
%   pace, so sweep it and report which value reproduces the trajectory.
combos = {0.05, 0.03, 0.02, 0.01};
fprintf('%-7s %8s %8s %8s %8s %8s %8s\n', ...
        'move','coal@','omega1','omega2','omega3','gap','w2 peak');
for i=1:numel(combos)
    mv = combos{i};
    cfg = defaultCfg();
    cfg.nelx=240; cfg.nely=30; cfg.rminEl=1.3; cfg.rminPhys=[];
    cfg.move=mv; cfg.tolMult=0.05; cfg.maxOuter=100;
    cfg.verbose=false; cfg.innerSolver='lp'; cfg.offDiag=false;
    res = olhoffOpt(cfg);
    h=res.hist; g=(h.omega(2,:)-h.omega(1,:))./h.omega(1,:);
    c = find(g<0.02,1);
    if isempty(c), c = NaN; end
    [pk,pki] = max(h.omega(2,:));
    fprintf('%-7.3f %8s %8.2f %8.2f %8.2f %8.4f %6.0f@%d\n', mv, ...
        num2str(c), res.omega(1),res.omega(2),res.omega(3), ...
        (res.omega(2)-res.omega(1))/res.omega(1), pk, pki);
    save(sprintf('results/fig4_mv%g.mat',mv),'res','-v7.3');
    compareHistory(res, sprintf('results/fig4_mv%g_vs_paper.png',mv), ...
        sprintf('REPRO 240X30 RMIN=1.3 MOVE=%.3f W1=%.1f W2=%.1f W3=%.1f', ...
                mv,res.omega(1),res.omega(2),res.omega(3)), 100);
end
fprintf('\nPaper: coalescence at ~20, omega2 peaks ~325 at iter 7, omega3 peaks ~527 at iter 9,\n');
fprintf('       converged by ~60-80 to omega1=omega2=174.7, omega3=284.9\n');
end
