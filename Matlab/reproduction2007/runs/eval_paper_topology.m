function eval_paper_topology(meshes)
%EVAL_PAPER_TOPOLOGY  Evaluate the paper's OWN printed Fig. 3a topology in our
%   FE model.  This separates two possible causes of an omega_1 shortfall:
%     - if the printed design gives ~174.7 here, our FE model is right and the
%       optimizer is underperforming;
%     - if it gives ~150, the FE model/idealization differs from the authors'.
maxNumCompThreads(1);
if nargin<1, meshes = {[160 20],[240 30]}; end
cfg0 = defaultCfg();
for mi = 1:numel(meshes)
    nelx = meshes{mi}(1); nely = meshes{mi}(2);
    rg = digitize_fig3('docs/figs/paper_fig3a.png', nelx, nely, true);
    cfg = cfg0; cfg.nelx = nelx; cfg.nely = nely;
    mdl = model2D(cfg);
    for mode = {'grey','binary-at-printed-vf','binary-at-50pct'}
        switch mode{1}
            case 'grey'
                r = max(cfg.rhomin, min(1, rg));
            case 'binary-at-printed-vf'
                r = cfg.rhomin*ones(size(rg)); r(rg>0.5) = 1;
            case 'binary-at-50pct'
                n1 = round(0.5*numel(rg));
                [~,ord] = sort(rg,'descend');
                r = cfg.rhomin*ones(size(rg)); r(ord(1:n1)) = 1;
        end
        [K,M] = assemble2D(mdl, r, cfg.p, cfg.massInterp);
        [w,Phi] = eigSolve(K,M,4,'eigs');
        T = classifyModes(mdl,M,Phi,w);
        fprintf('%dx%d  %-22s vf=%.3f  w = %7.2f %7.2f %7.2f %7.2f   (w2-w1)/w1=%.3f\n', ...
                nelx,nely,mode{1},mean(r),w(1),w(2),w(3),w(4),(w(2)-w(1))/w(1));
    end
    % also save what we digitized, for visual confirmation
    imwrite(topologyImage(min(max(rg,0),1),nelx,nely,8), ...
            sprintf('results/digitized_fig3a_%dx%d.png',nelx,nely));
end
fprintf('\nPaper reports omega1 = omega2 = 174.7 (bimodal), omega3 = 284.9\n');
end
