function test_threshold(matfiles)
%TEST_THRESHOLD  Is the omega_1 shortfall caused by grey material?
%   Project each saved design onto a 0/1 field at the SAME volume fraction and
%   re-analyse.  If the projected design jumps to ~174.7 then greyness is the
%   whole story; if it does not, the topology itself is the limitation.
maxNumCompThreads(1);
fprintf('%-28s %8s %8s %8s %8s %8s %8s\n', ...
        'run','Mnd','w1 grey','w2 grey','w1 0/1','w2 0/1','vol 0/1');
for i = 1:numel(matfiles)
    S = load(matfiles{i}); res = S.res; cfg = res.cfg;
    mdl = model2D(cfg);
    rho = res.rho;
    Mnd = sum(4*rho.*(1-rho))/numel(rho);
    [K,M] = assemble2D(mdl,rho,cfg.p,cfg.massInterp);
    wg = eigSolve(K,M,3,'eigs');
    % volume-preserving 0/1 projection
    nSolid = round(cfg.volfrac*numel(rho));
    [~,ord] = sort(rho,'descend');
    rb = cfg.rhomin*ones(size(rho));
    rb(ord(1:nSolid)) = 1;
    [K2,M2] = assemble2D(mdl,rb,cfg.p,cfg.massInterp);
    wb = eigSolve(K2,M2,3,'eigs');
    [~,nm] = fileparts(matfiles{i});
    fprintf('%-28s %8.3f %8.2f %8.2f %8.2f %8.2f %8.3f\n', ...
            nm, Mnd, wg(1), wg(2), wb(1), wb(2), mean(rb));
end
fprintf('\nPaper optimum: omega1 = omega2 = 174.7 (bimodal)\n');
end
