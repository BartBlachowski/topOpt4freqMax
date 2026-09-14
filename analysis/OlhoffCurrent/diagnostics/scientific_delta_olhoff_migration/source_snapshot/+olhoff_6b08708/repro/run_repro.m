function out = run_repro(label, preset, nelx, nely, varargin)
%RUN_REPRO  One Fig. 2a reproduction run, fully recorded.
%
%   out = RUN_REPRO(label, preset, nelx, nely, 'path.to.field', value, ...)
%
%   Resolves `preset` at the given mesh with the overrides, solves, and writes
%   into repro/results/<label>/:
%     res.mat            res (solver output) and cfg (effective configuration)
%     describe.txt       olh.config.describe(cfg)
%     summary.json       the numbers below
%     hist_vs_paper.png  iteration history stacked over the paper's Fig. 4a
%     topo_vs_paper.png  final topology stacked over the paper's Fig. 3a
%     topo.png           final topology alone
%
%   Every run is single-threaded (cfg.runtime.singleThread) and names its
%   preset and overrides in cfg.provenance.

root = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(root,'setpaths.m'));
outDir = fullfile(root,'repro','results',label);
if ~isfolder(outDir), mkdir(outDir); end

cfg = olh.config.resolve(preset, 'domain.mesh.nelx', nelx, 'domain.mesh.nely', nely, ...
                         'runtime.name', label, 'runtime.verbose', true, varargin{:});
txt = olh.config.describe(cfg);
fid = fopen(fullfile(outDir,'describe.txt'),'w'); fprintf(fid,'%s\n',txt); fclose(fid);
fprintf('%s\n', txt);

t0 = tic;
res = olhoffSolve(cfg);
wall = toc(t0);

h = res.hist;
w = res.omega;
gap12 = 100*(w(2)-w(1))/w(1);
conv  = any(contains(res.log,'converged at outer iteration'));
Mnd   = 4*mean(res.rho.*(1-res.rho));
grey  = mean(res.rho > 0.1 & res.rho < 0.9);
coal  = find((h.omega(2,:)-h.omega(1,:))./h.omega(1,:) < 0.02, 1);
if isempty(coal), coal = NaN; end
[w2pk, w2at] = max(h.omega(2,:));
[w3pk, w3at] = max(h.omega(3,:));

out = struct('label',label,'preset',preset,'mesh',[nelx nely], ...
    'status',res.status,'converged',conv,'nOuter',res.nOuter, ...
    'innerTotal',h.cumInner(end),'omega1',w(1),'omega2',w(2),'omega3',w(3), ...
    'gap12_pct',gap12,'Mnd',Mnd,'greyFraction',grey,'coalescenceIter',coal, ...
    'omega2_peak',w2pk,'omega2_peak_iter',w2at,'omega3_peak',w3pk,'omega3_peak_iter',w3at, ...
    'omega1_initial',h.omega(1,1),'final_dxNorm2',h.dxNorm2(end),'eps',cfg.stop.tolerance, ...
    'final_maxdrho',h.dxOuter(end),'move',h.move(end),'wall_s',wall, ...
    'tEig_s',sum(h.tEig),'tInner_s',sum(h.tInner),'MndHist',res.aux.Mnd, ...
    'overrides',{varargin},'log',{res.log});

fid = fopen(fullfile(outDir,'summary.json'),'w');
fprintf(fid,'%s\n', jsonencode(out,'PrettyPrint',true)); fclose(fid);
save(fullfile(outDir,'res.mat'),'res','cfg','out','-v7.3');

fprintf('\n=== %s  %dx%d  %s ===\n', label, nelx, nely, preset);
fprintf('status %s  conv %d  outer %d  inner %d  wall %.0f s\n', res.status, conv, res.nOuter, h.cumInner(end), wall);
fprintf('omega 1/2/3 = %.2f / %.2f / %.2f   gap12 %.2f%%   (paper 174.7 / 174.7 / 284.9)\n', w(1), w(2), w(3), gap12);
fprintf('M_nd %.3f  grey(0.1<rho<0.9) %.3f  coalescence@%s  w2 peak %.0f@%d  w3 peak %.0f@%d\n', ...
    Mnd, grey, num2str(coal), w2pk, w2at, w3pk, w3at);
for i = 1:numel(res.log), fprintf('LOG: %s\n', res.log{i}); end

% ---- figures -----------------------------------------------------------
xmax = 20*ceil(max(res.nOuter,80)/20);
paperHist = fullfile(root,'docs','figs','paper_fig4_hist.png');
paperTopo = fullfile(root,'docs','figs','paper_fig3a.png');
ttl = upper(sprintf('%s %dx%d  W1=%.1f W2=%.1f (PAPER 174.7) %s', label, nelx, nely, w(1), w(2), res.status));
compareHistoryTo(res, paperHist, fullfile(outDir,'hist_vs_paper.png'), ttl, xmax);
compareTopology(res.rho, nelx, nely, paperTopo, fullfile(outDir,'topo_vs_paper.png'), ...
    upper(sprintf('%s %dx%d  W1=%.1f W2=%.1f GAP=%.2f%% MND=%.3f', label, nelx, nely, w(1), w(2), gap12, Mnd)));
imwrite(topologyImage(res.rho, nelx, nely, 4), fullfile(outDir,'topo.png'));
end
