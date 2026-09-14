function report_run(matfile, paperPng, tag)
%REPORT_RUN  Summarise one saved run: frequencies, multiplicity, iteration
%   counts, AND a stacked topology comparison against the paper's figure.
S = load(matfile);
res = S.res; cfg = res.cfg; h = res.hist;

fprintf('\n======== %s ========\n', matfile);
fprintf('mesh %dx%d  bc=%s  support=%s/%s  elem=%s  massInterp=%s\n', ...
        cfg.nelx,cfg.nely,cfg.bc,cfg.support,cfg.axial,cfg.elemType,cfg.massInterp);
fprintf('SWEPT: rmin=%.2f el  tolMult=%.4f  move=%.4f  offDiag=%d  filter=%s\n', ...
        cfg.rminEl,cfg.tolMult,cfg.move,cfg.offDiag,cfg.filterMode);
fprintf('       maxInner=%d tolInner=%.1e  maxOuter=%d tolOuter=%.1e  solver=%s threads=%d\n', ...
        cfg.maxInner,cfg.tolInner,cfg.maxOuter,cfg.tolOuter,cfg.solver,cfg.threads);
if isfield(h,'cumInner') && ~isempty(h.cumInner)
    fprintf('outer iterations: %d   total inner sub-iterates: %d   inner converged: %d/%d\n', ...
            res.nOuter, h.cumInner(end), sum(h.innerConv), numel(h.innerConv));
else
    fprintf('outer iterations: %d   total inner sub-iterates: %d   (run predates conv logging)\n', ...
            res.nOuter, sum(h.nInner));
end
fprintf('inner sub-iterates per outer: mean %.1f  max %d  (cap maxInner=%d)\n', ...
        mean(h.nInner), max(h.nInner), cfg.maxInner);
fprintf('wallclock %.1f s   (eig %.1f%%  grad %.1f%%  inner %.1f%%)\n', ...
        res.wallclock, 100*sum(h.tEig)/res.wallclock, ...
        100*sum(h.tGrad)/res.wallclock, 100*sum(h.tInner)/res.wallclock);
fprintf('final omega: '); fprintf('%8.2f',res.omega(1:min(4,end))); fprintf('\n');
fprintf('mode Ex frac:'); fprintf('%8.2f',res.modeTable(1:min(4,end),2)); fprintf('\n');
last = max(1,res.nOuter-19):res.nOuter;
fprintf('multiplicity N over last 20 outer iters: min=%d max=%d  bimodal %.0f%% of the time\n', ...
        min(h.N(last)),max(h.N(last)),100*mean(h.N(last)>=2));
rg = abs(res.omega(2)-res.omega(1))/res.omega(1);
fprintf('final relative gap (w2-w1)/w1 = %.4f  -> %s\n', rg, ...
        ternary(rg < cfg.tolMult,'BIMODAL','SIMPLE'));
fprintf('volume fraction: %.4f (target %.4f)\n', mean(res.rho), cfg.volfrac);
gr = sum(4*res.rho.*(1-res.rho))/numel(res.rho);   % standard Mnd measure
fprintf('greyness (0=pure 0/1, 1=all grey): %.3f\n', gr);
for i=1:numel(res.log), fprintf('LOG: %s\n',res.log{i}); end

if nargin>=2 && ~isempty(paperPng)
    if nargin<3, [~,tag]=fileparts(matfile); end
    out = fullfile('results',[tag '_vs_paper.png']);
    lbl = sprintf('REPRO %dX%d W1=%.1f MOVE=%.3f RMIN=%.1f', ...
                  cfg.nelx,cfg.nely,res.omega(1),cfg.move,cfg.rminEl);
    compareTopology(res.rho,cfg.nelx,cfg.nely,paperPng,out,lbl, ...
                    'PAPER FIG 3A W1=174.7');
end
end

function o = ternary(c,a,b)
if c, o=a; else, o=b; end
end
