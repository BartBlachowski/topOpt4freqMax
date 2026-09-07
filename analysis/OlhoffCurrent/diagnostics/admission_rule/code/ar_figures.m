function files = ar_figures(R, figDir)
%AR_FIGURES  The eight preregistered figures.  Descents and admission points
%   are marked on every per-iteration panel.  Semantic labels only.
if ~isfolder(figDir); mkdir(figDir); end
files = {};
cands = ar_candidates();
CLR = struct('prod',[0.20 0.35 0.75], 'c1',[0.85 0.35 0.10], 'c2',[0.15 0.55 0.25]);
meshes = cell2mat(arrayfun(@(r) r.mesh(:).', R, 'UniformOutput', false)');

% Stop iterations per mesh
info = struct('mesh',{},'kProd',{},'k1',{},'k2',{});
for i = 1:numel(R)
    P = R(i).per;
    kP = find((P.l2 < R(i).prodTol) & (P.itersSinceMoveChange>=1), 1);
    E1 = ar_predicate(P, cands(1)); E2 = ar_predicate(P, cands(2));
    info(i) = struct('mesh',R(i).mesh,'kProd',kP,'k1',E1.stopIter,'k2',E2.stopIter);
end

% ---- 1. topology ------------------------------------------------------
f = figure('Position',[60 60 1500 620],'Color','w','Visible','off');
t = tiledlayout(numel(R), 2, 'TileSpacing','tight','Padding','compact');
for i = 1:numel(R)
    S = load(fullfile(fileparts(figDir),'runs', ...
        sprintf('unstopped_%dx%d.mat', R(i).mesh(1), R(i).mesh(2))), 'RHO');
    for which = {'kProd','k1'}
        k = info(i).(which{1});
        ax = nexttile; imagesc(reshape(1-S.RHO(:,k), R(i).mesh(2), R(i).mesh(1)));
        colormap(gray); clim([0 1]); axis image; set(ax,'XTick',[],'YTick',[]);
        P = R(i).per;
        if strcmp(which{1},'kProd'); nm = 'Production admission (L2 + settledMove)';
        else; nm = 'Candidate C1 (move-settled + local + objective)'; end
        title(ax, {sprintf('%s  -  %dx%d  stop at iter %d', nm, R(i).mesh(1), R(i).mesh(2), k), ...
                   sprintf('M_nd = %.2f%%   omega_1 = %.5g   max|drho|/move = %.3f', ...
                       P.Mnd(k), P.omega1(k), P.ratio(k))}, ...
            'FontSize',10,'Interpreter','none');
    end
end
title(t,'Final topology at each admission point (black = solid)','FontSize',11,'Interpreter','none');
files{end+1} = local_save(f, figDir, 'fig1_topology_comparison');

% ---- 2..5,7,8 per-iteration panels ------------------------------------
specs = { ...
 'fig2_maxabs',      'maxAbs', 'max|\Delta\rho|',                 'Largest single design change', true; ...
 'fig3_ratio',       'ratio',  'max|\Delta\rho| / move',          'Bound saturation - the invariant-enforcing quantity', true; ...
 'fig5_move',        'move',   'move limit',                      'Move limit', true; ...
 'fig7_Mnd',         'Mnd',    'M_{nd}  (%)',                     'Measure of non-discreteness', false; ...
 'fig8_omega1',      'omega1', '\omega_1',                        'First eigenfrequency', false };
for s = 1:size(specs,1)
    f = figure('Position',[60 60 1200 460],'Color','w','Visible','off');
    tl = tiledlayout(1,numel(R),'TileSpacing','compact','Padding','compact');
    for i = 1:numel(R)
        ax = nexttile; hold(ax,'on'); grid(ax,'on'); P = R(i).per;
        plot(P.outer, P.(specs{s,2}), '-', 'Color',[0.45 0.45 0.45], 'LineWidth',1.2, ...
            'DisplayName','unstopped trajectory');
        d = find(P.descent);
        plot(P.outer(d), P.(specs{s,2})(d), 'v','MarkerSize',8,'MarkerFaceColor',[0.9 0.75 0.1], ...
            'MarkerEdgeColor','k','DisplayName','move descent');
        local_stops(ax, info(i), CLR);
        if strcmp(specs{s,2},'ratio')
            yline(ax, 0.50, '--','\tau_{rel} = 0.50','Color',CLR.c1,'HandleVisibility','off');
            yline(ax, 0.25, ':','\tau_{rel} = 0.25','Color',CLR.c2,'HandleVisibility','off');
        end
        if any(strcmp(specs{s,2},{'maxAbs','move'})); set(ax,'YScale','log'); end
        xlabel('outer iteration'); ylabel(specs{s,3});
        title(sprintf('%dx%d', R(i).mesh(1), R(i).mesh(2)));
        if i==1; legend('Location','best','FontSize',8); end
    end
    title(tl, specs{s,4},'FontSize',11);
    files{end+1} = local_save(f, figDir, specs{s,1});
end

% ---- 4. objective relative change --------------------------------------
f = figure('Position',[60 60 1200 460],'Color','w','Visible','off');
tl = tiledlayout(1,numel(R),'TileSpacing','compact','Padding','compact');
for i = 1:numel(R)
    ax = nexttile; hold(ax,'on'); grid(ax,'on');
    E = ar_predicate(R(i).per, cands(1));
    plot(R(i).per.outer, E.objRelRange, '-','Color',[0.45 0.45 0.45],'LineWidth',1.2, ...
        'DisplayName','\omega_1 relative range over W=10');
    yline(ax, cands(1).tauObj, '--', '\tau_{obj} = 5e-3','Color',CLR.c1,'HandleVisibility','off');
    local_stops(ax, info(i), CLR);
    set(ax,'YScale','log'); xlabel('outer iteration'); ylabel('relative range of \omega_1');
    title(sprintf('%dx%d', R(i).mesh(1), R(i).mesh(2)));
    if i==1; legend('Location','best','FontSize',8); end
end
title(tl,'Objective stability component C','FontSize',11);
files{end+1} = local_save(f, figDir, 'fig4_objective_change');

% ---- 6. predicate components -------------------------------------------
f = figure('Position',[60 60 1200 520],'Color','w','Visible','off');
tl = tiledlayout(numel(R),1,'TileSpacing','compact','Padding','compact');
for i = 1:numel(R)
    ax = nexttile; hold(ax,'on'); grid(ax,'on'); P = R(i).per;
    E = ar_predicate(P, cands(1));
    rows = {E.A,'A: move settled (dwell 10)'; E.Babs,'B1: max|drho| < 0.01'; ...
            E.Brel,'B2: max|drho|/move < 0.50'; E.C,'C: omega1 stable'; E.admit,'ADMIT = A&B&C'};
    for q = 1:size(rows,1)
        y = double(rows{q,1})*0.8 + (size(rows,1)-q);
        stairs(P.outer, y, 'LineWidth',1.3);
    end
    d = find(P.descent);
    for q = 1:numel(d); xline(ax,d(q),'-','Color',[0.9 0.75 0.1],'HandleVisibility','off'); end
    xline(ax, info(i).kProd,'-','Color',CLR.prod,'LineWidth',1.6,'HandleVisibility','off');
    if ~isempty(info(i).k1); xline(ax, info(i).k1,'--','Color',CLR.c1,'LineWidth',1.6,'HandleVisibility','off'); end
    set(ax,'YTick',(0:size(rows,1)-1)+0.4,'YTickLabel',flip(rows(:,2)));
    xlim([1 min(250, P.outer(end))]); xlabel('outer iteration');
    title(sprintf('%dx%d   (yellow = move descent, blue = production stop, orange = C1 admits)', ...
        R(i).mesh(1), R(i).mesh(2)),'FontSize',9);
end
title(tl,'Convergence predicate components (C1)','FontSize',11);
files{end+1} = local_save(f, figDir, 'fig6_predicate_components');
end

function local_stops(ax, inf1, CLR)
%LOCAL_STOPS  Mark the three admission points, labels staggered so they do not
%   overprint each other when the stops are only a few iterations apart.
specs = { inf1.kProd, '-',  'production stop', CLR.prod, 'top'; ...
          inf1.k1,    '--', 'C1 admits',       CLR.c1,   'middle'; ...
          inf1.k2,    ':',  'C2 admits',       CLR.c2,   'bottom' };
for q = 1:size(specs,1)
    k = specs{q,1};
    if isempty(k); continue; end
    xline(ax, k, specs{q,2}, sprintf('%s (%d)', specs{q,3}, k), 'Color', specs{q,4}, ...
        'LineWidth', 1.6, 'LabelOrientation','horizontal', 'FontSize', 8, ...
        'LabelVerticalAlignment', specs{q,5}, 'LabelHorizontalAlignment','right', ...
        'HandleVisibility','off');
end
end

function p = local_save(f,d,name)
p = fullfile(d,[name '.png']); exportgraphics(f,p,'Resolution',150); close(f);
fprintf('  figure: %s\n', name);
end
