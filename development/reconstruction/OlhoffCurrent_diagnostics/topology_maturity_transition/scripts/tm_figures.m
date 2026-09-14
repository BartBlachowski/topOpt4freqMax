function tm_figures(A, outdir)
%TM_FIGURES  Phase A figures.  Purely descriptive; no rule is being fitted.
if ~exist(outdir,'dir'), mkdir(outdir); end
ms = {'m160x20','m320x40'}; lbl = {'160\times20','320\times40'}; col = {[0.85 0.33 0.10],[0 0.45 0.74]};

% --- F1: the two meshes are in different dynamical regimes -----------------
f = figure('Position',[100 100 900 380],'Color','w');
for i=1:2
    r = A.(ms{i}); subplot(1,2,i); hold on; grid on
    plot(r.coherence2,'-','Color',col{i},'LineWidth',1.0);
    yline(2,'k--','coherent (d_2=2d_1)');
    yline(1,'k:','');
    xline(r.kP,'r-','LineWidth',1.5);
    xlim([1 min(r.nK,400)]); ylim([0 3]);
    xlabel('outer iteration'); ylabel('d_2/d_1');
    title(sprintf('%s  (production descends at %d)', lbl{i}, r.kP));
end
sgtitle('F1  Two-step / one-step displacement ratio: regime diagnosis (ARM U, move = 0.04)');
exportgraphics(f, fullfile(outdir,'F1_regime_diagnosis.png'), 'Resolution',150); close(f);

% --- F2: candidate statistics vs TRUE remaining evolution ------------------
cands = {'RW_end','DW_unsat','CW_l1','inst_max'};
f = figure('Position',[100 100 1000 700],'Color','w');
for c = 1:numel(cands)
    subplot(2,2,c); hold on; grid on
    for i=1:2
        r = A.(ms{i});
        plot(100*r.rem, r.C.(cands{c}), '.', 'Color', col{i}, 'MarkerSize',5);
        plot(100*r.rem(r.kP), r.C.(cands{c})(r.kP), 'p', 'MarkerSize',16, ...
             'MarkerFaceColor',col{i},'MarkerEdgeColor','k');
    end
    set(gca,'XDir','reverse');
    xlabel('TRUE remaining evolution  [%]  (mature \rightarrow)');
    ylabel(strrep(cands{c},'_','\_'));
    title(strrep(cands{c},'_','\_'));
    if c==1, legend({'160\times20','160 at k_P','320\times40','320 at k_P'},'Location','northwest'); end
end
sgtitle('F2  Candidate statistic vs ground-truth maturity.  A usable signal would place the two stars at DIFFERENT heights.');
exportgraphics(f, fullfile(outdir,'F2_candidates_vs_maturity.png'), 'Resolution',150); close(f);

% --- F3: M_nd trajectories, showing what production discards ---------------
f = figure('Position',[100 100 900 380],'Color','w');
for i=1:2
    r = A.(ms{i}); subplot(1,2,i); hold on; grid on
    plot(r.Mnd,'-','Color',col{i},'LineWidth',1.2);
    xline(r.kP,'r-','LineWidth',1.5,'Label','production descent');
    xline(r.kM,'k--','LineWidth',1.2,'Label','mature reference');
    xlim([1 min(r.nK,400)]);
    xlabel('outer iteration'); ylabel('M_{nd}  [%]');
    title(sprintf('%s:  M_{nd} %.2f%% \\rightarrow %.2f%%', lbl{i}, r.Mnd_kP, r.Mnd_kM));
end
sgtitle('F3  Topology still available after the production move descent (ARM U, move = 0.04)');
exportgraphics(f, fullfile(outdir,'F3_Mnd_discarded.png'), 'Resolution',150); close(f);
end
