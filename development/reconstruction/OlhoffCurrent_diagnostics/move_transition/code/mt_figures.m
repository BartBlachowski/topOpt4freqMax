function files = mt_figures(R, figDir)
%MT_FIGURES  The eight preregistered figures (brief sec. 21).
%
%   R is a struct array of analysed runs with fields: arm, mesh, per, rhoFinal,
%   stopIter, label, and (for the 320x40 panels) the historical fixed-move
%   reference.  Curves are labelled semantically, never as S2/R2/M4.

if ~isfolder(figDir); mkdir(figDir); end
files = {};
CP = [0.20 0.35 0.75];   % ARM P
CU = [0.85 0.35 0.10];   % ARM U
CF = [0.35 0.65 0.30];   % historical fixed move
meshes = unique(cell2mat(arrayfun(@(r) r.mesh(:).', R, 'UniformOutput', false)'), 'rows');

pick = @(arm, m) R(arrayfun(@(r) strcmp(r.arm,arm) && isequal(r.mesh(:).',m), R));

% ================= 1. topology comparison ==============================
f = figure('Position',[60 60 1500 700],'Color','w','Visible','off');
tl = tiledlayout(size(meshes,1), 2, 'TileSpacing','tight','Padding','compact');
for m = 1:size(meshes,1)
    for a = {'P','U'}
        r = pick(a{1}, meshes(m,:));
        ax = nexttile;
        if isempty(r); axis off; title('(missing)'); continue; end
        imagesc(reshape(1-r.rhoAtStop, r.mesh(2), r.mesh(1)));
        colormap(gray); clim([0 1]); axis image; set(ax,'XTick',[],'YTick',[]);
        title(ax, {sprintf('%s   -   %dx%d   [%s]', r.label, r.mesh(1), r.mesh(2), r.status), ...
                   sprintf('iter %d    M_nd = %.2f%%    mid = %.4f    omega_1 = %.6g', ...
                       r.stopIter, r.MndAtStop, r.midAtStop, r.omega1AtStop)}, ...
              'FontSize', 10, 'Interpreter','none');
    end
end
title(tl, 'Topology: production vs utilization-gated move transition', ...
      'FontWeight','bold');
files{end+1} = local_save(f, figDir, 'fig1_topology_comparison.png');

% ================= per-iteration panels =================================
panels = { ...
 'fig2_Mnd_vs_iteration.png',       'Mnd',        'M_{nd} (%)',                     'M_{nd} vs iteration'; ...
 'fig3_omega1_vs_iteration.png',    'omega1',     '\omega_1',                       '\omega_1 vs iteration'; ...
 'fig4_rrho_vs_iteration.png',      'ratio',      'r_\rho = max|\Delta\rho| / move','Design utilization r_\rho vs iteration'; ...
 'fig5_persistence_vs_iteration.png','utilCount', 'consecutive iterations r_\rho<0.5','Utilization persistence counter'; ...
 'fig6_move_vs_iteration.png',      'move',       'move limit',                     'Move limit vs iteration'; ...
 'fig7_mid_vs_iteration.png',       'mid',        'mid-density fraction (0.4\leq\rho\leq0.6)','Mid-density fraction vs iteration'};

for p = 1:size(panels,1)
    f = figure('Position',[60 60 1350 520],'Color','w','Visible','off');
    tiledlayout(1, size(meshes,1), 'TileSpacing','compact','Padding','compact');
    for m = 1:size(meshes,1)
        ax = nexttile; hold(ax,'on'); grid(ax,'on');
        for a = {'P','U'}
            r = pick(a{1}, meshes(m,:));
            if isempty(r); continue; end
            c = CP; if strcmp(a{1},'U'); c = CU; end
            plot(r.per.outer, r.per.(panels{p,2}), '-', 'Color', c, 'LineWidth', 1.3, ...
                 'DisplayName', r.label);
            d = find(r.per.descent);
            plot(r.per.outer(d), r.per.(panels{p,2})(d), 'v', 'Color', c, ...
                 'MarkerFaceColor', c, 'MarkerSize', 6, 'HandleVisibility','off');
            xline(r.stopIter, '--', 'Color', c, 'LineWidth', 1.0, 'HandleVisibility','off');
        end
        if strcmp(panels{p,2},'ratio')
            yline(0.5, 'k:', 'LineWidth', 1.4, 'DisplayName','threshold 0.5');
        end
        if strcmp(panels{p,2},'utilCount')
            yline(10, 'k:', 'LineWidth', 1.4, 'DisplayName','persistence 10');
        end
        % historical fixed-move mature reference (evidence, not an arm)
        rf = R(arrayfun(@(r) strcmp(r.arm,'FIXED') && isequal(r.mesh(:).',meshes(m,:)), R));
        if ~isempty(rf) && any(strcmp(panels{p,2}, {'Mnd','omega1','mid','ratio'}))
            plot(rf.per.outer, rf.per.(panels{p,2}), ':', 'Color', CF, 'LineWidth', 1.2, ...
                 'DisplayName', 'historical fixed move 0.04 (evidence)');
        end
        if strcmp(panels{p,2},'move'); set(ax,'YScale','log'); end
        xlabel('outer iteration'); ylabel(panels{p,3});
        title(sprintf('%dx%d', meshes(m,1), meshes(m,2)));
        legend('Location','best','FontSize',8);
    end
    sgtitle(panels{p,4}, 'FontWeight','bold');
    files{end+1} = local_save(f, figDir, panels{p,1}); %#ok<AGROW>
end

% ================= 8. stage-by-stage M_nd reduction =====================
f = figure('Position',[60 60 1350 520],'Color','w','Visible','off');
tiledlayout(1, size(meshes,1), 'TileSpacing','compact','Padding','compact');
for m = 1:size(meshes,1)
    ax = nexttile; hold(ax,'on'); grid(ax,'on');
    lab = {}; vals = []; grp = [];
    for a = {'P','U'}
        r = pick(a{1}, meshes(m,:));
        if isempty(r); continue; end
        S = r.stages;
        for k = 1:numel(S)
            vals(end+1) = -S(k).dMnd; %#ok<AGROW>
            lab{end+1}  = sprintf('%s m=%.3g (%d it)', a{1}, S(k).move, S(k).nIter); %#ok<AGROW>
            grp(end+1)  = strcmp(a{1},'U'); %#ok<AGROW>
        end
    end
    b = bar(vals, 'FaceColor','flat');
    for k = 1:numel(vals)
        if grp(k); b.CData(k,:) = CU; else; b.CData(k,:) = CP; end
    end
    set(ax,'XTick',1:numel(lab),'XTickLabel',lab,'XTickLabelRotation',60,'FontSize',8);
    ylabel('M_{nd} reduction achieved in stage (points)');
    title(sprintf('%dx%d', meshes(m,1), meshes(m,2)));
end
sgtitle('Stage-by-stage M_{nd} reduction (blue = production, orange = utilization-gated)', ...
        'FontWeight','bold');
files{end+1} = local_save(f, figDir, 'fig8_stage_Mnd_reduction.png');
end

function f = local_save(fig, figDir, name)
f = fullfile(figDir, name);
exportgraphics(fig, f, 'Resolution', 130);
close(fig);
fprintf('[mt_figures] %s\n', f);
end
