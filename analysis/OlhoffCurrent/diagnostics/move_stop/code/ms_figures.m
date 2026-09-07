function files = ms_figures(runs, figDir)
%MS_FIGURES  The seven preregistered figures.
%
%   Curves are labelled semantically -- "Production: move ladder",
%   "Diagnostic: fixed move 0.04" -- never as M4/S2/R2.  Historical aliases
%   appear only in a secondary note on the topology figure.

if ~isfolder(figDir); mkdir(figDir); end
files = {};
LBL = struct('baseline','Production: move ladder', 'fixedmove','Diagnostic: fixed move 0.04');
CLR = struct('baseline',[0.20 0.35 0.75], 'fixedmove',[0.85 0.35 0.10]);
meshes = unique(cell2mat(arrayfun(@(r) r.mesh(:).', runs,'UniformOutput',false)'),'rows');

% ---- 1. topology comparison ------------------------------------------
f = figure('Position',[60 60 1500 620],'Color','w','Visible','off');
t = tiledlayout(size(meshes,1), 2, 'TileSpacing','tight','Padding','compact');
for m = 1:size(meshes,1)
    for a = {'baseline','fixedmove'}
        r = local_pick(runs, a{1}, meshes(m,:));
        ax = nexttile;
        if isempty(r); axis off; title('(missing)'); continue; end
        imagesc(reshape(1-r.rhoFinal, r.mesh(2), r.mesh(1)));
        colormap(gray); clim([0 1]); axis image; set(ax,'XTick',[],'YTick',[]);
        % Interpreter 'none' so status strings with underscores are not
        % rendered as subscripts (NATIVE_CONVERGED must read literally).
        title(ax, {sprintf('%s   -   %dx%d   [%s]', LBL.(a{1}), r.mesh(1), r.mesh(2), r.status), ...
                   sprintf('M_nd = %.2f%%    gray = %.3f    omega_1 = %.5g', ...
                       r.Mnd_final, r.gray_final, r.omega(1))}, ...
            'FontSize', 10, 'Interpreter','none');
    end
end
title(t, ['Final topology, physical density (black = solid).   ' ...
    'Historical alias of the production arm: M4 / S2 ladder.'], ...
    'FontSize', 10, 'Interpreter','none');
files{end+1} = local_save(f, figDir, 'fig1_topology_comparison');

% ---- 2..6 per-iteration panels ---------------------------------------
specs = { ...
 'fig2_Mnd_vs_iteration',     'Mnd',    'M_{nd}  (%)',                 'Measure of non-discreteness', false; ...
 'fig3_omega1_vs_iteration',  'omega1', '\omega_1',                    'First eigenfrequency',        false; ...
 'fig4_l2_vs_iteration',      'l2',     '||\Delta\rho||_2',            'Convergence statistic vs its tolerance', true; ...
 'fig5_maxabs_vs_iteration',  'maxAbs', 'max|\Delta\rho|',             'Largest single design change', true; ...
 'fig6_move_vs_iteration',    'move',   'move limit',                  'Move limit',                  true };
for s = 1:size(specs,1)
    f = figure('Position',[80 80 1120 460],'Color','w','Visible','off');
    tl = tiledlayout(1, size(meshes,1), 'TileSpacing','compact','Padding','compact');
    for m = 1:size(meshes,1)
        ax = nexttile; hold(ax,'on'); grid(ax,'on');
        for a = {'baseline','fixedmove'}
            r = local_pick(runs, a{1}, meshes(m,:));
            if isempty(r); continue; end
            plot(r.per.outer, r.per.(specs{s,2}), '-', 'LineWidth', 1.5, ...
                'Color', CLR.(a{1}), 'DisplayName', LBL.(a{1}));
            if specs{s,5}
                d = find(r.per.descent);
                if ~isempty(d)
                    plot(r.per.outer(d), r.per.(specs{s,2})(d), 'v', 'MarkerSize', 7, ...
                        'MarkerFaceColor', CLR.(a{1}), 'MarkerEdgeColor','k', ...
                        'HandleVisibility','off');
                end
            end
            if strcmp(specs{s,2},'l2')
                yline(ax, r.tolOuter, '--', sprintf('\\epsilon = %.4g', r.tolOuter), ...
                    'Color',[0.4 0.4 0.4], 'HandleVisibility','off');
            end
        end
        if any(strcmp(specs{s,2}, {'l2','maxAbs','move'})); set(ax,'YScale','log'); end
        xlabel('outer iteration'); ylabel(specs{s,3});
        title(sprintf('%dx%d', meshes(m,1), meshes(m,2)));
        if m==1; legend('Location','best','FontSize',8); end
    end
    note = specs{s,4};
    if specs{s,5}; note = [note '   (\nabla marks a move descent)']; end
    title(tl, note, 'FontSize', 11);
    files{end+1} = local_save(f, figDir, specs{s,1});
end

% ---- 7. active set ----------------------------------------------------
f = figure('Position',[80 80 1120 460],'Color','w','Visible','off');
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
ax = nexttile; hold(ax,'on'); grid(ax,'on');
for m = 1:size(meshes,1)
    for a = {'baseline','fixedmove'}
        r = local_pick(runs, a{1}, meshes(m,:));
        if isempty(r); continue; end
        ls = '-'; if m==2; ls = '--'; end
        plot(r.per.outer, r.per.nActive(:,1), ls, 'LineWidth',1.4, 'Color', CLR.(a{1}), ...
            'DisplayName', sprintf('%s, %dx%d', LBL.(a{1}), r.mesh(1), r.mesh(2)));
    end
end
set(ax,'YScale','log'); xlabel('outer iteration');
ylabel('N_{active}   ( |\Delta\rho_e| > \epsilon_{RMS} )');
title('Active-element count vs iteration'); legend('Location','best','FontSize',8);

ax = nexttile; hold(ax,'on'); grid(ax,'on');
for a = {'baseline','fixedmove'}
    NEs = []; fr = []; nn = [];
    for m = 1:size(meshes,1)
        r = local_pick(runs, a{1}, meshes(m,:));
        if isempty(r); continue; end
        NEs(end+1) = r.NE; fr(end+1) = r.per.nActive(end,1)/r.NE; nn(end+1) = r.per.nActive(end,1);
    end
    if isempty(NEs); continue; end
    yyaxis left;  plot(NEs, nn, 'o-','LineWidth',1.6,'Color',CLR.(a{1}), ...
        'MarkerFaceColor',CLR.(a{1}),'DisplayName',[LBL.(a{1}) ' (count)']);
    ylabel('N_{active} at termination');
    yyaxis right; plot(NEs, fr, 's--','LineWidth',1.2,'Color',CLR.(a{1}), ...
        'DisplayName',[LBL.(a{1}) ' (fraction)']);
    ylabel('N_{active}/N_E');
end
xlabel('N_E'); title('Active set vs mesh size'); legend('Location','best','FontSize',8);
title(tl, 'Active set: measured, not assumed', 'FontSize', 11);
files{end+1} = local_save(f, figDir, 'fig7_active_set');
end

function r = local_pick(runs, arm, mesh)
r = [];
for i=1:numel(runs)
    if strcmp(runs(i).arm,arm) && isequal(runs(i).mesh(:).',mesh(:).'); r=runs(i); return; end
end
end

function p = local_save(f, d, name)
p = fullfile(d, [name '.png']);
exportgraphics(f, p, 'Resolution', 150);
close(f);
fprintf('  figure: %s\n', name);
end
