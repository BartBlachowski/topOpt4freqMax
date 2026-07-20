function figs = a4_plots(outDir, res)
%A4_PLOTS  Nine figures required by Phase-2 specification §10.5.
figs = {};
if isempty(res.arms), return; end
arms = res.arms; c = a4_phase2_constants();
colors = lines(numel(arms));

% 1 — tracked endpoint vs N with the complete ±delta band.
f = localFigure(); hold on; grid on;
Ns = [arms.N]; finiteN = Ns(isfinite(Ns)); if isempty(finiteN), finiteN = 1; end
x = Ns; x(isinf(x)) = 2*max(finiteN);
ref = localFrozen(arms, 'omega1_tracked');
xl = [max(min(x)/1.5, 0.5), max(x)*1.5];
if isfinite(ref)
    fill([xl fliplr(xl)], [ref*(1-res.delta)*[1 1], ref*(1+res.delta)*[1 1]], ...
        [0.85 0.92 0.85], 'EdgeColor','none', 'DisplayName','5% equivalence band');
    yline(ref,'k--','DisplayName','N=inf reference');
end
eligible = ~strcmp({arms.phase2_status}, 'UNAVAILABLE') & ...
    ~strcmp({arms.phase2_status}, 'REJECTED');
plot(x(eligible), [arms(eligible).omega1_tracked], 'o-', 'LineWidth',1.5, ...
    'DisplayName','eligible endpoints');
for i=find(~eligible), text(x(i),ref,sprintf('N=%s %s',arms(i).tag,arms(i).phase2_status)); end
set(gca,'XScale','log','XTick',sort(x)); xlim(xl);
if isfinite(ref), ylim([ref*(1-res.delta)*0.995, ref*(1+res.delta)*1.005]); end
xlabel('refresh interval N'); ylabel('tracked omega1 [rad/s]');
title('A4 Phase 2 — tracked endpoint and 5% equivalence band'); legend('Location','best');
figs{end+1}=localSave(f,outDir,'a4_fig1_omega1_vs_N.png');

% 2 — selected-reference MAC histories; deferrals marked.
f=localFigure(); hold on; grid on;
for i=1:numel(arms)
    ev=arms(i).screening_events; if isempty(ev), continue; end
    plot([ev.iteration],[ev.selected_mac_phi0],'-o','Color',colors(i,:), ...
        'DisplayName',sprintf('N=%s',arms(i).tag));
    d=ev([ev.deferred]); if ~isempty(d), scatter([d.iteration],[d.selected_mac_phi0],55,'x','MarkerEdgeColor',colors(i,:)); end
end
yline(c.tau_mac,'r--','DisplayName','MAC threshold'); xlabel('iteration'); ylabel('MAC to solid Phi0');
title('Selected-mode continuity; x marks deferred refresh'); legend('Location','best');
figs{end+1}=localSave(f,outDir,'a4_fig2_mac_vs_iteration.png');

% 3 — design change.
f=localFigure(); hold on; grid on;
for i=1:numel(arms), h=arms(i).iteration_histories; if ~isempty(h), semilogy([h.iteration],[h.max_design_change], ...
        'Color',colors(i,:),'DisplayName',sprintf('N=%s',arms(i).tag)); end; end
xlabel('iteration'); ylabel('max absolute density change'); title('Optimization design-change histories'); legend('Location','best');
figs{end+1}=localSave(f,outDir,'a4_fig3_design_change.png');

% 4 — reference index, five panels.
f=figure('Visible','off','Position',[100 100 900 900]);
for i=1:numel(arms), subplot(numel(arms),1,i); h=arms(i).iteration_histories; grid on;
    if ~isempty(h), stairs([h.iteration],[h.reference_mode_index],'Color',colors(i,:)); end
    ylabel(sprintf('N=%s',arms(i).tag)); end
xlabel('iteration'); sgtitle('Tracked index j* of reference in force','Interpreter','none');
figs{end+1}=localSave(f,outDir,'a4_fig4_tracked_index.png');

% 5 — full final-window spectra and admissibility.
f=localFigure(); hold on; grid on;
for i=1:numel(arms), rows=arms(i).candidate_telemetry; if isempty(rows), continue; end
    adm=[rows.admissible]; scatter([rows(~adm).iteration],[rows(~adm).omega],5,colors(i,:),'filled','MarkerFaceAlpha',0.15);
    scatter([rows(adm).iteration],[rows(adm).omega],16,colors(i,:),'o','DisplayName',sprintf('N=%s admissible',arms(i).tag)); end
xlabel('iteration'); ylabel('candidate omega [rad/s]'); title('Final-window spectra and admissibility'); legend('Location','best');
figs{end+1}=localSave(f,outDir,'a4_fig5_spectrum_screen.png');

% 6 — final topologies.
f=figure('Visible','off','Position',[100 100 1000 800]);
for i=1:numel(arms), subplot(numel(arms),1,i); a=arms(i);
    if ~isempty(a.topology), imagesc(reshape(1-a.topology,res.nely,res.nelx)); axis equal off; colormap(gray); end
    title(sprintf('N=%s; %s; warnings=%s',a.tag,a.phase2_status,strjoin(a.warnings,',')),'Interpreter','none'); end
figs{end+1}=localSave(f,outDir,'a4_fig6_topologies.png');

% 7 — omega1/omega2 separation at common/operational events.
f=localFigure(); hold on; grid on;
for i=1:numel(arms), ev=arms(i).screening_events; if ~isempty(ev), plot([ev.iteration],[ev.omega1_omega2_gap],'-o', ...
        'Color',colors(i,:),'DisplayName',sprintf('N=%s',arms(i).tag)); end; end
xlabel('iteration'); ylabel('omega2 - omega1 [rad/s]'); title('First-mode spectral separation'); legend('Location','best');
figs{end+1}=localSave(f,outDir,'a4_fig7_omega_gap.png');

% 8 — required final window (central C-1 figure).
f=localFigure(); hold on; grid on;
for i=1:numel(arms), ev=arms(i).screening_events; if ~isempty(ev), stairs([ev.iteration],[ev.m_final],'-o', ...
        'Color',colors(i,:),'DisplayName',sprintf('N=%s',arms(i).tag)); end; end
yline(c.m0,'k--','DisplayName','old window m0=20'); yline(c.M_max,'r--','DisplayName','ceiling Mmax=320');
xlabel('iteration'); ylabel('m final'); title('Adaptive mode window required'); legend('Location','best');
figs{end+1}=localSave(f,outDir,'a4_fig8_required_window.png');

% 9 — selected mode index.
f=localFigure(); hold on; grid on;
for i=1:numel(arms), ev=arms(i).screening_events; if ~isempty(ev), plot([ev.iteration],[ev.selected_index],'-o', ...
        'Color',colors(i,:),'DisplayName',sprintf('N=%s',arms(i).tag)); end; end
yline(c.m0,'k--','DisplayName','old window m0=20'); xlabel('iteration'); ylabel('selected mode index');
title('Selected physical-mode index'); legend('Location','best');
figs{end+1}=localSave(f,outDir,'a4_fig9_selected_index.png');
end

function f=localFigure(), f=figure('Visible','off','Position',[100 100 950 560]); end
function p=localSave(f,outDir,name)
set(findall(f,'Type','text'),'Interpreter','none');
ax=findall(f,'Type','axes'); for i=1:numel(ax), ax(i).TickLabelInterpreter='none'; end
lg=findall(f,'Type','legend'); for i=1:numel(lg), lg(i).Interpreter='none'; end
p=fullfile(outDir,name); exportgraphics(f,p,'Resolution',150); close(f);
end
function v=localFrozen(arms,field)
v=NaN; i=find(isinf([arms.N]),1); if ~isempty(i), v=arms(i).(field); end
end
