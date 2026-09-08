function tb_figures(A, outdir)
%TB_FIGURES  The twelve required figures (Phase 18).
if ~exist(outdir,'dir'), mkdir(outdir); end
t = A.T.f240; p = t.per; d = t.dyn; B = t.B; s = A.branch.f240;
n = numel(p.move); col = [0.49 0.18 0.56];
kEv = s.event; kN = s.nativeStop; kBeta = s.betaStallFirst;

    function marks(nn)
        if ~isnan(kN)    && kN<=nn,    xline(kN,'--','Color',[0.5 0.5 0.5],'LineWidth',1.2); end
        if ~isnan(kBeta) && kBeta<=nn, xline(kBeta,':','Color',[0.85 0.6 0.0],'LineWidth',1.4); end
        if ~isnan(kEv)   && kEv<=nn,   xline(kEv,'m-','LineWidth',2.0); end
    end

% F1-F5 : cos, net/path, normalized amplitude, M_nd, omega1
specs = { {d.cosT,'cos\theta','F1_cos',[-1 1],true}, ...
          {d.net_ratio,'net / path (W=10)','F2_netpath',[0 1],true}, ...
          {p.ratio,'max|\Delta\rho| / move','F3_amplitude',[0 1.05],true}, ...
          {p.Mnd,'M_{nd} [%]','F4_Mnd',[],false}, ...
          {p.omega1,'\omega_1','F5_omega1',[],false} };
for i = 1:numel(specs)
    q = specs{i}; f = figure('Position',[100 100 980 340],'Color','w'); hold on; grid on
    plot(q{1},'-','Color',col,'LineWidth',0.9);
    if q{5}, plot(movmedian(q{1},20,'omitnan'),'k-','LineWidth',1.6); end
    if i<=2, yline(0,'k:'); end
    if i==2, yline(0.5,'r:','Branch A cut 0.5'); end
    marks(n); if ~isempty(q{4}), ylim(q{4}); end
    xlabel('outer iteration'); ylabel(q{2});
    title(sprintf('%s  240\\times30 fixed move 0.04 -- %s   (magenta = exhaustion event %s, grey = native stop %s)', ...
        q{3}(1:2), q{2}, mat2str(kEv), mat2str(kN)));
    exportgraphics(f, fullfile(outdir,[q{3} '.png']),'Resolution',150); close(f);
end

% F6 : all events marked on one axis
f = figure('Position',[100 100 980 560],'Color','w');
subplot(2,1,1); hold on; grid on
plot(B.medcos,'-','Color',[0.2 0.35 0.75],'LineWidth',1.4); yline(0,'k:');
plot(B.mednet,'-','Color',[0.85 0.33 0.10],'LineWidth',1.4); yline(0.5,'r:');
marks(n); ylim([-1 1.05]); ylabel('median_{20}');
legend({'median cos\theta','','median net/path'},'Location','best');
title('F6  Frozen predicate inputs, with native-stop (grey), \beta-stall (orange) and event (magenta)');
subplot(2,1,2); hold on; grid on
semilogy(max(p.l2,1e-18),'-','Color',[0.1 0.5 0.2],'LineWidth',1.2);
yline(B.tol,'r--',sprintf('tol = %.4g',B.tol)); set(gca,'YScale','log');
marks(n); xlabel('outer iteration'); ylabel('||\Delta\rho||_2');
exportgraphics(f, fullfile(outdir,'F6_events.png'),'Resolution',150); close(f);

% F7 : spatial reversal map near the event ; F8 : amplitude field near the event
nx=240; ny=30; X=[0.5*ones(nx*ny,1), t.RHO]; Dd=diff(X,1,2);
kk = kEv; if isnan(kk), kk = min(n-1, round(0.8*n)); end
kk = max(2,min(kk,n-1));
f = figure('Position',[80 100 1100 420],'Color','w');
subplot(2,1,1);
sgn = sign(Dd(:,kk).*Dd(:,kk-1));
imagesc(reshape(sgn,ny,nx)); axis image; axis off
colormap(gca,[0.80 0.10 0.10; 1 1 1; 0.10 0.35 0.75]); clim([-1 1]);
title(sprintf('F7  sign(\\Delta\\rho_k \\cdot \\Delta\\rho_{k-1}) at the event k=%d  --  reversing %.1f%%, boundFrac %.4f', ...
    kk, 100*mean(sgn<0), d.boundFrac(kk)));
subplot(2,1,2);
imagesc(reshape(abs(Dd(:,kk)),ny,nx)); axis image; axis off
colormap(gca,'parula'); cb=colorbar; cb.Label.String='|\Delta\rho|';
title(sprintf('F8  amplitude field |\\Delta\\rho| at k=%d   (move = 0.04, max = %.4g)', kk, max(abs(Dd(:,kk)))));
exportgraphics(f, fullfile(outdir,'F7_F8_spatial.png'),'Resolution',150); close(f);

% F9-F11 : cross-mesh
keys={'f160','f240','f320','f400'}; labs={'160\times20','240\times30 (withheld)','320\times40','400\times50'};
cols={[0.85 0.33 0.10],col,[0 0.45 0.74],[0.47 0.67 0.19]};
% F9 endpoint comparison
f=figure('Position',[100 100 980 380],'Color','w');
vals=zeros(4,3);
for i=1:4, b=A.branch.(keys{i}); vals(i,:)=[b.term_maxAbs_over_move, b.term_cosT, b.term_net_path]; end
bar(vals); grid on; set(gca,'XTickLabel',labs);
legend({'terminal max|\Delta\rho|/move','terminal cos\theta','terminal net/path'},'Location','best');
ylabel('value'); yline(0,'k-');
title('F9  Fixed-move endpoint type by mesh');
exportgraphics(f, fullfile(outdir,'F9_xmesh_endpoint.png'),'Resolution',150); close(f);
% F10 exhaustion-event comparison
f=figure('Position',[100 100 980 380],'Color','w'); hold on; grid on
for i=1:4
    b=A.branch.(keys{i});
    if ~isnan(b.event)
        bar(i, b.event, 'FaceColor', cols{i});
        text(i, b.event, sprintf('%d (%s)', b.event, b.branch),'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    if ~isnan(b.nativeStop), plot(i, b.nativeStop,'kv','MarkerFaceColor','k','MarkerSize',9); end
end
set(gca,'XTick',1:4,'XTickLabel',labs); ylabel('outer iteration');
title('F10  First exhaustion event by mesh (bar) and native stop (black triangle)');
exportgraphics(f, fullfile(outdir,'F10_xmesh_event.png'),'Resolution',150); close(f);
% F11 post-event useful evolution
f=figure('Position',[100 100 980 380],'Color','w');
v=nan(4,2);
for i=1:4, b=A.branch.(keys{i}); if ~isnan(b.event), v(i,:)=[b.remUseful, b.postRelImp]; end, end
bar(v); grid on; set(gca,'XTickLabel',labs); yline(5,'r--','P2 bound 5%'); yline(25,'m--','P3 bound 25%');
legend({'remUseful %','postRelImp %'},'Location','best'); ylabel('%');
title('F11  Useful topology evolution remaining after the exhaustion event');
exportgraphics(f, fullfile(outdir,'F11_xmesh_useful.png'),'Resolution',150); close(f);

% F12 : union classification diagram
f=figure('Position',[100 100 900 460],'Color','w'); hold on; grid on
for i=1:4
    b=A.branch.(keys{i});
    x=b.term_cosT; y=b.term_maxAbs_over_move;
    plot(x,y,'o','MarkerSize',14,'MarkerFaceColor',cols{i},'MarkerEdgeColor','k');
    text(x,y+0.045,sprintf('%s\n%s @ %s',labs{i},b.branch,mat2str(b.event)),'HorizontalAlignment','center','FontSize',9);
end
xline(0,'k--'); ylim([-0.1 1.2]); xlim([-1.1 1.1]);
xlabel('terminal median cos\theta   (<0 = cancelling)'); ylabel('terminal max|\Delta\rho| / move');
text(-0.95,1.12,'BRANCH A region: cancelling, amplitude non-negligible','FontSize',9,'Color',[0.6 0 0]);
text(0.05,0.12,'BRANCH B region: coherent, amplitude converged','FontSize',9,'Color',[0 0.4 0]);
title('F12  Union classification: where each fixed-move arm terminates');
exportgraphics(f, fullfile(outdir,'F12_union_diagram.png'),'Resolution',150); close(f);
fprintf('[tb_figures] wrote 12 figures\n');
end
