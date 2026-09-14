function dr_figures(A, outdir)
%DR_FIGURES  The twelve preregistered figures (Phase P).
if ~exist(outdir,'dir'), mkdir(outdir); end
T = A.T;
keys = {'f160','f320','p400'};
labs = {'160\times20 fixed 0.04','320\times40 fixed 0.04 (RUN B)','400\times50 production (RUN A)'};
cols = {[0.85 0.33 0.10],[0.00 0.45 0.74],[0.47 0.67 0.19]};
mesh3 = {[160 20],[320 40],[400 50]};

% ---- F1-F3: q2 per trajectory ------------------------------------------
for i = 1:3
    t = T.(keys{i}); f = figure('Position',[100 100 950 340],'Color','w'); hold on; grid on
    plot(t.dyn.q2,'-','Color',cols{i},'LineWidth',0.9);
    plot(movmedian(t.dyn.q2,20,'omitnan'),'k-','LineWidth',1.6);
    yline(2,'k--','coherent  q_2=2'); yline(1,'r:','cancellation  q_2<1'); yline(sqrt(2),'b:','reorient  q_2=\surd2');
    markEvents(t, numel(t.dyn.q2));
    ylim([0 3]); xlabel('outer iteration'); ylabel('q_2 = d_2/d_1');
    title(sprintf('F%d  q_2 -- %s   (final regime: %s)', i, labs{i}, t.C.labelFinal));
    legend({'q_2','20-iter median'},'Location','northeast');
    exportgraphics(f, fullfile(outdir,sprintf('F%d_q2_%s.png',i,keys{i})),'Resolution',150); close(f);
end

% ---- F4: cos_theta, all three ------------------------------------------
f = figure('Position',[100 100 1000 720],'Color','w');
for i=1:3
    t=T.(keys{i}); subplot(3,1,i); hold on; grid on
    plot(t.dyn.cosT,'-','Color',cols{i},'LineWidth',0.7);
    plot(movmedian(t.dyn.cosT,20,'omitnan'),'k-','LineWidth',1.6);
    yline(0,'k:'); markEvents(t, numel(t.dyn.cosT));
    ylim([-1 1]); ylabel('cos\theta'); title(labs{i});
    if i==3, xlabel('outer iteration'); end
end
sgtitle('F4  Directional correlation of successive density steps');
exportgraphics(f, fullfile(outdir,'F4_costheta_all.png'),'Resolution',150); close(f);

% ---- F5: net_ratio, all three ------------------------------------------
f = figure('Position',[100 100 1000 720],'Color','w');
for i=1:3
    t=T.(keys{i}); subplot(3,1,i); hold on; grid on
    plot(t.dyn.net_ratio,'-','Color',cols{i},'LineWidth',0.9);
    plot(t.dyn.net_ratio_unsat,'--','Color',[0.4 0.4 0.4],'LineWidth',0.9);
    markEvents(t, numel(t.dyn.net_ratio));
    ylim([0 1]); ylabel('net / path  (W=10)'); title(labs{i});
    if i==1, legend({'all elements','bound-saturated removed'},'Location','northeast'); end
    if i==3, xlabel('outer iteration'); end
end
sgtitle('F5  Net progress divided by path length -- 1 = pure progress, 0 = pure cancellation');
exportgraphics(f, fullfile(outdir,'F5_netratio_all.png'),'Resolution',150); close(f);

% ---- F6/F7: M_nd and omega1 with regime events -------------------------
for v = 1:2
    f = figure('Position',[100 100 1000 720],'Color','w');
    for i=1:3
        t=T.(keys{i}); subplot(3,1,i); hold on; grid on
        if v==1, y=t.per.Mnd; ylb='M_{nd} [%]'; else, y=t.per.omega1; ylb='\omega_1'; end
        plot(y,'-','Color',cols{i},'LineWidth',1.2);
        markEvents(t, numel(y));
        ylabel(ylb); title(labs{i}); if i==3, xlabel('outer iteration'); end
    end
    if v==1, sgtitle('F6  M_{nd} with dynamical-regime events'); nm='F6_Mnd_events.png';
    else,    sgtitle('F7  \omega_1 with dynamical-regime events'); nm='F7_omega1_events.png'; end
    exportgraphics(f, fullfile(outdir,nm),'Resolution',150); close(f);
end

% ---- F8: move with beta descents and regime changes --------------------
f = figure('Position',[100 100 1000 720],'Color','w');
for i=1:3
    t=T.(keys{i}); subplot(3,1,i); hold on; grid on
    stairs(t.per.move,'-','Color',cols{i},'LineWidth',1.4);
    markEvents(t, numel(t.per.move));
    set(gca,'YScale','log'); ylabel('move'); title(labs{i});
    if i==3, xlabel('outer iteration'); end
end
sgtitle('F8  Move level, \beta-stall descents (red) and regime onsets (magenta = period-2)');
exportgraphics(f, fullfile(outdir,'F8_move_events.png'),'Resolution',150); close(f);

% ---- F9-F11: spatial reversal maps -------------------------------------
picks = zeros(1,3);
for i=1:3
    t=T.(keys{i});
    if i<=2
        k = t.C.onset.PERIOD2; if isnan(k), k = min(numel(t.per.move), round(0.9*numel(t.per.move))); end
        k = min(k+20, numel(t.per.move));
    else
        k = t.ev.firstDescent; if isnan(k), k = numel(t.per.move); end
    end
    picks(i)=k;
    nx=mesh3{i}(1); ny=mesh3{i}(2);
    X=[A.rho0*ones(nx*ny,1), t.RHO]; D=diff(X,1,2);
    if k<2, k=2; end
    sgn = sign(D(:,k).*D(:,k-1));
    f=figure('Position',[100 100 1000 260],'Color','w');
    imagesc(reshape(sgn,ny,nx)); axis image; axis off
    colormap([0.80 0.10 0.10; 1 1 1; 0.10 0.35 0.75]); clim([-1 1]);
    cb=colorbar('Ticks',[-2/3 0 2/3],'TickLabels',{'reversing','still','continuing'});
    title(sprintf('F%d  %s -- sign(\\Delta\\rho_k \\cdot \\Delta\\rho_{k-1}) at k = %d   (reversing %.1f%% of elements)', ...
        8+i, labs{i}, k, 100*mean(sgn<0)));
    exportgraphics(f, fullfile(outdir,sprintf('F%d_spatial_%s.png',8+i,keys{i})),'Resolution',150); close(f);
end

% ---- F12: matched-event comparison -------------------------------------
f=figure('Position',[100 100 1000 420],'Color','w');
qn={'q2','cosT','net_ratio'}; ql={'q_2','cos\theta','net/path'};
for j=1:3
    subplot(1,3,j); hold on; grid on
    for i=1:3
        t=T.(keys{i}); k=t.ev.firstDescent; if isnan(k), k=numel(t.per.move); end
        v=t.dyn.(qn{j})(k);
        bar(i, v, 'FaceColor', cols{i});
        text(i, v, sprintf('%.2f',v),'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    set(gca,'XTick',1:3,'XTickLabel',{'160','320','400'});
    ylabel(ql{j}); title(sprintf('%s at first move descent', ql{j}));
end
sgtitle('F12  Dynamical state at the matched event "first move descent"');
exportgraphics(f, fullfile(outdir,'F12_matched_events.png'),'Resolution',150); close(f);
fprintf('[dr_figures] wrote 12 figures to %s\n', outdir);
end

function markEvents(t, n)
d = t.ev.moveDescents; d = d(d<=n);
for k = d, xline(k,'r-','LineWidth',1.0); end
if ~isnan(t.ev.period2Onset)  && t.ev.period2Onset<=n,  xline(t.ev.period2Onset,'m-','LineWidth',2.0); end
end
