function fm_figures(A, outdir)
%FM_FIGURES  The twelve required figures (Phase 19).
if ~exist(outdir,'dir'), mkdir(outdir); end
t4 = A.T.f400; p4 = t4.per; d4 = t4.dyn; on4 = t4.C.onset.PERIOD2;
kProd = 138; nC = numel(p4.move);
col4 = [0.47 0.67 0.19];

    function marks(n)
        xline(kProd,'r-','LineWidth',1.6);
        if ~isnan(on4) && on4<=n, xline(on4,'m-','LineWidth',2.0); end
        if ~isnan(A.runC.nativeStopIter), xline(A.runC.nativeStopIter,'--','Color',[0.5 0.5 0.5],'LineWidth',1.2); end
    end

% F1 cos(theta) raw ; F2 cos unsat ; F3 net/path ; F4 M_nd ; F5 omega1
specs = { {d4.cosT,'cos\theta (raw)','F1_cos_raw',[-1 1]}, ...
          {d4.cosT_unsat,'cos\theta (unsaturated only)','F2_cos_unsat',[-1 1]}, ...
          {d4.net_ratio,'net / path (W=10)','F3_netpath',[0 1]}, ...
          {p4.Mnd,'M_{nd} [%]','F4_Mnd',[]}, ...
          {p4.omega1,'\omega_1','F5_omega1',[]} };
for i = 1:numel(specs)
    s = specs{i}; f = figure('Position',[100 100 980 340],'Color','w'); hold on; grid on
    plot(s{1},'-','Color',col4,'LineWidth',0.9);
    if i<=3, plot(movmedian(s{1},20,'omitnan'),'k-','LineWidth',1.6); end
    if i<=2, yline(0,'k:'); end
    marks(nC); if ~isempty(s{4}), ylim(s{4}); end
    xlabel('outer iteration'); ylabel(s{2});
    if isnan(on4), onTxt = 'NONE'; else, onTxt = mat2str(on4); end
    title(sprintf('%s  400\\times50 fixed move 0.04 -- %s   (red = production descent 138, grey dashed = native stop 369, onset = %s)', ...
        s{3}(1:2), s{2}, onTxt));
    exportgraphics(f, fullfile(outdir,[s{3} '.png']),'Resolution',150); close(f);
end

% F6 common-prefix comparison
DIAGD = fileparts(fileparts(fileparts(mfilename('fullpath'))));
PA = load(fullfile(DIAGD,'dynamical_regime','runs','runA_400x50.mat'));
q = PA.out.per; n = numel(q.move);
f = figure('Position',[100 100 980 640],'Color','w');
subplot(2,1,1); hold on; grid on
plot(1:n, q.Mnd(1:n),'-','Color',[0.2 0.35 0.75],'LineWidth',2.2);
plot(1:min(nC,260), p4.Mnd(1:min(nC,260)),'--','Color',col4,'LineWidth',1.4);
xline(kProd,'r-','LineWidth',1.6); xlim([1 260]);
legend({'production (RUN A)','fixed move (RUN C)'},'Location','northeast');
ylabel('M_{nd} [%]'); title('F6  Common prefix: identical to iteration 137, diverging at the production descent (138)');
subplot(2,1,2); hold on; grid on
dd = arrayfun(@(k) max(abs(A.T.f400.RHO(:,k)-PA.RHO(:,k))), 1:n);
semilogy(max(dd,1e-18),'k-','LineWidth',1.2); set(gca,'YScale','log');
xline(kProd,'r-','LineWidth',1.6); xlim([1 n]);
xlabel('outer iteration'); ylabel('max |\Delta\rho| between arms');
exportgraphics(f, fullfile(outdir,'F6_common_prefix.png'),'Resolution',150); close(f);

% F7 production descent and onset marked together on move + cos
f = figure('Position',[100 100 980 560],'Color','w');
subplot(2,1,1); hold on; grid on
stairs(1:n, q.move(1:n),'-','Color',[0.2 0.35 0.75],'LineWidth',2.0);
stairs(p4.move,'--','Color',col4,'LineWidth',1.4);
marks(nC); set(gca,'YScale','log'); ylabel('move');
legend({'production','fixed move'},'Location','east');
title('F7  Production \beta-descent (red, 138) vs fixed-move cancellation onset (magenta)');
subplot(2,1,2); hold on; grid on
plot(movmedian(d4.cosT,20,'omitnan'),'k-','LineWidth',1.6); yline(0,'k:');
marks(nC); ylim([-1 1]); xlabel('outer iteration'); ylabel('median cos\theta');
exportgraphics(f, fullfile(outdir,'F7_descent_vs_onset.png'),'Resolution',150); close(f);

% F8-F10 cross-mesh aligned at onset
keys = {'f160','f320','f400'}; labs = {'160\times20','320\times40','400\times50'};
cols = {[0.85 0.33 0.10],[0 0.45 0.74],col4};
qty  = { {'cosT','cos\theta','F8_xmesh_cos',[-1 1]}, ...
         {'net_ratio','net / path','F9_xmesh_netpath',[0 1]}, ...
         {'Mnd','M_{nd} [%]','F10_xmesh_Mnd',[]} };
for j = 1:3
    f = figure('Position',[100 100 980 380],'Color','w'); hold on; grid on
    for i = 1:3
        t = A.T.(keys{i}); o = t.C.onset.PERIOD2;
        if isnan(o)
            o = A.runC.nativeStopIter;      % 400x50 has NO onset: align at native stop
            if isnan(o), continue; end
        end
        if strcmp(qty{j}{1},'Mnd'), y = t.per.Mnd; else, y = movmedian(t.dyn.(qty{j}{1}),20,'omitnan'); end
        x = (1:numel(y)) - o;
        m = x >= -200 & x <= 500;
        plot(x(m), y(m), '-', 'Color', cols{i}, 'LineWidth',1.6);
    end
    xline(0,'m-','LineWidth',2.0,'Label','cancellation onset');
    if j<=2, yline(0,'k:'); end
    if ~isempty(qty{j}{4}), ylim(qty{j}{4}); end
    xlabel('iterations relative to cancellation onset'); ylabel(qty{j}{2});
    legend({'160\times20 (onset 81)','320\times40 (onset 253)','400\times50 (NO onset; aligned at native stop 369)'},'Location','best');
    title(sprintf('%s  Cross-mesh %s, aligned at cancellation onset (400\times50 has none)', qty{j}{3}(1:3), qty{j}{2}));
    exportgraphics(f, fullfile(outdir,[qty{j}{3} '.png']),'Resolution',150); close(f);
end

% F11 spatial reversal maps at four stages
nx=400; ny=50; X=[0.5*ones(nx*ny,1), t4.RHO]; D=diff(X,1,2);
if isnan(on4)
    kN = A.runC.nativeStopIter;
    stages = [kProd, kN, 600, nC-1];
    names  = {'production descent (138)','native stop would fire (369)','M_{nd} plateau (600)','terminal state (1199)'};
else
    stages = [kProd, max(2,round(on4*0.75)), on4, min(nC-1, on4+300)];
    names  = {'production descent (138)','shortly before onset','cancellation onset','established regime'};
end
f = figure('Position',[80 60 1150 760],'Color','w');
for i = 1:4
    k = stages(i); if isnan(k)||k<2, continue; end
    sgn = sign(D(:,k).*D(:,k-1));
    subplot(4,1,i); imagesc(reshape(sgn,ny,nx)); axis image; axis off
    colormap([0.80 0.10 0.10; 1 1 1; 0.10 0.35 0.75]); clim([-1 1]);
    title(sprintf('%s  k=%d   reversing %.1f%% of elements, boundFrac %.4f', ...
        names{i}, k, 100*mean(sgn<0), d4.boundFrac(k)));
end
sgtitle('F11  400\times50 sign(\Delta\rho_k \cdot \Delta\rho_{k-1}):  red = reversing, blue = continuing, white = still');
exportgraphics(f, fullfile(outdir,'F11_spatial_reversal.png'),'Resolution',150); close(f);

% F12 confirmation tail
f = figure('Position',[100 100 980 560],'Color','w');
subplot(2,1,1); hold on; grid on
plot(p4.Mnd,'-','Color',col4,'LineWidth',1.4); marks(nC);
if isnan(on4)
    ylabel('M_{nd} [%]');
    title('F12  Post-descent tail: NO cancellation onset in 1200 iterations -- M_{nd} plateaus and then slowly worsens');
    yline(min(p4.Mnd),':','Color',[0.4 0.4 0.4],'LineWidth',1.2,'Label','best M_{nd} (15.05 at k=547)');
    xlim([100 nC]);
else
    ylabel('M_{nd} [%]');
    title(sprintf('F12  Confirmation tail after onset (%d iterations)', nC-on4));
    yline(p4.Mnd(on4),':','Color',[0.4 0.4 0.4],'LineWidth',1.2,'Label','M_{nd} at onset');
    xlim([max(1,on4-150) nC]);
end
subplot(2,1,2); hold on; grid on
plot(p4.omega1,'-','Color',[0.6 0.2 0.6],'LineWidth',1.4); marks(nC);
if isnan(on4), xlim([100 nC]); else, xlim([max(1,on4-150) nC]); end
xlabel('outer iteration'); ylabel('\omega_1');
exportgraphics(f, fullfile(outdir,'F12_confirmation_tail.png'),'Resolution',150); close(f);
fprintf('[fm_figures] wrote 12 figures\n');
end
