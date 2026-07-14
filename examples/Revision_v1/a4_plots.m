function figs = a4_plots(outDir, res)
%A4_PLOTS  Figure generation for A4 (A4_SPECIFICATION_V3 §7.6).
%
%   figs = A4_PLOTS(outDir, res)
%
%   Figure 1 is the PRIMARY figure: omega1_tracked vs N, with the +/-delta
%   equivalence band around the frozen (N=inf) arm.
%
%   GOVERNANCE (spec §7.6): arms classified B3 (contaminated) or B4 (unstable)
%   are plotted in a visually distinct, EXPLICITLY DISQUALIFIED style -- shown,
%   but never allowed to read as accuracy evidence.  This is the graphical form
%   of the EXP4 lesson: the -62% point was real, and publishing it as an
%   accuracy result was the error.
%
%   Headless-safe: figures are created invisible and closed after saving.

figs = {};
if ~exist(outDir, 'dir'), mkdir(outDir); end

arms = res.arms;
if isempty(arms), return; end

Ns = [arms.N];
% Plot on a finite axis: inf -> the largest finite N, times 2 (log-friendly).
finiteNs = Ns(~isinf(Ns));
if isempty(finiteNs), finiteNs = 1; end
infPos = max(finiteNs) * 2;
xpos = Ns; xpos(isinf(xpos)) = infPos;

clean = arrayfun(@(a) strcmp(a.class, 'ACCEPTED'), arms);
disq  = arrayfun(@(a) any(strcmp(a.breakdown, {'B3','B4'})), arms);

omega = [arms.omega1_tracked];
omegaMin = [arms.omega1_min];

% ---- Figure 1: PRIMARY -- omega1_tracked vs N ---------------------------
f = figure('Visible', 'off', 'Position', [100 100 900 560]);
hold on; grid on;

ref = res.decision.reference;
if isfinite(ref)
    xl = [min(xpos)/1.5, infPos*1.5];
    fill([xl(1) xl(2) xl(2) xl(1)], ...
         [ref*(1-res.delta) ref*(1-res.delta) ref*(1+res.delta) ref*(1+res.delta)], ...
         [0.85 0.92 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.6, ...
         'DisplayName', sprintf('\\pm%.0f%% equivalence band', 100*res.delta));
    yline(ref, 'k--', 'LineWidth', 1.2, 'DisplayName', 'frozen (N=\infty) reference');
end

if any(clean)
    plot(xpos(clean), omega(clean), 'o', 'MarkerSize', 10, 'LineWidth', 2, ...
        'MarkerFaceColor', [0.20 0.45 0.75], 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'ACCEPTED (eligible as accuracy reference)');
end
if any(disq)
    % Disqualified arms: red crosses, explicitly labelled. NOT accuracy evidence.
    plot(xpos(disq), omega(disq), 'x', 'MarkerSize', 14, 'LineWidth', 3, ...
        'Color', [0.80 0.10 0.10], ...
        'DisplayName', 'DISQUALIFIED (B3/B4) — NOT accuracy evidence');
end
other = ~clean & ~disq;
if any(other)
    plot(xpos(other), omega(other), 's', 'MarkerSize', 9, 'LineWidth', 1.5, ...
        'MarkerFaceColor', [0.95 0.75 0.20], 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'breakdown B1/B2 (a result)');
end

set(gca, 'XScale', 'log');
set(gca, 'XTick', sort(unique(xpos)));
lbl = arrayfun(@(a) a.tag, arms, 'UniformOutput', false);
[~, ord] = sort(xpos);
set(gca, 'XTickLabel', lbl(ord));
xlabel('refresh interval N  (\infty = frozen = the published method)');
ylabel('\omega_1 of the tracked \Phi_1-type mode  [rad/s]   (TRUE, exact eigensolve)');
title({'A4 — accuracy cost of freezing the reference eigenpair', ...
       sprintf('decision: %s', strrep(res.decision.outcome, '_', ' '))}, ...
       'Interpreter', 'none');
legend('Location', 'best');
p = fullfile(outDir, 'a4_fig1_omega1_vs_N.png');
exportgraphics(f, p, 'Resolution', 150); close(f);
figs{end+1} = p;

% ---- Figure 2: omega1_tracked vs omega1_min (contamination signature) ----
f = figure('Visible', 'off', 'Position', [100 100 900 520]);
hold on; grid on;
b = bar(1:numel(arms), [omega(:), omegaMin(:)], 'grouped');
b(1).DisplayName = '\omega_1 tracked (\Phi_1-type)';
b(2).DisplayName = '\omega_1 min (lowest mode, whatever it is)';
set(gca, 'XTick', 1:numel(arms), 'XTickLabel', {arms.tag});
xlabel('refresh interval N'); ylabel('\omega_1  [rad/s]');
title({'A4 — spurious-mode signature', ...
       'a large gap means a non-physical mode descended below the design mode'});
legend('Location', 'best');
p = fullfile(outDir, 'a4_fig2_tracked_vs_min.png');
exportgraphics(f, p, 'Resolution', 150); close(f);
figs{end+1} = p;

% ---- Figure 3: refresh events (MAC continuity + admissibility) ----------
f = figure('Visible', 'off', 'Position', [100 100 950 560]);
hold on; grid on;
anyEvents = false;
for i = 1:numel(arms)
    ev = arms(i).refresh_events;
    if isempty(ev), continue; end
    anyEvents = true;
    plot([ev.iter], [ev.mac_phi0], '-o', 'LineWidth', 1.5, 'MarkerSize', 4, ...
        'DisplayName', sprintf('N = %s', arms(i).tag));
end
if anyEvents
    yline(0.8, 'r--', 'LineWidth', 1.2, 'DisplayName', 'MAC threshold 0.8');
    xlabel('iteration'); ylabel('MAC( refreshed \Phi , solid \Phi_0 )');
    title({'A4 — refresh events: does the refreshed reference stay the same physical mode?', ...
           'each marker is one recorded refresh event'});
    legend('Location', 'best');
else
    text(0.5, 0.5, 'no refresh events (all arms frozen)', 'HorizontalAlignment', 'center');
    axis off;
end
p = fullfile(outDir, 'a4_fig3_refresh_events.png');
exportgraphics(f, p, 'Resolution', 150); close(f);
figs{end+1} = p;

% ---- Figure 4: final topologies ----------------------------------------
haveTopo = arrayfun(@(a) ~isempty(a.topology), arms);
if any(haveTopo)
    idx = find(haveTopo);
    f = figure('Visible', 'off', 'Position', [100 100 1000 130*numel(idx)+80]);
    % nelx/nely are fixed across arms by construction (single base config).
    nelx = 400; nely = 50;
    if isfield(res, 'nelx'), nelx = res.nelx; end
    if isfield(res, 'nely'), nely = res.nely; end
    for k = 1:numel(idx)
        a = arms(idx(k));
        subplot(numel(idx), 1, k);
        try
            imagesc(reshape(1 - a.topology, nely, nelx));
        catch
            imagesc(1 - a.topology(:)');
        end
        colormap(gray); axis equal off;
        ttl = sprintf('N = %s   \\omega_1 = %.2f   %s%s', ...
            a.tag, a.omega1_tracked, a.class, localBd(a.breakdown));
        title(ttl, 'Interpreter', 'tex');
    end
    p = fullfile(outDir, 'a4_fig4_topologies.png');
    exportgraphics(f, p, 'Resolution', 150); close(f);
    figs{end+1} = p;
end
end

function s = localBd(bd)
if isempty(bd), s = ''; else, s = ['/' bd]; end
end
