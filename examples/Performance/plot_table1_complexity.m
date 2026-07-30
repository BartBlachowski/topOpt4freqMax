function plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C, complexity_exp, outDir, filePrefix, titleBase)
% PLOT_TABLE1_COMPLEXITY  Plot measured Table 1 run times together with the
% fitted power-law curves C * N_e^exp, both on log-log and linear axes.
%
%   plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C, ...
%                           complexity_exp, outDir, filePrefix, titleBase)
%
% Ne             : [nRes x 1] number of elements per mesh
% methodLabels   : {1 x nMethods} cell of legend labels
% tTotal_all     : [nRes x nMethods] total run time (s)
% complexity_C   : [1 x nMethods] fitted prefactor C (NaN if not available)
% complexity_exp : [1 x nMethods] fitted exponent (NaN if not available)
% outDir         : folder where the PNG files are saved
% filePrefix     : (optional) output file basename, default
%                  'table1_complexity_fit'
% titleBase      : (optional) plot title (before the axis-type suffix),
%                  default 'Computational complexity fit:  T(N_e) = C \cdot N_e^{exp}'
%
% Saves:
%   <filePrefix>.png        (log-log axes)
%   <filePrefix>_linear.png (linear axes)

if nargin < 7 || isempty(filePrefix)
    filePrefix = 'table1_complexity_fit';
end
if nargin < 8 || isempty(titleBase)
    titleBase = 'Computational complexity fit:  T(N_e) = C \cdot N_e^{exp}';
end

colors  = [0.0000, 0.4470, 0.7410; ...
           0.8500, 0.3250, 0.0980; ...
           0.4660, 0.6740, 0.1880];
markers = {'o', 's', '^'};

% ---- Log-log plot ----
fig1 = figure('Color', 'white');
ax1 = axes('Parent', fig1);
hold(ax1, 'on');
plotSeries(ax1, Ne, tTotal_all, methodLabels, complexity_C, complexity_exp, colors, markers);
set(ax1, 'XScale', 'log', 'YScale', 'log');
xlabel(ax1, 'Number of elements N_e', 'FontSize', 12);
ylabel(ax1, 'Total run time T (s)', 'FontSize', 12);
title(ax1, [titleBase, '  (log-log)'], 'Interpreter', 'tex', 'FontSize', 12);
grid(ax1, 'on');
box(ax1, 'on');
legend(ax1, 'Location', 'best', 'FontSize', 9);

logPath = fullfile(outDir, [filePrefix, '.png']);
try
    exportgraphics(fig1, logPath, 'Resolution', 180, 'BackgroundColor', 'white');
    fprintf('Complexity fit plot (log-log) saved to: %s\n', logPath);
catch plotErr
    warning('performance_comparison:PlotSaveFailed', ...
        'Failed to save log-log complexity fit plot (%s).', plotErr.message);
end

% ---- Linear-linear plot ----
fig2 = figure('Color', 'white');
ax2 = axes('Parent', fig2);
hold(ax2, 'on');
plotSeries(ax2, Ne, tTotal_all, methodLabels, complexity_C, complexity_exp, colors, markers);
xlabel(ax2, 'Number of elements N_e', 'FontSize', 12);
ylabel(ax2, 'Total run time T (s)', 'FontSize', 12);
title(ax2, [titleBase, '  (linear)'], 'Interpreter', 'tex', 'FontSize', 12);
grid(ax2, 'on');
box(ax2, 'on');
legend(ax2, 'Location', 'best', 'FontSize', 9);

linPath = fullfile(outDir, [filePrefix, '_linear.png']);
try
    exportgraphics(fig2, linPath, 'Resolution', 180, 'BackgroundColor', 'white');
    fprintf('Complexity fit plot (linear) saved to: %s\n', linPath);
catch plotErr
    warning('performance_comparison:PlotSaveFailed', ...
        'Failed to save linear complexity fit plot (%s).', plotErr.message);
end
end

function plotSeries(ax, Ne, tTotal_all, methodLabels, complexity_C, complexity_exp, colors, markers)
nMethods = numel(methodLabels);
for m = 1:nMethods
    validMask = isfinite(tTotal_all(:,m)) & tTotal_all(:,m) > 0 & Ne > 0;
    if ~any(validMask)
        continue;
    end

    % Measured points from Table 1.
    plot(ax, Ne(validMask), tTotal_all(validMask, m), markers{m}, ...
        'MarkerSize', 8, 'LineWidth', 1.5, ...
        'MarkerFaceColor', colors(m,:), 'MarkerEdgeColor', colors(m,:), ...
        'LineStyle', 'none', ...
        'DisplayName', sprintf('%s (data)', methodLabels{m}));

    % Fitted curve C * N_e^exp.
    if ~isnan(complexity_exp(m))
        NeFine = linspace(min(Ne(validMask)), max(Ne(validMask)), 100);
        Tfit = complexity_C(m) * NeFine .^ complexity_exp(m);
        plot(ax, NeFine, Tfit, '-', 'Color', colors(m,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('%s fit: C=%.3g, exp=%.2f', ...
                methodLabels{m}, complexity_C(m), complexity_exp(m)));
    end
end
end
