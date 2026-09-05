function plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C, complexity_exp, outDir, filePrefix, titleBase, fitMask)
% PLOT_TABLE1_COMPLEXITY  Plot measured Table 1 run times together with the
% fitted power-law curves C * N_e^exp, both on log-log and linear axes.
%
%   plot_table1_complexity(Ne, methodLabels, tTotal_all, complexity_C, ...
%                           complexity_exp, outDir, filePrefix, titleBase, fitMask)
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
% fitMask        : (optional) [nRes x nMethods] logical, true where the point
%                  ENTERED the fit.  Points that did not are drawn as hollow
%                  markers and labelled "(excluded from fit)", so a censored or
%                  refused row can never be mistaken for one the curve was
%                  fitted through.  Default: all true (previous behaviour).
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
if nargin < 9 || isempty(fitMask)
    fitMask = true(size(tTotal_all));
end

colors  = [0.0000, 0.4470, 0.7410; ...
           0.8500, 0.3250, 0.0980; ...
           0.4660, 0.6740, 0.1880];
markers = {'o', 's', '^'};

% ---- Log-log plot ----
fig1 = figure('Color', 'white');
ax1 = axes('Parent', fig1);
hold(ax1, 'on');
plotSeries(ax1, Ne, tTotal_all, methodLabels, complexity_C, complexity_exp, colors, markers, fitMask);
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
plotSeries(ax2, Ne, tTotal_all, methodLabels, complexity_C, complexity_exp, colors, markers, fitMask);
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

function plotSeries(ax, Ne, tTotal_all, methodLabels, complexity_C, complexity_exp, colors, markers, fitMask)
nMethods = numel(methodLabels);
for m = 1:nMethods
    ci = mod(m-1, size(colors,1)) + 1;
    mk = markers{mod(m-1, numel(markers)) + 1};

    validMask = isfinite(tTotal_all(:,m)) & tTotal_all(:,m) > 0 & Ne > 0;
    if ~any(validMask)
        continue;
    end
    usedMask = validMask &  fitMask(:,m);
    dropMask = validMask & ~fitMask(:,m);

    % Measured points that entered the fit.
    if any(usedMask)
        plot(ax, Ne(usedMask), tTotal_all(usedMask, m), mk, ...
            'MarkerSize', 8, 'LineWidth', 1.5, ...
            'MarkerFaceColor', colors(ci,:), 'MarkerEdgeColor', colors(ci,:), ...
            'LineStyle', 'none', ...
            'DisplayName', sprintf('%s (data)', methodLabels{m}));
    end

    % Measured points that were refused by the fit rule.  Same colour and
    % marker, hollow, so they read as the same series without pretending the
    % curve was fitted through them.
    if any(dropMask)
        plot(ax, Ne(dropMask), tTotal_all(dropMask, m), mk, ...
            'MarkerSize', 8, 'LineWidth', 1.5, ...
            'MarkerFaceColor', 'none', 'MarkerEdgeColor', colors(ci,:), ...
            'LineStyle', 'none', ...
            'DisplayName', sprintf('%s (excluded from fit)', methodLabels{m}));
    end

    % Fitted curve C * N_e^exp, drawn across every measured mesh so that an
    % excluded point's departure from the model is visible rather than cropped.
    if ~isnan(complexity_exp(m))
        NeFine = linspace(min(Ne(validMask)), max(Ne(validMask)), 100);
        Tfit = complexity_C(m) * NeFine .^ complexity_exp(m);
        plot(ax, NeFine, Tfit, '-', 'Color', colors(ci,:), 'LineWidth', 2, ...
            'DisplayName', sprintf('%s fit: C=%.3g, exp=%.2f', ...
                methodLabels{m}, complexity_C(m), complexity_exp(m)));
    end
end
end
