function [pngPath, figPath] = save_frequency_iteration_plot(freqIterOmega, approachName, nelx, nely, outDir)
%SAVE_FREQUENCY_ITERATION_PLOT  Plot omega_1..omega_3 against outer iteration.
%
%   [pngPath, figPath] = SAVE_FREQUENCY_ITERATION_PLOT(freqIterOmega, approachName, nelx, nely, outDir)
%
%   freqIterOmega : nIter x k matrix of circular frequencies [rad/s]; the first
%                   three columns are plotted (missing columns are NaN-padded).
%   approachName  : shown in the title as "<name> frequency history" and used in
%                   the file names <name>_<nelx>x<nely>_freq_iterations.{png,fig}.
%   outDir        : destination folder, created when missing.
%
%   Returns the written paths, or '' for a file that could not be written.
%   Extracted from run_topopt_from_json so that runners whose solver is not
%   dispatched through it (e.g. analysis/Olhoff) produce the same figure.

    pngPath = '';
    figPath = '';
    if isempty(freqIterOmega)
        return;
    end
    if ~isnumeric(freqIterOmega)
        warning('run_topopt_from_json:InvalidFrequencyHistoryType', ...
            'Frequency history must be numeric to save iteration plot.');
        return;
    end

    freqIterOmega = double(freqIterOmega);
    nIter = size(freqIterOmega, 1);
    if nIter < 1
        return;
    end
    if size(freqIterOmega, 2) < 3
        tmp = NaN(nIter, 3);
        tmp(:,1:size(freqIterOmega,2)) = freqIterOmega;
        freqIterOmega = tmp;
    else
        freqIterOmega = freqIterOmega(:,1:3);
    end

    if exist(outDir, 'dir') ~= 7
        mkdir(outDir);
    end

    nameRaw = char(string(approachName));
    nameDisplay = strrep(nameRaw, '_', ' ');
    nameSafe = regexprep(nameRaw, '[^\w\-]', '_');
    outPath = fullfile(outDir, sprintf('%s_%dx%d_freq_iterations.png', nameSafe, nelx, nely));

    fig = figure('Color', 'white', 'Visible', 'off');
    if exist('theme', 'file') == 2 || exist('theme', 'builtin') == 5
        try
            theme("light");
        catch
            % Some MATLAB releases/toolboxes may not expose theme in scripts.
        end
    end
    ax = axes('Parent', fig);
    set(ax, 'FontSize', 22);
    hold(ax, 'on');
    colors = [0.0000, 0.4470, 0.7410; ...
              0.8500, 0.3250, 0.0980; ...
              0.4660, 0.6740, 0.1880];
    xIter = (1:nIter)';
    for j = 1:3
        plot(ax, xIter, freqIterOmega(:,j), '-', 'LineWidth', 3.2, ...
            'Color', colors(j,:), 'DisplayName', sprintf('\\omega_{%d}', j));
    end

    xlabel(ax, 'Outer iteration', 'FontSize', 22);
    ylabel(ax, 'Frequency (rad/s)', 'FontSize', 22);
    title(ax, sprintf('%s frequency history', nameDisplay), 'Interpreter', 'none', 'FontSize', 22);
    grid(ax, 'on');
    box(ax, 'on');
    % MATLAB requires strictly increasing limits; handle single-iteration runs.
    if nIter == 1
        xlim(ax, [0.5, 1.5]);
    else
        xlim(ax, [1, nIter]);
    end
    legend(ax, 'Location', 'best', 'FontSize', 22);

    try
        exportgraphics(fig, outPath, 'Resolution', 180, 'BackgroundColor', 'white');
        pngPath = outPath;
    catch pngErr
        warning('run_topopt_from_json:ExportGraphicsFailed', ...
            'exportgraphics failed (%s); falling back to print().', pngErr.message);
        try
            print(fig, outPath, '-dpng', '-r180');
            pngPath = outPath;
        catch pngErr2
            warning('run_topopt_from_json:PrintFailed', ...
                'Failed to save frequency iteration PNG (%s).', pngErr2.message);
        end
    end
    outFig = fullfile(outDir, sprintf('%s_%dx%d_freq_iterations.fig', nameSafe, nelx, nely));
    try
        set(fig, 'Visible', 'on');   % savefig records visibility; keep it 'on' so the file opens correctly
        savefig(fig, outFig);
        figPath = outFig;
    catch figErr
        warning('run_topopt_from_json:SaveFigFailed', ...
            'Failed to save MATLAB figure file (%s).', figErr.message);
    end
    close(fig);
    if ~isempty(pngPath)
        fprintf('Saved frequency iteration plot: %s\n', pngPath);
    end
    if ~isempty(figPath)
        fprintf('Saved frequency iteration figure: %s\n', figPath);
    end
end
