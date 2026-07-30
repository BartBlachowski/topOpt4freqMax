function extract_legacy_fig_histories()
%EXTRACT_LEGACY_FIG_HISTORIES Extract plotted arrays from pre-existing .fig files.
% These files have unknown run provenance and are intentionally kept separate
% from the regenerated authoritative audit diagnostics.

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(thisDir));
sourceDir = fullfile(repoRoot, 'results');
outDir = fullfile(thisDir, 'results', 'legacy_fig_extracted');
if ~exist(outDir, 'dir'), mkdir(outDir); end

figFiles = dir(fullfile(sourceDir, '*_freq_iterations.fig'));
inventory = table('Size', [numel(figFiles), 4], ...
    'VariableTypes', {'string','string','double','double'}, ...
    'VariableNames', {'SourceFig','ExtractedCsv','HistoryLength','SourceDatenum'});

for k = 1:numel(figFiles)
    sourcePath = fullfile(figFiles(k).folder, figFiles(k).name);
    fig = openfig(sourcePath, 'invisible');
    cleanup = onCleanup(@() close(fig));
    lines = findobj(fig, 'Type', 'Line');

    omega = cell(1, 3);
    for j = 1:numel(lines)
        name = string(get(lines(j), 'DisplayName'));
        y = get(lines(j), 'YData');
        if contains(name, '\omega_{1}'), omega{1} = y(:); end
        if contains(name, '\omega_{2}'), omega{2} = y(:); end
        if contains(name, '\omega_{3}'), omega{3} = y(:); end
    end
    lengths = cellfun(@numel, omega);
    n = max(lengths);
    if n == 0
        warning('No frequency lines found in %s', sourcePath);
        continue;
    end
    values = NaN(n, 3);
    for j = 1:3
        values(1:lengths(j),j) = omega{j};
    end
    T = array2table([(1:n)', values], ...
        'VariableNames', {'Iteration','Omega1','Omega2','Omega3'});
    [~, base] = fileparts(figFiles(k).name);
    csvPath = fullfile(outDir, [base '.csv']);
    writetable(T, csvPath);

    inventory.SourceFig(k) = string(sourcePath);
    inventory.ExtractedCsv(k) = string(csvPath);
    inventory.HistoryLength(k) = n;
    inventory.SourceDatenum(k) = figFiles(k).datenum;
    clear cleanup;
end

writetable(inventory, fullfile(outDir, 'inventory.csv'));
disp(inventory);
end
