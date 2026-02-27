function [corr, rowLabels, colLabels] = readCorrelationCSV(filename)
%READCORRELATIONCSV  Load a correlation CSV written by run_topopt_from_json.
%
%   [corr, rowLabels, colLabels] = readCorrelationCSV(filename)
%
% Inputs:
%   filename  - path to a correlation CSV file
%
% Outputs:
%   corr      - (nI x nT) double matrix of correlation values in [0,1]
%   rowLabels - (nI x 1) cell array of initial-mode label strings
%   colLabels - (nT x 1) cell array of topology-mode label strings
%
% The expected CSV formats (produced by run_topopt_from_json):
%   New:    "initial_mode,max_correlation_mode,topology_mode_1(omega1),..."
%           "initial_mode_i(omegaI),bestMode,c_i1,c_i2,..."
%   Legacy: "initial_mode,topology_mode_1(omega1),topology_mode_2(omega2),..."
%           "initial_mode_i(omegaI),c_i1,c_i2,..."
%
% Example:
%   [C, rows, cols] = readCorrelationCSV('BeamTopOptFreq_correlation.csv');
%   imagesc(C); colorbar;

    if ~isfile(filename)
        error('readCorrelationCSV:FileNotFound', 'File not found: %s', filename);
    end

    fid = fopen(filename, 'r');
    if fid < 0
        error('readCorrelationCSV:CannotOpen', 'Cannot open file: %s', filename);
    end
    cleanupObj = onCleanup(@() fclose(fid));

    % --- Read header row ---
    headerLine = fgetl(fid);
    if ~ischar(headerLine)
        error('readCorrelationCSV:EmptyFile', 'File is empty: %s', filename);
    end
    headerCells = strsplit(strtrim(headerLine), ',');
    if numel(headerCells) < 2
        error('readCorrelationCSV:MalformedHeader', ...
            'Header must have at least 2 columns (row-label + one topology mode).');
    end
    hasBestModeCol = numel(headerCells) >= 2 && ...
        strcmpi(strtrim(headerCells{2}), 'max_correlation_mode');
    if hasBestModeCol && numel(headerCells) < 3
        error('readCorrelationCSV:MalformedHeader', ...
            'Header with max_correlation_mode must include at least one topology mode column.');
    end
    if hasBestModeCol
        colStart = 3;  % skip row-label and max-correlation metadata columns
    else
        colStart = 2;  % legacy format: skip only row-label header
    end
    colLabels = headerCells(colStart:end)';   % (nT x 1), topology-mode headers only
    nT = numel(colLabels);
    expectedCols = colStart - 1 + nT;

    % --- Read data rows ---
    corr      = zeros(0, nT);
    rowLabels = {};
    lineNum   = 1;
    while true
        line = fgetl(fid);
        if ~ischar(line), break; end
        lineNum = lineNum + 1;
        line = strtrim(line);
        if isempty(line), continue; end

        cells = strsplit(line, ',');
        if numel(cells) < expectedCols
            warning('readCorrelationCSV:MalformedRow', ...
                'Line %d has %d columns but expected %d (skipped).', ...
                lineNum, numel(cells), expectedCols);
            continue;
        end

        rowLabels{end+1, 1} = strtrim(cells{1}); %#ok<AGROW>
        vals = cellfun(@str2double, cells(colStart : colStart + nT - 1));
        corr(end+1, :) = vals; %#ok<AGROW>
    end

    % --- Validate ---
    if isempty(corr)
        error('readCorrelationCSV:NoData', 'No data rows found in: %s', filename);
    end

    corr = double(corr);
    [nI, nTActual] = size(corr);

    if nTActual ~= nT
        warning('readCorrelationCSV:DimensionMismatch', ...
            'Header has %d topology columns but rows have %d values.', nT, nTActual);
    end
    if numel(rowLabels) ~= nI
        warning('readCorrelationCSV:LabelMismatch', ...
            'Row label count (%d) does not match data row count (%d).', ...
            numel(rowLabels), nI);
    end
end
