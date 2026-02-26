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
% The expected CSV format (produced by run_topopt_from_json):
%   First row: "initial_mode,topology_mode_1(omega1),topology_mode_2(omega2),..."
%   Subsequent rows: "initial_mode_i(omegaI),c_i1,c_i2,..."
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
    colLabels = headerCells(2:end)';   % (nT x 1), skip first (row-label header)
    nT = numel(colLabels);

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
        if numel(cells) < nT + 1
            warning('readCorrelationCSV:MalformedRow', ...
                'Line %d has %d columns but expected %d (skipped).', ...
                lineNum, numel(cells), nT + 1);
            continue;
        end

        rowLabels{end+1, 1} = strtrim(cells{1}); %#ok<AGROW>
        vals = cellfun(@str2double, cells(2 : nT + 1));
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
