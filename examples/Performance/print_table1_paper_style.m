function print_table1_paper_style(resolutions, groupLabels, tIter_all, nIter_all, tTotal_all, texPath, nIterStage1_all, nIterStage2_all)
% PRINT_TABLE1_PAPER_STYLE  Print the run-time comparison table with mesh
% resolutions as rows and one column GROUP per method (each group holding
% t_iter, n_iter, T (s) sub-columns), matching the paper's Table 1 layout:
%
%                Olhoff-Du          Yuksel-Yilmaz         Proposed
%   Mesh     t_iter n_iter T(s)   t_iter n_iter T(s)   t_iter n_iter T(s)
%
%   print_table1_paper_style(resolutions, groupLabels, tIter_all, ...
%                             nIter_all, tTotal_all, texPath, ...
%                             nIterStage1_all, nIterStage2_all)
%
% resolutions : [nRes x 2] (nelx, nely) per row
% groupLabels : {1 x nMethods} method group names, e.g.
%               {'Olhoff--Du', 'Yuksel--Yilmaz', 'Proposed'}
%               ('--' is rendered as an en-dash in the LaTeX export and
%               collapsed to '-' for the console table)
% tIter_all   : [nRes x nMethods] run time per iteration (s/iter)
% nIter_all   : [nRes x nMethods] number of iterations
% tTotal_all  : [nRes x nMethods] total run time (s)
% texPath     : (optional) if given, also write a booktabs LaTeX table there
% nIterStage1_all, nIterStage2_all : (optional) [nRes x nMethods] two-stage
%               iteration split (Yuksel). Where both are finite the n_iter
%               cell is printed as "n_total (n_stage1 + n_stage2)"; NaN
%               entries fall back to the plain total.

nRes     = size(resolutions, 1);
nGroups  = numel(groupLabels);

if nargin < 7 || isempty(nIterStage1_all)
    nIterStage1_all = NaN(nRes, nGroups);
end
if nargin < 8 || isempty(nIterStage2_all)
    nIterStage2_all = NaN(nRes, nGroups);
end

nIterStr_all = cell(nRes, nGroups);
for r = 1:nRes
    for g = 1:nGroups
        if isnan(tTotal_all(r,g))
            nIterStr_all{r,g} = 'N/A';
        else
            nIterStr_all{r,g} = formatIterCount(nIter_all(r,g), ...
                nIterStage1_all(r,g), nIterStage2_all(r,g));
        end
    end
end

% The stage split widens the n_iter column; size it to the widest cell.
nIterW = max([8, cellfun(@length, nIterStr_all(:))']);

colW = [8, nIterW, 9];      % widths for t_iter, n_iter, T (s)
gap  = '  ';
groupGap = '   ';
groupWidth = sum(colW) + 2*length(gap);
meshW = 10;

fprintf('\n');
fprintf('Table 1. Run time comparison between methods for maximizing the first\n');
fprintf('natural frequency of a simply supported beam (paper-style layout).\n');
fprintf('\n');

% ---- Header line 1: method group names ----
line1 = sprintf('%-*s', meshW, '');
for g = 1:nGroups
    consoleLabel = strrep(groupLabels{g}, '--', '-');
    line1 = [line1, groupGap, centerStr(consoleLabel, groupWidth)]; %#ok<AGROW>
end

% ---- Underline beneath each group name ----
line1u = sprintf('%-*s', meshW, '');
for g = 1:nGroups
    line1u = [line1u, groupGap, repmat('-', 1, groupWidth)]; %#ok<AGROW>
end

% ---- Header line 2: sub-column names ----
line2 = sprintf('%-*s', meshW, 'Mesh');
for g = 1:nGroups
    line2 = [line2, groupGap, formatTriple('t_iter', 'n_iter', 'T (s)', colW, gap)]; %#ok<AGROW>
end

fullSep = repmat('-', 1, length(line2));

fprintf('%s\n', line1);
fprintf('%s\n', line1u);
fprintf('%s\n', line2);
fprintf('%s\n', fullSep);

for r = 1:nRes
    meshStr = sprintf('%dx%d', resolutions(r,1), resolutions(r,2));
    rowStr = sprintf('%-*s', meshW, meshStr);
    for g = 1:nGroups
        if isnan(tTotal_all(r,g))
            rowStr = [rowStr, groupGap, formatTriple('N/A', 'N/A', 'N/A', colW, gap)]; %#ok<AGROW>
        else
            rowStr = [rowStr, groupGap, formatTriple( ...
                sprintf('%.2f', tIter_all(r,g)), ...
                nIterStr_all{r,g}, ...
                sprintf('%.1f', tTotal_all(r,g)), colW, gap)]; %#ok<AGROW>
        end
    end
    fprintf('%s\n', rowStr);
end
fprintf('%s\n', fullSep);
fprintf('\n');

if nargin >= 6 && ~isempty(texPath)
    writeLatexTable(texPath, resolutions, groupLabels, tIter_all, nIterStr_all, tTotal_all);
    fprintf('Paper-style LaTeX table saved to: %s\n', texPath);
end
end

function s = formatIterCount(nTotal, nStage1, nStage2)
% Two-stage methods report "total (stage1 + stage2)"; single-stage methods
% keep the bare total.
if isnan(nTotal)
    s = 'N/A';
elseif isnan(nStage1) || isnan(nStage2)
    s = sprintf('%d', nTotal);
else
    % A truncated stage 1 silently inflates stage 2 while leaving the total
    % plausible, so the decomposition is asserted wherever it is printed.
    assert(nStage1 + nStage2 == nTotal, ...
        'print_table1_paper_style:StageSumMismatch', ...
        'Stage 1 + Stage 2 (%d + %d = %d) must equal the total (%d).', ...
        nStage1, nStage2, nStage1 + nStage2, nTotal);
    s = sprintf('%d (%d + %d)', nTotal, nStage1, nStage2);
end
end

function s = centerStr(str, width)
n = length(str);
if n >= width
    s = str(1:width);
    return;
end
totalPad = width - n;
leftPad  = floor(totalPad/2);
rightPad = totalPad - leftPad;
s = [repmat(' ', 1, leftPad), str, repmat(' ', 1, rightPad)];
end

function s = formatTriple(a, b, c, colW, gap)
s = sprintf('%*s%s%*s%s%*s', colW(1), a, gap, colW(2), b, gap, colW(3), c);
end

function writeLatexTable(texPath, resolutions, groupLabels, tIter_all, nIterStr_all, tTotal_all)
nRes    = size(resolutions, 1);
nGroups = numel(groupLabels);

fid = fopen(texPath, 'w');
cleanupObj = onCleanup(@() fclose(fid));

fprintf(fid, '%% Auto-generated by print_table1_paper_style.m -- do not edit by hand\n');
fprintf(fid, '\\begin{table}[t]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Run time comparison between methods for maximizing the first natural frequency of a simply supported beam.}\n');
fprintf(fid, '\\label{tab:performance}\n');
fprintf(fid, '\\begin{tabular}{l%s}\n', repmat('c', 1, 3*nGroups));
fprintf(fid, '\\toprule\n');

headerRow1 = '';
cmidrules  = '';
col = 2;
for g = 1:nGroups
    headerRow1 = [headerRow1, sprintf(' & \\multicolumn{3}{c}{\\textbf{%s}}', groupLabels{g})]; %#ok<AGROW>
    cmidrules  = [cmidrules, sprintf(' \\cmidrule(lr){%d-%d}', col, col+2)]; %#ok<AGROW>
    col = col + 3;
end
fprintf(fid, '%s \\\\\n', headerRow1);
fprintf(fid, '%s\n', cmidrules);

headerRow2 = 'Mesh';
for g = 1:nGroups
    headerRow2 = [headerRow2, ' & $t_{\mathrm{iter}}$ & $n_{\mathrm{iter}}$ & $T$ (s)']; %#ok<AGROW>
end
fprintf(fid, '%s \\\\\n', headerRow2);
fprintf(fid, '\\midrule\n');

for r = 1:nRes
    rowStr = sprintf('%d$\\times$%d', resolutions(r,1), resolutions(r,2));
    for g = 1:nGroups
        if isnan(tTotal_all(r,g))
            rowStr = [rowStr, ' & N/A & N/A & N/A']; %#ok<AGROW>
        else
            rowStr = [rowStr, sprintf(' & %.2f & %s & %.1f', ...
                tIter_all(r,g), nIterStr_all{r,g}, tTotal_all(r,g))]; %#ok<AGROW>
        end
    end
    fprintf(fid, '%s \\\\\n', rowStr);
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');
end
