function print_complexity_fit_table(methodLabels, displayNames, C, expOut, R2, nValid, titleLines, csvPath)
% PRINT_COMPLEXITY_FIT_TABLE  Print a complexity-fit table (Method, C, exp,
% R^2, N pts) and optionally save it as CSV.
%
%   print_complexity_fit_table(methodLabels, displayNames, C, expOut, R2, ...
%       nValid, titleLines, csvPath)
%
% methodLabels : {1 x nMethods} labels used in the console table
% displayNames : {1 x nMethods} short names used in the CSV file
% C, expOut, R2, nValid : [1 x nMethods] as returned by fit_complexity_model
% titleLines   : cell array of strings printed above the table
% csvPath      : (optional) if given, also save the table as CSV there

sepWidth = 131;
sep = repmat('-', 1, sepWidth);
nMethods = numel(methodLabels);

fprintf('\n');
for k = 1:numel(titleLines)
    fprintf('%s\n', titleLines{k});
end
fprintf('\n');
fprintf('%-20s  %14s  %10s  %10s  %8s\n', 'Method', 'C', 'exp', 'R^2', 'N pts');
fprintf('%s\n', sep);

for m = 1:nMethods
    if isnan(expOut(m))
        fprintf('%-20s  %14s  %10s  %10s  %8d\n', methodLabels{m}, 'N/A', 'N/A', 'N/A', nValid(m));
    else
        fprintf('%-20s  %14.4e  %10.3f  %10.3f  %8d\n', ...
            methodLabels{m}, C(m), expOut(m), R2(m), nValid(m));
    end
end
fprintf('%s\n', sep);
fprintf('\n');

if nargin >= 8 && ~isempty(csvPath)
    fid = fopen(csvPath, 'w');
    fprintf(fid, 'Method,C,exp,R2,NPoints\n');
    for m = 1:nMethods
        if isnan(expOut(m))
            fprintf(fid, '%s,,,,%d\n', displayNames{m}, nValid(m));
        else
            fprintf(fid, '%s,%.6e,%.4f,%.4f,%d\n', displayNames{m}, C(m), expOut(m), R2(m), nValid(m));
        end
    end
    fclose(fid);
    fprintf('Complexity fit saved to: %s\n', csvPath);
end
end
