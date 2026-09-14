function bg_write_rows(rows, csvPath)
%BG_WRITE_ROWS  Write a cell array of scalar-field structs to CSV (vector fields dropped
%   except hist20/hist10/gray_by_row, which go to a sidecar .json).
scalarFields = {};
f0 = fieldnames(rows{1});
for k = 1:numel(f0)
    v = rows{1}.(f0{k});
    if (isnumeric(v) || islogical(v)) && isscalar(v) || ischar(v)
        scalarFields{end+1} = f0{k}; %#ok<AGROW>
    end
end
fid = fopen(csvPath, 'w');
fprintf(fid, '%s\n', strjoin(scalarFields, ','));
for i = 1:numel(rows)
    parts = cell(1, numel(scalarFields));
    for k = 1:numel(scalarFields)
        v = rows{i}.(scalarFields{k});
        if ischar(v), parts{k} = ['"' strrep(v, '"', '''') '"'];
        elseif islogical(v), parts{k} = sprintf('%d', v);
        else, parts{k} = sprintf('%.10g', v); end
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
end
fclose(fid);
% sidecar with the vector fields
J = struct('rows', {cell(1, numel(rows))});
for i = 1:numel(rows)
    s = struct();
    for fn = {'formulation','nelx','nely','hist20','hist10','gray_by_row','gray_by_col','hist20_edges','arm'}
        if isfield(rows{i}, fn{1}), s.(fn{1}) = rows{i}.(fn{1}); end
    end
    J.rows{i} = s;
end
[p, n] = fileparts(csvPath);
fid = fopen(fullfile(p, [n '_vectors.json']), 'w'); fwrite(fid, jsonencode(J)); fclose(fid);
end
