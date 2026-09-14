function L = sd_flatten(s, prefix)
%SD_FLATTEN  Struct -> N x 2 cell {dotted path, value}, leaves only.
if nargin < 2, prefix = ''; end
L = cell(0,2);
f = fieldnames(s);
for i = 1:numel(f)
    p = f{i};
    if ~isempty(prefix), p = [prefix '.' f{i}]; end
    v = s.(f{i});
    if isstruct(v) && isscalar(v)
        L = [L; sd_flatten(v, p)]; %#ok<AGROW>
    else
        L(end+1,:) = {p, v}; %#ok<AGROW>
    end
end
end
