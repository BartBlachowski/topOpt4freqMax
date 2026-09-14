function p = paths(s, prefix)
%PATHS  Every leaf path of a nested struct, as a cellstr.
if nargin < 2, prefix = ''; end
p = {};
f = fieldnames(s);
for k = 1:numel(f)
    v = s.(f{k});
    if isempty(prefix), q = f{k}; else, q = [prefix '.' f{k}]; end
    if isstruct(v) && isscalar(v) && ~isempty(fieldnames(v))
        p = [p; olh.config.paths(v, q)];  %#ok<AGROW>
    else
        p = [p; {q}];                     %#ok<AGROW>
    end
end
end
