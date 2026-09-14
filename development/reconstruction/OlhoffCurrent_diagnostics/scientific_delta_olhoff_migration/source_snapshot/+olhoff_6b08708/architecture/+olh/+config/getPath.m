function [value, found] = getPath(s, path)
%GETPATH  Read a nested struct by dotted path.  found=false if absent.
parts = strsplit(path, '.');
value = []; found = false;
for k = 1:numel(parts)
    if ~isstruct(s) || ~isfield(s, parts{k}), return; end
    s = s.(parts{k});
end
value = s; found = true;
end
