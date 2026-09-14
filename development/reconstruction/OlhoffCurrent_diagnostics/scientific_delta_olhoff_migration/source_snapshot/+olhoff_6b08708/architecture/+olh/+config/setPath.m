function s = setPath(s, path, value)
%SETPATH  Assign into a nested struct by dotted path, creating branches.
parts = strsplit(path, '.');
s = local_set(s, parts, value);
end

function s = local_set(s, parts, value)
if isscalar(parts)
    s.(parts{1}) = value;
    return
end
if ~isfield(s, parts{1}) || ~isstruct(s.(parts{1}))
    s.(parts{1}) = struct();
end
s.(parts{1}) = local_set(s.(parts{1}), parts(2:end), value);
end
