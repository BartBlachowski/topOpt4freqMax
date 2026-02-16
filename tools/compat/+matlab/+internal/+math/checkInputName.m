function tf = checkInputName(name, validNames)
%CHECKINPUTNAME Compatibility fallback for older MATLAB releases.
% This mirrors the internal helper used by newer built-ins to parse
% name-like dimension flags (for example: 'all').

    if isstring(name)
        if ~isscalar(name)
            tf = false;
            return;
        end
        name = char(name);
    end
    if ~(ischar(name) && isrow(name))
        tf = false;
        return;
    end

    if isstring(validNames)
        validNames = cellstr(validNames(:));
    elseif ischar(validNames)
        validNames = {validNames};
    elseif ~iscell(validNames)
        tf = false;
        return;
    end

    name = strtrim(name);
    tf = false;
    for k = 1:numel(validNames)
        item = validNames{k};
        if isstring(item)
            if ~isscalar(item)
                continue;
            end
            item = char(item);
        end
        if ischar(item) && strcmpi(name, strtrim(item))
            tf = true;
            return;
        end
    end
end
