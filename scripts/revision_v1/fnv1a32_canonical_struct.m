function h = fnv1a32_canonical_struct(s)
%FNV1A32_CANONICAL_STRUCT  Hash a recursively field-sorted MATLAB value.
%
%   This preserves the campaign runner's documented canonical JSON hashing
%   convention while using correct wrapping FNV-1a arithmetic.

ordered = localOrderStruct(s);
try
    txt = jsonencode(ordered, PrettyPrint=true);
catch
    txt = jsonencode(ordered);
end
txt = [txt newline];
hash = fnv1a32_bytes(uint8(txt));
h = sprintf('fnv1a32_%08x', hash);
end

function out = localOrderStruct(in)
if isstruct(in)
    out = in;
    if isscalar(in)
        f = sort(fieldnames(in));
        out = struct();
        for k = 1:numel(f)
            out.(f{k}) = localOrderStruct(in.(f{k}));
        end
    else
        for k = 1:numel(in)
            out(k) = localOrderStruct(in(k));
        end
    end
elseif iscell(in)
    out = in;
    for k = 1:numel(in)
        out{k} = localOrderStruct(in{k});
    end
else
    out = in;
end
end
