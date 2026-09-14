function d = anchorDigest(rec)
%ANCHORDIGEST  Order-independent SHA-256 over the exact bytes of an anchor.
%
%   Numeric content is hashed by its IEEE-754 bytes (typecast), so the digest
%   changes if any double changes in its last bit.  Field names are sorted, so
%   the digest does not depend on struct field order.

md = java.security.MessageDigest.getInstance('SHA-256');
feed(md, rec);
d = lower(reshape(dec2hex(typecast(md.digest(),'uint8')).',1,[]));
end

function feed(md, v)
if isstruct(v)
    if numel(v) ~= 1
        for i = 1:numel(v), feed(md, v(i)); end
        return
    end
    f = sort(fieldnames(v));
    for k = 1:numel(f)
        md.update(unicode2native(f{k},'UTF-8'));
        feed(md, v.(f{k}));
    end
elseif iscell(v)
    md.update(unicode2native(sprintf('cell%d',numel(v)),'UTF-8'));
    for k = 1:numel(v), feed(md, v{k}); end
elseif ischar(v)
    md.update(unicode2native(['char:' v],'UTF-8'));
elseif islogical(v)
    md.update(typecast(uint8(v(:)),'uint8'));
elseif isnumeric(v)
    md.update(unicode2native(sprintf('num%s%dx%d',class(v),size(v,1),size(v,2)),'UTF-8'));
    if ~isempty(v)
        md.update(typecast(double(v(:)),'uint8'));
    end
else
    error('anchorDigest:type','unhashable class %s',class(v));
end
end
