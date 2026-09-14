function [h, lines] = mig_hash_rows(cfg, rows)
%MIG_HASH_ROWS  olhoffcurrent_config_hash over an EXPLICIT list of schema rows.
%   Character-identical algorithm to analysis/OlhoffCurrent/olhoffcurrent_config_hash.m
%   (both before and after the migration); only the row list is a parameter, so
%   the pre-migration 81-row hash can be recomputed from a post-migration config.
lines = cell(numel(rows),1);
for k = 1:numel(rows)
    p = rows{k};
    if strcmp(p, 'runtime.name'); lines{k} = sprintf('%s=<excluded>', p); continue; end
    lines{k} = sprintf('%s=%s', p, local_show(olh.config.getPath(cfg, p)));
end
joined = strjoin(lines, newline);
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(uint8(joined(:)));
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end

function s = local_show(v)
if ischar(v);            s = v;
elseif isstring(v);      s = char(v);
elseif islogical(v);     s = mat2str(v);
elseif isnumeric(v);     s = mat2str(v, 17);
elseif iscell(v);        s = ['{' strjoin(cellfun(@local_show, v, 'UniformOutput', false), ',') '}'];
elseif isempty(v);       s = '[]';
else,                    s = class(v);
end
end
