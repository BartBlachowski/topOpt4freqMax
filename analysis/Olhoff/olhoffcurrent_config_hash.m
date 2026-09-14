function h = olhoffcurrent_config_hash(cfg)
%OLHOFFCURRENT_CONFIG_HASH  Stable SHA-256 of an EFFECTIVE configuration.
%
%   Two runs of the same formulation must produce the same hash, so the digest
%   is taken over the SCIENTIFIC content only.  Excluded:
%
%     provenance.resolvedAt   a timestamp; it differs on every resolve
%     runtime.name            a free-text label, never read by the mathematics
%
%   Everything else -- every field the solver can branch on -- is included, in
%   schema order, so the hash answers "was this the same formulation?" and not
%   "was this the same struct literal?".
%
%   See also OLHOFFCURRENT_PROVENANCE, OLH.CONFIG.SCHEMA.

S = olh.config.schema();
lines = cell(size(S,1),1);
for k = 1:size(S,1)
    p = S{k,1};
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
