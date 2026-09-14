function h = fp_hash(v)
%FP_HASH  SHA-256 of the raw little-endian doubles of a vector (same convention
%   as the prior audits' state-identity hashes).
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
