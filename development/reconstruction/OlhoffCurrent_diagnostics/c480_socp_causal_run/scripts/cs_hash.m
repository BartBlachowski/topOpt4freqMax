function h = cs_hash(v)
%CS_HASH  SHA-256 of a numeric array as little-endian float64 bytes, column
%   major -- the convention of cp_run's local_vecHash and fp_hash.
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(double(v(:)),'uint8'));
d = typecast(md.digest(),'uint8');
h = lower(reshape(dec2hex(d,2).',1,[]));
end
