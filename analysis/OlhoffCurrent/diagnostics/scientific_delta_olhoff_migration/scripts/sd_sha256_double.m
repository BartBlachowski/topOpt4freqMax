function h = sd_sha256_double(x)
%SD_SHA256_DOUBLE  SHA-256 of a double array's little-endian IEEE bytes, column order.
b = typecast(double(x(:)).', 'uint8');
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(b);
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end
