function digest = sha256_file(path)
%SHA256_FILE Return a lowercase SHA-256 digest for a file.
fid = fopen(path, 'rb');
assert(fid >= 0, 'ie2a:MissingFile', 'Cannot open required file: %s', path);
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(bytes);
digest = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end
