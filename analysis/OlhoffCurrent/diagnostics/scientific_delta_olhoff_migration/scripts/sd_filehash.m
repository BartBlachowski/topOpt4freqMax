function h = sd_filehash(f)
%SD_FILEHASH  SHA-256 of a file's bytes (streamed).
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(f, 'r'); assert(fid > 0, 'sd:hash', 'cannot open %s', f);
c = onCleanup(@() fclose(fid));
while true
    b = fread(fid, 2^24, '*uint8');
    if isempty(b), break; end
    md.update(b);
end
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end
