function h = cs_filehash(file)
%CS_FILEHASH  SHA-256 of a file's bytes.
fid = fopen(file, 'r'); assert(fid > 0, 'cs_filehash:open', 'cannot open %s', file);
md = java.security.MessageDigest.getInstance('SHA-256');
while true
    b = fread(fid, 2^24, '*uint8');
    if isempty(b), break; end
    md.update(b);
end
fclose(fid);
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end
