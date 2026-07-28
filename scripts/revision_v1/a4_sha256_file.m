function sha = a4_sha256_file(path)
%A4_SHA256_FILE  SHA-256 of exact file bytes.

if isstring(path), path = char(path); end
if ~ischar(path) || ~isfile(path)
    error('a4:Sha256FileMissing', 'Cannot hash missing file: %s', string(path));
end

fid = fopen(path, 'rb');
if fid < 0
    error('a4:Sha256ReadFailed', 'Cannot open file for SHA-256: %s', path);
end
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes = fread(fid, Inf, '*uint8');

digest = java.security.MessageDigest.getInstance('SHA-256');
digest.update(bytes);
raw = typecast(digest.digest(), 'uint8');
sha = lower(reshape(dec2hex(raw, 2).', 1, []));
end
