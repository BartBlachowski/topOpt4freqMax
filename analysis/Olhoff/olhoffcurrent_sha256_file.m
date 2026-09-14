function h = olhoffcurrent_sha256_file(filePath)
%OLHOFFCURRENT_SHA256_FILE  Lowercase hex SHA-256 of a file's RAW bytes.
%
%   Matches `shasum -a 256` exactly, so a hash written by tooling outside
%   MATLAB and a hash checked from inside it are the same string.
%
%   (Deliberately NOT sha256_hex, which prefixes a class tag and hashes MATLAB
%   values rather than file bytes.)

fid = fopen(filePath, 'r', 'n');
if fid < 0
    error('olhoffcurrent_sha256_file:Unreadable', 'Cannot read %s', filePath);
end
c = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');

md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes); md.update(bytes); end
digest = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(digest, 2).', 1, []));
end
