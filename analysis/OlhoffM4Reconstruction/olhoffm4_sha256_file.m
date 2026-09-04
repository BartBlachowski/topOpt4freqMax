function h = olhoffm4_sha256_file(filePath)
%OLHOFFM4_SHA256_FILE  Lowercase hex SHA-256 of a file's RAW bytes.
%
%   Matches `shasum -a 256` exactly, so a hash recorded by the import tooling
%   and a hash checked from MATLAB are the same string.  (This is deliberately
%   NOT sha256_hex, which prefixes a class tag and hashes MATLAB values, not
%   files.)

fid = fopen(filePath, 'r');
if fid < 0
    error('olhoffm4_sha256_file:CannotOpen', 'Cannot open %s', filePath);
end
c = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');

md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes); md.update(bytes); end
digest = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(digest, 2).', 1, []));
end
