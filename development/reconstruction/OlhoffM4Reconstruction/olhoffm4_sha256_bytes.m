function h = olhoffm4_sha256_bytes(data)
%OLHOFFM4_SHA256_BYTES  Lowercase hex SHA-256 of raw bytes held in memory.
%
%   Same digest as `shasum -a 256` over a file with those bytes, so an
%   in-memory reconstruction can be compared directly against a hash recorded
%   by the import tooling.  char input is taken as one byte per character,
%   which is what MATLAB's file I/O produces for these ASCII sources.
%
%   (Deliberately NOT sha256_hex, which prefixes a class tag and hashes MATLAB
%   values rather than bytes.)
%
%   See also OLHOFFM4_SHA256_FILE, OLHOFFM4_APPLY_UNIFIED_DIFF.

if ischar(data) || isstring(data)
    data = uint8(char(data));
end
bytes = reshape(uint8(data), [], 1);

md = java.security.MessageDigest.getInstance('SHA-256');
if ~isempty(bytes); md.update(bytes); end
digest = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(digest, 2).', 1, []));
end
