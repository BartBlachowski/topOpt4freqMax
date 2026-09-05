function h = olhoffm4_sha256_file(filePath)
%OLHOFFM4_SHA256_FILE  Lowercase hex SHA-256 of a file's RAW bytes.
%
%   Matches `shasum -a 256` exactly, so a hash recorded by the import tooling
%   and a hash checked from MATLAB are the same string.  (This is deliberately
%   NOT sha256_hex, which prefixes a class tag and hashes MATLAB values, not
%   files.)
%
%   See also OLHOFFM4_SHA256_BYTES.

h = olhoffm4_sha256_bytes(olhoffm4_read_bytes(filePath));
end
