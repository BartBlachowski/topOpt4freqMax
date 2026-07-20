function h = a4_hash_file(path)
%A4_HASH_FILE  Deterministic FNV-1a 32-bit hash of the exact file bytes.
%
%   h = A4_HASH_FILE(path) returns 'fnv1a32_XXXXXXXX'. Files are opened in
%   binary mode so the hash is independent of text decoding and newline
%   translation.

if isstring(path), path = char(path); end
if ~ischar(path) || ~isfile(path)
    error('a4_hash_file:FileNotFound', 'Cannot hash missing file: %s', string(path));
end

fid = fopen(path, 'rb');
if fid < 0
    error('a4_hash_file:OpenFailed', 'Cannot open file for hashing: %s', path);
end
cleanup = onCleanup(@() fclose(fid));
bytes = fread(fid, Inf, '*uint8');
hash = fnv1a32_bytes(bytes);
h = sprintf('fnv1a32_%08x', hash);
end
