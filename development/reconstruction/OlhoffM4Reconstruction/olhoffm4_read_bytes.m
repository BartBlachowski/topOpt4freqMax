function bytes = olhoffm4_read_bytes(filePath)
%OLHOFFM4_READ_BYTES  Read a file as raw uint8, with no encoding translation.
%
%   Used wherever a byte-exact comparison is the point: hashing, and rebuilding
%   the declared patch.  fileread would apply the platform encoding and could
%   silently change what is being compared.

fid = fopen(filePath, 'r');
if fid < 0
    error('olhoffm4_read_bytes:CannotOpen', 'Cannot open %s', filePath);
end
c = onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes = fread(fid, Inf, '*uint8');
end
